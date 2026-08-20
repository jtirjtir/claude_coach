#!/usr/bin/env python3
"""
Virtual Coaching Bot
- Daily 6am AEST training briefing via Telegram
- Post-workout analysis after each Intervals.icu upload
"""

import os
import asyncio
import json
import re
import time
import base64
import logging
import logging.handlers
import shutil
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Load .env from same directory as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

import transcript_db

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.handlers.RotatingFileHandler(
            os.path.join(os.path.dirname(__file__), "coach_bot.log"),
            maxBytes=5 * 1024 * 1024,
            backupCount=3,
        ),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# The Anthropic/OpenAI SDKs log every HTTP request at INFO via httpx/httpcore —
# pure noise here; drop to WARNING so coach_bot.log stays about coaching events.
for _noisy_logger in ("httpx", "httpcore", "openai", "anthropic"):
    logging.getLogger(_noisy_logger).setLevel(logging.WARNING)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN        = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID      = os.environ["TELEGRAM_CHAT_ID"]
INTERVALS_API_KEY     = os.environ["INTERVALS_API_KEY"]
INTERVALS_ATHLETE_ID  = os.environ["INTERVALS_ATHLETE_ID"]

STRAVA_CLIENT_ID     = os.environ.get("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN = os.environ.get("STRAVA_REFRESH_TOKEN", "")
STRAVA_ENABLED       = all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN])

# Garmin Connect — currently *only* used by the read-only freshness probe (see
# probe_garmin_freshness). Nothing the athlete sees is sourced from Garmin yet;
# the probe exists to measure whether Garmin Connect has last night's sleep/HRV
# at 05:00 when Intervals.icu demonstrably does not.
GARMIN_EMAIL      = os.environ.get("GARMIN_EMAIL", "")
GARMIN_PASSWORD   = os.environ.get("GARMIN_PASSWORD", "")
GARMIN_TOKENSTORE = os.environ.get("GARMIN_TOKENSTORE", "~/.garminconnect")
GARMIN_ENABLED    = all([GARMIN_EMAIL, GARMIN_PASSWORD])

AEST           = ZoneInfo("Australia/Sydney")
STATE_FILE     = os.path.join(os.path.dirname(__file__), "state.json")
MCP_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "intervals-mcp-server")
# uv lives in different places depending on install (pipx user-local, /usr/local,
# minimal systemd PATH). Explicit override wins, then PATH lookup, then the
# running user's ~/.local/bin — the last one covers systemd services whose
# PATH doesn't include user-local dirs, for whichever user the unit runs as.
UV_PATH = (
    os.environ.get("UV_PATH")
    or shutil.which("uv")
    or os.path.expanduser("~/.local/bin/uv")
)

BRIEFING_MAX_ATTEMPTS = 5    # per-day cap on briefing generation/send retries
ANALYSIS_MAX_ATTEMPTS = 3    # per-activity cap on post-workout analysis retries
ANALYSIS_RETRY_BACKOFF_SECS = 30 * 60
KOM_CHECK_HOUR = 18          # AEST hour the daily KOM check fires at
KOM_CHECK_BATCH = 50         # starred segments checked per daily run (window rotates)
ACTIVITY_SCAN_LIMIT = 10     # activities examined per poll for unseen uploads


def _env_int(name: str, default: int) -> int:
    """Env override that tolerates blank/garbage values — a typo'd knob in .env
    must not stop the bot booting."""
    raw = (os.environ.get(name) or "").strip()
    try:
        return int(raw)
    except ValueError:
        if raw:
            log.warning(f"{name}={raw!r} is not an integer — using {default}")
        return default


def _env_bool(name: str, default: bool) -> bool:
    """Boolean env override. Anything in the falsey set turns a feature off;
    anything else non-blank turns it on."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    return raw not in ("0", "false", "no", "off")

# ── Brief activity summary (Intervals.icu comment / Strava description) ───────
# Answer to "does an Intervals.icu comment reach Strava?": no. Intervals.icu
# calls activity comments "messages" (POST /activity/{id}/messages) and they
# stay inside Intervals.icu — the linked Strava activity's comment list and
# description are untouched. The only text Intervals.icu ever pushes to Strava
# is an edit to an activity's *name/description*, and only for activities it
# imported *from* Strava; this athlete's arrive from Garmin Connect (source=
# GARMIN_CONNECT), so even that path would never fire. Hence the Strava mirror.
COMMENT_SYNCS_TO_STRAVA = False

POST_ACTIVITY_COMMENT = _env_bool("POST_ACTIVITY_COMMENT", True)
# Mirror the same text into the Strava activity description. Strava has no
# public write endpoint for comments (POST /activities/{id}/comments is not in
# the v3 API and answers 401 whatever the scopes), so the description is the
# only free-text field an app can set — and it needs the activity:write scope.
STRAVA_MIRROR_SUMMARY = _env_bool("STRAVA_MIRROR_SUMMARY", True)
# Off by default: writing the description REPLACES whatever is there, so we
# only touch an empty one unless the athlete opts in to overwriting.
STRAVA_OVERWRITE_DESCRIPTION = _env_bool("STRAVA_OVERWRITE_DESCRIPTION", False)
ICU_COMMENT_MAX_CHARS = 900  # keep the comment glanceable, not a second essay


# ── Conversation memory ───────────────────────────────────────────────────────
# Prior chat turns replayed into each new question, so follow-ups like "and what
# about tomorrow?" resolve. Read back from the transcript DB (the same rows
# check_transcripts.py inspects), so memory survives restarts — but it is only
# as available as Postgres: with TRANSCRIPT_DB_URL unset, every turn is a fresh
# conversation exactly as before.
#
# Only the question and final reply of each turn are replayed, never tool output
# or images. The window is short by design: a coaching question from three days
# ago is rarely the context for today's, and stale context reads as the bot
# misremembering.
MEMORY_TURNS        = _env_int("MEMORY_TURNS", 6)          # turn pairs replayed
MEMORY_WINDOW_HOURS = _env_int("MEMORY_WINDOW_HOURS", 24)  # how far back to look
MEMORY_MAX_CHARS    = _env_int("MEMORY_MAX_CHARS", 2000)   # per replayed message

# Fixed AEST (hour, minute) slots the Strava fallback activity check fires at —
# typical post-workout windows. Intervals.icu is the primary source and polls
# every 5 min; Strava only exists to catch activities that synced there first,
# so polling it on the same cadence burns ~290 API calls/day for nothing.
STRAVA_FALLBACK_SLOTS = [
    (7, 0), (7, 15), (7, 45), (8, 0), (8, 15), (8, 30),
    (9, 0), (10, 0), (10, 30), (11, 30), (12, 15),
]

# ── Training goal ─────────────────────────────────────────────────────────────
# Every date-relative behaviour hangs off the goal event: the countdown in each
# prompt, the race-protection window that keeps the taper untouched, and the
# sport the coach speaks in. The goal lives in state.json so /goal can change it
# at runtime, and is mirrored into a module-level cache (see get_goal) so the
# many functions that need it don't all have to thread `state` through.

GOAL_SPORTS = ("run", "ride", "swim", "triathlon", "other")

# What the coach calls itself, per sport — a cycling goal shouldn't be briefed
# by "an expert running coach".
COACH_PERSONA = {
    "run":       "expert running coach",
    "ride":      "expert cycling coach",
    "swim":      "expert swimming coach",
    "triathlon": "expert triathlon coach",
    "other":     "expert endurance coach",
}

SPORT_NOUN = {
    "run":       "run",
    "ride":      "ride",
    "swim":      "swim",
    "triathlon": "triathlon",
    "other":     "event",
}

# Free text -> canonical sport. Keys are slugs (see _slug), so "Gran Fondo"
# arrives here as "granfondo".
SPORT_ALIASES = {
    "run": "run", "runs": "run", "running": "run", "runner": "run",
    "road": "run", "roadrace": "run", "trail": "run", "ultra": "run",
    "marathon": "run", "halfmarathon": "run", "parkrun": "run",
    "ride": "ride", "rides": "ride", "riding": "ride", "bike": "ride",
    "biking": "ride", "cycle": "ride", "cycling": "ride", "cyclist": "ride",
    "granfondo": "ride", "fondo": "ride", "gravel": "ride", "mtb": "ride",
    "swim": "swim", "swims": "swim", "swimming": "swim", "ows": "swim",
    "tri": "triathlon", "triathlon": "triathlon", "ironman": "triathlon",
    "duathlon": "triathlon", "aquabike": "triathlon", "multisport": "triathlon",
}

# Events the athlete can name directly ("/goal bowral"). `date` is a default,
# not a fact — "/goal bowral 2027-10-24" overrides it, and the generic distance
# presets carry no date at all, so they always require one. city2surf keeps the
# date this bot was originally hardcoded to, so nothing about the current plan
# shifts underneath the athlete; once it passes, /goal is how they move on.
GOAL_PRESETS = {
    "city2surf": {
        "name": "City2Surf Sydney",
        "sport": "run",
        "distance_km": 14,
        "date": "2026-08-09",
        "location": "Sydney, Australia",
        "challenge": "Heartbreak Hill at ~10km — a steep 1.8km climb that breaks most runners",
        "focus": "strong finish time, negative split strategy, surviving the hill with energy to sprint into Bondi",
    },
    "msgong": {
        "name": "MS Sydney to the Gong Ride",
        "sport": "ride",
        "distance_km": 82,
        "date": "2026-11-01",
        "location": "Sydney to Wollongong, NSW, Australia",
        "challenge": "the Royal National Park climbs through the middle third, then holding bunch pace down Grand Pacific Drive",
        "focus": "sustained aerobic endurance, riding safely and efficiently in a large bunch, and fuelling across 3-4 hours",
    },
    "bowral": {
        "name": "Bowral Classic (Gran Fondo)",
        "sport": "ride",
        "distance_km": 160,
        "date": "2026-10-25",
        "location": "Southern Highlands, NSW, Australia",
        "challenge": "~2,000m of climbing with Kangaroo Valley and the Range Road ascent coming late, on tired legs",
        "focus": "repeatable climbing, threshold power over 10-20 minute efforts, and nutrition across 5+ hours",
    },
    "marathon": {
        "name": "Marathon",
        "sport": "run",
        "distance_km": 42.2,
        "challenge": "the last 10km — holding form and pace once glycogen runs low",
        "focus": "aerobic durability, marathon-pace specificity, and a fuelling plan rehearsed in training",
    },
    "halfmarathon": {
        "name": "Half Marathon",
        "sport": "run",
        "distance_km": 21.1,
        "challenge": "holding threshold-adjacent pace for 90+ minutes without fading in the final 5km",
        "focus": "threshold work, race-pace volume, and controlled early pacing",
    },
    "10k": {
        "name": "10km Race",
        "sport": "run",
        "distance_km": 10,
        "challenge": "sitting just above threshold for the full distance without blowing up before 8km",
        "focus": "threshold and VO2 work, plus the discipline to pace the first 2km honestly",
    },
    "5k": {
        "name": "5km Race",
        "sport": "run",
        "distance_km": 5,
        "challenge": "the third kilometre, where the effort stops feeling sustainable",
        "focus": "VO2 max intervals, running economy, and a strong finishing kick",
    },
    "ironman": {
        "name": "Ironman (full distance triathlon)",
        "sport": "triathlon",
        "distance_km": 226,
        "challenge": "3.8km swim, 180km bike and a 42.2km run — pacing the bike so the marathon is still runnable",
        "focus": "aerobic volume across all three disciplines, brick sessions, and a fuelling plan that survives 10+ hours",
    },
    "703": {
        "name": "Ironman 70.3 (half distance triathlon)",
        "sport": "triathlon",
        "distance_km": 113,
        "challenge": "1.9km swim, 90km bike and a 21.1km run — holding sub-threshold on the bike so the half marathon holds together",
        "focus": "sustained sub-threshold work, brick sessions, and race-day nutrition over 4-6 hours",
    },
    "olympictri": {
        "name": "Olympic Distance Triathlon",
        "sport": "triathlon",
        "distance_km": 51.5,
        "challenge": "1.5km swim, 40km bike and a 10km run held close to threshold throughout, with fast transitions",
        "focus": "threshold work in all three disciplines, brick running, and transition practice",
    },
    "sprinttri": {
        "name": "Sprint Distance Triathlon",
        "sport": "triathlon",
        "distance_km": 25.75,
        "challenge": "750m swim, 20km bike and a 5km run at near-maximal effort with no room to recover",
        "focus": "high-intensity work in all three disciplines and slick transitions",
    },
}

# Slug -> preset key. Everything the athlete might plausibly type for an event
# whose canonical key isn't what came to mind.
GOAL_ALIASES = {
    "c2s": "city2surf", "citytosurf": "city2surf", "city2surfsydney": "city2surf",
    "gong": "msgong", "sydneytogong": "msgong", "sydney2gong": "msgong",
    "sydneytothegong": "msgong", "gongride": "msgong", "msride": "msgong",
    "bowralclassic": "bowral", "theclassic": "bowral", "granfondo": "bowral",
    "full": "marathon", "fullmarathon": "marathon", "42k": "marathon",
    "half": "halfmarathon", "halfmara": "halfmarathon", "21k": "halfmarathon",
    "10km": "10k", "5km": "5k",
    "im": "ironman", "fullironman": "ironman", "ironmanfull": "ironman",
    "70": "703", "halfironman": "703", "ironman703": "703", "halfim": "703",
    "olympic": "olympictri", "olympictriathlon": "olympictri", "standard": "olympictri",
    "sprint": "sprinttri", "sprinttriathlon": "sprinttri",
}

# Ordered most- to least-specific: ISO first, so "2026-08-09" is never offered
# to the day-first patterns (which would reject it anyway, but silently).
_GOAL_DATE_FORMATS = (
    "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%d-%m-%Y",
    "%d %b %Y", "%d %B %Y", "%b %d %Y", "%B %d %Y",
)


def _slug(text: str) -> str:
    """Lowercase alphanumerics only: 'City2Surf' -> 'city2surf', '70.3' -> '703'."""
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def parse_goal_date(text: str):
    """Parse an athlete-typed date, or None. Day-first on ambiguity (AU convention)."""
    text = (text or "").strip().strip(",")
    for fmt in _GOAL_DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _fmt_km(value) -> str:
    """'14 km' / '42.2 km' — no trailing '.0' on whole numbers."""
    if not value:
        return ""
    return f"{float(value):g} km"


def _infer_sport(text: str) -> str:
    """Best-guess sport from an event name: 'Sydney Marathon' -> run."""
    for word in str(text or "").split():
        sport = SPORT_ALIASES.get(_slug(word))
        if sport:
            return sport
    return "other"


def normalise_goal(raw) -> dict | None:
    """Coerce a stored or parsed goal into the canonical shape.

    Returns None when the goal lacks either of the two things every downstream
    consumer assumes it has — a name and a parseable date — so callers can fall
    back to DEFAULT_GOAL rather than crash a briefing on a hand-edited
    state.json."""
    if not isinstance(raw, dict):
        return None
    name = str(raw.get("name") or "").strip()
    date = parse_goal_date(str(raw.get("date") or ""))
    if not name or date is None:
        return None
    sport = raw.get("sport")
    sport = SPORT_ALIASES.get(_slug(sport), sport if sport in GOAL_SPORTS else "other")
    try:
        distance = float(raw["distance_km"]) if raw.get("distance_km") else None
    except (TypeError, ValueError):
        distance = None
    return {
        "key":         str(raw.get("key") or _slug(name)),
        "name":        name,
        "sport":       sport,
        "date":        date.strftime("%Y-%m-%d"),
        "distance_km": distance,
        "location":    str(raw.get("location") or "").strip(),
        "challenge":   str(raw.get("challenge") or "").strip(),
        "focus":       str(raw.get("focus") or "").strip(),
    }


def goal_from_preset(name: str, date_str: str | None = None) -> dict | None:
    """Build a goal from a known event name, or None if it isn't one. The
    returned goal may have no date (generic distance presets) — callers must
    check before persisting it."""
    key = _slug(name)
    key = GOAL_ALIASES.get(key, key)
    preset = GOAL_PRESETS.get(key)
    if preset is None:
        return None
    goal = dict(preset, key=key)
    if date_str:
        goal["date"] = date_str
    return goal


DEFAULT_GOAL = normalise_goal(goal_from_preset("city2surf"))

# Active goal, mirrored from state.json by load_goal() at startup and by
# save_goal() on every /goal change. Read it through get_goal().
_active_goal: dict = dict(DEFAULT_GOAL)


def get_goal() -> dict:
    return _active_goal


def set_active_goal(goal) -> dict:
    """Swap in a new active goal, falling back to the default if it's unusable."""
    global _active_goal
    _active_goal = normalise_goal(goal) or dict(DEFAULT_GOAL)
    return _active_goal


def load_goal(state: dict) -> dict:
    """Adopt the goal recorded in state, writing the normalised form back so
    state.json always holds the canonical shape."""
    goal = set_active_goal(state.get("goal"))
    state["goal"] = goal
    return goal


def save_goal(state: dict, goal: dict) -> dict:
    goal = set_active_goal(goal)
    state["goal"] = goal
    save_state(state)
    return goal


def race_date(goal: dict | None = None) -> datetime:
    """The goal event's date as an AEST datetime — the reference point for every
    countdown and for the race-protection window."""
    g = goal or get_goal()
    d = parse_goal_date(g["date"]) or parse_goal_date(DEFAULT_GOAL["date"])
    return datetime(d.year, d.month, d.day, tzinfo=AEST)


def days_to_goal(now: datetime | None = None, goal: dict | None = None) -> int:
    """Whole days from today to goal day. Negative once the event has passed."""
    now = now or datetime.now(AEST)
    return (race_date(goal).date() - now.date()).days


def goal_countdown(now: datetime | None = None, goal: dict | None = None) -> str:
    """One prompt-ready countdown line. Says so plainly once the date is behind
    us, rather than feeding a negative week count into the LLM."""
    g = goal or get_goal()
    days = days_to_goal(now, g)
    if days < 0:
        return (
            f"{g['name']} was {abs(days)} days ago — the goal event has passed and no "
            f"new one is set. Encourage the athlete to set one with /goal."
        )
    if days == 0:
        return f"{g['name']} is TODAY"
    return f"Weeks to {g['name']}: {days // 7} ({days} days)"


def coach_persona(goal: dict | None = None) -> str:
    """'expert cycling coach' — the sport-correct opening of every system prompt."""
    g = goal or get_goal()
    return COACH_PERSONA.get(g["sport"], COACH_PERSONA["other"])


def athlete_context(goal: dict | None = None) -> str:
    """The athlete profile block injected into every Claude prompt."""
    g = goal or get_goal()
    descriptor = ", ".join(
        x for x in (_fmt_km(g.get("distance_km")), SPORT_NOUN.get(g["sport"], "event")) if x
    )
    lines = [
        f"- Goal event: {g['name']}, {race_date(g).strftime('%d %B %Y')} ({descriptor})",
    ]
    if g.get("challenge"):
        lines.append(f"- Key challenge: {g['challenge']}")
    if g.get("focus"):
        lines.append(f"- Training for: {g['focus']}")
    if g.get("location"):
        lines.append(f"- Location: {g['location']}")
    lines.append("- Training platform: Intervals.icu with structured plan already loaded")
    return "\nAthlete profile:\n" + "\n".join(lines) + "\n"


# ── /goal command ─────────────────────────────────────────────────────────────
GOAL_HELP = (
    "*Setting your training goal*\n\n"
    "`/goal` — show the current goal\n"
    "`/goal list` — events I already know\n"
    "`/goal <event>` — e.g. `/goal bowral`, `/goal gong`\n"
    "`/goal <event> <date>` — e.g. `/goal marathon 2027-05-02`\n"
    "`/goal date <date>` — move the current goal's date\n"
    "`/goal reset` — back to the default\n\n"
    "*Anything I don't know:*\n"
    "`/goal <name> | <date> | <sport> | <distance> | <notes>`\n"
    "e.g. `/goal Six Foot Track | 2027-03-13 | run | 45km | brutal descent into Cox's River`\n"
    "Only the name and date are required.\n\n"
    "Dates: `2027-03-13` or `13/03/2027`. "
    f"Sports: {', '.join(GOAL_SPORTS[:-1])}."
)


def _split_date(text: str) -> tuple[str | None, str]:
    """Pull the first date-looking token out of `text`.

    Returns (ISO date or None, text with that token removed). Multi-token forms
    like '13 Mar 2027' are tried first so they aren't half-consumed."""
    words = text.split()
    for size in (3, 1):
        for i in range(len(words) - size + 1):
            found = parse_goal_date(" ".join(words[i:i + size]))
            if found:
                return (
                    found.strftime("%Y-%m-%d"),
                    " ".join(words[:i] + words[i + size:]),
                )
    return None, text


_GOAL_DISTANCE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*k(?:m|ms)?\b", re.IGNORECASE)


def _split_distance(text: str) -> tuple[float | None, str]:
    """Pull a '42.2km' / '90 km' distance out of `text`."""
    m = _GOAL_DISTANCE_RE.search(text)
    if not m:
        return None, text
    return float(m.group(1)), (text[: m.start()] + " " + text[m.end():]).strip()


def _titleish(text: str) -> str:
    """Capitalise a typed event name without flattening deliberate casing:
    'six foot track' -> 'Six Foot Track', but 'MS Gong Ride' is left alone."""
    return " ".join(
        w if any(c.isupper() for c in w) else w.capitalize() for w in text.split()
    )


def _goal_from_fields(arg: str) -> dict | None:
    """Parse the pipe-delimited free-form: name | date | sport | distance | notes."""
    parts = [p.strip() for p in arg.split("|")]
    name = parts[0] if parts else ""
    date = parse_goal_date(parts[1]) if len(parts) > 1 else None
    if not name or date is None:
        return None
    sport_field = parts[2] if len(parts) > 2 else ""
    distance, _ = _split_distance(parts[3]) if len(parts) > 3 else (None, "")
    if distance is None and len(parts) > 3:
        try:
            distance = float(parts[3])
        except ValueError:
            distance = None
    return normalise_goal({
        "name":        _titleish(name),
        "date":        date.strftime("%Y-%m-%d"),
        "sport":       _infer_sport(sport_field) if sport_field else _infer_sport(name),
        "distance_km": distance,
        "challenge":   parts[4] if len(parts) > 4 else "",
    })


def format_goal(goal: dict, today=None) -> str:
    """The athlete-facing summary of a goal, for /goal and every confirmation."""
    today = today or datetime.now(AEST).date()
    date = parse_goal_date(goal["date"])
    days = (date - today).days
    descriptor = " · ".join(
        x for x in (_fmt_km(goal.get("distance_km")), SPORT_NOUN.get(goal["sport"], "")) if x
    )
    lines = [f"🎯 *{goal['name']}*"]
    if descriptor:
        lines.append(descriptor)
    lines.append(f"📅 {date.strftime('%A, %d %B %Y')}")
    if days > 0:
        weeks = days // 7
        lines.append(f"⏳ {days} days away ({weeks} week{'s' if weeks != 1 else ''})")
    elif days == 0:
        lines.append("⏳ *Today.* Go get it.")
    else:
        lines.append(f"⚠️ That date passed {abs(days)} days ago — set a new goal with `/goal`.")
    if goal.get("location"):
        lines.append(f"📍 {goal['location']}")
    if goal.get("challenge"):
        lines.append(f"⛰ {goal['challenge']}")
    return "\n".join(lines)


def format_goal_presets() -> str:
    lines = ["*Events I know* — name any of these directly:\n"]
    for key, preset in GOAL_PRESETS.items():
        date = parse_goal_date(preset.get("date", ""))
        when = date.strftime("%d %b %Y") if date else "date required"
        detail = ", ".join(x for x in (_fmt_km(preset.get("distance_km")), when) if x)
        lines.append(f"`/goal {key}` — {preset['name']} ({detail})")
    lines.append("\nAnything else: `/goal help`")
    return "\n".join(lines)


def parse_goal_command(arg: str, current: dict, today=None) -> tuple[dict | None, str]:
    """Interpret the text following `/goal`.

    Returns (goal, reply). `goal` is a goal to persist, or None when the command
    was informational (show/list/help) or couldn't be understood — `reply` then
    explains why. Deliberately pure: no state, no I/O, so the parsing rules are
    testable without a Telegram or Intervals.icu round trip."""
    today = today or datetime.now(AEST).date()
    arg = (arg or "").strip()

    if not arg:
        return None, format_goal(current, today) + "\n\n_`/goal help` to change it._"

    head, _, rest = arg.partition(" ")
    head, rest = _slug(head), rest.strip()

    if head in ("help", "usage", "h", "?"):
        return None, GOAL_HELP
    if head in ("show", "current", "status"):
        return None, format_goal(current, today)
    if head in ("list", "presets", "events", "options"):
        return None, format_goal_presets()
    if head in ("reset", "default", "clear"):
        return dict(DEFAULT_GOAL), "🎯 Goal reset.\n\n" + format_goal(DEFAULT_GOAL, today)
    if head in ("date", "when", "on", "moveto"):
        date = parse_goal_date(rest)
        if date is None:
            return None, (
                f"⚠️ Couldn't read a date in “{rest}”.\n"
                "Try `/goal date 2026-11-01` or `/goal date 01/11/2026`."
            )
        goal = normalise_goal(dict(current, date=date.strftime("%Y-%m-%d")))
        return goal, "🎯 Goal date updated.\n\n" + format_goal(goal, today)

    if "|" in arg:
        goal = _goal_from_fields(arg)
        if goal is None:
            return None, (
                "⚠️ A free-form goal needs at least a name and a date:\n"
                "`/goal Six Foot Track | 2027-03-13 | run | 45km | brutal river descent`"
            )
        return goal, "🎯 Goal set.\n\n" + format_goal(goal, today)

    date_str, remainder = _split_date(arg)
    remainder = remainder.strip(" -–—,:")

    # Preset lookup runs before any distance is stripped out: "10k" and "70.3"
    # are preset names that _split_distance would otherwise eat.
    distance = None
    goal = goal_from_preset(remainder, date_str)
    if goal is None:
        distance, stripped = _split_distance(remainder)
        stripped = stripped.strip(" -–—,:")
        if distance is not None:
            goal = goal_from_preset(stripped, date_str)
            if stripped:
                remainder = stripped

    if goal is not None:
        if distance is not None:
            goal["distance_km"] = distance
        if not goal.get("date"):
            return None, (
                f"📅 *{goal['name']}* has no fixed date — tell me when yours is:\n"
                f"`/goal {remainder} 2027-05-02`"
            )
        return normalise_goal(goal), "🎯 Goal set.\n\n" + format_goal(normalise_goal(goal), today)

    if not remainder:
        return None, "⚠️ Tell me what the goal is — `/goal bowral`, or `/goal help`."
    if not date_str:
        return None, (
            f"📅 I don't know “{remainder}” — give me a date and I'll take it as a one-off:\n"
            f"`/goal {remainder} 2027-05-02`\n\n"
            "Or `/goal list` for the events I do know."
        )
    goal = normalise_goal({
        "name":        _titleish(remainder),
        "date":        date_str,
        "sport":       _infer_sport(remainder),
        "distance_km": distance,
    })
    return goal, "🎯 Goal set.\n\n" + format_goal(goal, today)


def handle_goal_command(arg: str, state: dict) -> str:
    """Run a `/goal ...` message, persisting the goal when the command sets one."""
    previous = get_goal()
    goal, reply = parse_goal_command(arg, previous)
    if goal is not None:
        save_goal(state, goal)
        log.info(
            f"Training goal changed: {previous['name']} ({previous['date']}) "
            f"→ {goal['name']} ({goal['date']})"
        )
    return reply

# ── LLM provider (model-agnostic) ─────────────────────────────────────────────
def _read_config_file(relative_path: str) -> str:
    with open(os.path.join(os.path.dirname(__file__), relative_path)) as f:
        return f.read().strip()

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()

if LLM_PROVIDER == "azure_foundry":
    from openai import OpenAI

    FOUNDRY_MODEL = os.environ["FOUNDRY_MODEL"]
    _foundry_endpoint = _read_config_file(os.environ.get("FOUNDRY_ENDPOINT_FILE", "foundry_endpoint"))
    _foundry_api_key  = _read_config_file(os.environ.get("FOUNDRY_APIKEY_FILE", "foundry_apikey"))
    # The SDK appends "/responses" itself — strip it if the configured endpoint already has it
    _foundry_base_url = (
        _foundry_endpoint[: -len("/responses")]
        if _foundry_endpoint.endswith("/responses")
        else _foundry_endpoint
    )
    _llm_client = OpenAI(base_url=_foundry_base_url, api_key=_foundry_api_key)
elif LLM_PROVIDER == "anthropic":
    import anthropic

    ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")
    _llm_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
else:
    raise ValueError(f"Unknown LLM_PROVIDER '{LLM_PROVIDER}' — expected 'anthropic' or 'azure_foundry'")

log.info(f"LLM provider: {LLM_PROVIDER}")
transcript_db.ensure_schema()

# ── Telegram ──────────────────────────────────────────────────────────────────
def _log_telegram_error(context: str, exc: Exception) -> None:
    """Log a Telegram API failure WITHOUT the exception message: requests
    exceptions embed the full request URL, which contains the bot token."""
    status = getattr(getattr(exc, "response", None), "status_code", None)
    detail = f"HTTP {status}" if status else type(exc).__name__
    log.error(f"{context}: {detail}")

def _post_send_message(payload: dict) -> None:
    """POST sendMessage. On HTTP 400 — almost always unbalanced Markdown in
    LLM-generated text — retry once without parse_mode so the message still
    lands as plain text instead of being dropped silently."""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    r = requests.post(url, json=payload, timeout=15)
    if r.status_code == 400 and payload.get("parse_mode"):
        log.warning("Telegram 400 with Markdown — retrying as plain text")
        r = requests.post(url, json={k: v for k, v in payload.items() if k != "parse_mode"}, timeout=15)
    r.raise_for_status()

TELEGRAM_MAX_CHARS = 4096  # Telegram's hard limit on a single message's text length

def _chunk_message(text: str, limit: int = TELEGRAM_MAX_CHARS) -> list[str]:
    """Split text into <=limit-char pieces so a long LLM reply doesn't get
    rejected outright by Telegram's per-message cap (previously: HTTP 400,
    logged, and the whole reply silently dropped — even the plain-text retry
    in _post_send_message can't save an over-length message, since removing
    parse_mode doesn't shorten it).

    Breaks on the latest paragraph, then line, then word boundary within the
    limit, so words and (usually) Markdown entities aren't torn mid-token;
    hard-cuts only if a single unbroken run of text exceeds the limit outright."""
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        cut = -1
        for sep in ("\n\n", "\n", " "):
            idx = remaining.rfind(sep, 0, limit)
            if idx > 0:
                cut = idx + len(sep)
                break
        if cut == -1:
            cut = limit
        chunks.append(remaining[:cut].rstrip())
        remaining = remaining[cut:]
    if remaining:
        chunks.append(remaining)
    return chunks

def send_telegram(message: str) -> bool:
    chunks = _chunk_message(message)
    for i, chunk in enumerate(chunks, 1):
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": chunk, "parse_mode": "Markdown"}
        try:
            _post_send_message(payload)
        except Exception as e:
            _log_telegram_error(f"Telegram send failed (part {i}/{len(chunks)})", e)
            return False
    log.info("Telegram sent ✓" if len(chunks) == 1 else f"Telegram sent ✓ ({len(chunks)} parts)")
    return True

def download_telegram_photo(file_id: str) -> tuple[bytes, str] | None:
    """Download a photo or document from Telegram. Returns (bytes, media_type) or None."""
    try:
        r = requests.get(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getFile",
            params={"file_id": file_id},
            timeout=15,
        )
        r.raise_for_status()
        file_path = r.json()["result"]["file_path"]
        img = requests.get(
            f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file_path}",
            timeout=30,
        )
        img.raise_for_status()
        ext = file_path.rsplit(".", 1)[-1].lower()
        media_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg",
                      "png": "image/png", "gif": "image/gif",
                      "webp": "image/webp"}.get(ext, "image/jpeg")
        return img.content, media_type
    except Exception as e:
        _log_telegram_error("Photo download failed", e)
        return None

# ── State ─────────────────────────────────────────────────────────────────────
def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            # A crash mid-write (pre-atomic-save history, or disk issues) can
            # leave state.json truncated/corrupt — don't let that crash the bot
            # on every restart. Fall back to a fresh state; the next save_state
            # overwrites the bad file.
            log.error(f"state.json corrupt/unreadable ({e}) — starting from fresh state")
    return {"last_activity_id": None, "last_briefing_date": None}

def save_state(state: dict):
    """Write state atomically: a torn write (crash/power-loss mid-write) must
    never leave state.json in a half-written, unparseable state."""
    tmp_path = f"{STATE_FILE}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp_path, STATE_FILE)

MAX_TRACKED_IDS = 200  # cap for dedup-guard id lists below, so state.json doesn't grow forever

def _cap_id_list(ids, max_size: int = MAX_TRACKED_IDS) -> list[str]:
    """Bound the size of an accumulating dedup-guard id list. These ids are only
    ever checked against a small rolling lookback/lookahead window (days, not
    months), so once the list exceeds max_size the oldest entries are provably
    irrelevant — which specific ones are dropped doesn't affect correctness."""
    ids = list(ids)
    return ids[-max_size:] if len(ids) > max_size else ids

# ── Intervals.icu API ─────────────────────────────────────────────────────────
def _auth_header() -> dict:
    token = base64.b64encode(f"API_KEY:{INTERVALS_API_KEY}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def intervals_get(path: str):
    try:
        r = requests.get(
            f"https://intervals.icu/api/v1{path}",
            headers=_auth_header(),
            timeout=15,
        )
        if r.status_code == 404:
            log.warning(f"Intervals 404 — no data at {path}")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Intervals API error ({path}): {e}")
        return None

def intervals_put(path: str, data: dict) -> dict | None:
    try:
        r = requests.put(
            f"https://intervals.icu/api/v1{path}",
            headers={**_auth_header(), "Content-Type": "application/json"},
            json=data,
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Intervals PUT error ({path}): {e}")
        return None

def intervals_post(path: str, data: dict) -> dict | None:
    try:
        r = requests.post(
            f"https://intervals.icu/api/v1{path}",
            headers={**_auth_header(), "Content-Type": "application/json"},
            json=data,
            timeout=15,
        )
        r.raise_for_status()
        return r.json() if r.content else {}
    except Exception as e:
        log.error(f"Intervals POST error ({path}): {e}")
        return None

def get_event_for_date(date_str: str) -> dict | None:
    """First non-NOTE/RACE event planned for the given YYYY-MM-DD date."""
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/events?oldest={date_str}&newest={date_str}"
    )
    if isinstance(data, list):
        for ev in data:
            if ev.get("category") not in ("NOTE", "RACE"):
                return ev
    return None

def get_todays_event() -> dict | None:
    return get_event_for_date(datetime.now(AEST).strftime("%Y-%m-%d"))

def wellness_age_days(wellness: dict | None, now: datetime | None = None) -> int | None:
    """Age in days of the *data* in a wellness record — 0 means it's today's.

    Distinct from how long ago the record was fetched: the 05:00 prefetch can
    return a perfectly fresh HTTP response containing a day-old record, which is
    exactly the failure this guards against. Returns None if the record carries
    no usable date.
    """
    record_date = (wellness or {}).get("id")
    if not record_date:
        return None
    try:
        d = datetime.strptime(record_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return ((now or datetime.now(AEST)).date() - d).days

def get_wellness(lookback_days: int = 5) -> dict | None:
    """Most recent wellness record that actually has HRV in it.

    Intervals.icu pre-creates empty records for today and future dates (verified:
    tomorrow's record exists with every biometric field null), so "the newest
    record" is normally an empty shell. Hence the scan back for a populated one.

    That scan is why the briefing can silently run on yesterday's numbers: Garmin
    overnight data lands in Intervals.icu hours after the 05:15 briefing, so at
    05:00 the newest *populated* record is the previous day's. Callers must check
    wellness_age_days() and tell the athlete when they're being shown stale data —
    do not present a returned record as "this morning" without checking.
    """
    today    = datetime.now(AEST).strftime("%Y-%m-%d")
    earliest = (datetime.now(AEST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/wellness?oldest={earliest}&newest={today}"
    )
    if not isinstance(data, list):
        return None

    record = None
    for candidate in reversed(data):
        if candidate and candidate.get("hrv") is not None:
            record = candidate
            break
    else:
        # No HRV anywhere in the window — fall back to the newest non-empty record
        for candidate in reversed(data):
            if candidate:
                record = candidate
                break

    age = wellness_age_days(record)
    if age:  # not None and not 0
        log.warning(
            f"Wellness data is {age} day(s) old (record {record.get('id')}) — "
            f"today's Garmin sync has not reached Intervals.icu yet"
        )
    return record

def get_recent_activities(n: int = 5) -> list:
    today    = datetime.now(AEST).strftime("%Y-%m-%d")
    week_ago = (datetime.now(AEST) - timedelta(days=7)).strftime("%Y-%m-%d")
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/activities?oldest={week_ago}&newest={today}"
    )
    if isinstance(data, list):
        return sorted(data, key=lambda x: x.get("start_date_local", ""), reverse=True)[:n]
    return []

def get_activity_detail(activity_id: str) -> dict | None:
    data = intervals_get(f"/athlete/{INTERVALS_ATHLETE_ID}/activities/{activity_id}")
    if isinstance(data, list):
        return data[0] if data else None
    return data

def get_latest_activity() -> dict | None:
    acts = get_recent_activities(1)
    return acts[0] if acts else None

def get_activity_streams(activity_id: str) -> dict | None:
    data = intervals_get(f"/athlete/{INTERVALS_ATHLETE_ID}/activities/{activity_id}/streams")
    return data if isinstance(data, dict) else None

def get_activity_intervals(activity_id: str) -> dict | None:
    """Fetch Intervals.icu's detected lap/interval breakdown for an activity.

    Unlike per-km GPS-distance splits, this reflects the actual rep/recovery
    structure (e.g. 4x1km + jog recoveries) that Intervals.icu matched against
    the planned workout or the device's lap markers.
    """
    data = intervals_get(f"/activity/{activity_id}/intervals")
    return data if isinstance(data, dict) else None

def get_activity_comments(activity_id: str) -> list:
    """Comments already on an Intervals.icu activity.

    Intervals.icu calls these "messages" — the coach/athlete thread under an
    activity, distinct from the activity's own description field."""
    data = intervals_get(f"/activity/{activity_id}/messages")
    return data if isinstance(data, list) else []


def post_intervals_comment(activity_id: str, content: str) -> bool:
    """Add a comment to an Intervals.icu activity. See COMMENT_SYNCS_TO_STRAVA
    for why this never reaches Strava on its own."""
    result = intervals_post(f"/activity/{activity_id}/messages", {"content": content})
    if result is None:
        log.error(f"Failed to comment on Intervals.icu activity {activity_id}")
        return False
    msg_id = result.get("id") if isinstance(result, dict) else None
    log.info(f"Commented on Intervals.icu activity {activity_id} (message {msg_id})")
    return True


def find_intervals_activity_by_strava_id(strava_id) -> dict | None:
    """Locate the Intervals.icu copy of a Strava activity.

    Needed on the Strava-fallback path, where the id in hand is a Strava one but
    the comment has to land on the Intervals.icu activity. Returns None while
    Intervals.icu has not synced the activity yet."""
    if not strava_id:
        return None
    for act in get_recent_activities(ACTIVITY_SCAN_LIMIT):
        if str(act.get("strava_id") or "") == str(strava_id):
            return act
    return None


def summarize_streams(streams: dict) -> str:
    """Extract coaching-relevant metrics from per-second activity streams."""
    time_arr  = streams.get("time") or []
    hr_arr    = streams.get("heartrate") or []
    spd_arr   = streams.get("speed") or []      # m/s
    alt_arr   = streams.get("altitude") or []
    cad_arr   = streams.get("cadence") or []
    dist_arr  = streams.get("distance") or []   # cumulative metres

    n = len(time_arr)
    if n < 30:
        return ""

    lines = []

    # ── Km splits ─────────────────────────────────────────────────────────────
    if dist_arr and spd_arr:
        km_splits                 = []
        km                        = 1
        bkt_spd, bkt_hr, bkt_cad = [], [], []
        prev_i                    = 0

        for i in range(len(dist_arr)):
            if i < len(spd_arr) and spd_arr[i] > 0:
                bkt_spd.append(spd_arr[i])
            if hr_arr and i < len(hr_arr) and hr_arr[i] > 0:
                bkt_hr.append(hr_arr[i])
            if cad_arr and i < len(cad_arr) and cad_arr[i] > 0:
                bkt_cad.append(cad_arr[i])

            if dist_arr[i] >= km * 1000:
                if bkt_spd:
                    avg_spd  = sum(bkt_spd) / len(bkt_spd)
                    pace_s   = 1000 / avg_spd if avg_spd > 0 else 0
                    hr_str   = f"{round(sum(bkt_hr)/len(bkt_hr))} bpm" if bkt_hr else "N/A"
                    cad_str  = f"{round(sum(bkt_cad)/len(bkt_cad))} spm" if bkt_cad else ""
                    elev_str = ""
                    if alt_arr and i < len(alt_arr):
                        seg  = alt_arr[prev_i:i + 1]
                        gain = sum(max(0, seg[j] - seg[j - 1]) for j in range(1, len(seg)))
                        if gain > 2:
                            elev_str = f" ↑{round(gain)}m"
                    row = f"  km {km}: {_fmt_time(pace_s)}/km | {hr_str}"
                    if cad_str:
                        row += f" | {cad_str}"
                    row += elev_str
                    km_splits.append(row)

                bkt_spd, bkt_hr, bkt_cad = [], [], []
                prev_i = i
                km    += 1

        if km_splits:
            lines.append("*Km splits (pace | avg HR | cadence | elev gain):*\n" + "\n".join(km_splits))

    # ── Pacing shape (negative/positive split) ────────────────────────────────
    mid = n // 2
    if spd_arr:
        h1 = [s for s in spd_arr[:mid] if s > 0]
        h2 = [s for s in spd_arr[mid:] if s > 0]
        if h1 and h2:
            p1   = 1000 / (sum(h1) / len(h1)) / 60
            p2   = 1000 / (sum(h2) / len(h2)) / 60
            diff = p2 - p1  # positive = slower second half
            if diff < -0.05:
                label = "negative split ✅"
            elif diff > 0.08:
                label = "positive split"
            else:
                label = "even split"
            lines.append(
                f"*Pacing:* {label} — "
                f"first half {int(p1)}:{int((p1 % 1) * 60):02d}/km, "
                f"second half {int(p2)}:{int((p2 % 1) * 60):02d}/km"
            )

    # ── HR drift ──────────────────────────────────────────────────────────────
    if hr_arr:
        h1 = [h for h in hr_arr[:mid] if h > 0]
        h2 = [h for h in hr_arr[mid:] if h > 0]
        if h1 and h2:
            hr1, hr2 = sum(h1) / len(h1), sum(h2) / len(h2)
            drift    = hr2 - hr1
            sign     = "+" if drift >= 0 else ""
            lines.append(f"*HR drift:* {round(hr1)} → {round(hr2)} bpm ({sign}{round(drift, 1)} bpm)")

    return "\n\n".join(lines)


def summarize_intervals(intervals_data: dict) -> str:
    """Format Intervals.icu's detected work/recovery reps into a rep-by-rep block.

    This is the authoritative source for verifying structured-workout execution
    (e.g. did the 4x1km reps land near target pace?) — it separates work reps
    from recoveries, which raw per-km GPS-distance splits cannot do.
    """
    icu_intervals = intervals_data.get("icu_intervals") or []
    if not icu_intervals:
        return ""

    lines = []
    for i, iv in enumerate(icu_intervals, 1):
        label      = iv.get("label") or iv.get("type") or f"Interval {i}"
        itype      = iv.get("type", "")
        distance_m = iv.get("distance") or 0
        moving_s   = iv.get("moving_time") or 0
        avg_hr     = iv.get("average_heartrate")
        avg_spd    = iv.get("average_speed")  # m/s

        if avg_spd and avg_spd > 0:
            pace_str = f"{_fmt_time(1000 / avg_spd)}/km"
        elif distance_m and moving_s:
            pace_str = f"{_fmt_time(moving_s / (distance_m / 1000))}/km"
        else:
            pace_str = "N/A"

        dist_str = f"{distance_m / 1000:.2f}km" if distance_m else "N/A"
        hr_str   = f"{round(avg_hr)} bpm" if avg_hr else "N/A"
        time_str = _fmt_time(moving_s) if moving_s else "N/A"

        lines.append(f"  [{i}] {label} ({itype}) — {dist_str} in {time_str} | {pace_str} | {hr_str}")

    return "*Detected intervals (work/recovery reps):*\n" + "\n".join(lines)

# ── Strava API ────────────────────────────────────────────────────────────────
_strava_access_token: str | None = None
_strava_token_expiry: float      = 0.0
# Scopes the refresh token was granted. Strava fixes these at authorisation
# time — a token minted from a refresh token carries exactly what was granted
# then and cannot widen itself, so writing needs a re-authorisation, not a
# retry. Populated by the first _get_strava_token() call.
_strava_scopes: set                = set()

def _get_strava_token() -> str | None:
    global _strava_access_token, _strava_token_expiry, _strava_scopes
    if _strava_access_token and time.time() < _strava_token_expiry - 60:
        return _strava_access_token
    try:
        r = requests.post("https://www.strava.com/oauth/token", json={
            "client_id":     STRAVA_CLIENT_ID,
            "client_secret": STRAVA_CLIENT_SECRET,
            "refresh_token": STRAVA_REFRESH_TOKEN,
            "grant_type":    "refresh_token",
        }, timeout=15)
        r.raise_for_status()
        data = r.json()
        _strava_access_token = data["access_token"]
        _strava_token_expiry = data["expires_at"]
        _strava_scopes       = set((data.get("scope") or "").replace(",", " ").split())
        log.info("Strava token refreshed")
        return _strava_access_token
    except Exception as e:
        log.error(f"Strava token refresh failed: {e}")
        return None

def strava_get(path: str, params: dict | None = None):
    if not STRAVA_ENABLED:
        return None
    token = _get_strava_token()
    if not token:
        return None
    try:
        r = requests.get(
            f"https://www.strava.com/api/v3{path}",
            headers={"Authorization": f"Bearer {token}"},
            params=params or {},
            timeout=15,
        )
        if r.status_code == 403:
            # Friends leaderboard requires Strava Summit — skip silently
            log.warning(f"Strava 403 Forbidden — {path} (Summit or scope restriction)")
            return None
        if r.status_code == 404:
            log.warning(f"Strava 404 Not Found — {path}")
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Strava API error ({path}): {e}")
        return None

STRAVA_WRITE_SCOPE = "activity:write"


def strava_can_write() -> bool:
    """True when the connected Strava token carries activity:write.

    Forces a token fetch the first time, since the granted scope list only
    comes back on the refresh response."""
    if not STRAVA_ENABLED:
        return False
    if not _strava_scopes:
        _get_strava_token()
    return STRAVA_WRITE_SCOPE in _strava_scopes


def strava_put(path: str, data: dict) -> dict | None:
    if not STRAVA_ENABLED:
        return None
    token = _get_strava_token()
    if not token:
        return None
    try:
        r = requests.put(
            f"https://www.strava.com/api/v3{path}",
            headers={"Authorization": f"Bearer {token}"},
            data=data,
            timeout=15,
        )
        if r.status_code in (401, 403):
            log.warning(
                f"Strava {r.status_code} on PUT {path} — the token is missing "
                f"{STRAVA_WRITE_SCOPE}"
            )
            return None
        r.raise_for_status()
        return r.json()
    except Exception as e:
        log.error(f"Strava PUT error ({path}): {e}")
        return None


def update_strava_description(strava_id, text: str) -> bool:
    """Put `text` into a Strava activity's description.

    Strava exposes no way to *post a comment* from an app — POST
    /activities/{id}/comments is not part of the v3 API and answers 401 with
    every scope granted — so the description is the only free-text field an
    app can write, via PUT /activities/{id} with the activity:write scope.

    Because that write REPLACES the description rather than appending to it,
    an activity the athlete has already described is left alone unless
    STRAVA_OVERWRITE_DESCRIPTION is set."""
    if not strava_id:
        return False
    if not strava_can_write():
        log.warning(
            f"Strava description not written for {strava_id}: token scope is "
            f"{sorted(_strava_scopes) or 'unknown'}, needs {STRAVA_WRITE_SCOPE}. "
            "Re-run strava_setup.py to re-authorise."
        )
        return False

    if not STRAVA_OVERWRITE_DESCRIPTION:
        existing = (get_strava_activity_detail(str(strava_id)) or {}).get("description")
        if (existing or "").strip():
            log.info(
                f"Strava activity {strava_id} already has a description — leaving it "
                "(set STRAVA_OVERWRITE_DESCRIPTION=1 to replace)"
            )
            return False

    if strava_put(f"/activities/{strava_id}", {"description": text}) is None:
        return False
    log.info(f"Wrote summary into Strava activity {strava_id} description")
    return True


def get_strava_athlete_id() -> int | None:
    data = strava_get("/athlete")
    return data.get("id") if data else None

def get_latest_strava_activity() -> dict | None:
    acts = strava_get("/athlete/activities", {"per_page": 1})
    return acts[0] if isinstance(acts, list) and acts else None

def get_strava_activity_detail(activity_id: str) -> dict | None:
    return strava_get(f"/activities/{activity_id}", {"include_all_efforts": "true"})

def get_segment_details(segment_id: str) -> dict | None:
    return strava_get(f"/segments/{segment_id}")

def _cached_segment_details(segment_id: str, cache: dict | None) -> dict | None:
    """get_segment_details, memoised in `cache` for the duration of a single
    activity's processing — build_segment_report and get_pr_kom_chases both
    look up the same segments and would otherwise double the API calls."""
    if cache is None:
        return get_segment_details(segment_id)
    if segment_id not in cache:
        cache[segment_id] = get_segment_details(segment_id)
    return cache[segment_id]

def get_segment_leaderboard(segment_id: str, following: bool = True) -> list:
    data = strava_get(
        f"/segments/{segment_id}/leaderboard",
        {"following": "true" if following else "false", "per_page": 10},
    )
    return (data or {}).get("entries", [])

def get_starred_segments() -> list:
    data = strava_get("/segments/starred", {"per_page": 100})
    return data if isinstance(data, list) else []

def _parse_time_str(s: str) -> int | None:
    """Parse '3:24' or '1:23:45' into total seconds."""
    try:
        parts = str(s).strip().split(":")
        if len(parts) == 2:
            return int(parts[0]) * 60 + int(parts[1])
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        pass
    return None

def _fmt_time(secs: int | float) -> str:
    """Format seconds as M:SS."""
    secs = int(abs(secs))
    m, s = divmod(secs, 60)
    return f"{m}:{s:02d}"

def find_strava_match(intervals_activity: dict) -> dict | None:
    """Find the Strava activity that corresponds to an Intervals.icu activity."""
    if not STRAVA_ENABLED:
        return None
    icu_start = intervals_activity.get("start_date_local", "")[:13]  # "2026-06-02T17"
    icu_dist  = intervals_activity.get("distance", 0)
    acts = strava_get("/athlete/activities", {"per_page": 10})
    if not isinstance(acts, list):
        return None
    for act in acts:
        strava_start = act.get("start_date_local", "")[:13]
        strava_dist  = act.get("distance", 0)
        if strava_start == icu_start and abs(strava_dist - icu_dist) < 500:
            return act
    return None

def _activity_signature(act: dict) -> tuple[str, float]:
    """Hour-truncated start time + distance — enough to recognise 'the same run'
    across Intervals.icu and Strava, whose numeric IDs never match each other."""
    return (act.get("start_date_local", "")[:13], act.get("distance", 0) or 0)

def _same_activity(sig_a, sig_b) -> bool:
    if not sig_a or not sig_b:
        return False
    return sig_a[0] == sig_b[0] and abs(sig_a[1] - sig_b[1]) < 500

def build_segment_report(strava_detail: dict, athlete_id: int | None, segment_cache: dict | None = None) -> str:
    """
    Build a segment performance summary for post-workout analysis.
    Covers PR gaps, KOM gaps, and friend leaderboard position.
    Capped at 8 segments (by elapsed time) to stay within rate limits.
    """
    efforts = strava_detail.get("segment_efforts", [])
    if not efforts:
        return ""

    top_efforts = sorted(efforts, key=lambda e: e.get("elapsed_time", 0), reverse=True)[:8]
    lines       = []

    for effort in top_efforts:
        seg      = effort.get("segment", {})
        seg_id   = str(seg.get("id", ""))
        seg_name = effort.get("name") or seg.get("name", "Unknown")
        elapsed  = effort.get("elapsed_time", 0)
        if not seg_id or not elapsed:
            continue

        details = _cached_segment_details(seg_id, segment_cache)
        if not details:
            continue

        pr_secs  = (details.get("athlete_segment_stats") or {}).get("pr_elapsed_time")
        kom_str  = (details.get("xoms") or {}).get("kom")
        kom_secs = _parse_time_str(kom_str) if kom_str else None

        line = f"• *{seg_name}* — {_fmt_time(elapsed)}"

        if pr_secs:
            gap = elapsed - pr_secs
            if gap <= 0:
                line += " 🏅 *PR!*"
            else:
                pct = round(gap / pr_secs * 100, 1)
                line += f" (PR +{_fmt_time(gap)}, {pct}% off)"

        if kom_secs:
            gap = elapsed - kom_secs
            if gap <= 0:
                line += " 🏆 *KOM!*"
            else:
                line += f" | KOM: {_fmt_time(kom_secs)} (-{_fmt_time(gap)})"

        lines.append(line)

    if not lines:
        return ""

    # Friend leaderboard for top 3 segments only (rate limit)
    friend_lines = []
    for effort in top_efforts[:3]:
        seg_id   = str((effort.get("segment") or {}).get("id", ""))
        seg_name = effort.get("name") or (effort.get("segment") or {}).get("name", "")
        if not seg_id:
            continue
        entries = get_segment_leaderboard(seg_id, following=True)
        if not entries:
            continue
        my_entry = next((e for e in entries if e.get("athlete_id") == athlete_id), None)
        if my_entry:
            rank  = my_entry.get("rank", "?")
            total = len(entries)
            friend_lines.append(f"• *{seg_name}*: #{rank} of {total} friends")

    result = "*Segment breakdown:*\n" + "\n".join(lines)
    if friend_lines:
        result += "\n\n*Friends leaderboard:*\n" + "\n".join(friend_lines)
    return result

def get_pr_kom_chases(strava_detail: dict, segment_cache: dict | None = None) -> list[str]:
    """
    Return alert strings for any segment where the athlete was within
    5% of the KOM or within 3% of their own PR — prime opportunities to chase.
    """
    efforts = strava_detail.get("segment_efforts", [])
    alerts  = []
    for effort in efforts[:15]:
        if len(alerts) >= 3:  # cap at 3 to keep messages tidy — stop burning API calls once hit
            break
        seg_id   = str((effort.get("segment") or {}).get("id", ""))
        seg_name = effort.get("name") or (effort.get("segment") or {}).get("name", "")
        elapsed  = effort.get("elapsed_time", 0)
        if not seg_id or not elapsed:
            continue
        details  = _cached_segment_details(seg_id, segment_cache)
        if not details:
            continue
        kom_str  = (details.get("xoms") or {}).get("kom")
        kom_secs = _parse_time_str(kom_str) if kom_str else None
        pr_secs  = (details.get("athlete_segment_stats") or {}).get("pr_elapsed_time")

        if kom_secs and elapsed > kom_secs:
            pct = (elapsed - kom_secs) / kom_secs * 100
            if pct <= 5:
                alerts.append(
                    f"🎯 *{seg_name}*: {_fmt_time(elapsed - kom_secs)} off the KOM "
                    f"({pct:.1f}%) — within striking distance!"
                )
        elif pr_secs and elapsed > pr_secs:
            pct = (elapsed - pr_secs) / pr_secs * 100
            if pct <= 3:
                alerts.append(
                    f"⚡ *{seg_name}*: just {_fmt_time(elapsed - pr_secs)} off your PR ({pct:.1f}%)"
                )

    return alerts

def check_kom_alerts(state: dict) -> list[str]:
    """
    Check starred segments for KOM changes since last run.
    Returns Telegram-ready alert strings. Updates state in-place.
    """
    starred = get_starred_segments()
    if not starred:
        return []

    prev_koms   = state.get("segment_kom_times", {})
    new_koms    = {}
    athlete_id  = state.get("strava_athlete_id")
    alerts      = []

    # get_starred_segments fetches up to 100 but checking all of them daily would
    # cost 100 of Strava's 1000 req/day. Check a window of KOM_CHECK_BATCH and
    # rotate its start by day, so every starred segment is covered within a few
    # days at unchanged API cost. Previously the first 50 were checked and state
    # was then *replaced* with only those, so segments 51+ were evicted and never
    # alerted on again.
    starred_ids = {str(s.get("id", "")) for s in starred if s.get("id")}
    offset = (datetime.now(AEST).timetuple().tm_yday * KOM_CHECK_BATCH) % max(len(starred), 1)
    window = (starred + starred)[offset:offset + KOM_CHECK_BATCH]

    for seg in window:
        seg_id   = str(seg.get("id", ""))
        seg_name = seg.get("name", "Unknown segment")
        if not seg_id:
            continue

        details = get_segment_details(seg_id)
        if not details:
            continue

        kom_str = (details.get("xoms") or {}).get("kom")
        if kom_str:
            new_koms[seg_id] = kom_str

        pr_secs  = (details.get("athlete_segment_stats") or {}).get("pr_elapsed_time")
        kom_secs = _parse_time_str(kom_str) if kom_str else None
        is_kom   = pr_secs and kom_secs and pr_secs <= kom_secs

        prev_kom_str  = prev_koms.get(seg_id)
        prev_kom_secs = _parse_time_str(prev_kom_str) if prev_kom_str else None

        # Only a *faster* KOM time is a change worth reporting, and the two cases
        # are distinguished by whether the athlete's own PR now matches it:
        #   is_kom True  — the new leading time is the athlete's own effort
        #                  (pr_secs <= kom_secs), i.e. they took the KOM.
        #   is_kom False — someone else went faster than the athlete's PR.
        # This relies on pr_secs being current, which holds because `details` is
        # re-fetched from Strava on every run rather than read from state.
        if prev_kom_str and kom_str and kom_str != prev_kom_str and prev_kom_secs:
            if kom_secs and kom_secs < prev_kom_secs:
                if is_kom:
                    alerts.append(f"🏆 You set a new KOM on *{seg_name}*! ({kom_str})")
                else:
                    alerts.append(
                        f"⚠️ Your KOM was beaten on *{seg_name}*!\n"
                        f"New KOM: {kom_str} (was {prev_kom_str})"
                    )

    # Merge rather than replace, so segments outside this run's window keep their
    # recorded times; then drop any segment the athlete has since unstarred so the
    # dict can't grow forever.
    merged = {**prev_koms, **new_koms}
    state["segment_kom_times"] = {
        sid: kom for sid, kom in merged.items() if sid in starred_ids and kom
    }
    return alerts

# ── Garmin Connect freshness probe (measurement only) ─────────────────────────
# Intervals.icu does not have last night's sleep/HRV at 05:15 — measured 11/11
# mornings, the newest populated wellness record is the *previous* day's. Two
# hops could be responsible: watch → Garmin Connect, or Garmin Connect →
# Intervals.icu. Only the second is worth engineering around; if the watch itself
# hasn't synced by 05:00 then a direct Garmin integration buys nothing.
#
# This probe answers that question and nothing else. It never feeds the briefing.
# Once a few mornings of verdicts are logged, decide whether to promote Garmin to
# a real wellness source and then delete this.

def _garmin_client():
    """Authenticated Garmin client, or None if unavailable.

    Imported lazily and behind a broad except: garminconnect is an optional dep
    against an unofficial API, and a probe must never be able to take the bot's
    briefing down.
    """
    if not GARMIN_ENABLED:
        return None
    try:
        from garminconnect import Garmin
    except ImportError:
        log.warning("Garmin probe: garminconnect not installed (pip install garminconnect) — skipping")
        return None
    try:
        client = Garmin(GARMIN_EMAIL, GARMIN_PASSWORD)
        # Token store avoids a full SSO login (and its CAPTCHA risk) on every run;
        # cached OAuth tokens are reused and refreshed in place.
        client.login(tokenstore=os.path.expanduser(GARMIN_TOKENSTORE))
        return client
    except Exception as e:
        log.warning(f"Garmin probe: login failed ({type(e).__name__}: {e}) — skipping")
        return None

def _garmin_overnight(client, date_str: str) -> dict:
    """Sleep/HRV/RHR that Garmin Connect holds for date_str. Missing pieces come
    back as None rather than raising — a partial sync is itself a useful signal."""
    out = {"hrv": None, "rhr": None, "sleep_secs": None, "sleep_score": None}

    try:
        hrv = client.get_hrv_data(date_str) or {}
        summary = hrv.get("hrvSummary") or {}
        out["hrv"] = summary.get("lastNightAvg")
    except Exception as e:
        log.debug(f"Garmin probe: HRV fetch failed: {e}")

    try:
        sleep = client.get_sleep_data(date_str) or {}
        daily = sleep.get("dailySleepDTO") or {}
        out["sleep_secs"] = daily.get("sleepTimeSeconds")
        out["sleep_score"] = ((daily.get("sleepScores") or {}).get("overall") or {}).get("value")
    except Exception as e:
        log.debug(f"Garmin probe: sleep fetch failed: {e}")

    try:
        rhr = client.get_rhr_day(date_str) or {}
        # Shape varies by endpoint version: sometimes a flat restingHeartRate,
        # sometimes nested under allMetrics.metricsMap.
        nested = (rhr.get("allMetrics") or {}).get("metricsMap") or {}
        series = nested.get("WELLNESS_RESTING_HEART_RATE") or [{}]
        out["rhr"] = rhr.get("restingHeartRate") or series[0].get("value")
    except Exception as e:
        log.debug(f"Garmin probe: RHR fetch failed: {e}")

    return out

def probe_garmin_freshness(intervals_record: dict | None, now: datetime) -> None:
    """Log what Garmin Connect has for *today* alongside what Intervals.icu gave us.

    Read-only and best-effort: every failure path degrades to a log line.
    """
    if not GARMIN_ENABLED:
        return

    today = now.strftime("%Y-%m-%d")
    client = _garmin_client()
    if client is None:
        return

    g = _garmin_overnight(client, today)
    garmin_has_today = any(g[k] is not None for k in ("hrv", "sleep_secs", "sleep_score"))

    iv_age = wellness_age_days(intervals_record, now)
    intervals_has_today = iv_age == 0

    def _hrs(secs):
        return f"{secs/3600:.1f}h" if secs else "N/A"

    log.info(f"── Garmin freshness probe @ {now.strftime('%Y-%m-%d %H:%M')} ──")
    log.info(
        f"   intervals.icu: record={(intervals_record or {}).get('id', 'none')} "
        f"({'TODAY' if intervals_has_today else f'{iv_age} day(s) stale' if iv_age is not None else 'undated'}) "
        f"| hrv={(intervals_record or {}).get('hrv')} "
        f"rhr={(intervals_record or {}).get('restingHR')} "
        f"sleep={_hrs((intervals_record or {}).get('sleepSecs'))} "
        f"score={(intervals_record or {}).get('sleepScore')}"
    )
    log.info(
        f"   garmin connect: date={today} "
        f"({'POPULATED' if garmin_has_today else 'EMPTY'}) "
        f"| hrv={g['hrv']} rhr={g['rhr']} "
        f"sleep={_hrs(g['sleep_secs'])} score={g['sleep_score']}"
    )

    if garmin_has_today and not intervals_has_today:
        verdict = ("Garmin HAS today's data, Intervals.icu does NOT — the lag is the "
                   "Garmin→Intervals.icu sync. A direct Garmin integration would fix the briefing.")
    elif not garmin_has_today and not intervals_has_today:
        verdict = ("NEITHER has today's data — the watch itself has not synced by now. "
                   "A direct Garmin integration would NOT help at this hour.")
    elif garmin_has_today and intervals_has_today:
        verdict = "Both have today's data — no staleness to fix at this hour."
    else:
        verdict = "Intervals.icu has today's data but Garmin does not — unexpected; check the probe."
    log.info(f"   => VERDICT: {verdict}")

# ── Missed session detection & rebaseline ────────────────────────────────────
def get_activities_range(oldest: str, newest: str) -> list[dict]:
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/activities?oldest={oldest}&newest={newest}"
    )
    return data if isinstance(data, list) else []

def get_strava_activities_since(days_back: int) -> list[dict]:
    """Strava activities from the trailing `days_back` days, in one range call
    (Strava's activities endpoint takes an epoch 'after' cursor rather than a
    date range) — used to cross-check missed-session detection against sync lag."""
    if not STRAVA_ENABLED:
        return []
    after_epoch = int((datetime.now(AEST) - timedelta(days=days_back)).timestamp())
    data = strava_get("/athlete/activities", {"after": after_epoch, "per_page": 50})
    return data if isinstance(data, list) else []

def detect_missed_sessions(lookback_days: int = 3, processed_ids: set | None = None) -> list[dict]:
    """
    Compare planned events to actual activities for the past N days, via a
    single range fetch per side rather than per-day API calls. Matches events
    to activities per-day by closest duration (greedy) instead of treating any
    activity that day as covering every planned event that day. Before
    declaring an unmatched event missed, cross-checks Strava for that date —
    an activity that's synced to Strava but not yet Intervals.icu isn't a real
    miss, just sync lag. Already-processed event IDs (from state) are skipped
    to avoid re-processing.
    """
    processed_ids = processed_ids or set()
    today  = datetime.now(AEST)
    oldest = (today - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    newest = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    events     = get_events_range(oldest, newest)
    activities = get_activities_range(oldest, newest)
    strava_dates = {
        act.get("start_date_local", "")[:10]
        for act in get_strava_activities_since(lookback_days)
    }

    events_by_date: dict[str, list[dict]] = {}
    for ev in events:
        if ev.get("category") in ("NOTE", "RACE"):
            continue
        d = ev.get("start_date_local", "")[:10]
        if d:
            events_by_date.setdefault(d, []).append(ev)

    activities_by_date: dict[str, list[dict]] = {}
    for act in activities:
        d = act.get("start_date_local", "")[:10]
        if d:
            activities_by_date.setdefault(d, []).append(act)

    missed = []
    for check_date, day_events in events_by_date.items():
        unclaimed = list(activities_by_date.get(check_date, []))

        for ev in day_events:
            ev_id = str(ev.get("id", ""))
            if not ev_id or ev_id in processed_ids:
                continue

            match = None
            if unclaimed:
                target_secs = ev.get("moving_time") or 0
                match = (
                    min(unclaimed, key=lambda a: abs((a.get("moving_time") or 0) - target_secs))
                    if target_secs else unclaimed[0]
                )
                unclaimed.remove(match)

            if match is None and check_date not in strava_dates:
                missed.append({"date": check_date, "event": ev})

    missed.sort(key=lambda m: m["date"], reverse=True)
    return missed

def rebaseline_schedule(missed_sessions: list[dict], state: dict | None = None) -> dict:
    """
    For each missed session, move it to the next free day within the following
    7 days by updating the event via the Intervals.icu API.

    Returns a dict:
      adjustments  — human-readable strings of what changed
      missed_count — total missed sessions found
      resolved_ids — event ids that reached a final outcome (rescheduled, or
                     deliberately kept in plan). Excludes sessions whose PUT
                     failed, so the caller can retry those tomorrow instead of
                     marking them processed forever.

    `state` is threaded in only so a successful reschedule can be persisted
    immediately; the caller makes a slow LLM call before its own save.
    """
    if not missed_sessions:
        return {"adjustments": [], "missed_count": 0, "resolved_ids": set()}

    today = datetime.now(AEST)
    today_str = today.strftime("%Y-%m-%d")
    week_end  = (today + timedelta(days=7)).strftime("%Y-%m-%d")

    # Build the set of days already carrying a workout in the upcoming week
    upcoming_raw = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/events?oldest={today_str}&newest={week_end}"
    ) or []
    occupied_dates: set[str] = set()
    if isinstance(upcoming_raw, list):
        for ev in upcoming_raw:
            if ev.get("category") not in ("NOTE",):
                d = ev.get("start_date_local", "")[:10]
                if d:
                    occupied_dates.add(d)

    adjustments: list[str] = []
    resolved_ids: set[str] = set()

    for missed in missed_sessions:
        date       = missed["date"]
        event      = missed["event"]
        event_id   = event.get("id")
        event_name = event.get("name", "Session")

        if not event_id:
            continue

        # Find the first free day in the next 7 days
        rescheduled  = False
        race_blocked = False
        put_failed   = False
        for days_ahead in range(1, 8):
            candidate_date = today + timedelta(days=days_ahead)
            candidate      = candidate_date.strftime("%Y-%m-%d")
            if candidate in occupied_dates:
                continue

            # Race-week protection (same rule as apply_training_adjustment):
            # never shuffle a session into the RACE_PROTECT_DAYS window before race day
            days_to_race = (race_date().date() - candidate_date.date()).days
            if 0 <= days_to_race <= RACE_PROTECT_DAYS:
                race_blocked = True
                continue

            # Preserve any time component from the original event
            orig_start = event.get("start_date_local", "")
            if "T" in orig_start:
                new_start = f"{candidate}T{orig_start.split('T')[1]}"
            else:
                new_start = candidate

            updated = {**event, "start_date_local": new_start}
            result  = intervals_put(
                f"/athlete/{INTERVALS_ATHLETE_ID}/events/{event_id}", updated
            )
            if result:
                adjustments.append(
                    f"Rescheduled '{event_name}' from {date} → {candidate}"
                )
                occupied_dates.add(candidate)
                rescheduled = True
                resolved_ids.add(str(event_id))
                log.info(f"Rebaseline: moved event {event_id} '{event_name}' {date} → {candidate}")
                # Persist the resolution as soon as the remote write lands — the
                # caller runs an LLM call before saving, and a crash in between
                # would leave the event moved on Intervals.icu but unrecorded.
                if state is not None:
                    state["processed_missed_ids"] = _cap_id_list(
                        set(state.get("processed_missed_ids", [])) | {str(event_id)}
                    )
                    save_state(state)
            else:
                adjustments.append(
                    f"Failed to reschedule '{event_name}' from {date} (API error)"
                )
                put_failed = True
                log.warning(f"Rebaseline: PUT failed for event {event_id}")
            break  # attempt once; move to next missed session regardless

        # An explicit flag, not a substring scan of the last adjustment: two
        # sessions missed on the SAME date used to collide, because the first
        # session's "Rescheduled ... from <date>" line contains this session's
        # date and silently suppressed its "kept in plan" message.
        if not rescheduled and not put_failed:
            reason = "race-week protection" if race_blocked else "no free slot in next 7 days"
            adjustments.append(
                f"Missed '{event_name}' on {date} — kept in plan ({reason})"
            )
            resolved_ids.add(str(event_id))
            log.info(f"Rebaseline: {reason} for '{event_name}' from {date}")

    return {
        "adjustments": adjustments,
        "missed_count": len(missed_sessions),
        "resolved_ids": resolved_ids,
    }

# ── Training-load trend reconciliation & dynamic adjustment ──────────────────
RECONCILE_LOOKBACK_DAYS  = 7
RECONCILE_LOOKAHEAD_DAYS = 7
RACE_PROTECT_DAYS        = 10    # never touch or propose changes to events this close to race day
HRV_SUPPRESSION_PCT      = 7.5   # % below rolling baseline considered suppressed
TSB_REDUCE_THRESHOLD     = -25   # Form (TSB) at/below this = deep fatigue
TSB_PROGRESS_THRESHOLD   = -5    # Form (TSB) at/above this = room to load up
CTL_RAMP_LIMIT           = 8.0   # CTL points/week considered a safe ceiling to progress further
MISSED_RATE_REDUCE       = 0.3   # 30%+ of planned sessions missed -> back off
MISSED_RATE_PROGRESS     = 0.1   # <=10% missed -> adherence supports progressing
MINOR_ADJUST_PCT         = 0.12  # +/-12% volume tweak for auto-applied minor adjustments
MAX_PROPOSALS_PER_RUN    = 2     # cap approval requests sent per briefing
# Sport-neutral: the goal event can be a ride or a triathlon, and a run-only
# keyword list would classify every bike session as filler and quietly
# auto-adjust key work that should have gone out for approval.
KEY_SESSION_KEYWORDS = (
    "tempo", "interval", "threshold", "hill", "race", "long run",
    "time trial", "repeats", "fartlek", "vo2", "speed",
    "long ride", "sweet spot", "sweetspot", "ftp", "climb", "brick",
    "race pace", "swim set", "open water",
)

def get_wellness_series(days_back: int = RECONCILE_LOOKBACK_DAYS + 3) -> list[dict]:
    """Wellness records for the trailing window, oldest first — used for trend analysis
    (as opposed to get_wellness(), which returns a single latest-with-HRV snapshot)."""
    today    = datetime.now(AEST).strftime("%Y-%m-%d")
    earliest = (datetime.now(AEST) - timedelta(days=days_back)).strftime("%Y-%m-%d")
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/wellness?oldest={earliest}&newest={today}"
    )
    return sorted(data, key=lambda r: r.get("id", "")) if isinstance(data, list) else []

def get_events_range(oldest: str, newest: str) -> list[dict]:
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/events?oldest={oldest}&newest={newest}"
    )
    return data if isinstance(data, list) else []

def _sport_of(item: dict) -> str:
    """Normalised sport for an event or activity, for adherence matching."""
    return str(item.get("type") or "").strip().lower()

def count_missed_in_window(days: int = RECONCILE_LOOKBACK_DAYS) -> tuple[int, int]:
    """(missed, planned) session counts over the trailing window — an adherence signal
    for the trend decision, independent of the rebaseline dedup logic above.
    Uses a single range fetch per side instead of per-day API calls, and counts
    per-day shortfall (planned minus actual) rather than treating any activity
    that day as covering every planned event that day.

    Buckets by (date, sport): a planned run is only satisfied by a run. Counting
    per-day totals alone let a 30-minute bike ride satisfy a planned 60-minute
    run, *under*-reporting misses — which feeds an over-stated adherence figure
    into compute_training_trend and can produce a 'progress' signal (load the
    athlete up) during a week they were actually skipping sessions.

    Still coarser than detect_missed_sessions, which additionally matches on
    duration; this is a trend signal, not a per-session verdict.
    """
    today  = datetime.now(AEST)
    oldest = (today - timedelta(days=days)).strftime("%Y-%m-%d")
    newest = (today - timedelta(days=1)).strftime("%Y-%m-%d")

    events     = [ev for ev in get_events_range(oldest, newest) if ev.get("category") not in ("NOTE", "RACE")]
    activities = get_activities_range(oldest, newest)

    activities_by_date: dict[str, list[str]] = {}
    for act in activities:
        d = act.get("start_date_local", "")[:10]
        if d:
            activities_by_date.setdefault(d, []).append(_sport_of(act))

    events_by_date: dict[str, list[str]] = {}
    for ev in events:
        d = ev.get("start_date_local", "")[:10]
        if d:
            events_by_date.setdefault(d, []).append(_sport_of(ev))

    planned = sum(len(v) for v in events_by_date.values())
    missed  = 0
    for d, planned_sports in events_by_date.items():
        available = list(activities_by_date.get(d, []))
        # Typed events first, so a same-sport activity isn't consumed by an
        # untyped event that would have matched anything.
        for sport in sorted(planned_sports, key=lambda s: not s):
            if sport and sport in available:
                available.remove(sport)
            elif not sport and available:
                # Event carries no sport — fall back to the old day-level match
                # rather than declaring it missed on a technicality.
                available.pop(0)
            else:
                missed += 1
    return missed, planned

def compute_training_trend() -> dict:
    """
    Reconcile HRV vs its rolling baseline, sleep trend, fitness (CTL ramp rate),
    Form (TSB), and session adherence into a single signal: 'reduce', 'hold', or
    'progress'. This drives whether the upcoming week's schedule gets dialed back,
    held steady, or nudged up.
    """
    series = get_wellness_series()

    hrv_records = [r for r in series if r.get("hrv") is not None]
    hrv_dev_pct = None
    hrv_date    = None
    if len(hrv_records) >= 4:
        # Deliberately *latest*, not "today's" — at 05:15 the newest populated
        # record is normally yesterday's (see get_wellness). The comparison is
        # still valid, it's just shifted a day, so the date is returned alongside
        # and reported rather than being passed off as this morning's reading.
        latest_hrv = hrv_records[-1]["hrv"]
        hrv_date   = hrv_records[-1].get("id")
        baseline   = sum(r["hrv"] for r in hrv_records[:-1]) / len(hrv_records[:-1])
        if baseline:
            hrv_dev_pct = (latest_hrv - baseline) / baseline * 100

    sleep_records = [r["sleepSecs"] for r in series if r.get("sleepSecs")]
    sleep_dev_pct = None
    if len(sleep_records) >= 5:
        recent_avg  = sum(sleep_records[-3:]) / 3
        overall_avg = sum(sleep_records) / len(sleep_records)
        if overall_avg:
            sleep_dev_pct = (recent_avg - overall_avg) / overall_avg * 100

    # CTL_RAMP_LIMIT is documented as a points-*per-week* ceiling, but `series`
    # spans RECONCILE_LOOKBACK_DAYS+3 days (extra buffer for the HRV/sleep
    # baselines above) — a raw first-vs-last delta over that window would be a
    # ~10-day ramp compared against a 7-day threshold. Normalise to an actual
    # 7-day-equivalent rate using the real gap between the anchor records.
    ctl_records = [(r.get("id", ""), r["ctl"]) for r in series if r.get("ctl") is not None]
    tsb_latest  = next((r.get("tsb") for r in reversed(series) if r.get("tsb") is not None), None)
    ctl_ramp    = None
    if len(ctl_records) >= 2:
        first_date, first_ctl = ctl_records[0]
        last_date, last_ctl   = ctl_records[-1]
        try:
            day_span = (datetime.fromisoformat(last_date) - datetime.fromisoformat(first_date)).days
        except ValueError:
            day_span = None
        if day_span:
            ctl_ramp = (last_ctl - first_ctl) / day_span * 7

    missed, planned = count_missed_in_window()
    missed_rate = (missed / planned) if planned else 0.0

    reduce_triggers = []
    if hrv_dev_pct is not None and hrv_dev_pct <= -HRV_SUPPRESSION_PCT:
        reduce_triggers.append(f"HRV {hrv_dev_pct:.1f}% below {RECONCILE_LOOKBACK_DAYS}-day baseline")
    if tsb_latest is not None and tsb_latest <= TSB_REDUCE_THRESHOLD:
        reduce_triggers.append(f"Form (TSB) at {tsb_latest:.1f} — deep fatigue")
    if missed_rate >= MISSED_RATE_REDUCE:
        reduce_triggers.append(f"{missed}/{planned} planned sessions missed in the last {RECONCILE_LOOKBACK_DAYS} days")
    if sleep_dev_pct is not None and sleep_dev_pct <= -15:
        reduce_triggers.append(f"sleep {sleep_dev_pct:.1f}% below recent average")

    progress_ok = (
        not reduce_triggers
        and tsb_latest is not None and tsb_latest >= TSB_PROGRESS_THRESHOLD
        and (hrv_dev_pct is None or hrv_dev_pct >= -3)
        and (ctl_ramp is None or ctl_ramp <= CTL_RAMP_LIMIT)
        and missed_rate <= MISSED_RATE_PROGRESS
    )

    if reduce_triggers:
        signal, reasons = "reduce", reduce_triggers
    elif progress_ok:
        signal  = "progress"
        reasons = [f"Form (TSB) {tsb_latest:.1f}, HRV/sleep stable, adherence {100 - missed_rate*100:.0f}% — room to progress"]
    else:
        signal  = "hold"
        reasons = ["No strong signal either way — holding current plan"]

    return {
        "signal": signal,
        "reasons": reasons,
        "hrv_dev_pct": hrv_dev_pct,
        "hrv_date": hrv_date,
        "hrv_stale_days": wellness_age_days({"id": hrv_date}) if hrv_date else None,
        "sleep_dev_pct": sleep_dev_pct,
        "tsb": tsb_latest,
        "ctl_ramp": ctl_ramp,
        "missed_rate": missed_rate,
    }

def _is_key_session(event: dict) -> bool:
    text = f"{event.get('name','')} {event.get('description','')}".lower()
    return any(k in text for k in KEY_SESSION_KEYWORDS)

def _new_pending_id(state: dict) -> str:
    state["pending_counter"] = state.get("pending_counter", 0) + 1
    return f"adj{state['pending_counter']}"

def send_telegram_proposal(summary: str, pending_id: str) -> bool:
    """Send a schedule-change proposal with inline Approve/Deny buttons. Nothing is
    written to Intervals.icu until the athlete taps Approve — see handle_callback()."""
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": f"🔧 *Suggested schedule change*\n\n{summary}",
        "parse_mode": "Markdown",
        "reply_markup": {
            "inline_keyboard": [[
                {"text": "✅ Approve", "callback_data": f"approve:{pending_id}"},
                {"text": "❌ Deny",    "callback_data": f"deny:{pending_id}"},
            ]]
        },
    }
    try:
        _post_send_message(payload)
        return True
    except Exception as e:
        _log_telegram_error("Telegram proposal failed", e)
        return False

def _prune_pending_adjustments(state: dict, today_str: str) -> None:
    """Drop proposals whose event date has passed or that are older than 7 days.
    Stale entries can never be sensibly applied, and they permanently block
    re-proposal of that event via the pending_event_ids check below."""
    pending = state.get("pending_adjustments", {})
    if not pending:
        return
    cutoff = (datetime.now(AEST) - timedelta(days=7)).strftime("%Y-%m-%d")
    stale = []
    for pid, p in pending.items():
        ev_date = (p.get("updated_event") or {}).get("start_date_local", "")[:10]
        if (ev_date and ev_date < today_str) or p.get("created", "") < cutoff:
            stale.append(pid)
    for pid in stale:
        log.info(f"Pruning stale pending adjustment {pid}: {pending[pid].get('summary', '')}")
        pending.pop(pid, None)

def apply_training_adjustment(trend: dict, state: dict) -> dict:
    """
    Turn a trend signal into concrete schedule changes over the lookahead window.
    - Non-key (easy/filler) sessions: minor volume tweak, auto-applied straight away.
    - Key sessions (tempo/interval/hill/long run/etc.): proposed via Telegram with
      Approve/Deny buttons — never auto-written.
    - Anything within RACE_PROTECT_DAYS of race day: never touched or proposed, only
      noted so the briefing can mention it.
    """
    signal = trend["signal"]
    applied: list[str]    = []
    proposed: list[str]   = []
    race_notes: list[str] = []

    today     = datetime.now(AEST)
    today_str = today.strftime("%Y-%m-%d")
    _prune_pending_adjustments(state, today_str)

    if signal == "hold":
        return {"applied": applied, "proposed": proposed, "race_notes": race_notes}

    week_end  = (today + timedelta(days=RECONCILE_LOOKAHEAD_DAYS)).strftime("%Y-%m-%d")
    events    = sorted(
        get_events_range(today_str, week_end),
        key=lambda e: e.get("start_date_local", ""),
    )
    events = [e for e in events if e.get("category") not in ("NOTE", "RACE")]

    handled_ids = set(state.get("load_adjusted_ids", []))
    decided_ids = set(state.get("load_decided_ids", []))
    pending     = state.setdefault("pending_adjustments", {})
    pending_event_ids = {p["event_id"] for p in pending.values()}

    minor_done       = False
    proposals_sent   = 0
    direction        = -1 if signal == "reduce" else 1
    pct              = MINOR_ADJUST_PCT * direction

    for event in events:
        event_id = str(event.get("id") or "")
        if (
            not event_id
            or event_id in handled_ids
            or event_id in decided_ids
            or event_id in pending_event_ids
        ):
            continue

        ev_date = event.get("start_date_local", "")[:10]
        days_to_race = (
            (race_date().date() - datetime.strptime(ev_date, "%Y-%m-%d").date()).days
            if ev_date else 999
        )
        race_protected = 0 <= days_to_race <= RACE_PROTECT_DAYS
        is_key = _is_key_session(event)
        name   = event.get("name", "Session")

        if race_protected:
            if signal == "reduce" and is_key:
                race_notes.append(
                    f"{ev_date}: '{name}' is inside race prep — holding as planned; "
                    f"ease pace/effort on the day rather than changing the plan ({trend['reasons'][0]})."
                )
            continue

        updates: dict = {}
        for field in ("moving_time", "distance"):
            val = event.get(field)
            if val:
                updates[field] = round(val * (1 + pct))
        if not updates:
            continue

        note = (
            f"\n\n_Adjusted {('down' if direction < 0 else 'up')} {abs(pct)*100:.0f}% by coaching bot "
            f"— {trend['reasons'][0]}._"
        )
        updated_event = {**event, **updates, "description": (event.get("description") or "") + note}
        summary = f"{ev_date}: '{name}' — {'reduce' if direction < 0 else 'increase'} volume {abs(pct)*100:.0f}%"

        if is_key:
            if proposals_sent >= MAX_PROPOSALS_PER_RUN:
                continue
            pid = _new_pending_id(state)
            pending[pid] = {
                "event_id": event_id,
                "updated_event": updated_event,
                "summary": summary,
                "created": today_str,
            }
            # Persist BEFORE the message goes out. A proposal whose button is
            # live in Telegram but whose state entry was never written would be
            # answered with "This suggestion has expired" — and worse, the
            # unsaved pending_counter would re-mint the same id for a different
            # proposal later, so the stale button would apply the wrong change.
            save_state(state)
            if send_telegram_proposal(summary, pid):
                proposed.append(summary)
                proposals_sent += 1
                log.info(f"Load adjustment proposed (awaiting approval): {summary}")
            else:
                # Roll back: an entry whose proposal never reached the athlete
                # would block re-proposal of this event until it expired.
                pending.pop(pid, None)
                save_state(state)
                log.warning(f"Proposal send failed — rolled back pending entry: {summary}")
        elif not minor_done:
            result = intervals_put(f"/athlete/{INTERVALS_ATHLETE_ID}/events/{event_id}", updated_event)
            if result:
                applied.append(summary)
                handled_ids.add(event_id)
                log.info(f"Load adjustment auto-applied: {summary}")
                # Only touch one non-key session per run to avoid cascading edits.
                # Set only on success: a failed PUT used to consume the single
                # slot for the whole run, so an Intervals blip meant no session
                # was adjusted at all while the briefing reported normally.
                minor_done = True
                # Persist immediately — the caller (generate_briefing) then makes
                # a slow LLM call before its own save, and a crash in that window
                # would replay this adjustment on the next run. The tweak is
                # multiplicative on the current value, so a restart loop would
                # compound it (0.88 -> 0.77 -> 0.68).
                state["load_adjusted_ids"] = _cap_id_list(handled_ids)
                save_state(state)
            else:
                log.warning(f"Load adjustment PUT failed, slot not consumed: {summary}")

    state["load_adjusted_ids"] = _cap_id_list(handled_ids)
    save_state(state)
    return {"applied": applied, "proposed": proposed, "race_notes": race_notes}

# ── LLM helpers ────────────────────────────────────────────────────────────────
class LLMUnavailable(RuntimeError):
    """The LLM call failed. Raised rather than returned as an apology string:
    callers can't distinguish a failure string from a real briefing, so they
    used to send it to the athlete AND mark the day's briefing delivered (or
    dequeue a post-workout analysis), defeating every retry path above."""

def ask_llm(system: str, user: str, max_tokens: int = 1000) -> str:
    try:
        if LLM_PROVIDER == "azure_foundry":
            response = _llm_client.responses.create(
                model=FOUNDRY_MODEL,
                instructions=system,
                input=user,
                max_output_tokens=max_tokens,
            )
            return response.output_text
        else:
            msg = _llm_client.messages.create(
                model=ANTHROPIC_MODEL,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return msg.content[0].text
    except Exception as e:
        log.error(f"LLM API error: {e}")
        raise LLMUnavailable(str(e)) from e

# ── Incoming Telegram messages ─────────────────────────────────────────────────
def get_telegram_updates(offset: int | None = None, timeout: int = 30) -> list:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": timeout, "allowed_updates": ["message", "callback_query"]}
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=timeout + 10)
        r.raise_for_status()
        return r.json().get("result", [])
    except Exception as e:
        _log_telegram_error("Telegram getUpdates error", e)
        time.sleep(30)
        return []

def answer_callback(callback_query_id: str, text: str = "") -> None:
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/answerCallbackQuery",
            json={"callback_query_id": callback_query_id, "text": text},
            timeout=15,
        )
    except Exception as e:
        _log_telegram_error("Telegram answerCallbackQuery error", e)

def handle_callback(callback_query: dict, state: dict) -> None:
    """Handle an Approve/Deny tap on a schedule-change proposal (see send_telegram_proposal)."""
    callback_id = callback_query.get("id", "")
    chat_id = str((callback_query.get("message") or {}).get("chat", {}).get("id", ""))
    if chat_id != str(TELEGRAM_CHAT_ID):
        return

    action, _, pending_id = callback_query.get("data", "").partition(":")
    pending = state.get("pending_adjustments", {}).get(pending_id)
    if not pending:
        answer_callback(callback_id, "This suggestion has expired.")
        return

    # Never apply a proposal whose event date has already passed
    ev_date = (pending.get("updated_event") or {}).get("start_date_local", "")[:10]
    if ev_date and ev_date < datetime.now(AEST).strftime("%Y-%m-%d"):
        state.get("pending_adjustments", {}).pop(pending_id, None)
        answer_callback(callback_id, "Expired — that session date has passed.")
        return

    if action == "approve":
        result = intervals_put(
            f"/athlete/{INTERVALS_ATHLETE_ID}/events/{pending['event_id']}",
            pending["updated_event"],
        )
        if not result:
            # Keep the pending entry so the buttons stay live — the athlete is
            # told to try again, so "again" must actually work.
            answer_callback(callback_id, "Failed — try again")
            send_telegram(f"⚠️ Couldn't update Intervals.icu for: {pending['summary']} — tap Approve to retry")
            return
        answer_callback(callback_id, "Applied ✓")
        send_telegram(f"✅ Applied: {pending['summary']}")
        log.info(f"Load adjustment approved & applied: {pending['summary']}")
    elif action == "deny":
        answer_callback(callback_id, "Dismissed")
        send_telegram(f"👍 Kept as planned: {pending['summary']}")
    else:
        answer_callback(callback_id, "Unrecognized action")
        return  # nothing decided — keep the entry so valid buttons stay live

    # Reached only on a definite decision (approved & applied, or denied)
    decided_ids = set(state.get("load_decided_ids", []))
    decided_ids.add(pending["event_id"])
    state["load_decided_ids"] = _cap_id_list(decided_ids)
    state.get("pending_adjustments", {}).pop(pending_id, None)

STRAVA_TOOLS = [
    {
        "name": "strava_list_activities",
        "description": "List the athlete's recent Strava activities. Returns id, name, sport type, date, distance, moving time, elevation gain, average speed, and HR.",
        "input_schema": {
            "type": "object",
            "properties": {
                "per_page": {"type": "integer", "description": "Number of activities to return (max 30, default 10)"},
                "before":   {"type": "integer", "description": "Unix timestamp — return activities before this time"},
                "after":    {"type": "integer", "description": "Unix timestamp — return activities after this time"},
            },
        },
    },
    {
        "name": "strava_get_activity",
        "description": "Get full detail for a single Strava activity including all segment efforts, laps, best efforts (PR distances), average HR/watts/cadence, and PR achievements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_id": {"type": "string", "description": "Strava activity ID"},
            },
            "required": ["activity_id"],
        },
    },
    {
        "name": "strava_get_segment",
        "description": "Get details for a Strava segment including current KOM time, athlete's personal record time and effort count, distance, elevation, and location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "string", "description": "Strava segment ID"},
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "strava_get_segment_leaderboard",
        "description": "Get the leaderboard for a Strava segment. Use following=true to see only athletes the user follows (friends comparison). Returns rank, name, time, and date for each entry.",
        "input_schema": {
            "type": "object",
            "properties": {
                "segment_id": {"type": "string", "description": "Strava segment ID"},
                "following":  {"type": "boolean", "description": "If true, limit to athletes the user follows (default true)"},
            },
            "required": ["segment_id"],
        },
    },
    {
        "name": "strava_get_starred_segments",
        "description": "List all segments the athlete has starred on Strava — these are segments the athlete cares about, including any they hold KOMs on.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

INTERVALS_STREAM_TOOLS = [
    {
        "name": "intervals_get_streams",
        "description": (
            "Fetch per-second time-series data for an Intervals.icu activity. "
            "Returns km splits (pace, HR, cadence, elevation gain per km), pacing shape "
            "(negative/positive/even split), and HR drift. Use this when the athlete asks "
            "about pacing, effort distribution, HR trends, or how they ran each km."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "activity_id": {"type": "string", "description": "Intervals.icu activity ID"},
            },
            "required": ["activity_id"],
        },
    },
]


def _dispatch_intervals_tool(name: str, inputs: dict) -> str:
    try:
        if name == "intervals_get_streams":
            streams = get_activity_streams(str(inputs["activity_id"]))
            if not streams:
                return "No stream data available for this activity."
            summary = summarize_streams(streams)
            return summary or "Stream data present but too short to summarise."
        return f"Unknown intervals tool: {name}"
    except Exception as e:
        return f"Intervals stream tool error: {e}"


def _dispatch_strava_tool(name: str, inputs: dict) -> str:
    """Execute a Strava tool call and return the result as a JSON string."""
    try:
        if name == "strava_list_activities":
            params = {"per_page": inputs.get("per_page", 10)}
            if inputs.get("before"):
                params["before"] = inputs["before"]
            if inputs.get("after"):
                params["after"] = inputs["after"]
            result = strava_get("/athlete/activities", params)

        elif name == "strava_get_activity":
            result = get_strava_activity_detail(str(inputs["activity_id"]))

        elif name == "strava_get_segment":
            result = get_segment_details(str(inputs["segment_id"]))

        elif name == "strava_get_segment_leaderboard":
            following = inputs.get("following", True)
            result    = get_segment_leaderboard(str(inputs["segment_id"]), following=following)

        elif name == "strava_get_starred_segments":
            result = get_starred_segments()

        else:
            return f"Unknown Strava tool: {name}"

        return json.dumps(result) if result is not None else "No data returned"
    except Exception as e:
        return f"Strava tool error: {e}"


async def _dispatch_tool_call(name: str, inputs: dict, session: ClientSession) -> str:
    """Route a tool call to the Strava dispatcher, the local stream dispatcher, or the MCP session.
    Shared by every LLM provider's tool loop below."""
    if name.startswith("strava_"):
        return _dispatch_strava_tool(name, inputs)
    if name == "intervals_get_streams":
        return _dispatch_intervals_tool(name, inputs)
    try:
        result = await session.call_tool(name, inputs)
        return "\n".join(item.text for item in result.content if hasattr(item, "text")) or "No result"
    except Exception as e:
        return f"Error calling {name}: {e}"


# Cap on tool-call rounds per question — a looping model would otherwise burn
# API calls indefinitely. On hitting the cap we force a final answer with no tools.
MAX_TOOL_ROUNDS = 8


def _history_messages(history: list[dict] | None) -> list[dict]:
    """Flatten stored turns into alternating user/assistant messages.

    One builder serves both providers: {"role": ..., "content": <str>} is valid
    as an Anthropic `messages` entry and as a Foundry Responses `input` item.

    Any turn missing either half is dropped rather than half-replayed — an
    assistant message with no preceding user message breaks Anthropic's
    required alternation and fails the whole call.
    """
    messages = []
    for turn in history or []:
        user      = (turn.get("user") or "").strip()
        assistant = (turn.get("assistant") or "").strip()
        if not user or not assistant:
            continue
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})
    return messages


async def _run_anthropic_tool_loop(
    system: str,
    user_content,
    tool_defs: list[dict],
    session: ClientSession,
    turn_id: int | None = None,
    history: list[dict] | None = None,
) -> tuple[str, dict | None]:
    # Prior turns first, then the question being asked now. Tool-call rounds
    # append to the same list, so the model keeps both across the whole turn.
    messages = _history_messages(history) + [{"role": "user", "content": user_content}]
    sequence = 0
    rounds = 0
    last_response = None

    while True:
        response = _llm_client.messages.create(
            model=ANTHROPIC_MODEL,
            max_tokens=2000,
            system=system,
            tools=tool_defs,
            messages=messages,
        )
        last_response = response

        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text, last_response.model_dump()
            return "Done.", last_response.model_dump()

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    log.info(f"Tool: {block.name} {json.dumps(block.input)[:120]}")
                    content = await _dispatch_tool_call(block.name, block.input, session)
                    sequence += 1
                    transcript_db.log_tool_call(turn_id, sequence, block.name, block.input, content)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": content,
                    })
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

            rounds += 1
            if rounds >= MAX_TOOL_ROUNDS:
                log.warning(f"Tool loop hit {MAX_TOOL_ROUNDS} rounds — forcing final answer")
                final = _llm_client.messages.create(
                    model=ANTHROPIC_MODEL,
                    max_tokens=2000,
                    system=system + "\nYou have reached the data-gathering limit — answer now using only what you already have.",
                    messages=messages,  # no tools → the model must answer
                )
                for block in final.content:
                    if hasattr(block, "text"):
                        return block.text, final.model_dump()
                return "Done.", final.model_dump()
        else:
            break

    # Any other stop_reason (max_tokens, stop_sequence, pause_turn...) still
    # usually carries usable text — max_tokens in particular means the model
    # produced a long answer and ran out of room. Returning the error string and
    # discarding that text loses a complete, useful reply.
    if last_response is not None:
        salvaged = "".join(
            block.text for block in last_response.content if hasattr(block, "text")
        ).strip()
        if salvaged:
            log.warning(f"Salvaged partial reply on stop_reason={last_response.stop_reason}")
            return salvaged, last_response.model_dump()

    return "⚠️ Unexpected response from coach.", (last_response.model_dump() if last_response else None)


def _foundry_tool_defs(tool_defs: list[dict]) -> list[dict]:
    """Convert Anthropic-shaped tool defs (name/description/input_schema) to the
    OpenAI Responses API function-tool shape (name/description/parameters)."""
    return [
        {
            "type": "function",
            "name": t["name"],
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}},
        }
        for t in tool_defs
    ]


async def _run_foundry_tool_loop(
    system: str,
    text: str,
    image_bytes: bytes | None,
    media_type: str,
    tool_defs: list[dict],
    session: ClientSession,
    turn_id: int | None = None,
    history: list[dict] | None = None,
) -> tuple[str, dict | None]:
    openai_tools = _foundry_tool_defs(tool_defs)

    # Prior turns lead the input list; only this first call needs them, as the
    # tool rounds below chain off previous_response_id and carry them forward.
    input_content = _history_messages(history)
    if image_bytes:
        input_content.append({
            "type": "message",
            "role": "user",
            "content": [
                {"type": "input_text", "text": text or "Please analyse this workout image."},
                {
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}",
                },
            ],
        })
    else:
        input_content.append({"role": "user", "content": text})

    response = _llm_client.responses.create(
        model=FOUNDRY_MODEL,
        instructions=system,
        input=input_content,
        tools=openai_tools,
        max_output_tokens=2000,
    )

    sequence = 0
    rounds = 0
    while True:
        function_calls = [item for item in response.output if item.type == "function_call"]
        if not function_calls:
            return response.output_text or "Done.", response.model_dump()

        tool_outputs = []
        for call in function_calls:
            # Malformed JSON here used to raise straight out of the tool loop and
            # be caught by handle_incoming_message's outer except, which answered
            # the athlete with the generic "Coach is unavailable" and dropped the
            # message. Feed the error back instead so the model can correct itself.
            try:
                inputs = json.loads(call.arguments or "{}")
            except json.JSONDecodeError as e:
                log.warning(f"Malformed tool arguments for {call.name}: {e}")
                tool_outputs.append({
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": (
                        f"Error: arguments for {call.name} were not valid JSON "
                        f"({e}). Retry this call with valid JSON arguments."
                    ),
                })
                continue
            log.info(f"Tool: {call.name} {json.dumps(inputs)[:120]}")
            content = await _dispatch_tool_call(call.name, inputs, session)
            sequence += 1
            transcript_db.log_tool_call(turn_id, sequence, call.name, inputs, content)
            tool_outputs.append({
                "type": "function_call_output",
                "call_id": call.call_id,
                "output": content,
            })

        rounds += 1
        if rounds >= MAX_TOOL_ROUNDS:
            log.warning(f"Tool loop hit {MAX_TOOL_ROUNDS} rounds — forcing final answer")
            response = _llm_client.responses.create(
                model=FOUNDRY_MODEL,
                previous_response_id=response.id,
                input=tool_outputs,
                max_output_tokens=2000,  # no tools → the model must answer
            )
        else:
            response = _llm_client.responses.create(
                model=FOUNDRY_MODEL,
                previous_response_id=response.id,
                input=tool_outputs,
                tools=openai_tools,
                max_output_tokens=2000,
            )


async def _handle_message_async(
    text: str,
    system: str,
    image_bytes: bytes | None = None,
    media_type: str = "image/jpeg",
    turn_id: int | None = None,
    history: list[dict] | None = None,
) -> str:
    server_params = StdioServerParameters(
        command=UV_PATH,
        args=["run", "--directory", MCP_SERVER_DIR, "python", "-m", "intervals_mcp_server.server"],
        # Minimal env — passing **os.environ would hand the subprocess every
        # secret in .env; it only needs its own credentials (+ PATH/HOME for uv).
        env={
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/root"),
            "API_KEY": INTERVALS_API_KEY,
            "ATHLETE_ID": INTERVALS_ATHLETE_ID,
        },
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Intervals.icu MCP tools + Strava tools, in Anthropic-shaped tool defs
            tools_result = await session.list_tools()
            tool_defs = [
                {"name": t.name, "description": t.description, "input_schema": t.inputSchema}
                for t in tools_result.tools
            ]
            tool_defs.extend(INTERVALS_STREAM_TOOLS)
            if STRAVA_ENABLED:
                tool_defs.extend(STRAVA_TOOLS)

            if LLM_PROVIDER == "azure_foundry":
                reply, raw_response = await _run_foundry_tool_loop(
                    system, text, image_bytes, media_type, tool_defs, session, turn_id, history
                )
                transcript_db.log_reply(turn_id, reply, raw_response)
                return reply

            # Anthropic — build first message, including image if provided
            if image_bytes:
                user_content = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": base64.b64encode(image_bytes).decode(),
                        },
                    },
                    {"type": "text", "text": text or "Please analyse this workout image."},
                ]
            else:
                user_content = text

            reply, raw_response = await _run_anthropic_tool_loop(
                system, user_content, tool_defs, session, turn_id, history
            )
            transcript_db.log_reply(turn_id, reply, raw_response)
            return reply


def handle_incoming_message(
    text: str,
    image_bytes: bytes | None = None,
    media_type: str = "image/jpeg",
    chat_id: str = "",
    telegram_message_id: int | None = None,
) -> str:
    today_str = datetime.now(AEST).strftime("%A, %d %B %Y")
    goal      = get_goal()

    strava_note = (
        " You also have direct Strava access via strava_* tools — use these for segment analysis, "
        "PR/KOM comparisons, friend leaderboards, and activity history that may not be in Intervals.icu."
        if STRAVA_ENABLED else ""
    )
    image_note = (
        " When given a workout image (Garmin, Strava screenshot, training summary), extract all visible "
        "metrics (pace, HR, power, distance, splits, zones) and provide a detailed coaching analysis. "
        "Cross-reference with Intervals.icu and Strava data via tools when relevant."
        if image_bytes else ""
    )
    system = (
        f"You are an {coach_persona(goal)} preparing this athlete for {goal['name']}. "
        "You have direct access to the athlete's Intervals.icu data via tools — use them to answer questions accurately. "
        "When the athlete asks to modify their plan (add, reschedule, or delete sessions), use the appropriate tools."
        f"{strava_note}{image_note} "
        "Be specific, data-driven, and conversational — like a trusted coach responding to a text. "
        "Format for Telegram: *bold* for key points, emoji sparingly. Under 300 words unless detail is needed.\n\n"
        f"{athlete_context(goal)}\n"
        f"Today: {today_str} | {goal_countdown(goal=goal)}"
    )
    model = FOUNDRY_MODEL if LLM_PROVIDER == "azure_foundry" else ANTHROPIC_MODEL

    # Read memory *before* start_turn() inserts this turn, so the athlete's
    # current question can never come back as part of its own history.
    history = transcript_db.get_recent_turns(
        chat_id,
        limit=MEMORY_TURNS,
        window_hours=MEMORY_WINDOW_HOURS,
        max_chars=MEMORY_MAX_CHARS,
    )
    if history:
        log.info(f"Conversation memory: replaying {len(history)} prior turn(s)")

    turn_id = transcript_db.start_turn(
        chat_id, telegram_message_id, text, image_bytes, media_type, LLM_PROVIDER, model
    )
    try:
        return asyncio.run(
            _handle_message_async(text, system, image_bytes, media_type, turn_id, history)
        )
    except Exception as e:
        log.error(f"MCP handler error: {e}")
        transcript_db.log_reply(turn_id, None, None, error=str(e))
        return "⚠️ Coach is unavailable right now — check back shortly."

# ── Daily Briefing ─────────────────────────────────────────────────────────────
def generate_briefing(state: dict | None = None) -> str:
    today_str = datetime.now(AEST).strftime("%A, %d %B %Y")
    event     = get_todays_event()
    recent    = get_recent_activities(3)

    # Wellness comes from the cache refreshed at 22:00 / 00:01 / 05:00. Note this
    # bounds how long ago the record was *fetched*, not how old the data in it is —
    # all three slots run before Garmin's overnight sync reaches Intervals.icu, so
    # a freshly-fetched record routinely contains yesterday's numbers. That is what
    # wellness_age_days() below measures, and what the athlete gets told about.
    wellness  = None
    cached     = (state or {}).get("cached_wellness")
    fetched_at = (state or {}).get("cached_wellness_fetched_at", "")
    if cached and fetched_at:
        try:
            fetch_age = datetime.now(AEST) - datetime.fromisoformat(fetched_at)
            if fetch_age <= timedelta(hours=24):
                log.info(f"Using cached wellness (fetched {fetched_at}, data date={cached.get('id')})")
                wellness = cached
        except ValueError:
            pass
    if wellness is None:
        log.info("Wellness cache missing/stale — falling back to live fetch")
        wellness = get_wellness()

    wellness_stale_days = wellness_age_days(wellness)
    if wellness_stale_days:
        log.warning(
            f"Briefing is running on {wellness_stale_days}-day-old wellness data "
            f"(record {wellness.get('id')}) — flagging it as stale in the prompt"
        )

    goal = get_goal()
    ctx  = [f"Date: {today_str}", goal_countdown(goal=goal)]

    if wellness:
        hrv        = wellness.get("hrv")
        hrv_sdnn   = wellness.get("hrvSDNN")
        rhr        = wellness.get("restingHR")
        ctl        = wellness.get("ctl")
        atl        = wellness.get("atl")
        tsb        = wellness.get("tsb")
        sleep_secs = wellness.get("sleepSecs")
        sleep_score  = wellness.get("sleepScore")
        sleep_qual   = wellness.get("sleepQuality")
        readiness    = wellness.get("readiness")
        sleep_qual_map = {1: "poor", 2: "fair", 3: "good", 4: "excellent"}
        sleep_hrs  = f"{sleep_secs/3600:.1f} hrs" if sleep_secs else "N/A"

        # The record's own date, always — the numbers below are only "this
        # morning's" when it happens to be today's record, which at 05:15 it
        # usually is not.
        if wellness_stale_days:
            measured_on = datetime.strptime(wellness["id"], "%Y-%m-%d").strftime("%A %d %B")
            wellness_header = (
                f"\nWellness [STALE — measured {measured_on}, "
                f"{wellness_stale_days} day(s) ago; this morning's Garmin data had not "
                f"synced to Intervals.icu by briefing time]:"
            )
        else:
            wellness_header = "\nWellness [measured this morning]:"

        ctx.append(
            wellness_header
            + f"\nHRV {hrv or 'N/A'} ms (SDNN {hrv_sdnn or 'N/A'}) | RHR {rhr or 'N/A'} bpm"
            + (f" | Readiness {readiness}/10" if readiness else "")
            + f"\nSleep: {sleep_hrs}"
            + (f" | Score {sleep_score}/100" if sleep_score else "")
            + (f" | Quality {sleep_qual_map.get(sleep_qual, 'N/A')}" if sleep_qual else "")
            + f"\nForm scores: Fitness (CTL) {round(ctl,1) if ctl else 'N/A'} | "
            f"Fatigue (ATL) {round(atl,1) if atl else 'N/A'} | "
            f"Form (TSB) {round(tsb,1) if tsb else 'N/A'}"
        )

    if event:
        name  = event.get("name", "Run")
        desc  = event.get("description", "No description")
        dur_s = event.get("moving_time")
        dist  = event.get("distance")
        dur_str  = f"{dur_s//60} min" if dur_s else ""
        dist_str = f"{round(dist/1000,1)} km" if dist else ""
        ctx.append(f"\nToday's plan: {name} {dist_str} {dur_str}\n{desc}")
    else:
        ctx.append("\nNo session scheduled — rest or easy recovery day.")

    if recent:
        ctx.append("\nRecent sessions:")
        for a in recent:
            d    = a.get("start_date_local","")[:10]
            nm   = a.get("name","Run")
            km   = f"{round(a.get('distance',0)/1000,1)} km" if a.get("distance") else ""
            mins = f"{a.get('moving_time',0)//60} min" if a.get("moving_time") else ""
            load = a.get("training_load")
            ctx.append(f"  • {d}: {nm} {km} {mins}" + (f" | Load {round(load)}" if load else ""))

    # ── Missed session detection & rebaseline ─────────────────────────────────
    missed_block = ""
    if state is not None:
        try:
            processed_ids = set(state.get("processed_missed_ids", []))
            missed = detect_missed_sessions(lookback_days=3, processed_ids=processed_ids)
            if missed:
                rebaseline = rebaseline_schedule(missed, state)
                # Record only the IDs that actually reached an outcome. Marking
                # every missed session processed meant a transient Intervals
                # error (PUT failed) permanently retired that session: it was
                # never rescheduled and never mentioned again.
                new_ids = processed_ids | rebaseline["resolved_ids"]
                state["processed_missed_ids"] = _cap_id_list(new_ids)

                missed_lines = [
                    f"  • {m['date']}: {m['event'].get('name', 'Session')}"
                    for m in missed
                ]
                missed_block = (
                    f"\nMISSED SESSIONS ({len(missed)}):\n" + "\n".join(missed_lines)
                )
                if rebaseline["adjustments"]:
                    missed_block += "\nSchedule adjustments made:\n" + "\n".join(
                        f"  • {adj}" for adj in rebaseline["adjustments"]
                    )
                log.info(
                    f"Rebaseline: {len(missed)} missed session(s); "
                    f"{len(rebaseline['adjustments'])} adjustment(s)"
                )
        except Exception as e:
            log.error(f"Missed-session rebaseline error: {e}")

    if missed_block:
        ctx.append(missed_block)

    # ── Training-load trend reconciliation & dynamic adjustment ───────────────
    trend_block = ""
    if state is not None:
        try:
            trend      = compute_training_trend()
            adjustment = apply_training_adjustment(trend, state)

            trend_lines = [f"Signal: {trend['signal'].upper()} — " + "; ".join(trend["reasons"])]
            if trend.get("hrv_dev_pct") is not None:
                hrv_line = f"HRV vs {RECONCILE_LOOKBACK_DAYS}-day baseline: {trend['hrv_dev_pct']:+.1f}%"
                if trend.get("hrv_stale_days"):
                    hrv_line += f" (from {trend['hrv_date']}, NOT this morning — latest reading available)"
                trend_lines.append(hrv_line)
            if trend.get("ctl_ramp") is not None:
                trend_lines.append(f"CTL ramp ({RECONCILE_LOOKBACK_DAYS}d): {trend['ctl_ramp']:+.1f}")
            if adjustment["applied"]:
                trend_lines.append("Auto-adjusted (minor, non-key sessions): " + "; ".join(adjustment["applied"]))
            if adjustment["proposed"]:
                trend_lines.append(
                    "Sent for approval (key sessions, awaiting Telegram Approve/Deny): "
                    + "; ".join(adjustment["proposed"])
                )
            if adjustment["race_notes"]:
                trend_lines.append("Race-window hold (no changes made): " + "; ".join(adjustment["race_notes"]))

            trend_block = "\nTRAINING TREND & LOAD ADJUSTMENT:\n" + "\n".join(f"  • {l}" for l in trend_lines)
            log.info(
                f"Training trend: {trend['signal']} | applied={len(adjustment['applied'])} "
                f"proposed={len(adjustment['proposed'])} race_notes={len(adjustment['race_notes'])}"
            )
        except Exception as e:
            log.error(f"Training trend/adjustment error: {e}")

    if trend_block:
        ctx.append(trend_block)

    context_block = "\n".join(ctx)

    readiness_instruction = "*Readiness* — what the wellness numbers say (use plain English, not just numbers)"
    if wellness_stale_days:
        readiness_instruction += (
            ". IMPORTANT: the wellness block is flagged STALE — those numbers are NOT from this "
            "morning. Say so in one short clause (e.g. \"last night's data hasn't synced yet, so "
            "this is Tuesday's\"), treat them as background rather than today's readiness, and do "
            "not tell the athlete how they slept last night or how their HRV is today. Lean on how "
            "they actually feel instead."
        )

    instructions = [
        readiness_instruction,
        "*Today's session* — what to do, how to pace it, what to focus on",
        f"*{goal['name']} context* — brief mention of how today fits the goal-event prep "
        "(the key challenge named in the profile, pacing strategy, or countdown milestone "
        "if notable)",
    ]
    if missed_block:
        instructions.append(
            "*Missed session* — briefly acknowledge the skipped session(s) listed above, "
            "confirm what was rescheduled, and advise how to approach it without doubling up on fatigue."
        )
    if trend_block:
        instructions.append(
            "*Training load* — state plainly whether things are trending well (progressing), holding "
            "steady, or need backing off, referencing the signal above. If a change was auto-applied, "
            "mention it. If a change was sent for approval, tell the athlete to check the message with "
            "Approve/Deny buttons rather than restating the full detail."
        )
    numbered_instructions = "\n".join(f"{i + 1}. {ins}" for i, ins in enumerate(instructions))
    sign_off_line = f"{len(instructions) + 1}. One-line motivational sign-off. Do NOT use generic filler."

    system = (
        f"You are an {coach_persona(goal)} preparing this athlete for {goal['name']}. "
        "You deliver sharp, motivating, data-driven daily briefings. "
        "Tone: direct, encouraging, like a trusted coach who knows the athlete well. "
        "Format for Telegram: use *bold* for headings, emoji sparingly. Length: 220-280 words."
    )
    user = (
        f"{athlete_context(goal)}\n\nData for today:\n{context_block}\n\n"
        f"Write the daily briefing. Cover:\n{numbered_instructions}\n{sign_off_line}"
    )
    return ask_llm(system, user)

# ── Post-Workout Analysis ──────────────────────────────────────────────────────
# ── Brief activity summary ────────────────────────────────────────────────────
def _fmt_duration(secs) -> str:
    """5976 -> '1h39m'; 2100 -> '35m'."""
    secs = int(secs or 0)
    hours, minutes = divmod(secs // 60, 60)
    return f"{hours}h{minutes:02d}m" if hours else f"{minutes}m"


def _first(activity: dict, *keys):
    """First of `keys` present on the activity with a non-None value.

    Activities reach these functions in two shapes — the Intervals.icu record
    (icu_ctl, icu_training_load, …) and the Strava record (suffer_score, …) —
    and most of the fields that matter here are named differently in each.
    Reading only one shape's names is why the form line used to be silently
    absent from every analysis that came through the Intervals.icu path."""
    for key in keys:
        value = activity.get(key)
        if value is not None:
            return value
    return None


def _round_or_none(value, digits: int = 0):
    if value is None:
        return None
    return round(value, digits) if digits else round(value)


# Words Garmin and Strava pad auto-generated activity names with; whatever is
# left after removing them is usually the place ('Sydney Road Cycling').
_NON_PLACE_WORDS = {
    "road", "cycling", "ride", "riding", "running", "run", "walking", "walk",
    "swim", "swimming", "hiking", "hike", "indoor", "outdoor", "virtual",
    "gravel", "mtb", "mountain", "biking", "bike", "treadmill", "trail",
    "morning", "afternoon", "evening", "night", "lunch", "lunchtime", "e",
    "activity", "workout", "session", "commute", "rowing", "row", "yoga",
}


def _place_from_name(name: str) -> str:
    words = [w for w in re.split(r"[\s,/\-–—]+", str(name or "")) if w]
    kept  = [w for w in words if w.lower().strip(".") not in _NON_PLACE_WORDS]
    return " ".join(kept)


def activity_location(activity: dict, strava_detail: dict | None = None) -> str:
    """Best available place name for an activity.

    Neither Intervals.icu nor (for Garmin-sourced activities) Strava reliably
    fills the location fields, so fall back to the place Garmin bakes into the
    activity name."""
    for source in (strava_detail or {}, activity):
        parts = [source.get(k) for k in
                 ("location_city", "location_state", "location_country")]
        parts = [p for p in parts if p]
        if parts:
            return ", ".join(dict.fromkeys(parts))
    return _place_from_name(activity.get("name")) or _place_from_name(
        (strava_detail or {}).get("name")
    )


def _zone_summary(activity: dict) -> str:
    """Compact time-in-zone line, power zones preferred over HR zones."""
    power = activity.get("icu_zone_times")
    if isinstance(power, list) and power:
        parts = [f"{z.get('id')} {int((z.get('secs') or 0) / 60)}min"
                 for z in power if isinstance(z, dict) and (z.get("secs") or 0) >= 60]
        if parts:
            return "Power " + ", ".join(parts)
    hr = activity.get("icu_hr_zone_times")
    if isinstance(hr, list) and hr:
        parts = [f"Z{i + 1} {int(secs / 60)}min"
                 for i, secs in enumerate(hr) if (secs or 0) >= 60]
        if parts:
            return "HR " + ", ".join(parts)
    return ""


def activity_facts(activity: dict, strava_detail: dict | None = None) -> dict:
    """Numbers the athlete-facing summaries quote, normalised across the
    Intervals.icu and Strava activity shapes."""
    ctl = _first(activity, "icu_ctl", "ctl")
    atl = _first(activity, "icu_atl", "atl")
    tsb = _first(activity, "tsb")
    if tsb is None and ctl is not None and atl is not None:
        # Intervals.icu returns the two components but no form figure of its
        # own; form is simply fitness minus fatigue.
        tsb = ctl - atl

    distance_m  = _first(activity, "distance", "icu_distance") or 0
    moving_time = _first(activity, "moving_time", "elapsed_time") or 0

    pace = ""
    speed_ms = activity.get("average_speed")
    if speed_ms and speed_ms > 0:
        per_km = 1000 / speed_ms / 60
        pace = f"{int(per_km)}:{int((per_km % 1) * 60):02d} /km"

    return {
        "name":        activity.get("name") or "Activity",
        "type":        _first(activity, "type", "sport_type") or "Workout",
        "date":        (activity.get("start_date_local") or "")[:10],
        "start_local": activity.get("start_date_local") or "",
        "distance_m":  distance_m,
        "distance_km": round(distance_m / 1000, 1),
        "moving_time": moving_time,
        "duration":    _fmt_duration(moving_time),
        "pace":        pace,
        "location":    activity_location(activity, strava_detail),
        "hr_avg":      _round_or_none(_first(activity, "average_heartrate")),
        "hr_max":      _round_or_none(_first(activity, "max_heartrate")),
        "hr_lthr":     _first(activity, "lthr"),
        "load":        _round_or_none(_first(activity, "icu_training_load", "training_load")),
        "elevation":   _round_or_none(_first(activity, "total_elevation_gain")),
        "calories":    _round_or_none(_first(activity, "calories")),
        "ctl":         ctl,
        "atl":         atl,
        "tsb":         tsb,
        # eFTP from this ride's own power curve first, then the rolling
        # estimate, then whatever FTP the athlete has set.
        "ftp_estimate": _first(activity, "icu_pm_ftp", "icu_rolling_ftp", "icu_ftp"),
        "ftp_source":  ("this activity" if activity.get("icu_pm_ftp") is not None else
                        "rolling estimate" if activity.get("icu_rolling_ftp") is not None else
                        "athlete setting" if activity.get("icu_ftp") is not None else ""),
        "ftp_setting": _first(activity, "icu_ftp"),
        "np":          _round_or_none(_first(activity, "icu_weighted_avg_watts",
                                             "weighted_average_watts")),
        "avg_watts":   _round_or_none(_first(activity, "icu_average_watts", "average_watts")),
        "intensity":   _round_or_none(_first(activity, "icu_intensity")),
        "decoupling":  _round_or_none(_first(activity, "decoupling"), 1),
        "zones":       _zone_summary(activity),
    }


def _summary_header(facts: dict, purpose: str) -> str:
    """'Wed 20 August 05:44 1h39m 42.8 km Sydney - Z2 endurance ride'."""
    bits = []
    try:
        started = datetime.fromisoformat(facts["start_local"])
        bits.append(f"{started:%a} {started.day} {started:%B} {started:%H:%M}")
    except (ValueError, TypeError):
        if facts["date"]:
            bits.append(facts["date"])
    if facts["moving_time"]:
        bits.append(facts["duration"])
    if facts["distance_km"]:
        bits.append(_fmt_km(facts["distance_km"]))
    if facts["location"]:
        bits.append(facts["location"])
    header = " ".join(bits)
    return f"{header} - {purpose}" if purpose else header


def fallback_activity_comment(facts: dict) -> str:
    """Deterministic summary, used when the LLM is unavailable.

    The comment is worth posting even without prose: the point of it is that
    the numbers sit on the activity, and an LLM outage should not take that
    away."""
    purpose = facts["type"]
    stats = []
    if facts["hr_avg"]:
        stats.append(f"avg HR {facts['hr_avg']}")
    if facts["hr_max"]:
        stats.append(f"max HR {facts['hr_max']}")
    if facts["ftp_estimate"]:
        stats.append(f"eFTP {facts['ftp_estimate']}w")
    if facts["load"]:
        stats.append(f"load {facts['load']}")
    if facts["decoupling"] is not None:
        stats.append(f"decoupling {facts['decoupling']}%")
    if facts["tsb"] is not None:
        stats.append(f"Form {facts['tsb']:+.0f}")
    body = ", ".join(stats)
    body = body[0].upper() + body[1:] + "." if body else "Logged."
    return f"{_summary_header(facts, purpose)}\n\n{body}"


def generate_activity_comment(
    activity: dict,
    planned: dict | None = None,
    intervals_summary: str = "",
    strava_detail: dict | None = None,
) -> str:
    """A very brief summary of one activity, sized for an Intervals.icu comment.

    Deliberately not the Telegram analysis in miniature: a title naming what the
    session was and what it was for, then two or three sentences on how it went
    carrying eFTP, average and max HR, and form. Falls back to a numbers-only
    version rather than raising, so a comment always gets posted."""
    facts = activity_facts(activity, strava_detail)

    data_lines = [
        f"Activity name: {facts['name']} ({facts['type']})",
        f"Start: {facts['start_local']}",
        f"Duration: {facts['duration']} | Distance: {facts['distance_km']} km",
        f"Location: {facts['location'] or 'unknown'}",
    ]
    if facts["pace"]:
        data_lines.append(f"Average pace: {facts['pace']}")
    if facts["elevation"]:
        data_lines.append(f"Elevation gain: {facts['elevation']} m")
    if facts["hr_avg"] or facts["hr_max"]:
        data_lines.append(
            f"Avg HR: {facts['hr_avg'] or 'N/A'} | Max HR: {facts['hr_max'] or 'N/A'}"
            + (f" | LTHR: {facts['hr_lthr']}" if facts["hr_lthr"] else "")
        )
    if facts["ftp_estimate"]:
        data_lines.append(
            f"Estimated FTP: {facts['ftp_estimate']} W (from {facts['ftp_source']})"
            + (f"; FTP currently set to {facts['ftp_setting']} W"
               if facts["ftp_setting"] and facts["ftp_setting"] != facts["ftp_estimate"] else "")
        )
    if facts["np"] or facts["avg_watts"]:
        data_lines.append(
            f"Normalised power: {facts['np'] or 'N/A'} W | Average power: "
            f"{facts['avg_watts'] or 'N/A'} W"
            + (f" | Intensity: {facts['intensity']}%" if facts["intensity"] else "")
        )
    if facts["load"]:
        data_lines.append(f"Training load: {facts['load']}")
    if facts["decoupling"] is not None:
        data_lines.append(f"Aerobic decoupling: {facts['decoupling']}%")
    if facts["tsb"] is not None:
        data_lines.append(
            f"Form (TSB) after this session: {facts['tsb']:+.1f} "
            f"(Fitness {facts['ctl']:.1f}, Fatigue {facts['atl']:.1f})"
        )
    if facts["zones"]:
        data_lines.append(f"Time in zone: {facts['zones']}")
    if planned:
        data_lines.append(
            f"Planned session: {planned.get('name', '')} — "
            f"{(planned.get('description') or '')[:300]}"
        )
    if intervals_summary:
        data_lines.append(f"Detected intervals:\n{intervals_summary[:800]}")

    goal = get_goal()
    system = (
        f"You are an {coach_persona(goal)} writing a short note on the athlete's own "
        "training log, for them to re-read months later. Plain text only — no markdown, "
        "no bullet points, no emoji, no headings."
    )
    user = (
        f"{athlete_context(goal)}\n\nSession data:\n" + "\n".join(data_lines) + "\n\n"
        "Write exactly two parts, separated by a blank line.\n\n"
        "PART 1 — a single short title line naming what the session was and what it was "
        "for. Style: 'Z2 endurance ride', 'Threshold intervals 4x8min', "
        "'Activation (openers)', 'Long run — aerobic base'. No date, no duration, no "
        "distance, no place: those are added automatically around it. Under 60 characters.\n\n"
        "PART 2 — two or three sentences, 60 words maximum, on how the session actually "
        "went. Quote the estimated FTP, average HR and max HR, and finish with where form "
        "now sits. Mention decoupling or time in zone only if it says something. Say "
        "plainly if it was easier or harder than intended. Past tense, no preamble."
    )

    try:
        raw = ask_llm(system, user, max_tokens=400).strip()
    except LLMUnavailable as e:
        log.warning(f"Comment LLM unavailable ({e}) — posting the numbers-only summary")
        return fallback_activity_comment(facts)

    parts   = [block.strip() for block in raw.split("\n\n", 1)]
    purpose = parts[0].strip().strip("*#").strip() if parts else ""
    body    = parts[1].strip() if len(parts) > 1 else ""
    if not body:
        # Model ignored the blank line — treat the first line as the title.
        lines   = [ln.strip() for ln in raw.splitlines() if ln.strip()]
        purpose = lines[0].strip("*#").strip() if lines else ""
        body    = " ".join(lines[1:])
    if len(purpose) > 80:      # a paragraph, not a title — drop it from the header
        body, purpose = (purpose + " " + body).strip(), ""

    comment = f"{_summary_header(facts, purpose)}\n\n{body}".strip()
    if len(comment) > ICU_COMMENT_MAX_CHARS:
        comment = comment[:ICU_COMMENT_MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;") + "…"
    return comment


def publish_activity_summary(
    activity: dict,
    act_id: str,
    source: str,
    planned: dict | None = None,
    intervals_summary: str = "",
    strava_detail: dict | None = None,
) -> dict:
    """Write the brief summary onto the activity itself.

    Intervals.icu gets it as a comment. Strava gets the same text in the
    activity description, because the Intervals.icu comment does not travel
    there (see COMMENT_SYNCS_TO_STRAVA) and Strava has no comment-write API."""
    comment = generate_activity_comment(
        activity, planned, intervals_summary, strava_detail
    )

    if source == "strava":
        strava_id  = act_id
        icu_match  = find_intervals_activity_by_strava_id(strava_id)
        intervals_id = icu_match.get("id") if icu_match else None
    else:
        intervals_id = act_id
        strava_id    = activity.get("strava_id") or (strava_detail or {}).get("id")

    posted_intervals = False
    if intervals_id:
        posted_intervals = post_intervals_comment(str(intervals_id), comment)
    else:
        log.warning(
            f"No Intervals.icu activity for Strava {act_id} yet — comment not posted"
        )

    posted_strava = False
    if STRAVA_MIRROR_SUMMARY and strava_id:
        posted_strava = update_strava_description(str(strava_id), comment)

    return {
        "comment":      comment,
        "intervals_id": intervals_id,
        "strava_id":    strava_id,
        "intervals":    posted_intervals,
        "strava":       posted_strava,
    }


def generate_analysis(
    activity: dict,
    planned: dict | None,
    strava_segments: str = "",
    streams_summary: str = "",
    intervals_summary: str = "",
) -> str:
    # Via activity_facts because these fields carry different names on the
    # Intervals.icu record than on the Strava one, and the Intervals.icu path is
    # the normal one: reading only the Strava names meant training load and the
    # whole post-run form line were silently dropped from most analyses.
    facts     = activity_facts(activity)
    name      = facts["name"]
    wtype     = facts["type"]
    date      = facts["date"]
    dist_km   = round(facts["distance_m"] / 1000, 2)
    dur_min   = facts["moving_time"] // 60
    load      = facts["load"]
    hr_avg    = facts["hr_avg"]
    hr_max    = facts["hr_max"]
    pace_ms   = activity.get("average_speed")       # m/s
    ctl       = facts["ctl"]
    atl       = facts["atl"]
    tsb       = facts["tsb"]
    suffer    = activity.get("suffer_score")
    elevation = facts["elevation"]

    pace_str = "N/A"
    if pace_ms and pace_ms > 0:
        pm = 1000 / pace_ms / 60
        pace_str = f"{int(pm)}:{int((pm % 1)*60):02d} /km"

    actual = (
        f"Activity: {name} ({wtype}) — {date}\n"
        f"Distance: {dist_km} km | Duration: {dur_min} min | Pace: {pace_str}\n"
        f"Avg HR: {hr_avg or 'N/A'} bpm | Max HR: {hr_max or 'N/A'} bpm"
    )
    if elevation:
        actual += f" | Elevation: {round(elevation)} m"
    if load:
        actual += f"\nTraining load: {round(load)}"
    if suffer:
        actual += f" | Suffer score: {suffer}"
    if ctl:
        actual += (
            f"\nPost-run form: Fitness {round(ctl,1)} | "
            f"Fatigue {round(atl,1) if atl else 'N/A'} | "
            f"Form {round(tsb,1) if tsb else 'N/A'}"
        )

    plan_ctx = "No specific session was planned today."
    if planned:
        plan_ctx = (
            f"Planned: {planned.get('name','')} — "
            f"{planned.get('description','(no description)')}"
        )

    goal = get_goal()
    system = (
        f"You are an {coach_persona(goal)} preparing this athlete for {goal['name']}. "
        "Deliver post-session feedback that's specific, data-driven, and actionable. "
        "Format for Telegram: *bold* headings, emoji sparingly. Length: 280-340 words."
    )
    seg_block       = f"\n\nStrava segment data:\n{strava_segments}" if strava_segments else ""
    streams_block   = f"\n\nPer-km stream analysis:\n{streams_summary}" if streams_summary else ""
    intervals_block = f"\n\n{intervals_summary}" if intervals_summary else ""
    user = (
        f"{athlete_context(goal)}\n\nActual session:\n{actual}\n\nPlan:\n{plan_ctx}"
        f"{intervals_block}{seg_block}{streams_block}\n\n"
        "Write the post-workout analysis. Cover:\n"
        "1. *Execution* — how well did actual match the plan? What the numbers show. "
        "If detected intervals are provided, use those rep-by-rep numbers as the source of truth for "
        "whether a structured session (e.g. reps at target pace) was actually executed — they separate "
        "work reps from recoveries, unlike the per-km stream splits which are just fixed GPS-distance "
        "buckets and can blend reps with recovery jogging. Only fall back to per-km splits or hedge "
        "on execution if no interval data was provided.\n"
        "2. *Physiological read* — what HR, pace, and load tell us about effort and fitness.\n"
        f"3. *{goal['name']} relevance* — did this session build anything specific for the "
        "goal event (the key challenge named in the profile, threshold, aerobic base, "
        "fatigue management)?\n"
        "4. *Segment highlights* — if Strava data is provided, call out any notable PR gaps, "
        "KOM opportunities, or segments where the athlete was close to something special.\n"
        "5. *Next session tip* — one concrete, specific thing to focus on next time.\n"
        "Be honest — if it was too easy, too hard, or off-plan, say so constructively."
    )
    return ask_llm(system, user)

def _analyse_and_send(act_id: str, source: str, state: dict) -> bool:
    """Fetch enrichment data for one activity, generate the post-workout analysis,
    and send it via Telegram. Returns True on success (caller clears the pending
    retry); False leaves it queued for the next retry attempt.

    `source` is 'intervals' (the normal path — richer data via streams/detected
    intervals) or 'strava' (fallback when Strava has synced the activity before
    Intervals.icu has)."""
    try:
        if source == "strava":
            activity = get_strava_activity_detail(act_id)
            if not activity:
                log.warning(f"Strava fallback: no detail yet for activity {act_id} — will retry")
                return False
            strava_detail     = activity
            streams_summary   = ""
            intervals_summary = ""
        else:
            activity = get_activity_detail(act_id)
            if not activity:
                log.warning(f"Intervals: no detail yet for activity {act_id} — will retry")
                return False

            streams_summary = ""
            try:
                streams = get_activity_streams(act_id)
                if streams:
                    streams_summary = summarize_streams(streams)
                    log.info(f"Streams fetched for activity {act_id}")
            except Exception as e:
                log.error(f"Streams fetch error: {e}")

            intervals_summary = ""
            try:
                intervals_data = get_activity_intervals(act_id)
                if intervals_data:
                    intervals_summary = summarize_intervals(intervals_data)
                    if intervals_summary:
                        log.info(f"Intervals fetched for activity {act_id}")
            except Exception as e:
                log.error(f"Intervals fetch error: {e}")

            strava_detail = None
            if STRAVA_ENABLED:
                try:
                    strava_act = find_strava_match(activity)
                    if strava_act:
                        strava_detail = get_strava_activity_detail(str(strava_act["id"]))
                except Exception as e:
                    log.error(f"Strava match error: {e}")

        strava_segments = ""
        chase_alerts    = []
        if strava_detail:
            try:
                segment_cache   = {}
                athlete_id      = state.get("strava_athlete_id")
                strava_segments = build_segment_report(strava_detail, athlete_id, segment_cache)
                chase_alerts    = get_pr_kom_chases(strava_detail, segment_cache)
                log.info(f"Strava segments enriched for activity {strava_detail.get('id')}")
            except Exception as e:
                log.error(f"Strava enrichment error: {e}")

        # Compare against the plan for the activity's own date, not "today" —
        # with the initial processing delay and retry backoffs, the analysis can
        # run after midnight for a workout done the previous evening.
        act_date = (activity.get("start_date_local") or "")[:10]
        planned  = get_event_for_date(act_date) if act_date else get_todays_event()
        analysis = generate_analysis(activity, planned, strava_segments, streams_summary, intervals_summary)
        header   = "💪 *Post-Run Analysis*\n\n"
        if not send_telegram(header + analysis):
            log.warning(f"Analysis send failed for activity {act_id} — will retry")
            return False

        if chase_alerts:
            chase_msg = "🔥 *Chase these next run:*\n\n" + "\n".join(chase_alerts)
            send_telegram(chase_msg)

        # Brief summary written onto the activity itself. Runs after the
        # Telegram send and swallows its own errors on purpose: returning False
        # from here would re-queue the activity and re-send the full analysis to
        # the athlete, and a missing comment is not worth a duplicate briefing.
        if POST_ACTIVITY_COMMENT:
            try:
                publish_activity_summary(
                    activity, act_id, source, planned,
                    intervals_summary, strava_detail,
                )
            except Exception as e:
                log.error(f"Activity summary error (activity {act_id}): {e}")

        # Record what we just analysed so the other source (once it syncs the
        # same run) recognises it and skips sending a duplicate analysis.
        state["last_analysed_sig"] = list(_activity_signature(activity))
        return True
    except Exception as e:
        log.error(f"Analysis error (activity {act_id}, source={source}): {e}")
        return False

def queue_unseen_intervals_activities(state: dict) -> list[dict]:
    """Queue every Intervals.icu activity we haven't seen before, oldest first.

    Returns the pending-analysis queue (also stored on `state`).

    Checks the whole recent window rather than only the newest activity.
    Comparing a single `last_activity_id` meant that when two activities
    appeared between polls — or a backlog built up while the bot was down —
    only the most recent was ever analysed and the rest were skipped silently.
    `get_recent_activities` already fetches a week, so the older ones were being
    fetched and thrown away.
    """
    queue = state.get("pending_analysis") or []

    # Ordered list (not just a set) so _cap_id_list keeps the newest on trim.
    seen_list = [str(i) for i in state.get("seen_activity_ids", [])]
    seen_ids  = set(seen_list)

    recent = get_recent_activities(ACTIVITY_SCAN_LIMIT)
    for activity in sorted(recent, key=lambda a: a.get("start_date_local", "")):
        act_id = str(activity.get("id") or "")
        if not act_id or act_id in seen_ids:
            continue
        seen_ids.add(act_id)
        seen_list.append(act_id)
        state["last_activity_id"] = act_id  # retained so a rollback still works
        sig = _activity_signature(activity)
        if _same_activity(sig, state.get("last_analysed_sig")):
            log.info(f"Intervals activity {act_id} already analysed via Strava fallback — skipping")
        elif any(_same_activity(sig, e.get("sig")) for e in queue):
            log.info(f"Intervals activity {act_id} already queued for analysis — skipping")
        else:
            log.info(f"New activity detected via Intervals: {act_id}")
            queue.append({
                "act_id": act_id,
                "source": "intervals",
                "attempts": 0,
                # give Intervals ~90s to finish processing before the first attempt
                "next_attempt_at": (datetime.now(AEST) + timedelta(seconds=90)).isoformat(),
                # lets the Strava fallback recognise this run as already queued
                # before it's been analysed
                "sig": list(sig),
            })
            state["pending_analysis"] = queue

    state["seen_activity_ids"] = _cap_id_list(seen_list)
    save_state(state)
    return queue


def _due_strava_fallback_slot(now: datetime) -> str | None:
    """Return the key (e.g. '2026-08-02 07:15') of the most recent fallback slot
    due at `now`, or None before the first slot of the day. Keying on the latest
    due slot — rather than requiring an exact-minute match — means the check
    still fires (once) if the loop tick lands late or the bot was down when the
    slot passed."""
    due = [(h, m) for h, m in STRAVA_FALLBACK_SLOTS if (now.hour, now.minute) >= (h, m)]
    if not due:
        return None
    h, m = due[-1]
    return f"{now.strftime('%Y-%m-%d')} {h:02d}:{m:02d}"

# ── Main loop ──────────────────────────────────────────────────────────────────
def run():
    state = load_state()
    goal  = load_goal(state)
    save_state(state)
    log.info(
        f"🏃 Coaching bot started — goal: {goal['name']} on {goal['date']} "
        f"({days_to_goal()} days out, sport={goal['sport']})"
    )

    # Ensure processed_missed_ids list exists in state
    if "processed_missed_ids" not in state:
        state["processed_missed_ids"] = []
        save_state(state)

    # Normalise pending_analysis to a queue — older state files hold a single
    # dict (or None), and the single-slot design dropped activities uploaded
    # while one was being analysed.
    pa = state.get("pending_analysis")
    if pa is None:
        state["pending_analysis"] = []
    elif isinstance(pa, dict):
        state["pending_analysis"] = [pa]

    # On first run, record baseline activity and Strava athlete ID
    if state.get("last_activity_id") is None:
        latest = get_latest_activity()
        if latest:
            state["last_activity_id"] = str(latest.get("id"))
            save_state(state)
            log.info(f"Baseline activity set: {state['last_activity_id']}")

    # Seed the seen-activity set. On a fresh install the whole recent window is
    # baselined (nothing historical gets analysed); on an upgrade from the old
    # single-marker scheme, that marker becomes the baseline so the week of
    # activities behind it isn't queued in one go.
    if "seen_activity_ids" not in state:
        if state.get("last_activity_id"):
            baseline = [str(a.get("id")) for a in get_recent_activities(ACTIVITY_SCAN_LIMIT) if a.get("id")]
            if str(state["last_activity_id"]) not in baseline:
                baseline.append(str(state["last_activity_id"]))
        else:
            baseline = []
        state["seen_activity_ids"] = _cap_id_list(baseline)
        save_state(state)
        log.info(f"Seeded seen-activity baseline with {len(baseline)} id(s)")

    if STRAVA_ENABLED:
        if not state.get("strava_athlete_id"):
            athlete_id = get_strava_athlete_id()
            if athlete_id:
                state["strava_athlete_id"] = athlete_id
                save_state(state)
                log.info(f"Strava athlete ID: {athlete_id}")
        # Mirror the Intervals.icu baseline above — without this, the first
        # fallback slot after a fresh install would queue an analysis for
        # whatever the latest Strava activity happens to be, even if days old.
        if not state.get("last_strava_activity_id"):
            strava_latest = get_latest_strava_activity()
            if strava_latest:
                state["last_strava_activity_id"] = str(strava_latest.get("id"))
                save_state(state)
                log.info(f"Baseline Strava activity set: {state['last_strava_activity_id']}")
    else:
        log.info("Strava integration disabled — run strava_setup.py to enable")

    # Startup catch-up: with no recorded offset, the first getUpdates in the
    # loop below would return every message Telegram has queued over the last
    # 24h (fresh install, or a corrupt-state reset) — and the bot would reply
    # to all of them. Fetch once without long-polling and discard, keeping
    # only the new offset.
    if state.get("last_update_id") is None:
        backlog = get_telegram_updates(timeout=0)
        if backlog:
            state["last_update_id"] = max(u.get("update_id", 0) for u in backlog) + 1
            save_state(state)
        log.info(f"Startup catch-up: skipped {len(backlog)} queued update(s)")

    last_activity_check = 0.0

    while True:
        now = datetime.now(AEST)

        today_str = now.strftime("%Y-%m-%d")

        # ── Wellness cache refresh at 22:00, 00:01 and 05:00 AEST ────────────
        # The briefing reads this cache (see generate_briefing) so it always has
        # fresh data — never a live fetch, never a days-old stale cache.
        wellness_slot = None
        if now.hour == 22 and now.minute < 10:
            wellness_slot = f"{today_str}:22:00"
        elif now.hour == 0 and now.minute < 10:
            wellness_slot = f"{today_str}:00:01"
        elif now.hour == 5 and now.minute < 10:
            wellness_slot = f"{today_str}:05:00"

        if wellness_slot and state.get("last_wellness_prefetch_slot") != wellness_slot:
            log.info(f"Refreshing wellness cache ({wellness_slot})…")
            try:
                w = get_wellness()
                if w:
                    state["cached_wellness"] = w
                    state["cached_wellness_fetched_at"] = now.isoformat()
                    state["last_wellness_prefetch_slot"] = wellness_slot
                    save_state(state)
                    age = wellness_age_days(w, now)
                    log.info(
                        f"Wellness cached: HRV={w.get('hrv')} ms | "
                        f"RHR={w.get('restingHR')} bpm | "
                        f"Sleep={round(w.get('sleepSecs',0)/3600,1)}h | "
                        f"date={w.get('id')} | "
                        + ("data is TODAY's" if age == 0 else f"data is {age} day(s) STALE")
                    )
                else:
                    log.warning("Wellness cache refresh returned no data")

                # Measurement only — see probe_garmin_freshness. Runs at the 05:00
                # slot because that is the fetch the 05:15 briefing actually consumes.
                if now.hour == 5:
                    try:
                        probe_garmin_freshness(w, now)
                    except Exception as e:
                        log.warning(f"Garmin freshness probe error (ignored): {e}")
            except Exception as e:
                log.error(f"Wellness cache refresh error: {e}")

        # ── Daily briefing, fires once from 05:15 AEST ─────────────────────────
        # Retried on the natural loop cadence (~poll interval) up to
        # BRIEFING_MAX_ATTEMPTS times per day, rather than only within a narrow
        # 5-minute window that gives up silently if the first attempt fails.
        due_for_briefing = (
            (now.hour > 5 or (now.hour == 5 and now.minute >= 15))
            and state.get("last_briefing_date") != today_str
        )
        if due_for_briefing:
            if state.get("briefing_attempts_date") != today_str:
                state["briefing_attempts_date"] = today_str
                state["briefing_attempts"] = 0

            attempts = state.get("briefing_attempts", 0)
            if attempts >= BRIEFING_MAX_ATTEMPTS:
                pass  # exhausted for today — wait for tomorrow's date rollover
            else:
                log.info(f"Generating daily briefing… (attempt {attempts + 1}/{BRIEFING_MAX_ATTEMPTS})")
                try:
                    briefing = generate_briefing(state)
                    header   = f"🌅 *Good morning! Daily Training Briefing*\n_{now.strftime('%A, %d %B')}_\n\n"
                    if not send_telegram(header + briefing):
                        # send_telegram catches its own exceptions and returns False —
                        # raise here so the retry path below counts it as a failed attempt
                        raise RuntimeError("Telegram send failed")
                    state["last_briefing_date"] = today_str
                    save_state(state)
                except Exception as e:
                    state["briefing_attempts"] = attempts + 1
                    save_state(state)
                    if state["briefing_attempts"] >= BRIEFING_MAX_ATTEMPTS:
                        log.error(f"Briefing error (attempt {state['briefing_attempts']}/{BRIEFING_MAX_ATTEMPTS}, giving up for today): {e}")
                    else:
                        log.error(f"Briefing error (attempt {state['briefing_attempts']}/{BRIEFING_MAX_ATTEMPTS}, will retry): {e}")

        # ── New-activity detection — Intervals.icu poll every 5 minutes ────────
        # Intervals.icu is the primary source (richer data: streams, detected
        # intervals). Strava is checked as a fallback only at the fixed daily
        # slots in STRAVA_FALLBACK_SLOTS — it catches activities that synced to
        # Strava before Intervals.icu picked them up. Detections are appended to
        # the pending_analysis queue, so uploads that happen while an analysis
        # is pending/retrying are not dropped.
        if time.time() - last_activity_check >= 300:
            try:
                queue = queue_unseen_intervals_activities(state)

                # ── Strava fallback — once per fixed slot, not every tick ──
                slot = _due_strava_fallback_slot(now)
                if (
                    STRAVA_ENABLED
                    and slot is not None
                    and state.get("last_strava_fallback_slot") != slot
                ):
                    # Consume the slot even if the fetch fails — the schedule
                    # stays strict rather than retrying every loop tick.
                    #
                    # last_strava_activity_id is only advanced on a successful
                    # fetch, so a run uploaded during an outage is still detected
                    # at a later slot: it differs from the last id we actually
                    # saw, which is the correct baseline. If the outage spans
                    # several slots, the sig-based dedup below (and against
                    # last_analysed_sig) stops the same run being queued twice.
                    state["last_strava_fallback_slot"] = slot
                    strava_latest = get_latest_strava_activity()
                    if strava_latest:
                        strava_id = str(strava_latest.get("id"))
                        if strava_id != state.get("last_strava_activity_id"):
                            state["last_strava_activity_id"] = strava_id
                            sig = _activity_signature(strava_latest)
                            if _same_activity(sig, state.get("last_analysed_sig")):
                                log.info(f"Strava activity {strava_id} already analysed via Intervals — skipping")
                            elif any(_same_activity(sig, e.get("sig")) for e in queue):
                                log.info(f"Strava activity {strava_id} already queued via Intervals — skipping")
                            else:
                                log.info(f"New activity detected via Strava fallback (Intervals hasn't synced it yet): {strava_id}")
                                queue.append({
                                    "act_id": strava_id,
                                    "source": "strava",
                                    "attempts": 0,
                                    "next_attempt_at": (datetime.now(AEST) + timedelta(seconds=90)).isoformat(),
                                    "sig": list(sig),
                                })
                                state["pending_analysis"] = queue
                    save_state(state)
                last_activity_check = time.time()
            except Exception as e:
                log.error(f"Activity check error: {e}")

        # ── Post-workout analysis — process the activity queue, with retry ────
        # Processed head-first in detection order. Nothing is dequeued until
        # _analyse_and_send actually succeeds, so a failed generation/send gets
        # retried instead of silently dropped.
        queue = state.get("pending_analysis") or []
        if queue and datetime.now(AEST) >= datetime.fromisoformat(queue[0]["next_attempt_at"]):
            pending     = queue[0]
            attempt_num = pending["attempts"] + 1
            log.info(
                f"Running post-workout analysis for {pending['act_id']} "
                f"(source={pending['source']}, attempt {attempt_num}/{ANALYSIS_MAX_ATTEMPTS})"
            )
            success = _analyse_and_send(pending["act_id"], pending["source"], state)
            if success:
                queue.pop(0)
            else:
                pending["attempts"] = attempt_num
                if attempt_num >= ANALYSIS_MAX_ATTEMPTS:
                    log.error(f"Analysis for {pending['act_id']} failed {attempt_num} times — giving up")
                    queue.pop(0)
                else:
                    pending["next_attempt_at"] = (
                        datetime.now(AEST) + timedelta(seconds=ANALYSIS_RETRY_BACKOFF_SECS)
                    ).isoformat()
            state["pending_analysis"] = queue
            save_state(state)

        # ── Daily KOM check, fires once per day at/after 18:00 AEST ────────────
        # Date-gated (like the briefing) rather than a time.time()-elapsed
        # interval, which used to fire immediately on every bot restart
        # (last_kom_check started at 0.0) and drifted off any fixed time of day.
        if STRAVA_ENABLED and now.hour >= KOM_CHECK_HOUR and state.get("last_kom_check_date") != today_str:
            try:
                log.info("Running daily KOM check…")
                kom_alerts = check_kom_alerts(state)
                if kom_alerts:
                    msg = "🔔 *Strava KOM Update*\n\n" + "\n\n".join(kom_alerts)
                    send_telegram(msg)
                state["last_kom_check_date"] = today_str
                save_state(state)
            except Exception as e:
                log.error(f"KOM check error: {e}")

        # ── Incoming messages — long-poll (blocks up to 30s) ──────────────────
        updates = get_telegram_updates(state.get("last_update_id"))
        for update in updates:
            # Persist the offset BEFORE acting on the update — a crash between
            # the side effect (reply sent / plan change applied) and the save
            # would otherwise re-deliver the same update on restart and repeat
            # the action. Trade-off: a crash right after the save means this
            # update is never processed (at-most-once) — preferable to
            # duplicate replies and double-applied plan changes.
            state["last_update_id"] = update.get("update_id", 0) + 1
            save_state(state)

            callback_query = update.get("callback_query")
            if callback_query:
                handle_callback(callback_query, state)
                save_state(state)
                continue

            msg     = update.get("message", {})
            chat_id = str(msg.get("chat", {}).get("id", ""))
            if chat_id != str(TELEGRAM_CHAT_ID):
                continue

            text        = (msg.get("text") or msg.get("caption") or "").strip()
            image_bytes = None
            media_type  = "image/jpeg"

            # Photo sent as compressed image
            if msg.get("photo"):
                file_id = msg["photo"][-1]["file_id"]  # largest size
                result  = download_telegram_photo(file_id)
                if result:
                    image_bytes, media_type = result
                    log.info("Photo received — downloading for analysis")

            # Photo sent as uncompressed document
            elif msg.get("document"):
                doc  = msg["document"]
                mime = doc.get("mime_type", "")
                if mime.startswith("image/"):
                    result = download_telegram_photo(doc["file_id"])
                    if result:
                        image_bytes, media_type = result
                        log.info(f"Document image received ({mime})")

            if not text and not image_bytes:
                continue

            # Slash commands are handled here rather than by the LLM: setting the
            # goal has to change bot state deterministically, not depend on the
            # model deciding to call a tool.
            if text.startswith("/"):
                cmd, _, cmd_arg = text.partition(" ")
                cmd = cmd.split("@", 1)[0].lower()   # strip the @botname Telegram adds in groups
                if cmd in ("/goal", "/goals"):
                    log.info(f"Command: {text[:80]}")
                    send_telegram(handle_goal_command(cmd_arg, state))
                    save_state(state)
                    continue

            log.info(f"Incoming message: {text[:60]}" + (" [+image]" if image_bytes else ""))
            reply = handle_incoming_message(text, image_bytes, media_type, chat_id, msg.get("message_id"))
            send_telegram(reply)
            save_state(state)

if __name__ == "__main__":
    run()
