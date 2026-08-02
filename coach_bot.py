#!/usr/bin/env python3
"""
Virtual Coaching Bot — City2Surf Edition
- Daily 6am AEST training briefing via Telegram
- Post-workout analysis after each Intervals.icu upload
"""

import os
import asyncio
import json
import time
import base64
import logging
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
        logging.FileHandler(os.path.join(os.path.dirname(__file__), "coach_bot.log")),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TELEGRAM_TOKEN        = os.environ["TELEGRAM_BOT_TOKEN"]
TELEGRAM_CHAT_ID      = os.environ["TELEGRAM_CHAT_ID"]
INTERVALS_API_KEY     = os.environ["INTERVALS_API_KEY"]
INTERVALS_ATHLETE_ID  = os.environ["INTERVALS_ATHLETE_ID"]

STRAVA_CLIENT_ID     = os.environ.get("STRAVA_CLIENT_ID", "")
STRAVA_CLIENT_SECRET = os.environ.get("STRAVA_CLIENT_SECRET", "")
STRAVA_REFRESH_TOKEN = os.environ.get("STRAVA_REFRESH_TOKEN", "")
STRAVA_ENABLED       = all([STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN])

AEST           = ZoneInfo("Australia/Sydney")
STATE_FILE     = os.path.join(os.path.dirname(__file__), "state.json")
MCP_SERVER_DIR = os.path.join(os.path.dirname(__file__), "..", "intervals-mcp-server")
RACE_DATE      = datetime(2026, 8, 9, tzinfo=AEST)

# ── Athlete context (used in every Claude prompt) ─────────────────────────────
ATHLETE_CONTEXT = """
Athlete profile:
- Goal race: City2Surf Sydney, August 2026 (14km road race, iconic course from Hyde Park to Bondi Beach)
- Key challenge: Heartbreak Hill at ~10km — a steep 1.8km climb that breaks most runners
- Training for: strong finish time, negative split strategy, surviving the hill with energy to sprint Bondi
- Location: Sydney, Australia
- Training platform: Intervals.icu with structured plan already loaded
"""

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

def send_telegram(message: str) -> bool:
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    try:
        _post_send_message(payload)
        log.info("Telegram sent ✓")
        return True
    except Exception as e:
        _log_telegram_error("Telegram send failed", e)
        return False

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
        with open(STATE_FILE) as f:
            return json.load(f)
    return {"last_activity_id": None, "last_briefing_date": None}

def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

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

def get_todays_event() -> dict | None:
    today = datetime.now(AEST).strftime("%Y-%m-%d")
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/events?oldest={today}&newest={today}"
    )
    if isinstance(data, list):
        for ev in data:
            if ev.get("category") not in ("NOTE", "RACE"):
                return ev
    return None

def get_wellness(lookback_days: int = 5) -> dict | None:
    today    = datetime.now(AEST).strftime("%Y-%m-%d")
    earliest = (datetime.now(AEST) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/wellness?oldest={earliest}&newest={today}"
    )
    if not isinstance(data, list):
        return None
    # Prefer most recent record that has actual HRV data (device may not have synced yet at 5am)
    for record in reversed(data):
        if record and record.get("hrv") is not None:
            return record
    # Fall back to most recent non-empty record
    for record in reversed(data):
        if record:
            return record
    return None

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

def _get_strava_token() -> str | None:
    global _strava_access_token, _strava_token_expiry
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

def build_segment_report(strava_detail: dict, athlete_id: int | None) -> str:
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

        details = get_segment_details(seg_id)
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

def get_pr_kom_chases(strava_detail: dict) -> list[str]:
    """
    Return alert strings for any segment where the athlete was within
    5% of the KOM or within 3% of their own PR — prime opportunities to chase.
    """
    efforts = strava_detail.get("segment_efforts", [])
    alerts  = []
    for effort in efforts[:15]:
        seg_id   = str((effort.get("segment") or {}).get("id", ""))
        seg_name = effort.get("name") or (effort.get("segment") or {}).get("name", "")
        elapsed  = effort.get("elapsed_time", 0)
        if not seg_id or not elapsed:
            continue
        details  = get_segment_details(seg_id)
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

    return alerts[:3]  # cap at 3 to keep messages tidy

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

    for seg in starred[:50]:
        seg_id   = str(seg.get("id", ""))
        seg_name = seg.get("name", "Unknown segment")
        if not seg_id:
            continue

        details = get_segment_details(seg_id)
        if not details:
            continue

        kom_str = (details.get("xoms") or {}).get("kom")
        new_koms[seg_id] = kom_str

        pr_secs  = (details.get("athlete_segment_stats") or {}).get("pr_elapsed_time")
        kom_secs = _parse_time_str(kom_str) if kom_str else None
        is_kom   = pr_secs and kom_secs and pr_secs <= kom_secs

        prev_kom_str  = prev_koms.get(seg_id)
        prev_kom_secs = _parse_time_str(prev_kom_str) if prev_kom_str else None

        if prev_kom_str and kom_str and kom_str != prev_kom_str and prev_kom_secs:
            if kom_secs and kom_secs < prev_kom_secs:
                if is_kom:
                    alerts.append(f"🏆 You set a new KOM on *{seg_name}*! ({kom_str})")
                else:
                    alerts.append(
                        f"⚠️ Your KOM was beaten on *{seg_name}*!\n"
                        f"New KOM: {kom_str} (was {prev_kom_str})"
                    )

    state["segment_kom_times"] = new_koms
    return alerts

# ── Missed session detection & rebaseline ────────────────────────────────────
def get_events_on_date(date_str: str) -> list:
    """Return planned training events (non-note, non-race) for a single date."""
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/events?oldest={date_str}&newest={date_str}"
    )
    if isinstance(data, list):
        return [ev for ev in data if ev.get("category") not in ("NOTE", "RACE")]
    return []

def get_activities_on_date(date_str: str) -> list:
    """Return completed activities recorded on a single date."""
    data = intervals_get(
        f"/athlete/{INTERVALS_ATHLETE_ID}/activities?oldest={date_str}&newest={date_str}"
    )
    return data if isinstance(data, list) else []

def detect_missed_sessions(lookback_days: int = 3, processed_ids: set | None = None) -> list[dict]:
    """
    Compare planned events to actual activities for the past N days.
    Returns a list of {date, event} dicts for sessions with no matching activity.
    Already-processed event IDs (from state) are skipped to avoid re-processing.
    """
    processed_ids = processed_ids or set()
    missed = []
    today = datetime.now(AEST)

    for days_ago in range(1, lookback_days + 1):
        check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        events = get_events_on_date(check_date)
        if not events:
            continue
        activities = get_activities_on_date(check_date)
        if not activities:
            for ev in events:
                ev_id = str(ev.get("id", ""))
                if ev_id and ev_id not in processed_ids:
                    missed.append({"date": check_date, "event": ev})

    return missed

def rebaseline_schedule(missed_sessions: list[dict]) -> dict:
    """
    For each missed session, move it to the next free day within the following
    7 days by updating the event via the Intervals.icu API.

    Returns a dict:
      adjustments  — human-readable strings of what changed
      missed_count — total missed sessions found
    """
    if not missed_sessions:
        return {"adjustments": [], "missed_count": 0}

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
        for days_ahead in range(1, 8):
            candidate_date = today + timedelta(days=days_ahead)
            candidate      = candidate_date.strftime("%Y-%m-%d")
            if candidate in occupied_dates:
                continue

            # Race-week protection (same rule as apply_training_adjustment):
            # never shuffle a session into the RACE_PROTECT_DAYS window before race day
            days_to_race = (RACE_DATE.date() - candidate_date.date()).days
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
                log.info(f"Rebaseline: moved event {event_id} '{event_name}' {date} → {candidate}")
            else:
                adjustments.append(
                    f"Failed to reschedule '{event_name}' from {date} (API error)"
                )
                log.warning(f"Rebaseline: PUT failed for event {event_id}")
            break  # attempt once; move to next missed session regardless

        if not rescheduled and not any(date in a for a in adjustments[-1:]):
            reason = "race-week protection" if race_blocked else "no free slot in next 7 days"
            adjustments.append(
                f"Missed '{event_name}' on {date} — kept in plan ({reason})"
            )
            log.info(f"Rebaseline: {reason} for '{event_name}' from {date}")

    return {"adjustments": adjustments, "missed_count": len(missed_sessions)}

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
KEY_SESSION_KEYWORDS = (
    "tempo", "interval", "threshold", "hill", "race", "long run",
    "time trial", "repeats", "fartlek", "vo2", "speed",
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

def count_missed_in_window(days: int = RECONCILE_LOOKBACK_DAYS) -> tuple[int, int]:
    """(missed, planned) session counts over the trailing window — an adherence signal
    for the trend decision, independent of the rebaseline dedup logic above."""
    today = datetime.now(AEST)
    missed, planned = 0, 0
    for days_ago in range(1, days + 1):
        check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")
        events = get_events_on_date(check_date)
        if not events:
            continue
        planned += len(events)
        if not get_activities_on_date(check_date):
            missed += len(events)
    return missed, planned

def compute_training_trend(state: dict) -> dict:
    """
    Reconcile HRV vs its rolling baseline, sleep trend, fitness (CTL ramp rate),
    Form (TSB), and session adherence into a single signal: 'reduce', 'hold', or
    'progress'. This drives whether the upcoming week's schedule gets dialed back,
    held steady, or nudged up.
    """
    series = get_wellness_series()

    hrv_records = [r for r in series if r.get("hrv") is not None]
    hrv_dev_pct = None
    if len(hrv_records) >= 4:
        today_hrv = hrv_records[-1]["hrv"]
        baseline  = sum(r["hrv"] for r in hrv_records[:-1]) / len(hrv_records[:-1])
        if baseline:
            hrv_dev_pct = (today_hrv - baseline) / baseline * 100

    sleep_records = [r["sleepSecs"] for r in series if r.get("sleepSecs")]
    sleep_dev_pct = None
    if len(sleep_records) >= 5:
        recent_avg  = sum(sleep_records[-3:]) / 3
        overall_avg = sum(sleep_records) / len(sleep_records)
        if overall_avg:
            sleep_dev_pct = (recent_avg - overall_avg) / overall_avg * 100

    ctl_records = [r["ctl"] for r in series if r.get("ctl") is not None]
    tsb_latest  = next((r.get("tsb") for r in reversed(series) if r.get("tsb") is not None), None)
    ctl_ramp    = (ctl_records[-1] - ctl_records[0]) if len(ctl_records) >= 2 else None

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
            (RACE_DATE.date() - datetime.strptime(ev_date, "%Y-%m-%d").date()).days
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
            if send_telegram_proposal(summary, pid):
                proposed.append(summary)
                proposals_sent += 1
                log.info(f"Load adjustment proposed (awaiting approval): {summary}")
        elif not minor_done:
            result = intervals_put(f"/athlete/{INTERVALS_ATHLETE_ID}/events/{event_id}", updated_event)
            if result:
                applied.append(summary)
                handled_ids.add(event_id)
                log.info(f"Load adjustment auto-applied: {summary}")
            minor_done = True  # only touch one non-key session per run to avoid cascading edits

    state["load_adjusted_ids"] = list(handled_ids)
    return {"applied": applied, "proposed": proposed, "race_notes": race_notes}

# ── LLM helpers ────────────────────────────────────────────────────────────────
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
        return "⚠️ Coach is unavailable right now — check back shortly."

# ── Incoming Telegram messages ─────────────────────────────────────────────────
def get_telegram_updates(offset: int | None = None) -> list:
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates"
    params = {"timeout": 30, "allowed_updates": ["message", "callback_query"]}
    if offset is not None:
        params["offset"] = offset
    try:
        r = requests.get(url, params=params, timeout=40)
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

    decided_ids = set(state.get("load_decided_ids", []))
    decided_ids.add(pending["event_id"])
    state["load_decided_ids"] = list(decided_ids)

    if action == "approve":
        result = intervals_put(
            f"/athlete/{INTERVALS_ATHLETE_ID}/events/{pending['event_id']}",
            pending["updated_event"],
        )
        if result:
            answer_callback(callback_id, "Applied ✓")
            send_telegram(f"✅ Applied: {pending['summary']}")
            log.info(f"Load adjustment approved & applied: {pending['summary']}")
        else:
            answer_callback(callback_id, "Failed — try again")
            send_telegram(f"⚠️ Couldn't update Intervals.icu for: {pending['summary']}")
    elif action == "deny":
        answer_callback(callback_id, "Dismissed")
        send_telegram(f"👍 Kept as planned: {pending['summary']}")
    else:
        answer_callback(callback_id, "Unrecognized action")

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


async def _run_anthropic_tool_loop(
    system: str,
    user_content,
    tool_defs: list[dict],
    session: ClientSession,
    turn_id: int | None = None,
) -> tuple[str, dict | None]:
    messages = [{"role": "user", "content": user_content}]
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
) -> tuple[str, dict | None]:
    openai_tools = _foundry_tool_defs(tool_defs)

    if image_bytes:
        input_content = [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text or "Please analyse this workout image."},
                    {
                        "type": "input_image",
                        "image_url": f"data:{media_type};base64,{base64.b64encode(image_bytes).decode()}",
                    },
                ],
            }
        ]
    else:
        input_content = text

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
            inputs = json.loads(call.arguments or "{}")
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
) -> str:
    server_params = StdioServerParameters(
        command="/home/jt/.local/bin/uv",
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
                    system, text, image_bytes, media_type, tool_defs, session, turn_id
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

            reply, raw_response = await _run_anthropic_tool_loop(system, user_content, tool_defs, session, turn_id)
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
    days_out  = (RACE_DATE - datetime.now(AEST)).days
    weeks_out = days_out // 7

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
        "You are an expert running coach specialising in City2Surf preparation. "
        "You have direct access to the athlete's Intervals.icu data via tools — use them to answer questions accurately. "
        "When the athlete asks to modify their plan (add, reschedule, or delete sessions), use the appropriate tools."
        f"{strava_note}{image_note} "
        "Be specific, data-driven, and conversational — like a trusted coach responding to a text. "
        "Format for Telegram: *bold* for key points, emoji sparingly. Under 300 words unless detail is needed.\n\n"
        f"{ATHLETE_CONTEXT}\n"
        f"Today: {today_str} | Weeks to City2Surf: {weeks_out} ({days_out} days)"
    )
    model = FOUNDRY_MODEL if LLM_PROVIDER == "azure_foundry" else ANTHROPIC_MODEL
    turn_id = transcript_db.start_turn(
        chat_id, telegram_message_id, text, image_bytes, media_type, LLM_PROVIDER, model
    )
    try:
        return asyncio.run(_handle_message_async(text, system, image_bytes, media_type, turn_id))
    except Exception as e:
        log.error(f"MCP handler error: {e}")
        transcript_db.log_reply(turn_id, None, None, error=str(e))
        return "⚠️ Coach is unavailable right now — check back shortly."

# ── Daily Briefing ─────────────────────────────────────────────────────────────
def generate_briefing(state: dict | None = None) -> str:
    today_str = datetime.now(AEST).strftime("%A, %d %B %Y")
    event     = get_todays_event()
    recent    = get_recent_activities(3)

    # Wellness comes from the cache refreshed at 22:00 / 00:01 / 05:00 — so it's
    # always recent. Only fall back to a live fetch if the cache is missing or
    # older than 24h (e.g. the bot was down through every refresh slot).
    wellness  = None
    cached     = (state or {}).get("cached_wellness")
    fetched_at = (state or {}).get("cached_wellness_fetched_at", "")
    if cached and fetched_at:
        try:
            age = datetime.now(AEST) - datetime.fromisoformat(fetched_at)
            if age <= timedelta(hours=24):
                log.info(f"Using cached wellness (fetched {fetched_at}, data date={cached.get('id')})")
                wellness = cached
        except ValueError:
            pass
    if wellness is None:
        log.info("Wellness cache missing/stale — falling back to live fetch")
        wellness = get_wellness()

    # Weeks until City2Surf (first Sunday of August 2026)
    days_out  = (RACE_DATE - datetime.now(AEST)).days
    weeks_out = days_out // 7

    ctx = [f"Date: {today_str}", f"Weeks to City2Surf: {weeks_out} ({days_out} days)"]

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
        ctx.append(
            f"\nWellness: HRV {hrv or 'N/A'} ms (SDNN {hrv_sdnn or 'N/A'}) | RHR {rhr or 'N/A'} bpm"
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
                rebaseline = rebaseline_schedule(missed)
                # Record IDs so we never process the same miss twice
                new_ids = processed_ids | {str(m["event"].get("id", "")) for m in missed}
                state["processed_missed_ids"] = list(new_ids)

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
            trend      = compute_training_trend(state)
            adjustment = apply_training_adjustment(trend, state)

            trend_lines = [f"Signal: {trend['signal'].upper()} — " + "; ".join(trend["reasons"])]
            if trend.get("hrv_dev_pct") is not None:
                trend_lines.append(f"HRV vs {RECONCILE_LOOKBACK_DAYS}-day baseline: {trend['hrv_dev_pct']:+.1f}%")
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

    instructions = [
        "*Readiness* — what the wellness numbers say (use plain English, not just numbers)",
        "*Today's session* — what to do, how to pace it, what to focus on",
        "*City2Surf context* — brief mention of how today fits the race prep (Heartbreak Hill, "
        "pacing strategy, or countdown milestone if notable)",
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
        "You are an expert running coach specialising in road racing and Sydney events. "
        "You deliver sharp, motivating, data-driven daily briefings. "
        "Tone: direct, encouraging, like a trusted coach who knows the athlete well. "
        "Format for Telegram: use *bold* for headings, emoji sparingly. Length: 220-280 words."
    )
    user = (
        f"{ATHLETE_CONTEXT}\n\nData for today:\n{context_block}\n\n"
        f"Write the daily briefing. Cover:\n{numbered_instructions}\n{sign_off_line}"
    )
    return ask_llm(system, user)

# ── Post-Workout Analysis ──────────────────────────────────────────────────────
def generate_analysis(
    activity: dict,
    planned: dict | None,
    strava_segments: str = "",
    streams_summary: str = "",
    intervals_summary: str = "",
) -> str:
    name      = activity.get("name", "Run")
    wtype     = activity.get("type", "Run")
    date      = activity.get("start_date_local", "")[:10]
    dist_km   = round(activity.get("distance", 0) / 1000, 2)
    dur_min   = activity.get("moving_time", 0) // 60
    load      = activity.get("training_load")
    hr_avg    = activity.get("average_heartrate")
    hr_max    = activity.get("max_heartrate")
    pace_ms   = activity.get("average_speed")       # m/s
    ctl       = activity.get("ctl")
    atl       = activity.get("atl")
    tsb       = activity.get("tsb")
    suffer    = activity.get("suffer_score")
    elevation = activity.get("total_elevation_gain")

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

    system = (
        "You are an expert running coach specialising in City2Surf preparation. "
        "Deliver post-run feedback that's specific, data-driven, and actionable. "
        "Format for Telegram: *bold* headings, emoji sparingly. Length: 280-340 words."
    )
    seg_block       = f"\n\nStrava segment data:\n{strava_segments}" if strava_segments else ""
    streams_block   = f"\n\nPer-km stream analysis:\n{streams_summary}" if streams_summary else ""
    intervals_block = f"\n\n{intervals_summary}" if intervals_summary else ""
    user = (
        f"{ATHLETE_CONTEXT}\n\nActual run:\n{actual}\n\nPlan:\n{plan_ctx}"
        f"{intervals_block}{seg_block}{streams_block}\n\n"
        "Write the post-workout analysis. Cover:\n"
        "1. *Execution* — how well did actual match the plan? What the numbers show. "
        "If detected intervals are provided, use those rep-by-rep numbers as the source of truth for "
        "whether a structured session (e.g. reps at target pace) was actually executed — they separate "
        "work reps from recoveries, unlike the per-km stream splits which are just fixed GPS-distance "
        "buckets and can blend reps with recovery jogging. Only fall back to per-km splits or hedge "
        "on execution if no interval data was provided.\n"
        "2. *Physiological read* — what HR, pace, and load tell us about effort and fitness.\n"
        "3. *City2Surf relevance* — did this session build anything specific for race day "
        "(hill strength, threshold, aerobic base, fatigue management)?\n"
        "4. *Segment highlights* — if Strava data is provided, call out any notable PR gaps, "
        "KOM opportunities, or segments where the athlete was close to something special.\n"
        "5. *Next session tip* — one concrete, specific thing to focus on next time.\n"
        "Be honest — if it was too easy, too hard, or off-plan, say so constructively."
    )
    return ask_llm(system, user)

# ── Main loop ──────────────────────────────────────────────────────────────────
def run():
    log.info("🏃 City2Surf coaching bot started")
    state = load_state()

    # Ensure processed_missed_ids list exists in state
    if "processed_missed_ids" not in state:
        state["processed_missed_ids"] = []
        save_state(state)

    # On first run, record baseline activity and Strava athlete ID
    if state.get("last_activity_id") is None:
        latest = get_latest_activity()
        if latest:
            state["last_activity_id"] = str(latest.get("id"))
            save_state(state)
            log.info(f"Baseline activity set: {state['last_activity_id']}")

    if STRAVA_ENABLED and not state.get("strava_athlete_id"):
        athlete_id = get_strava_athlete_id()
        if athlete_id:
            state["strava_athlete_id"] = athlete_id
            save_state(state)
            log.info(f"Strava athlete ID: {athlete_id}")
    elif not STRAVA_ENABLED:
        log.info("Strava integration disabled — run strava_setup.py to enable")

    last_activity_check = 0.0
    last_kom_check      = 0.0

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
                    log.info(
                        f"Wellness cached: HRV={w.get('hrv')} ms | "
                        f"RHR={w.get('restingHR')} bpm | "
                        f"Sleep={round(w.get('sleepSecs',0)/3600,1)}h | "
                        f"date={w.get('id')}"
                    )
                else:
                    log.warning("Wellness cache refresh returned no data")
            except Exception as e:
                log.error(f"Wellness cache refresh error: {e}")

        # ── Daily briefing at 05:15 AEST ──────────────────────────────────────
        if now.hour == 5 and now.minute >= 15 and now.minute < 20 and state.get("last_briefing_date") != today_str:
            log.info("Generating daily briefing…")
            try:
                briefing = generate_briefing(state)
                header   = f"🌅 *Good morning! Daily Training Briefing*\n_{now.strftime('%A, %d %B')}_\n\n"
                send_telegram(header + briefing)
                state["last_briefing_date"] = today_str
                save_state(state)
            except Exception as e:
                log.error(f"Briefing error: {e}")

        # ── Post-workout analysis — poll every 5 minutes ───────────────────────
        if time.time() - last_activity_check >= 300:
            try:
                latest = get_latest_activity()
                if latest:
                    act_id = str(latest.get("id"))
                    if act_id != state.get("last_activity_id"):
                        log.info(f"New activity detected: {act_id}")
                        state["last_activity_id"] = act_id  # mark seen immediately to prevent retry loop
                        save_state(state)
                        time.sleep(90)  # wait for Intervals to finish processing
                        activity = get_activity_detail(act_id) or latest
                        planned  = get_todays_event()

                        # ── Streams analysis ───────────────────────────────────
                        streams_summary = ""
                        try:
                            streams = get_activity_streams(act_id)
                            if streams:
                                streams_summary = summarize_streams(streams)
                                log.info(f"Streams fetched for activity {act_id}")
                        except Exception as e:
                            log.error(f"Streams fetch error: {e}")

                        # ── Interval/lap analysis ──────────────────────────────
                        intervals_summary = ""
                        try:
                            intervals_data = get_activity_intervals(act_id)
                            if intervals_data:
                                intervals_summary = summarize_intervals(intervals_data)
                                if intervals_summary:
                                    log.info(f"Intervals fetched for activity {act_id}")
                        except Exception as e:
                            log.error(f"Intervals fetch error: {e}")

                        # ── Strava segment enrichment ──────────────────────────
                        strava_segments = ""
                        chase_alerts    = []
                        if STRAVA_ENABLED:
                            try:
                                strava_act = find_strava_match(activity)
                                if strava_act:
                                    strava_detail = get_strava_activity_detail(
                                        str(strava_act["id"])
                                    )
                                    if strava_detail:
                                        athlete_id      = state.get("strava_athlete_id")
                                        strava_segments = build_segment_report(strava_detail, athlete_id)
                                        chase_alerts    = get_pr_kom_chases(strava_detail)
                                        log.info(f"Strava segments enriched for activity {strava_act['id']}")
                            except Exception as e:
                                log.error(f"Strava enrichment error: {e}")

                        analysis = generate_analysis(
                            activity, planned, strava_segments, streams_summary, intervals_summary
                        )
                        header   = "💪 *Post-Run Analysis*\n\n"
                        send_telegram(header + analysis)

                        # Send PR/KOM chase highlights as a follow-up message
                        if chase_alerts:
                            chase_msg = "🔥 *Chase these next run:*\n\n" + "\n".join(chase_alerts)
                            send_telegram(chase_msg)

                last_activity_check = time.time()
            except Exception as e:
                log.error(f"Activity check error: {e}")

        # ── Daily KOM check ────────────────────────────────────────────────────
        if STRAVA_ENABLED and time.time() - last_kom_check >= 86400:
            try:
                log.info("Running daily KOM check…")
                kom_alerts = check_kom_alerts(state)
                if kom_alerts:
                    msg = "🔔 *Strava KOM Update*\n\n" + "\n\n".join(kom_alerts)
                    send_telegram(msg)
                save_state(state)
                last_kom_check = time.time()
            except Exception as e:
                log.error(f"KOM check error: {e}")

        # ── Incoming messages — long-poll (blocks up to 30s) ──────────────────
        updates = get_telegram_updates(state.get("last_update_id"))
        for update in updates:
            state["last_update_id"] = update.get("update_id", 0) + 1

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

            log.info(f"Incoming message: {text[:60]}" + (" [+image]" if image_bytes else ""))
            reply = handle_incoming_message(text, image_bytes, media_type, chat_id, msg.get("message_id"))
            send_telegram(reply)
            save_state(state)

if __name__ == "__main__":
    run()
