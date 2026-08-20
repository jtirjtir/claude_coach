"""Tests for the dynamic training goal and the /goal command."""

from datetime import date, datetime, timedelta

import pytest

from conftest import set_goal_days_out

TODAY = date(2026, 1, 15)


@pytest.fixture(autouse=True)
def _default_goal(bot, monkeypatch):
    """Every test starts from the shipped default, whatever state.json holds."""
    monkeypatch.setattr(bot, "_active_goal", dict(bot.DEFAULT_GOAL))


def _set(bot, arg, current=None):
    """Run a /goal argument through the parser and return (goal, reply)."""
    return bot.parse_goal_command(arg, current or bot.get_goal(), TODAY)


# ── Date parsing ─────────────────────────────────────────────────────────────
@pytest.mark.parametrize("text,expected", [
    ("2026-11-01",  date(2026, 11, 1)),
    ("2026/11/01",  date(2026, 11, 1)),
    ("01/11/2026",  date(2026, 11, 1)),   # day-first: AU convention, not US
    ("01-11-2026",  date(2026, 11, 1)),
    ("1 Nov 2026",  date(2026, 11, 1)),
    ("1 November 2026", date(2026, 11, 1)),
    ("Nov 1 2026",  date(2026, 11, 1)),
    ("next tuesday", None),
    ("", None),
    ("2026-13-45", None),
])
def test_parse_goal_date(bot, text, expected):
    assert bot.parse_goal_date(text) == expected


# ── Presets ──────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("typed,key,sport", [
    ("city2surf",  "city2surf",  "run"),
    ("C2S",        "city2surf",  "run"),
    ("gong",       "msgong",     "ride"),
    ("Sydney2Gong","msgong",     "ride"),
    ("bowral",     "bowral",     "ride"),
    ("Bowral Classic", "bowral", "ride"),
    ("ironman",    "ironman",    "triathlon"),
    ("70.3",       "703",        "triathlon"),
    ("half",       "halfmarathon", "run"),
])
def test_preset_aliases_resolve(bot, typed, key, sport):
    goal = bot.goal_from_preset(typed)
    assert goal["key"] == key
    assert goal["sport"] == sport


def test_every_preset_normalises_once_given_a_date(bot):
    for key in bot.GOAL_PRESETS:
        goal = bot.normalise_goal(bot.goal_from_preset(key, "2027-05-02"))
        assert goal is not None, key
        assert goal["sport"] in bot.GOAL_SPORTS, key
        assert goal["name"], key


def test_preset_aliases_all_point_at_real_presets(bot):
    unknown = set(bot.GOAL_ALIASES.values()) - set(bot.GOAL_PRESETS)
    assert not unknown


# ── Setting goals ────────────────────────────────────────────────────────────
def test_set_dated_preset(bot):
    goal, reply = _set(bot, "gong")
    assert goal["name"] == "MS Sydney to the Gong Ride"
    assert goal["date"] == "2026-11-01"
    assert goal["sport"] == "ride"
    assert "Goal set" in reply


def test_explicit_date_overrides_preset_default(bot):
    goal, _ = _set(bot, "bowral 2027-10-24")
    assert goal["key"] == "bowral"
    assert goal["date"] == "2027-10-24"


def test_dateless_preset_is_refused_until_given_a_date(bot):
    goal, reply = _set(bot, "marathon")
    assert goal is None
    assert "no fixed date" in reply

    goal, _ = _set(bot, "marathon 2027-05-02")
    assert goal["date"] == "2027-05-02"
    assert goal["distance_km"] == pytest.approx(42.2)


def test_distance_preset_names_are_not_eaten_as_distances(bot):
    """'10k' and '70.3' look like distances — the preset lookup must win."""
    goal, _ = _set(bot, "10k 2026-09-20")
    assert goal["key"] == "10k"
    assert goal["distance_km"] == 10

    goal, _ = _set(bot, "70.3 2026-09-20")
    assert goal["key"] == "703"


def test_unknown_event_with_a_date_becomes_a_one_off(bot):
    goal, reply = _set(bot, "six foot track 2027-03-13")
    assert goal["name"] == "Six Foot Track"
    assert goal["date"] == "2027-03-13"
    assert goal["sport"] == "other"
    assert "Goal set" in reply


def test_unknown_event_sport_is_inferred_from_the_name(bot):
    goal, _ = _set(bot, "Sydney Marathon 2026-08-30")
    assert goal["name"] == "Sydney Marathon"
    assert goal["sport"] == "run"


def test_unknown_event_without_a_date_asks_for_one(bot):
    goal, reply = _set(bot, "six foot track")
    assert goal is None
    assert "give me a date" in reply


def test_free_form_pipe_fields(bot):
    goal, _ = _set(
        bot,
        "Cairns Ironman | 14/06/2027 | triathlon | 226km | hot and humid, long bike",
    )
    assert goal["name"] == "Cairns Ironman"
    assert goal["date"] == "2027-06-14"
    assert goal["sport"] == "triathlon"
    assert goal["distance_km"] == 226
    assert "humid" in goal["challenge"]


def test_free_form_needs_name_and_date(bot):
    goal, reply = _set(bot, "Cairns Ironman |")
    assert goal is None
    assert "name and a date" in reply


def test_free_form_keeps_deliberate_casing(bot):
    goal, _ = _set(bot, "MS Gong Ride | 2026-11-01")
    assert goal["name"] == "MS Gong Ride"


# ── Subcommands ──────────────────────────────────────────────────────────────
def test_bare_goal_shows_current_without_changing_it(bot):
    goal, reply = _set(bot, "")
    assert goal is None
    assert bot.DEFAULT_GOAL["name"] in reply


def test_date_subcommand_moves_the_current_goal_only(bot):
    current = bot.goal_from_preset("bowral")
    current = bot.normalise_goal(current)
    goal, reply = _set(bot, "date 05/10/2027", current)
    assert goal["key"] == "bowral"
    assert goal["date"] == "2027-10-05"
    assert goal["distance_km"] == current["distance_km"]


def test_date_subcommand_rejects_junk(bot):
    goal, reply = _set(bot, "date sometime in spring")
    assert goal is None
    assert "Couldn't read a date" in reply


def test_reset_returns_the_default(bot):
    goal, reply = _set(bot, "reset", bot.normalise_goal(bot.goal_from_preset("bowral")))
    assert goal["key"] == bot.DEFAULT_GOAL["key"]
    assert "reset" in reply.lower()


def test_list_and_help_change_nothing(bot):
    for arg in ("list", "help"):
        goal, reply = _set(bot, arg)
        assert goal is None
        assert reply


def test_list_names_every_preset(bot):
    reply = bot.format_goal_presets()
    for key in bot.GOAL_PRESETS:
        assert f"/goal {key}" in reply


def test_empty_name_is_rejected(bot):
    goal, reply = _set(bot, "2027-05-02")
    assert goal is None
    assert "what the goal is" in reply


# ── Normalisation & fallbacks ────────────────────────────────────────────────
@pytest.mark.parametrize("raw", [
    None, {}, "city2surf", {"name": "No Date"}, {"date": "2027-05-02"},
    {"name": "Bad Date", "date": "whenever"},
])
def test_unusable_goals_normalise_to_none(bot, raw):
    assert bot.normalise_goal(raw) is None


def test_set_active_goal_falls_back_to_default(bot):
    assert bot.set_active_goal({"name": "No Date"})["key"] == bot.DEFAULT_GOAL["key"]


def test_corrupt_stored_goal_does_not_break_startup(bot):
    """A hand-edited state.json must not take the briefing down."""
    state = {"goal": {"name": "Broken", "date": "not-a-date"}}
    goal = bot.load_goal(state)
    assert goal["key"] == bot.DEFAULT_GOAL["key"]
    assert state["goal"] == goal   # canonical shape written back


def test_load_goal_adopts_a_stored_goal(bot):
    state = {"goal": {"name": "Bowral Classic", "date": "2027-10-24", "sport": "cycling"}}
    goal = bot.load_goal(state)
    assert goal["sport"] == "ride"          # alias normalised
    assert bot.get_goal()["name"] == "Bowral Classic"
    assert bot.race_date().date() == date(2027, 10, 24)


# ── Derived values ───────────────────────────────────────────────────────────
def test_race_date_and_countdown_follow_the_active_goal(bot, monkeypatch):
    set_goal_days_out(bot, monkeypatch, 200, name="Test Event")
    assert bot.days_to_goal() == 200
    assert "Weeks to Test Event: 28 (200 days)" == bot.goal_countdown()


def test_countdown_says_so_once_the_event_has_passed(bot, monkeypatch):
    set_goal_days_out(bot, monkeypatch, -3, name="Test Event")
    countdown = bot.goal_countdown()
    assert "3 days ago" in countdown
    assert "/goal" in countdown


def test_countdown_on_the_day(bot, monkeypatch):
    set_goal_days_out(bot, monkeypatch, 0, name="Test Event")
    assert bot.goal_countdown() == "Test Event is TODAY"


@pytest.mark.parametrize("sport,expected", [
    ("run", "expert running coach"),
    ("ride", "expert cycling coach"),
    ("triathlon", "expert triathlon coach"),
    ("other", "expert endurance coach"),
])
def test_coach_persona_follows_the_sport(bot, sport, expected):
    assert bot.coach_persona({"sport": sport}) == expected


def test_athlete_context_reflects_the_goal(bot, monkeypatch):
    bot.set_active_goal(bot.goal_from_preset("bowral"))
    ctx = bot.athlete_context()
    assert "Bowral Classic" in ctx
    assert "160 km, ride" in ctx
    assert "Southern Highlands" in ctx
    assert "City2Surf" not in ctx


def test_athlete_context_omits_fields_a_one_off_goal_lacks(bot):
    bot.set_active_goal({"name": "Six Foot Track", "date": "2027-03-13", "sport": "run"})
    ctx = bot.athlete_context()
    assert "Six Foot Track" in ctx
    assert "Key challenge" not in ctx
    assert "Location" not in ctx
    assert "Intervals.icu" in ctx


# ── Persistence via handle_goal_command ──────────────────────────────────────
def test_handle_goal_command_persists(bot, state_file, monkeypatch):
    state = {}
    reply = bot.handle_goal_command("bowral", state)

    assert "Bowral Classic" in reply
    assert state["goal"]["key"] == "bowral"
    assert bot.get_goal()["key"] == "bowral"          # module cache updated
    assert bot.race_date().date() == date(2026, 10, 25)

    import json
    assert json.loads(state_file.read_text())["goal"]["key"] == "bowral"


def test_handle_goal_command_leaves_state_alone_on_a_read(bot, state_file):
    state = {}
    bot.handle_goal_command("list", state)
    assert "goal" not in state
    assert not state_file.exists()


def test_goal_survives_a_save_load_round_trip(bot, state_file):
    state = {}
    bot.handle_goal_command("Sydney Marathon 2026-08-30", state)

    bot.set_active_goal(None)                          # simulate a restart
    assert bot.get_goal()["key"] == bot.DEFAULT_GOAL["key"]

    reloaded = bot.load_goal(bot.load_state())
    assert reloaded["name"] == "Sydney Marathon"
    assert reloaded["date"] == "2026-08-30"


# ── Race protection tracks the goal, not a constant ──────────────────────────
def test_race_protection_window_moves_with_the_goal(bot, monkeypatch, no_telegram):
    """A session 5 days before goal day is protected; the same session is
    fair game once the goal moves a year out."""
    ev_date = (datetime.now(bot.AEST) + timedelta(days=3)).strftime("%Y-%m-%d")
    event = {
        "id": 1, "name": "Threshold intervals",
        "start_date_local": f"{ev_date}T06:00:00",
        "moving_time": 3600, "distance": 12000,
    }
    monkeypatch.setattr(bot, "get_events_range", lambda o, n: [event])
    monkeypatch.setattr(bot, "intervals_put", lambda p, d: d)
    trend = {"signal": "reduce", "reasons": ["HRV suppressed"]}

    set_goal_days_out(bot, monkeypatch, 8)             # event falls inside the window
    result = bot.apply_training_adjustment(trend, {})
    assert result["race_notes"] and not result["applied"] and not result["proposed"]

    set_goal_days_out(bot, monkeypatch, 365)           # same event, distant goal
    result = bot.apply_training_adjustment(trend, {})
    assert not result["race_notes"]
    assert result["applied"] or result["proposed"]
