"""Characterisation tests for detect_missed_sessions (Phase 0 task 0.4 — this
function had zero coverage despite being named in the plan and touched by the
L3/N2 fixes in tests/test_schedule.py's rebaseline_schedule, which consumes
its output)."""

from datetime import datetime, timedelta


def _on(bot, days_ago):
    return (datetime.now(bot.AEST) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def _event(event_id, date, moving_time=1800):
    return {"id": event_id, "start_date_local": f"{date}T06:00:00", "moving_time": moving_time,
            "category": "WORKOUT"}


def _activity(date, moving_time=1800):
    return {"start_date_local": f"{date}T06:30:00", "moving_time": moving_time}


def _patch(bot, monkeypatch, events, activities, strava_activities=()):
    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)
    monkeypatch.setattr(bot, "get_activities_range", lambda oldest, newest: activities)
    monkeypatch.setattr(bot, "get_strava_activities_since", lambda days: list(strava_activities))


def test_matched_activity_is_not_missed(bot, monkeypatch):
    date = _on(bot, 1)
    events = [_event("e1", date)]
    activities = [_activity(date)]
    _patch(bot, monkeypatch, events, activities)

    assert bot.detect_missed_sessions() == []


def test_unmatched_event_with_no_strava_activity_is_missed(bot, monkeypatch):
    date = _on(bot, 1)
    events = [_event("e1", date)]
    _patch(bot, monkeypatch, events, activities=[])

    missed = bot.detect_missed_sessions()
    assert [m["event"]["id"] for m in missed] == ["e1"]
    assert missed[0]["date"] == date


def test_strava_sync_lag_excuses_an_unsynced_activity(bot, monkeypatch):
    """An activity that's landed on Strava but not yet Intervals.icu isn't a
    real miss — it's sync lag, and must not be reported."""
    date = _on(bot, 1)
    events = [_event("e1", date)]
    strava_activity = {"start_date_local": f"{date}T06:30:00"}
    _patch(bot, monkeypatch, events, activities=[], strava_activities=[strava_activity])

    assert bot.detect_missed_sessions() == []


def test_already_processed_event_is_skipped(bot, monkeypatch):
    date = _on(bot, 1)
    events = [_event("e1", date)]
    _patch(bot, monkeypatch, events, activities=[])

    assert bot.detect_missed_sessions(processed_ids={"e1"}) == []


def test_greedy_duration_matching_picks_closest_activity(bot, monkeypatch):
    """Two events and two activities on the same day: each event should claim
    the activity closest to its planned duration, not just the first one."""
    date = _on(bot, 1)
    events = [
        _event("short", date, moving_time=1200),   # 20 min
        _event("long", date, moving_time=5400),    # 90 min
    ]
    activities = [
        _activity(date, moving_time=5000),  # closer to "long"
        _activity(date, moving_time=1300),  # closer to "short"
    ]
    _patch(bot, monkeypatch, events, activities)

    assert bot.detect_missed_sessions() == []


def test_unmatched_second_event_on_a_partially_covered_day_is_missed(bot, monkeypatch):
    """One activity that day only covers one of two planned events."""
    date = _on(bot, 1)
    events = [_event("e1", date, moving_time=1800), _event("e2", date, moving_time=3600)]
    activities = [_activity(date, moving_time=1800)]
    _patch(bot, monkeypatch, events, activities)

    missed = bot.detect_missed_sessions()
    assert [m["event"]["id"] for m in missed] == ["e2"]
