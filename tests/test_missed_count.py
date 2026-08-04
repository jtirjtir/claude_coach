"""Regression tests for count_missed_in_window's sport-aware matching (L5)."""

from datetime import datetime, timedelta


def _on(bot, days_ago):
    return (datetime.now(bot.AEST) - timedelta(days=days_ago)).strftime("%Y-%m-%d")


def test_cross_sport_activity_does_not_satisfy_planned_session(bot, monkeypatch):
    """L5: a planned run is only satisfied by a run. Before this fix, any
    activity on the same day counted against any planned session that day, so
    a 30-minute bike ride would silently satisfy a missed 60-minute run —
    under-reporting misses and over-stating adherence to compute_training_trend."""
    date = _on(bot, 2)
    events = [{"start_date_local": f"{date}T06:00:00", "type": "Run", "category": "WORKOUT"}]
    activities = [{"start_date_local": f"{date}T07:00:00", "type": "Ride"}]

    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)
    monkeypatch.setattr(bot, "get_activities_range", lambda oldest, newest: activities)

    missed, planned = bot.count_missed_in_window(days=7)
    assert (missed, planned) == (1, 1)


def test_same_sport_activity_satisfies_planned_session(bot, monkeypatch):
    """Sanity check the matching direction: a same-sport activity on the same
    day does count as the session being completed."""
    date = _on(bot, 2)
    events = [{"start_date_local": f"{date}T06:00:00", "type": "Run", "category": "WORKOUT"}]
    activities = [{"start_date_local": f"{date}T07:00:00", "type": "Run"}]

    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)
    monkeypatch.setattr(bot, "get_activities_range", lambda oldest, newest: activities)

    missed, planned = bot.count_missed_in_window(days=7)
    assert (missed, planned) == (0, 1)


def test_untyped_event_falls_back_to_day_level_match(bot, monkeypatch):
    """An event with no sport type can't be matched by sport, so it falls back
    to the old any-activity-that-day behaviour rather than being declared
    missed on a technicality."""
    date = _on(bot, 2)
    events = [{"start_date_local": f"{date}T06:00:00", "type": "", "category": "WORKOUT"}]
    activities = [{"start_date_local": f"{date}T07:00:00", "type": "Ride"}]

    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)
    monkeypatch.setattr(bot, "get_activities_range", lambda oldest, newest: activities)

    missed, planned = bot.count_missed_in_window(days=7)
    assert (missed, planned) == (0, 1)
