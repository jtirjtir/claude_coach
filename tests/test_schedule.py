"""Regression tests for rebaseline scheduling (L3, N2, X1)."""

import json
from datetime import datetime, timedelta

import pytest

from conftest import set_goal_days_out


@pytest.fixture(autouse=True)
def _race_far_away(bot, monkeypatch):
    """Every test here reasons about which of the next 7 days are free, using
    the real wall clock for 'today'. RACE_PROTECT_DAYS blocks candidates within
    10 days of the goal event — as goal day approaches (or has just passed),
    that window can swallow part or all of the 7-day lookahead and silently
    change which branch a test exercises. Pin the goal far away so these tests
    reason only about occupied-date/PUT-outcome logic, independent of when the
    suite runs and of whatever goal is currently set."""
    set_goal_days_out(bot, monkeypatch, 200)


def _event(event_id, name, date, **extra):
    return {"id": event_id, "name": name, "start_date_local": f"{date}T06:00:00", **extra}


def _missed(event_id, name, date):
    return {"date": date, "event": _event(event_id, name, date)}


def _yesterday(bot):
    return (datetime.now(bot.AEST) - timedelta(days=1)).strftime("%Y-%m-%d")


def test_two_sessions_missed_on_same_date_both_reported(bot, monkeypatch):
    """L3: the first session's 'Rescheduled ... from <date>' line contains the
    second session's date too. The old substring guard saw that and silently
    suppressed the second session's message, so the athlete was never told."""
    date = _yesterday(bot)
    # Every upcoming day is occupied, so nothing can be rescheduled...
    upcoming = [
        _event(f"u{i}", "Planned", (datetime.now(bot.AEST) + timedelta(days=i)).strftime("%Y-%m-%d"))
        for i in range(1, 9)
    ]
    monkeypatch.setattr(bot, "intervals_get", lambda path: upcoming)
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})

    result = bot.rebaseline_schedule([_missed("e1", "Easy run", date), _missed("e2", "Strides", date)])
    text = "\n".join(result["adjustments"])
    assert "Easy run" in text
    assert "Strides" in text
    assert result["missed_count"] == 2


def test_same_date_one_rescheduled_one_kept(bot, monkeypatch):
    """The specific collision: session A gets a free slot, session B doesn't."""
    date = _yesterday(bot)
    # Only day+1 is free; A takes it, so B has nowhere to go.
    occupied = [
        _event(f"u{i}", "Planned", (datetime.now(bot.AEST) + timedelta(days=i)).strftime("%Y-%m-%d"))
        for i in range(2, 9)
    ]
    monkeypatch.setattr(bot, "intervals_get", lambda path: occupied)
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})

    result = bot.rebaseline_schedule([_missed("e1", "Session A", date), _missed("e2", "Session B", date)])
    text = "\n".join(result["adjustments"])
    assert "Rescheduled 'Session A'" in text
    assert "Missed 'Session B'" in text and "kept in plan" in text


def test_failed_put_does_not_mark_session_resolved(bot, monkeypatch):
    """N2: a transient Intervals error must leave the session retryable, not
    retire it permanently via processed_missed_ids."""
    date = _yesterday(bot)
    monkeypatch.setattr(bot, "intervals_get", lambda path: [])
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: None)  # API error

    result = bot.rebaseline_schedule([_missed("e1", "Tempo", date)])
    assert result["resolved_ids"] == set()
    assert any("Failed to reschedule" in a for a in result["adjustments"])
    # And no duplicate "kept in plan" line for the same session.
    assert not any("kept in plan" in a for a in result["adjustments"])


def test_successful_reschedule_is_resolved(bot, monkeypatch):
    date = _yesterday(bot)
    monkeypatch.setattr(bot, "intervals_get", lambda path: [])
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})

    result = bot.rebaseline_schedule([_missed("e1", "Tempo", date)])
    assert result["resolved_ids"] == {"e1"}


def test_kept_in_plan_is_resolved(bot, monkeypatch):
    """No free slot is a final outcome — don't re-report it every day."""
    date = _yesterday(bot)
    occupied = [
        _event(f"u{i}", "Planned", (datetime.now(bot.AEST) + timedelta(days=i)).strftime("%Y-%m-%d"))
        for i in range(1, 9)
    ]
    monkeypatch.setattr(bot, "intervals_get", lambda path: occupied)
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})

    result = bot.rebaseline_schedule([_missed("e1", "Easy", date)])
    assert result["resolved_ids"] == {"e1"}


def test_reschedule_persists_state_immediately(bot, monkeypatch, state_file):
    """X1: the remote write is live the moment intervals_put returns. If the
    caller crashes during its subsequent LLM call, the resolution must already
    be on disk."""
    date = _yesterday(bot)
    monkeypatch.setattr(bot, "intervals_get", lambda path: [])
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})

    state = {"processed_missed_ids": []}
    bot.rebaseline_schedule([_missed("e1", "Tempo", date)], state)

    on_disk = json.loads(state_file.read_text())
    assert "e1" in on_disk["processed_missed_ids"]


def test_no_state_threading_still_works(bot, monkeypatch):
    """state is optional — callers that don't need durability can omit it."""
    date = _yesterday(bot)
    monkeypatch.setattr(bot, "intervals_get", lambda path: [])
    monkeypatch.setattr(bot, "intervals_put", lambda path, data: {"ok": True})
    result = bot.rebaseline_schedule([_missed("e1", "Tempo", date)])
    assert result["missed_count"] == 1


def test_empty_missed_list_returns_resolved_ids_key(bot):
    result = bot.rebaseline_schedule([])
    assert result == {"adjustments": [], "missed_count": 0, "resolved_ids": set()}
