"""Regression tests for apply_training_adjustment's minor-adjustment slot (N3)."""

from datetime import datetime, timedelta


def _far_race(bot, monkeypatch):
    # None of these events should collide with race-week protection — keep the
    # race well outside the RECONCILE_LOOKAHEAD_DAYS window regardless of when
    # this suite runs (see tests/test_schedule.py for the same reasoning).
    monkeypatch.setattr(bot, "RACE_DATE", datetime.now(bot.AEST) + timedelta(days=200))


def _event(event_id, name, days_ahead, bot):
    date = (datetime.now(bot.AEST) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
    return {
        "id": event_id,
        "name": name,
        "start_date_local": f"{date}T06:00:00",
        "moving_time": 1800,
        "distance": 5000,
        "category": "WORKOUT",
    }


def test_failed_put_does_not_consume_minor_slot(bot, monkeypatch, state_file):
    """N3: a failed PUT for a non-key session must not consume the one-per-run
    minor-adjustment slot. The old bug set minor_done=True unconditionally
    (even when intervals_put returned None), so a single Intervals API blip
    meant zero adjustments were made for the entire run while the briefing
    still reported normally."""
    _far_race(bot, monkeypatch)

    events = [_event("e1", "Easy run", 1, bot), _event("e2", "Easy run 2", 2, bot)]
    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)

    put_calls = []

    def fake_put(path, data):
        put_calls.append(data)
        return None if len(put_calls) == 1 else {"ok": True}  # first fails, second succeeds

    monkeypatch.setattr(bot, "intervals_put", fake_put)

    trend = {"signal": "reduce", "reasons": ["testing"]}
    result = bot.apply_training_adjustment(trend, {})

    # Both events were attempted — the first failure didn't burn the slot.
    assert len(put_calls) == 2
    date2 = events[1]["start_date_local"][:10]
    assert result["applied"] == [f"{date2}: 'Easy run 2' — reduce volume 12%"]


def test_successful_put_still_consumes_minor_slot(bot, monkeypatch, state_file):
    """The one-per-run cap is intentional (L8) — only the failure path changed.
    A successful PUT on the first non-key session must still stop a second one
    from being auto-applied in the same run."""
    _far_race(bot, monkeypatch)

    events = [_event("e1", "Easy run", 1, bot), _event("e2", "Easy run 2", 2, bot)]
    monkeypatch.setattr(bot, "get_events_range", lambda oldest, newest: events)

    put_calls = []

    def fake_put(path, data):
        put_calls.append(data)
        return {"ok": True}

    monkeypatch.setattr(bot, "intervals_put", fake_put)

    trend = {"signal": "reduce", "reasons": ["testing"]}
    result = bot.apply_training_adjustment(trend, {})

    assert len(put_calls) == 1
    assert len(result["applied"]) == 1
    assert "Easy run'" in result["applied"][0]
