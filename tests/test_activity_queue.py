"""Regression tests for queue_unseen_intervals_activities (X2)."""


def _activity(act_id, date, distance=5000):
    return {"id": act_id, "start_date_local": f"{date}T06:00:00", "distance": distance}


def test_backlog_activities_are_queued_oldest_first(bot, monkeypatch, state_file):
    """X2: every unseen activity in the recent-activities window must be
    queued, not just the newest. The old bug compared a single
    `last_activity_id`, so any activity that appeared between polls — or a
    backlog built up while the bot was down — was silently skipped once a
    newer one advanced the marker past it."""
    a1 = _activity("a1", "2026-08-01")
    a2 = _activity("a2", "2026-08-02")  # already seen
    a3 = _activity("a3", "2026-08-03")

    # Deliberately out of date order, to confirm the function sorts rather
    # than relying on the API returning activities oldest-first.
    monkeypatch.setattr(bot, "get_recent_activities", lambda n: [a3, a1, a2])

    state = {"seen_activity_ids": ["a2"], "pending_analysis": []}
    queue = bot.queue_unseen_intervals_activities(state)

    assert [e["act_id"] for e in queue] == ["a1", "a3"]
    assert all(e["source"] == "intervals" for e in queue)
    assert state["pending_analysis"] == queue
    assert set(state["seen_activity_ids"]) == {"a1", "a2", "a3"}


def test_already_queued_activity_is_not_duplicated(bot, monkeypatch, state_file):
    """An activity already sitting in pending_analysis (e.g. queued by the
    Strava fallback first) must not be enqueued a second time when Intervals
    also picks it up."""
    a1 = _activity("a1", "2026-08-01")
    sig = list(bot._activity_signature(a1))

    monkeypatch.setattr(bot, "get_recent_activities", lambda n: [a1])

    state = {
        "seen_activity_ids": [],
        "pending_analysis": [
            {"act_id": "strava-a1", "source": "strava", "attempts": 0,
             "next_attempt_at": "2026-08-01T00:00:00+10:00", "sig": sig},
        ],
    }
    queue = bot.queue_unseen_intervals_activities(state)

    assert [e["act_id"] for e in queue] == ["strava-a1"]
    assert "a1" in state["seen_activity_ids"]  # marked seen even though not (re)queued


def test_no_new_activities_leaves_queue_untouched(bot, monkeypatch, state_file):
    a1 = _activity("a1", "2026-08-01")
    monkeypatch.setattr(bot, "get_recent_activities", lambda n: [a1])

    state = {"seen_activity_ids": ["a1"], "pending_analysis": []}
    queue = bot.queue_unseen_intervals_activities(state)

    assert queue == []
    assert state["pending_analysis"] == []
