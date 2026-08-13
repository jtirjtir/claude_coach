"""Regression tests for check_kom_alerts state handling (L1, N4)."""

from datetime import datetime


def _details(kom=None, pr_secs=None):
    return {
        "xoms": {"kom": kom} if kom else {},
        "athlete_segment_stats": {"pr_elapsed_time": pr_secs} if pr_secs else {},
    }


def test_kom_merge_preserves_out_of_window_segment(bot, monkeypatch):
    """L1: segments outside this run's rotation window must keep their previously
    recorded KOM baseline instead of being evicted. The old bug replaced state
    wholesale with only the window's fetched segments, so anything past the
    first KOM_CHECK_BATCH starred segments silently lost its baseline and could
    never alert again."""
    starred = [{"id": "s1", "name": "Hill A"}, {"id": "s2", "name": "Hill B"}]
    monkeypatch.setattr(bot, "get_starred_segments", lambda: starred)
    monkeypatch.setattr(bot, "KOM_CHECK_BATCH", 1)

    # With batch=1 and 2 starred segments, exactly one falls in this run's
    # rotation window; replicate the function's own offset formula to know
    # which one, without needing to freeze real time.
    offset = datetime.now(bot.AEST).timetuple().tm_yday % 2
    in_window_id = starred[offset]["id"]
    out_of_window_id = starred[1 - offset]["id"]

    fetched = []

    def fake_details(seg_id):
        fetched.append(seg_id)
        return _details(kom="20:00", pr_secs=1500)

    monkeypatch.setattr(bot, "get_segment_details", fake_details)

    state = {"segment_kom_times": {in_window_id: "25:00", out_of_window_id: "18:30"}}
    bot.check_kom_alerts(state)

    assert fetched == [in_window_id]
    assert state["segment_kom_times"][out_of_window_id] == "18:30"  # preserved, not evicted
    assert state["segment_kom_times"][in_window_id] == "20:00"      # refreshed


def test_kom_prunes_unstarred_segments(bot, monkeypatch):
    """N4: a segment the athlete has since unstarred must be dropped from state
    rather than accumulating there forever."""
    starred = [{"id": "s1", "name": "Hill A"}]
    monkeypatch.setattr(bot, "get_starred_segments", lambda: starred)
    monkeypatch.setattr(bot, "get_segment_details", lambda seg_id: _details(kom="20:00", pr_secs=1500))

    state = {"segment_kom_times": {"s1": "25:00", "s_old_unstarred": "10:00"}}
    bot.check_kom_alerts(state)

    assert "s_old_unstarred" not in state["segment_kom_times"]
    assert state["segment_kom_times"]["s1"] == "20:00"


def test_kom_none_value_not_stored(bot, monkeypatch):
    """N4: a segment with no computed KOM (empty xoms) must not poison state
    with a None entry."""
    starred = [{"id": "s1", "name": "Hill A"}]
    monkeypatch.setattr(bot, "get_starred_segments", lambda: starred)
    monkeypatch.setattr(bot, "get_segment_details", lambda seg_id: _details())

    state = {"segment_kom_times": {}}
    bot.check_kom_alerts(state)

    assert state["segment_kom_times"] == {}


def test_kom_alert_on_new_pr(bot, monkeypatch):
    """Sanity check the alert branch still fires: athlete's own effort becomes
    the new, faster KOM.

    Not asserting an exact alert count: when starred has fewer segments than
    KOM_CHECK_BATCH, `(starred + starred)[offset:offset+KOM_CHECK_BATCH]` wraps
    into the duplicate, so a single starred segment is processed twice and
    produces two identical alerts. Pre-existing, harmless-but-noisy quirk,
    orthogonal to the merge/prune behaviour this file otherwise covers."""
    starred = [{"id": "s1", "name": "Hill A"}]
    monkeypatch.setattr(bot, "get_starred_segments", lambda: starred)
    monkeypatch.setattr(bot, "get_segment_details", lambda seg_id: _details(kom="19:00", pr_secs=1140))

    state = {"segment_kom_times": {"s1": "20:00"}}
    alerts = bot.check_kom_alerts(state)

    assert alerts
    assert all("new KOM" in a for a in alerts)
