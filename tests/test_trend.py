"""Characterisation tests for compute_training_trend (Phase 0 task 0.4 — this
function had zero coverage despite being named in the plan, and its L5 input
(count_missed_in_window) was changed by this same round of fixes)."""


def _patch(bot, monkeypatch, series, missed_planned=(0, 10)):
    monkeypatch.setattr(bot, "get_wellness_series", lambda days_back=None: series)
    monkeypatch.setattr(bot, "count_missed_in_window", lambda: missed_planned)


def test_deep_fatigue_tsb_triggers_reduce(bot, monkeypatch):
    series = [{"id": "2026-08-01", "tsb": -30}]
    _patch(bot, monkeypatch, series)

    trend = bot.compute_training_trend()
    assert trend["signal"] == "reduce"
    assert "deep fatigue" in trend["reasons"][0]


def test_high_missed_rate_triggers_reduce(bot, monkeypatch):
    series = [{"id": "2026-08-01", "tsb": -3}]  # otherwise fine
    _patch(bot, monkeypatch, series, missed_planned=(4, 10))  # 40% missed

    trend = bot.compute_training_trend()
    assert trend["signal"] == "reduce"
    assert "missed" in trend["reasons"][0]


def test_hrv_suppression_triggers_reduce(bot, monkeypatch):
    # 4 baseline records at hrv=50, today's at hrv=40 -> -20% deviation.
    series = [
        {"id": "2026-07-29", "hrv": 50}, {"id": "2026-07-30", "hrv": 50},
        {"id": "2026-07-31", "hrv": 50}, {"id": "2026-08-01", "hrv": 50},
        {"id": "2026-08-02", "hrv": 40},
    ]
    _patch(bot, monkeypatch, series)

    trend = bot.compute_training_trend()
    assert trend["signal"] == "reduce"
    assert trend["hrv_dev_pct"] == -20.0


def test_good_form_and_adherence_triggers_progress(bot, monkeypatch):
    series = [{"id": "2026-08-01", "tsb": 5}]
    _patch(bot, monkeypatch, series, missed_planned=(0, 10))

    trend = bot.compute_training_trend()
    assert trend["signal"] == "progress"
    assert "room to progress" in trend["reasons"][0]


def test_middling_signal_holds(bot, monkeypatch):
    # TSB between the reduce and progress thresholds, adherence fine:
    # no reduce trigger, but not good enough to progress either.
    series = [{"id": "2026-08-01", "tsb": -10}]
    _patch(bot, monkeypatch, series, missed_planned=(0, 10))

    trend = bot.compute_training_trend()
    assert trend["signal"] == "hold"


def test_fast_ctl_ramp_blocks_progress_even_with_good_form(bot, monkeypatch):
    """A CTL ramp above CTL_RAMP_LIMIT (8 pts/week) should hold the athlete at
    'hold' rather than progressing further, even though TSB/HRV/adherence
    otherwise look fine."""
    series = [
        {"id": "2026-07-26", "tsb": 5, "ctl": 40},
        {"id": "2026-08-02", "tsb": 5, "ctl": 55},  # +15 CTL over 7 days
    ]
    _patch(bot, monkeypatch, series, missed_planned=(0, 10))

    trend = bot.compute_training_trend()
    assert trend["ctl_ramp"] == 15.0
    assert trend["signal"] == "hold"
