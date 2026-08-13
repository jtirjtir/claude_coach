"""Regression coverage for the stale-wellness bug.

The 05:15 briefing was silently running on the previous day's sleep/HRV — every
morning for 11 straight days, per coach_bot.log. Intervals.icu pre-creates empty
wellness records for today and future dates, so get_wellness()'s scan for a
record with non-null HRV always walked back past today's empty shell and landed
on yesterday's, with nothing downstream noticing.

The data now has to carry its own date, and anything shown to the athlete has to
say so when it isn't this morning's.
"""

from datetime import datetime


def _at(bot, date_str, hour=5, minute=15):
    y, m, d = (int(x) for x in date_str.split("-"))
    return datetime(y, m, d, hour, minute, tzinfo=bot.AEST)


# ── wellness_age_days ────────────────────────────────────────────────────────

def test_todays_record_is_zero_days_old(bot):
    now = _at(bot, "2026-08-13")
    assert bot.wellness_age_days({"id": "2026-08-13"}, now) == 0


def test_yesterdays_record_is_one_day_old(bot):
    now = _at(bot, "2026-08-13")
    assert bot.wellness_age_days({"id": "2026-08-12"}, now) == 1


def test_age_is_measured_in_calendar_days_not_elapsed_hours(bot):
    """A record dated today is 0 days old at 00:05, not "-1" or fractional — the
    briefing fires at 05:15 and only ever cares which date the data belongs to."""
    assert bot.wellness_age_days({"id": "2026-08-13"}, _at(bot, "2026-08-13", 0, 5)) == 0
    assert bot.wellness_age_days({"id": "2026-08-13"}, _at(bot, "2026-08-13", 23, 55)) == 0


def test_undatable_records_return_none(bot):
    now = _at(bot, "2026-08-13")
    assert bot.wellness_age_days(None, now) is None
    assert bot.wellness_age_days({}, now) is None
    assert bot.wellness_age_days({"id": None}, now) is None
    assert bot.wellness_age_days({"id": "not-a-date"}, now) is None


# ── get_wellness ─────────────────────────────────────────────────────────────

def test_skips_todays_empty_shell_and_returns_yesterdays_populated_record(bot, monkeypatch):
    """The real Intervals.icu shape at 05:00: today's record exists but every
    biometric is null because Garmin hasn't synced yet."""
    monkeypatch.setattr(bot, "intervals_get", lambda path: [
        {"id": "2026-08-11", "hrv": 31.0, "restingHR": 57},
        {"id": "2026-08-12", "hrv": 41.0, "restingHR": 55},
        {"id": "2026-08-13", "hrv": None, "restingHR": None},
    ])

    record = bot.get_wellness()
    assert record["id"] == "2026-08-12"


def test_prefers_todays_record_once_it_is_populated(bot, monkeypatch):
    monkeypatch.setattr(bot, "intervals_get", lambda path: [
        {"id": "2026-08-12", "hrv": 41.0},
        {"id": "2026-08-13", "hrv": 33.0},
    ])

    assert bot.get_wellness()["id"] == "2026-08-13"


def test_falls_back_to_newest_non_empty_record_when_no_hrv_anywhere(bot, monkeypatch):
    monkeypatch.setattr(bot, "intervals_get", lambda path: [
        {"id": "2026-08-12", "hrv": None, "restingHR": 55},
        {"id": "2026-08-13", "hrv": None, "restingHR": None},
    ])

    # Still returns something rather than None — a record with RHR but no HRV is
    # worth showing, it just has to be labelled with its date like any other.
    assert bot.get_wellness()["id"] == "2026-08-13"


def test_returns_none_when_api_fails(bot, monkeypatch):
    monkeypatch.setattr(bot, "intervals_get", lambda path: None)
    assert bot.get_wellness() is None


# ── compute_training_trend ───────────────────────────────────────────────────

def test_trend_reports_the_date_its_hrv_reading_came_from(bot, monkeypatch):
    """hrv_dev_pct used to be computed from a variable named `today_hrv` that was
    routinely yesterday's. The comparison is fine; passing it off as today's is
    not, so the date rides along with it."""
    series = [
        {"id": "2026-08-08", "hrv": 50}, {"id": "2026-08-09", "hrv": 50},
        {"id": "2026-08-10", "hrv": 50}, {"id": "2026-08-11", "hrv": 50},
        {"id": "2026-08-12", "hrv": 40},
    ]
    monkeypatch.setattr(bot, "get_wellness_series", lambda days_back=None: series)
    monkeypatch.setattr(bot, "count_missed_in_window", lambda: (0, 10))

    trend = bot.compute_training_trend()
    assert trend["hrv_dev_pct"] == -20.0
    assert trend["hrv_date"] == "2026-08-12"


# ── Briefing context ─────────────────────────────────────────────────────────

def _briefing_context(bot, monkeypatch, wellness, now):
    """Run generate_briefing far enough to capture the prompt it builds, without
    reaching the network or the LLM."""
    captured = {}

    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return now

    monkeypatch.setattr(bot, "datetime", _FrozenDatetime)
    monkeypatch.setattr(bot, "get_todays_event", lambda: None)
    monkeypatch.setattr(bot, "get_recent_activities", lambda n=5: [])
    monkeypatch.setattr(bot, "get_wellness", lambda lookback_days=5: wellness)
    monkeypatch.setattr(bot, "detect_missed_sessions", lambda *a, **k: [])
    monkeypatch.setattr(
        bot, "ask_llm",
        lambda system, user, **kw: captured.update(system=system, user=user) or "briefing text",
    )

    bot.generate_briefing(state=None)
    return captured["user"]


def test_stale_wellness_is_flagged_in_the_prompt(bot, monkeypatch):
    ctx = _briefing_context(
        bot, monkeypatch,
        wellness={"id": "2026-08-12", "hrv": 41.0, "restingHR": 55,
                  "sleepSecs": 24711, "sleepScore": 76.0},
        now=_at(bot, "2026-08-13"),
    )

    assert "STALE" in ctx
    assert "Wednesday 12 August" in ctx
    # And the model is told not to narrate it as last night
    assert "NOT from this morning" in ctx


def test_fresh_wellness_is_not_flagged(bot, monkeypatch):
    ctx = _briefing_context(
        bot, monkeypatch,
        wellness={"id": "2026-08-13", "hrv": 33.0, "restingHR": 59,
                  "sleepSecs": 24569, "sleepScore": 56.0},
        now=_at(bot, "2026-08-13"),
    )

    assert "STALE" not in ctx
    assert "measured this morning" in ctx


# ── Garmin freshness probe (measurement only) ────────────────────────────────

def _run_probe(bot, monkeypatch, caplog, intervals_record, garmin_data, enabled=True):
    import logging
    monkeypatch.setattr(bot, "GARMIN_ENABLED", enabled)
    monkeypatch.setattr(bot, "_garmin_client", lambda: object())
    monkeypatch.setattr(bot, "_garmin_overnight", lambda client, date_str: garmin_data)

    with caplog.at_level(logging.INFO):
        bot.probe_garmin_freshness(intervals_record, _at(bot, "2026-08-13", 5, 0))
    return caplog.text


_GARMIN_HAS_DATA = {"hrv": 33.0, "rhr": 59, "sleep_secs": 24569, "sleep_score": 56}
_GARMIN_EMPTY    = {"hrv": None, "rhr": None, "sleep_secs": None, "sleep_score": None}


def test_probe_flags_intervals_sync_lag_when_only_garmin_has_today(bot, monkeypatch, caplog):
    """The outcome that would justify building the Garmin integration."""
    text = _run_probe(
        bot, monkeypatch, caplog,
        intervals_record={"id": "2026-08-12", "hrv": 41.0},
        garmin_data=_GARMIN_HAS_DATA,
    )
    assert "Garmin HAS today's data, Intervals.icu does NOT" in text


def test_probe_exonerates_garmin_when_neither_source_has_today(bot, monkeypatch, caplog):
    """The outcome that would kill it — the watch itself hasn't synced by 05:00,
    so a direct integration would return the same emptiness."""
    text = _run_probe(
        bot, monkeypatch, caplog,
        intervals_record={"id": "2026-08-12", "hrv": 41.0},
        garmin_data=_GARMIN_EMPTY,
    )
    assert "NEITHER has today's data" in text
    assert "would NOT help" in text


def test_probe_reports_no_problem_when_both_are_current(bot, monkeypatch, caplog):
    text = _run_probe(
        bot, monkeypatch, caplog,
        intervals_record={"id": "2026-08-13", "hrv": 33.0},
        garmin_data=_GARMIN_HAS_DATA,
    )
    assert "Both have today's data" in text


def test_probe_treats_partial_garmin_sync_as_populated(bot, monkeypatch, caplog):
    """Sleep landed but HRV hasn't — still evidence Garmin is ahead of Intervals."""
    text = _run_probe(
        bot, monkeypatch, caplog,
        intervals_record={"id": "2026-08-12", "hrv": 41.0},
        garmin_data={"hrv": None, "rhr": None, "sleep_secs": 24569, "sleep_score": 56},
    )
    assert "Garmin HAS today's data, Intervals.icu does NOT" in text


def test_probe_is_a_noop_without_credentials(bot, monkeypatch, caplog):
    """Default state for the running bot — no creds configured, nothing logged,
    and crucially no login attempt."""
    def _explode():
        raise AssertionError("must not attempt a Garmin login when disabled")
    monkeypatch.setattr(bot, "_garmin_client", _explode)

    text = _run_probe(
        bot, monkeypatch, caplog,
        intervals_record={"id": "2026-08-12"},
        garmin_data=_GARMIN_EMPTY,
        enabled=False,
    )
    assert "Garmin freshness probe" not in text
