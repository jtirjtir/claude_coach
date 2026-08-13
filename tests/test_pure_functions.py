"""Characterisation tests for the pure helpers.

These pin down existing behaviour before the Phase 1 fixes touch the code around
them. Where a test encodes a quirk rather than a requirement, it says so.
"""

import pytest


# ── time formatting ──────────────────────────────────────────────────────────
@pytest.mark.parametrize("secs,expected", [
    (0, "0:00"),
    (59, "0:59"),
    (60, "1:00"),
    (3599, "59:59"),
    (3600, "60:00"),      # no hour rollover by design — M:SS only
    (330.7, "5:30"),      # truncates, doesn't round
    (-90, "1:30"),        # abs() first, so sign is lost
])
def test_fmt_time(bot, secs, expected):
    assert bot._fmt_time(secs) == expected


@pytest.mark.parametrize("text,expected", [
    ("3:24", 204),
    ("1:23:45", 5025),
    ("0:00", 0),
    ("", None),
    ("abc", None),
    ("12", None),         # bare number is not a time
    ("1:2:3:4", None),    # too many parts
])
def test_parse_time_str(bot, text, expected):
    assert bot._parse_time_str(text) == expected


def test_fmt_parse_round_trip(bot):
    for secs in (0, 61, 599, 3599):
        assert bot._parse_time_str(bot._fmt_time(secs)) == secs


# ── activity identity ────────────────────────────────────────────────────────
def test_activity_signature_truncates_to_hour(bot):
    act = {"start_date_local": "2026-06-02T17:43:21", "distance": 14000}
    assert bot._activity_signature(act) == ("2026-06-02T17", 14000)


def test_activity_signature_missing_fields(bot):
    assert bot._activity_signature({}) == ("", 0)
    assert bot._activity_signature({"distance": None}) == ("", 0)


def test_same_activity_within_tolerance(bot):
    a = ("2026-06-02T17", 14000)
    assert bot._same_activity(a, ("2026-06-02T17", 14400)) is True    # 400m < 500m
    assert bot._same_activity(a, ("2026-06-02T17", 14600)) is False   # 600m > 500m
    assert bot._same_activity(a, ("2026-06-02T18", 14000)) is False   # different hour


def test_same_activity_handles_empty(bot):
    assert bot._same_activity(None, ("2026-06-02T17", 1)) is False
    assert bot._same_activity((), ()) is False


# ── misc helpers ─────────────────────────────────────────────────────────────
def test_cap_id_list_keeps_newest(bot):
    assert bot._cap_id_list(list(range(10)), max_size=3) == [7, 8, 9]
    assert bot._cap_id_list([1, 2], max_size=5) == [1, 2]


def test_is_key_session_matches_keywords(bot):
    assert bot._is_key_session({"name": "Tempo run", "description": ""}) is True
    assert bot._is_key_session({"name": "", "description": "4x1km INTERVALS"}) is True
    assert bot._is_key_session({"name": "Easy jog", "description": "keep it steady"}) is False


# ── stream summarisation ─────────────────────────────────────────────────────
def _flat_run_streams(km=3, sec_per_km=300):
    """A synthetic 1 Hz run: constant speed, flat, steady HR/cadence."""
    n = km * sec_per_km
    speed = 1000 / sec_per_km
    return {
        "time":      list(range(n)),
        "distance":  [speed * i for i in range(n)],
        "speed":     [speed] * n,
        "heartrate": [150] * n,
        "cadence":   [180] * n,
        "altitude":  [100.0] * n,
    }


def test_summarize_streams_too_short_returns_empty(bot):
    assert bot.summarize_streams({"time": list(range(10))}) == ""


def test_summarize_streams_km_splits_and_pacing(bot):
    out = bot.summarize_streams(_flat_run_streams())
    assert "Km splits" in out
    assert "km 1: 5:00/km | 150 bpm | 180 spm" in out
    assert "even split" in out
    assert "HR drift:* 150 → 150 bpm (+0.0 bpm)" in out


def test_summarize_streams_flat_course_reports_no_elevation(bot):
    # Constant altitude: no per-km "↑Nm" annotation.
    assert "↑" not in bot.summarize_streams(_flat_run_streams())


def test_summarize_streams_negative_split(bot):
    s = _flat_run_streams(km=2)
    half = len(s["speed"]) // 2
    s["speed"] = [3.0] * half + [3.6] * (len(s["speed"]) - half)
    assert "negative split" in bot.summarize_streams(s)


def test_summarize_streams_skips_zero_samples(bot):
    # Zero speed/HR samples (GPS or strap dropouts) are excluded from averages,
    # not treated as real zeroes.
    s = _flat_run_streams(km=1)
    s["speed"][:100] = [0] * 100
    s["heartrate"][:100] = [0] * 100
    assert "150 bpm" in bot.summarize_streams(s)


# ── interval summarisation ───────────────────────────────────────────────────
def test_summarize_intervals_empty(bot):
    assert bot.summarize_intervals({}) == ""
    assert bot.summarize_intervals({"icu_intervals": []}) == ""


def test_summarize_intervals_formats_reps(bot):
    data = {"icu_intervals": [
        {"label": "1", "type": "WORK", "distance": 1000, "moving_time": 240,
         "average_heartrate": 168.4, "average_speed": 4.1667},
        {"label": "R", "type": "RECOVERY", "distance": 200, "moving_time": 90},
    ]}
    out = bot.summarize_intervals(data)
    assert "Detected intervals" in out
    # 1000/4.1667 = 239.998 s/km, and _fmt_time truncates rather than rounds —
    # so a 4:00/km rep displays as 3:59. Pinning the quirk, not endorsing it.
    assert "[1] 1 (WORK) — 1.00km in 4:00 | 3:59/km | 168 bpm" in out
    assert "[2] R (RECOVERY) — 0.20km in 1:30 | 7:30/km | N/A" in out
