"""Tests for the brief activity summary posted onto the activity itself.

Covers the Intervals.icu comment, the Strava description mirror (needed because
an Intervals.icu comment never reaches Strava), and the shared fact extraction —
including the form/load fields the analysis prompt used to drop.
"""

import pytest


# An Intervals.icu activity record, field names as the API actually returns them.
ICU_ACTIVITY = {
    "id": "i177741129",
    "name": "Sydney Road Cycling",
    "type": "Ride",
    "start_date_local": "2026-08-20T05:44:33",
    "distance": 42797.23,
    "moving_time": 5976,
    "elapsed_time": 8766,
    "average_heartrate": 146,
    "max_heartrate": 171,
    "lthr": 168,
    "icu_ftp": 205,
    "icu_pm_ftp": 173,
    "icu_rolling_ftp": 202,
    "icu_weighted_avg_watts": 194,
    "icu_average_watts": 137,
    "icu_training_load": 149,
    "icu_intensity": 94.63415,
    "icu_ctl": 48.64532,
    "icu_atl": 56.199005,
    "decoupling": 21.257246,
    "total_elevation_gain": 546.0,
    "average_speed": 7.145,
    "source": "GARMIN_CONNECT",
    "strava_id": 19815394001,
    "icu_zone_times": [{"id": "Z1", "secs": 2403}, {"id": "Z2", "secs": 1051}],
}

# The same ride as Strava returns it — different names for the same quantities.
STRAVA_ACTIVITY = {
    "id": 19815394001,
    "name": "Morning Ride",
    "sport_type": "Ride",
    "start_date_local": "2026-08-20T05:44:33Z",
    "distance": 42797.2,
    "moving_time": 6066,
    "average_heartrate": 146.4,
    "max_heartrate": 171.0,
    "weighted_average_watts": 176,
    "average_watts": 143.9,
    "total_elevation_gain": 546.0,
    "suffer_score": 120,
}


# ── fact extraction ───────────────────────────────────────────────────────────
def test_facts_read_the_intervals_shape(bot):
    """The Intervals.icu path is the normal one, and it names these fields
    icu_*. Reading only the Strava names is what silently dropped training load
    and the whole form line from almost every analysis."""
    facts = bot.activity_facts(ICU_ACTIVITY)

    assert facts["load"] == 149
    assert facts["hr_avg"] == 146
    assert facts["hr_max"] == 171
    assert facts["duration"] == "1h39m"
    assert facts["distance_km"] == 42.8


def test_form_is_derived_from_fitness_and_fatigue(bot):
    """Intervals.icu returns CTL and ATL but no form figure of its own."""
    facts = bot.activity_facts(ICU_ACTIVITY)
    assert facts["tsb"] == pytest.approx(48.64532 - 56.199005)


def test_form_uses_an_explicit_tsb_when_one_is_given(bot):
    facts = bot.activity_facts({**ICU_ACTIVITY, "tsb": -3.0})
    assert facts["tsb"] == -3.0


def test_facts_read_the_strava_shape_too(bot):
    facts = bot.activity_facts(STRAVA_ACTIVITY)
    assert facts["type"] == "Ride"
    assert facts["hr_max"] == 171
    assert facts["np"] == 176
    assert facts["tsb"] is None          # Strava carries no fitness/fatigue


def test_facts_survive_a_bare_activity(bot):
    """A just-uploaded activity can be missing nearly everything; the summary
    still has to be generatable rather than raising."""
    facts = bot.activity_facts({"name": "Run", "type": "Run"})
    assert facts["distance_km"] == 0
    assert facts["duration"] == "0m"
    assert facts["tsb"] is None


def test_ftp_estimate_prefers_this_activity_over_the_setting(bot):
    facts = bot.activity_facts(ICU_ACTIVITY)
    assert facts["ftp_estimate"] == 173
    assert facts["ftp_source"] == "this activity"
    assert facts["ftp_setting"] == 205


def test_ftp_estimate_falls_back_to_the_rolling_estimate(bot):
    facts = bot.activity_facts({k: v for k, v in ICU_ACTIVITY.items() if k != "icu_pm_ftp"})
    assert facts["ftp_estimate"] == 202
    assert facts["ftp_source"] == "rolling estimate"


# ── location ──────────────────────────────────────────────────────────────────
def test_location_falls_back_to_the_place_in_the_activity_name(bot):
    """Garmin-sourced rides leave every location field null on both services,
    but Garmin bakes the place into the name."""
    assert bot.activity_location(ICU_ACTIVITY) == "Sydney"


def test_location_prefers_strava_location_fields(bot):
    strava = {**STRAVA_ACTIVITY, "location_city": "Sydney", "location_state": "NSW"}
    assert bot.activity_location(ICU_ACTIVITY, strava) == "Sydney, NSW"


def test_generic_strava_name_yields_no_location(bot):
    """'Morning Ride' is all filler — better to say nothing than to call the
    ride 'Morning'."""
    assert bot.activity_location(STRAVA_ACTIVITY) == ""


# ── header composition ────────────────────────────────────────────────────────
def test_header_matches_the_log_line_format(bot):
    facts = bot.activity_facts(ICU_ACTIVITY)
    assert bot._summary_header(facts, "Z2 endurance ride") == (
        "Thu 20 August 05:44 1h39m 42.8 km Sydney - Z2 endurance ride"
    )


def test_header_omits_missing_pieces(bot):
    facts = bot.activity_facts({"name": "Ride", "type": "Ride",
                                "start_date_local": "2026-08-20T05:44:33"})
    assert bot._summary_header(facts, "Recovery spin") == "Thu 20 August 05:44 - Recovery spin"


# ── comment generation ────────────────────────────────────────────────────────
def _llm(bot, monkeypatch, reply):
    monkeypatch.setattr(bot, "ask_llm", lambda system, user, **kw: reply)


def test_comment_splits_title_from_body(bot, monkeypatch):
    _llm(bot, monkeypatch, "Z2 endurance ride\n\nSolid aerobic hours. Form now -8.")
    comment = bot.generate_activity_comment(ICU_ACTIVITY)

    assert comment.startswith("Thu 20 August 05:44 1h39m 42.8 km Sydney - Z2 endurance ride")
    assert comment.endswith("Solid aerobic hours. Form now -8.")
    assert "\n\n" in comment


def test_comment_handles_a_model_that_skips_the_blank_line(bot, monkeypatch):
    _llm(bot, monkeypatch, "Z2 endurance ride\nSolid aerobic hours.")
    assert "Sydney - Z2 endurance ride" in bot.generate_activity_comment(ICU_ACTIVITY)


def test_a_paragraph_where_the_title_belongs_is_not_jammed_into_the_header(bot, monkeypatch):
    """Guards the header against a model that ignores the title instruction and
    returns prose — the header must stay a header."""
    prose = "This was a really long rambling opening sentence that is clearly not a title at all."
    _llm(bot, monkeypatch, prose + "\n\nSecond paragraph.")
    comment = bot.generate_activity_comment(ICU_ACTIVITY)

    assert comment.splitlines()[0] == "Thu 20 August 05:44 1h39m 42.8 km Sydney"
    assert prose in comment


def test_comment_is_truncated_to_fit(bot, monkeypatch):
    _llm(bot, monkeypatch, "Z2 ride\n\n" + "word " * 500)
    comment = bot.generate_activity_comment(ICU_ACTIVITY)

    assert len(comment) <= bot.ICU_COMMENT_MAX_CHARS
    assert comment.endswith("…")


def test_llm_outage_still_produces_a_postable_comment(bot, monkeypatch):
    """The value of the comment is that the numbers land on the activity, so an
    LLM outage must not cost us the post."""
    def boom(system, user, **kw):
        raise bot.LLMUnavailable("503")
    monkeypatch.setattr(bot, "ask_llm", boom)

    comment = bot.generate_activity_comment(ICU_ACTIVITY)
    assert "Sydney" in comment
    assert "Avg HR 146" in comment
    assert "max HR 171" in comment
    assert "eFTP 173w" in comment
    assert "Form -8" in comment


def test_comment_carries_the_numbers_the_athlete_asked_for(bot, monkeypatch):
    """eFTP, average HR, max HR and form must all reach the prompt, otherwise
    the model cannot quote them however well it is asked to."""
    seen = {}
    monkeypatch.setattr(bot, "ask_llm",
                        lambda system, user, **kw: seen.setdefault("user", user) and "" or "T\n\nB")
    bot.generate_activity_comment(ICU_ACTIVITY)

    prompt = seen["user"]
    assert "Estimated FTP: 173 W" in prompt
    assert "Avg HR: 146" in prompt
    assert "Max HR: 171" in prompt
    assert "Form (TSB) after this session: -7.6" in prompt


# ── posting ───────────────────────────────────────────────────────────────────
def test_comment_posts_to_the_messages_endpoint(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "intervals_post",
                        lambda path, data: calls.append((path, data)) or {"id": 42})

    assert bot.post_intervals_comment("i1", "hello") is True
    assert calls == [("/activity/i1/messages", {"content": "hello"})]


def test_failed_comment_reports_failure(bot, monkeypatch):
    monkeypatch.setattr(bot, "intervals_post", lambda path, data: None)
    assert bot.post_intervals_comment("i1", "hello") is False


def test_publish_posts_to_intervals_and_mirrors_to_strava(bot, monkeypatch):
    """The Strava mirror exists because the Intervals.icu comment does not travel
    there — COMMENT_SYNCS_TO_STRAVA records that finding."""
    assert bot.COMMENT_SYNCS_TO_STRAVA is False

    posted, mirrored = [], []
    monkeypatch.setattr(bot, "generate_activity_comment", lambda *a, **kw: "SUMMARY")
    monkeypatch.setattr(bot, "post_intervals_comment",
                        lambda aid, text: posted.append((aid, text)) or True)
    monkeypatch.setattr(bot, "update_strava_description",
                        lambda sid, text: mirrored.append((sid, text)) or True)
    monkeypatch.setattr(bot, "STRAVA_MIRROR_SUMMARY", True)

    result = bot.publish_activity_summary(ICU_ACTIVITY, "i177741129", "intervals")

    assert posted == [("i177741129", "SUMMARY")]
    assert mirrored == [("19815394001", "SUMMARY")]
    assert result["intervals"] and result["strava"]


def test_strava_sourced_activity_finds_its_intervals_twin(bot, monkeypatch):
    """On the Strava-fallback path the id in hand is a Strava one, but the
    comment still belongs on the Intervals.icu activity."""
    posted = []
    monkeypatch.setattr(bot, "generate_activity_comment", lambda *a, **kw: "SUMMARY")
    monkeypatch.setattr(bot, "get_recent_activities",
                        lambda n=5: [{"id": "iOTHER", "strava_id": 1}, ICU_ACTIVITY])
    monkeypatch.setattr(bot, "post_intervals_comment",
                        lambda aid, text: posted.append(aid) or True)
    monkeypatch.setattr(bot, "STRAVA_MIRROR_SUMMARY", False)

    bot.publish_activity_summary(STRAVA_ACTIVITY, "19815394001", "strava")
    assert posted == ["i177741129"]


def test_unsynced_strava_activity_skips_the_comment(bot, monkeypatch):
    monkeypatch.setattr(bot, "generate_activity_comment", lambda *a, **kw: "SUMMARY")
    monkeypatch.setattr(bot, "get_recent_activities", lambda n=5: [])
    monkeypatch.setattr(bot, "STRAVA_MIRROR_SUMMARY", False)

    result = bot.publish_activity_summary(STRAVA_ACTIVITY, "19815394001", "strava")
    assert result["intervals"] is False


# ── Strava description mirror ─────────────────────────────────────────────────
def test_strava_write_refused_without_the_write_scope(bot, monkeypatch):
    """Strava fixes scopes at authorisation time, so a read-only token can never
    write however many times we retry — fail loudly, do not call the API."""
    called = []
    monkeypatch.setattr(bot, "STRAVA_ENABLED", True)
    monkeypatch.setattr(bot, "_strava_scopes", {"read", "activity:read_all"})
    monkeypatch.setattr(bot, "strava_put", lambda p, d: called.append(p))

    assert bot.update_strava_description("19815394001", "SUMMARY") is False
    assert called == []


def test_strava_write_leaves_an_existing_description_alone(bot, monkeypatch):
    """PUT replaces rather than appends, so the athlete's own words must not be
    silently destroyed."""
    called = []
    monkeypatch.setattr(bot, "STRAVA_ENABLED", True)
    monkeypatch.setattr(bot, "_strava_scopes", {"activity:write"})
    monkeypatch.setattr(bot, "STRAVA_OVERWRITE_DESCRIPTION", False)
    monkeypatch.setattr(bot, "get_strava_activity_detail",
                        lambda sid: {"description": "Felt great today"})
    monkeypatch.setattr(bot, "strava_put", lambda p, d: called.append(p))

    assert bot.update_strava_description("19815394001", "SUMMARY") is False
    assert called == []


def test_strava_write_fills_an_empty_description(bot, monkeypatch):
    called = []
    monkeypatch.setattr(bot, "STRAVA_ENABLED", True)
    monkeypatch.setattr(bot, "_strava_scopes", {"activity:write"})
    monkeypatch.setattr(bot, "STRAVA_OVERWRITE_DESCRIPTION", False)
    monkeypatch.setattr(bot, "get_strava_activity_detail", lambda sid: {"description": None})
    monkeypatch.setattr(bot, "strava_put",
                        lambda p, d: called.append((p, d)) or {"id": 1})

    assert bot.update_strava_description("19815394001", "SUMMARY") is True
    assert called == [("/activities/19815394001", {"description": "SUMMARY"})]


def test_overwrite_opt_in_skips_the_read_back(bot, monkeypatch):
    called = []
    monkeypatch.setattr(bot, "STRAVA_ENABLED", True)
    monkeypatch.setattr(bot, "_strava_scopes", {"activity:write"})
    monkeypatch.setattr(bot, "STRAVA_OVERWRITE_DESCRIPTION", True)
    monkeypatch.setattr(bot, "get_strava_activity_detail",
                        lambda sid: pytest.fail("should not read back when overwriting"))
    monkeypatch.setattr(bot, "strava_put", lambda p, d: called.append(p) or {"id": 1})

    assert bot.update_strava_description("19815394001", "SUMMARY") is True
    assert called == ["/activities/19815394001"]


# ── the analysis prompt, which shares the same fact extraction ────────────────
def test_analysis_prompt_now_carries_load_and_form(bot, monkeypatch):
    """Regression: generate_analysis read 'training_load'/'ctl'/'atl'/'tsb',
    which only exist on the Strava record. On the Intervals.icu path — the
    normal one — every one of those lookups missed, so the athlete's analysis
    never mentioned training load and never showed the post-run form line."""
    seen = {}
    monkeypatch.setattr(bot, "ask_llm",
                        lambda system, user, **kw: seen.setdefault("user", user) and "")
    bot.generate_analysis(ICU_ACTIVITY, planned=None)

    prompt = seen["user"]
    assert "Training load: 149" in prompt
    assert "Post-run form: Fitness 48.6 | Fatigue 56.2 | Form -7.6" in prompt


def test_analysis_still_reads_the_strava_shape(bot, monkeypatch):
    seen = {}
    monkeypatch.setattr(bot, "ask_llm",
                        lambda system, user, **kw: seen.setdefault("user", user) and "")
    bot.generate_analysis(STRAVA_ACTIVITY, planned=None)

    assert "Suffer score: 120" in seen["user"]


# ── the summary must never cost the athlete a duplicate analysis ──────────────
def _stub_analysis_path(bot, monkeypatch, sent):
    monkeypatch.setattr(bot, "get_activity_detail", lambda aid: ICU_ACTIVITY)
    monkeypatch.setattr(bot, "get_activity_streams", lambda aid: None)
    monkeypatch.setattr(bot, "get_activity_intervals", lambda aid: None)
    monkeypatch.setattr(bot, "get_event_for_date", lambda d: None)
    monkeypatch.setattr(bot, "generate_analysis", lambda *a, **kw: "ANALYSIS")
    monkeypatch.setattr(bot, "send_telegram", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(bot, "STRAVA_ENABLED", False)
    monkeypatch.setattr(bot, "POST_ACTIVITY_COMMENT", True)


def test_a_failed_comment_does_not_requeue_the_analysis(bot, monkeypatch):
    """_analyse_and_send returning False re-queues the activity and re-sends the
    full analysis to Telegram. A comment that could not be posted is not worth
    a duplicate briefing, so the publish step swallows its own errors."""
    sent = []
    _stub_analysis_path(bot, monkeypatch, sent)

    def boom(*a, **kw):
        raise RuntimeError("intervals down")
    monkeypatch.setattr(bot, "publish_activity_summary", boom)

    assert bot._analyse_and_send("i177741129", "intervals", {}) is True
    assert len(sent) == 1


def test_the_comment_is_published_on_the_happy_path(bot, monkeypatch):
    sent, published = [], []
    _stub_analysis_path(bot, monkeypatch, sent)
    monkeypatch.setattr(bot, "publish_activity_summary",
                        lambda *a, **kw: published.append(a[1]) or {})

    assert bot._analyse_and_send("i177741129", "intervals", {}) is True
    assert published == ["i177741129"]


def test_the_comment_can_be_switched_off(bot, monkeypatch):
    sent = []
    _stub_analysis_path(bot, monkeypatch, sent)
    monkeypatch.setattr(bot, "POST_ACTIVITY_COMMENT", False)
    monkeypatch.setattr(bot, "publish_activity_summary",
                        lambda *a, **kw: pytest.fail("should not publish when disabled"))

    assert bot._analyse_and_send("i177741129", "intervals", {}) is True
