"""State persistence: atomic writes and corrupt-file recovery."""

import json


def test_load_state_missing_file_returns_fresh(bot, state_file):
    assert bot.load_state() == {"last_activity_id": None, "last_briefing_date": None}


def test_save_then_load_round_trip(bot, state_file):
    bot.save_state({"a": 1, "nested": {"b": [1, 2]}})
    assert bot.load_state() == {"a": 1, "nested": {"b": [1, 2]}}


def test_load_state_recovers_from_corrupt_file(bot, state_file):
    state_file.write_text('{"truncated": ')
    assert bot.load_state() == {"last_activity_id": None, "last_briefing_date": None}


def test_save_state_leaves_no_temp_file(bot, state_file):
    bot.save_state({"x": 1})
    assert not state_file.with_suffix(".json.tmp").exists()
    assert json.loads(state_file.read_text()) == {"x": 1}


def test_save_state_overwrites_corrupt_file(bot, state_file):
    state_file.write_text("not json at all")
    bot.load_state()
    bot.save_state({"recovered": True})
    assert bot.load_state() == {"recovered": True}
