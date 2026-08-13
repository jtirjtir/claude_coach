"""Regression tests for the LLM call paths (N1, L2, L4)."""

import asyncio
import json

import pytest


# ── N1: an LLM outage must not look like a successful briefing ───────────────
class _BoomClient:
    class messages:
        @staticmethod
        def create(**kwargs):
            raise RuntimeError("API is down")


def test_ask_llm_raises_instead_of_returning_apology(bot, monkeypatch):
    monkeypatch.setattr(bot, "_llm_client", _BoomClient)
    monkeypatch.setattr(bot, "LLM_PROVIDER", "anthropic")
    with pytest.raises(bot.LLMUnavailable):
        bot.ask_llm("system", "user")


def test_llm_unavailable_is_catchable_as_exception(bot):
    # The briefing retry path in run() catches Exception; the analysis path
    # relies on _analyse_and_send's outer except. Both must still see this.
    assert issubclass(bot.LLMUnavailable, Exception)


def test_generate_analysis_propagates_llm_failure(bot, monkeypatch):
    """Previously returned the apology string, which _analyse_and_send then sent
    to the athlete and treated as success — dequeuing the activity forever."""
    monkeypatch.setattr(bot, "_llm_client", _BoomClient)
    monkeypatch.setattr(bot, "LLM_PROVIDER", "anthropic")
    with pytest.raises(bot.LLMUnavailable):
        bot.generate_analysis({"name": "Run", "distance": 5000, "moving_time": 1500}, None)


def test_analyse_and_send_reports_failure_on_llm_outage(bot, monkeypatch, state_file):
    """The end-to-end guarantee: a failed analysis returns False so the caller
    keeps it queued for retry, and nothing is sent to the athlete."""
    sent = []
    monkeypatch.setattr(bot, "get_activity_detail",
                        lambda act_id: {"name": "Run", "start_date_local": "2026-08-01T06:00:00",
                                        "distance": 5000, "moving_time": 1500})
    monkeypatch.setattr(bot, "get_activity_streams", lambda act_id: None)
    monkeypatch.setattr(bot, "get_activity_intervals", lambda act_id: None)
    monkeypatch.setattr(bot, "get_event_for_date", lambda d: None)
    monkeypatch.setattr(bot, "STRAVA_ENABLED", False)
    monkeypatch.setattr(bot, "send_telegram", lambda m: sent.append(m) or True)
    monkeypatch.setattr(bot, "generate_analysis",
                        lambda *a, **k: (_ for _ in ()).throw(bot.LLMUnavailable("down")))

    assert bot._analyse_and_send("123", "intervals", {}) is False
    assert sent == []


# ── L4: don't discard partial text on an unexpected stop_reason ──────────────
class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Response:
    def __init__(self, stop_reason, content):
        self.stop_reason = stop_reason
        self.content = content

    def model_dump(self):
        return {"stop_reason": self.stop_reason}


def _anthropic_client_returning(response):
    class _C:
        class messages:
            @staticmethod
            def create(**kwargs):
                return response
    return _C


def test_anthropic_loop_salvages_text_on_max_tokens(bot, monkeypatch):
    resp = _Response("max_tokens", [_Block("Your tempo run looked strong, and")])
    monkeypatch.setattr(bot, "_llm_client", _anthropic_client_returning(resp))
    reply, _ = asyncio.run(bot._run_anthropic_tool_loop("sys", "hi", [], session=None))
    assert reply == "Your tempo run looked strong, and"


def test_anthropic_loop_concatenates_multiple_text_blocks(bot, monkeypatch):
    resp = _Response("pause_turn", [_Block("part one "), _Block("part two")])
    monkeypatch.setattr(bot, "_llm_client", _anthropic_client_returning(resp))
    reply, _ = asyncio.run(bot._run_anthropic_tool_loop("sys", "hi", [], session=None))
    assert reply == "part one part two"


def test_anthropic_loop_still_errors_when_no_text(bot, monkeypatch):
    resp = _Response("max_tokens", [])
    monkeypatch.setattr(bot, "_llm_client", _anthropic_client_returning(resp))
    reply, _ = asyncio.run(bot._run_anthropic_tool_loop("sys", "hi", [], session=None))
    assert reply == "⚠️ Unexpected response from coach."


def test_anthropic_loop_normal_end_turn_unaffected(bot, monkeypatch):
    resp = _Response("end_turn", [_Block("Nice work today.")])
    monkeypatch.setattr(bot, "_llm_client", _anthropic_client_returning(resp))
    reply, _ = asyncio.run(bot._run_anthropic_tool_loop("sys", "hi", [], session=None))
    assert reply == "Nice work today."


# ── L2: malformed tool arguments must not kill the turn ─────────────────────
class _Call:
    type = "function_call"

    def __init__(self, name, arguments, call_id="c1"):
        self.name = name
        self.arguments = arguments
        self.call_id = call_id


class _FoundryResponse:
    def __init__(self, output, output_text="", rid="r1"):
        self.output = output
        self.output_text = output_text
        self.id = rid

    def model_dump(self):
        return {"id": self.id}


def test_foundry_loop_survives_malformed_arguments(bot, monkeypatch):
    """Bad JSON used to raise out of the loop, so the athlete got the generic
    'Coach is unavailable' and their message was dropped."""
    responses = [
        _FoundryResponse(output=[_Call("strava_get_segment", "{not valid json")]),
        _FoundryResponse(output=[], output_text="Here's your segment summary."),
    ]
    calls = []

    class _C:
        class responses:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return responses[len(calls) - 1]

    monkeypatch.setattr(bot, "_llm_client", _C)
    monkeypatch.setattr(bot, "FOUNDRY_MODEL", "test-model", raising=False)

    reply, _ = asyncio.run(
        bot._run_foundry_tool_loop("sys", "how did I do?", None, "image/jpeg", [], session=None)
    )
    assert reply == "Here's your segment summary."

    # The model is told what went wrong so it can retry the call.
    followup_input = calls[1]["input"]
    assert followup_input[0]["type"] == "function_call_output"
    assert "not valid JSON" in followup_input[0]["output"]


def test_foundry_loop_handles_empty_arguments(bot, monkeypatch):
    """`arguments` of None/"" is legitimate for a no-parameter tool."""
    responses = [
        _FoundryResponse(output=[_Call("strava_get_starred_segments", None)]),
        _FoundryResponse(output=[], output_text="done"),
    ]
    calls = []

    class _C:
        class responses:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return responses[len(calls) - 1]

    monkeypatch.setattr(bot, "_llm_client", _C)
    monkeypatch.setattr(bot, "FOUNDRY_MODEL", "test-model", raising=False)
    monkeypatch.setattr(bot, "_dispatch_strava_tool", lambda n, i: json.dumps([]))

    reply, _ = asyncio.run(
        bot._run_foundry_tool_loop("sys", "q", None, "image/jpeg", [], session=None)
    )
    assert reply == "done"
