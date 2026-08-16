"""Conversation memory — prior turns replayed into each new question.

Before this, every Telegram message was a fresh single-turn conversation: the
transcript DB was written but never read, so a follow-up like "and what about
tomorrow?" arrived with no idea what had just been discussed.
"""

import asyncio
import base64

import transcript_db


# ── The shared history builder (both providers use this shape) ───────────────
def test_history_builds_alternating_messages(bot):
    history = [
        {"user": "how was my long run?", "assistant": "Strong — negative split."},
        {"user": "and tomorrow?", "assistant": "Easy 5km recovery."},
    ]
    assert bot._history_messages(history) == [
        {"role": "user", "content": "how was my long run?"},
        {"role": "assistant", "content": "Strong — negative split."},
        {"role": "user", "content": "and tomorrow?"},
        {"role": "assistant", "content": "Easy 5km recovery."},
    ]


def test_history_drops_half_turns(bot):
    """A turn missing either half would break Anthropic's required
    user/assistant alternation and 400 the entire call."""
    history = [
        {"user": "q1", "assistant": ""},      # crashed mid-turn
        {"user": "", "assistant": "orphan"},  # reply with no question
        {"user": "q2", "assistant": "a2"},
    ]
    assert bot._history_messages(history) == [
        {"role": "user", "content": "q2"},
        {"role": "assistant", "content": "a2"},
    ]


def test_history_empty_when_no_memory(bot):
    assert bot._history_messages(None) == []
    assert bot._history_messages([]) == []


# ── Anthropic path ───────────────────────────────────────────────────────────
class _Block:
    def __init__(self, text):
        self.text = text
        self.type = "text"


class _Response:
    stop_reason = "end_turn"

    def __init__(self, text):
        self.content = [_Block(text)]

    def model_dump(self):
        return {}


def _recording_anthropic_client(calls):
    class _C:
        class messages:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return _Response("noted")
    return _C


def test_anthropic_loop_prepends_history(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_llm_client", _recording_anthropic_client(calls))

    history = [{"user": "how was my long run?", "assistant": "Strong."}]
    reply, _ = asyncio.run(
        bot._run_anthropic_tool_loop("sys", "and tomorrow?", [], session=None, history=history)
    )

    assert reply == "noted"
    assert calls[0]["messages"] == [
        {"role": "user", "content": "how was my long run?"},
        {"role": "assistant", "content": "Strong."},
        {"role": "user", "content": "and tomorrow?"},  # current question last
    ]


def test_anthropic_loop_without_history_unchanged(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_llm_client", _recording_anthropic_client(calls))

    asyncio.run(bot._run_anthropic_tool_loop("sys", "hi", [], session=None))
    assert calls[0]["messages"] == [{"role": "user", "content": "hi"}]


# ── Foundry path ─────────────────────────────────────────────────────────────
class _FoundryResponse:
    output = []

    def __init__(self, output_text="noted", rid="r1"):
        self.output_text = output_text
        self.id = rid

    def model_dump(self):
        return {"id": self.id}


def _recording_foundry_client(calls):
    class _C:
        class responses:
            @staticmethod
            def create(**kwargs):
                calls.append(kwargs)
                return _FoundryResponse()
    return _C


def test_foundry_loop_prepends_history(bot, monkeypatch):
    calls = []
    monkeypatch.setattr(bot, "_llm_client", _recording_foundry_client(calls))
    monkeypatch.setattr(bot, "FOUNDRY_MODEL", "test-model", raising=False)

    history = [{"user": "how was my long run?", "assistant": "Strong."}]
    reply, _ = asyncio.run(
        bot._run_foundry_tool_loop(
            "sys", "and tomorrow?", None, "image/jpeg", [], session=None, history=history
        )
    )

    assert reply == "noted"
    assert calls[0]["input"] == [
        {"role": "user", "content": "how was my long run?"},
        {"role": "assistant", "content": "Strong."},
        {"role": "user", "content": "and tomorrow?"},
    ]


def test_foundry_loop_prepends_history_before_image(bot, monkeypatch):
    """The image message item must stay last — it is the current question."""
    calls = []
    monkeypatch.setattr(bot, "_llm_client", _recording_foundry_client(calls))
    monkeypatch.setattr(bot, "FOUNDRY_MODEL", "test-model", raising=False)

    history = [{"user": "q", "assistant": "a"}]
    asyncio.run(
        bot._run_foundry_tool_loop(
            "sys", "analyse this", b"\x89PNG-bytes", "image/png", [], session=None, history=history
        )
    )

    sent = calls[0]["input"]
    assert sent[:2] == [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "a"},
    ]
    assert sent[-1]["type"] == "message"
    assert sent[-1]["content"][0]["text"] == "analyse this"
    encoded = base64.b64encode(b"\x89PNG-bytes").decode()
    assert sent[-1]["content"][1]["image_url"] == f"data:image/png;base64,{encoded}"


# ── Reading memory back out of the transcript DB ─────────────────────────────
def test_get_recent_turns_disabled_without_db(bot):
    """conftest sets TRANSCRIPT_DB_URL="" — memory degrades to none, and must
    not raise or attempt a connection."""
    assert transcript_db.get_recent_turns("123456") == []


def test_get_recent_turns_survives_db_failure(monkeypatch):
    monkeypatch.setattr(transcript_db, "_DB_URL", "postgresql://unreachable/db")
    monkeypatch.setattr(
        transcript_db, "_get_conn",
        lambda: (_ for _ in ()).throw(RuntimeError("connection refused")),
    )
    assert transcript_db.get_recent_turns("123456") == []


# ── Ordering: the current question must not appear in its own history ────────
def test_history_read_before_current_turn_is_logged(bot, monkeypatch):
    """start_turn() inserts the athlete's message before the LLM runs, so
    reading memory afterwards would feed their question back as context."""
    order = []
    captured = {}

    monkeypatch.setattr(
        bot.transcript_db, "get_recent_turns",
        lambda *a, **k: order.append("read") or [{"user": "q", "assistant": "a"}],
    )
    monkeypatch.setattr(
        bot.transcript_db, "start_turn",
        lambda *a, **k: order.append("write") or 42,
    )

    async def _fake_handler(text, system, image_bytes, media_type, turn_id, history):
        captured["history"] = history
        captured["turn_id"] = turn_id
        return "reply"

    monkeypatch.setattr(bot, "_handle_message_async", _fake_handler)

    assert bot.handle_incoming_message("and tomorrow?", chat_id="123456") == "reply"
    assert order == ["read", "write"]
    assert captured["history"] == [{"user": "q", "assistant": "a"}]
    assert captured["turn_id"] == 42
