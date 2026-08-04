"""Regression tests for send_telegram's message chunking.

A long LLM reply (observed with the gpt-5.6-terra Foundry deployment) can
exceed Telegram's 4096-char per-message cap. Before this fix, _post_send_message
raised on the resulting HTTP 400 even after its own Markdown-stripping retry
(removing parse_mode doesn't shorten the text), and send_telegram just
reported failure — the whole reply was dropped, with no way to fix a
too-long message by retrying it unchanged.
"""


def test_short_message_is_a_single_chunk(bot):
    assert bot._chunk_message("hello") == ["hello"]


def test_message_at_exactly_the_limit_is_a_single_chunk(bot):
    text = "a" * bot.TELEGRAM_MAX_CHARS
    assert bot._chunk_message(text) == [text]


def test_long_message_splits_on_paragraph_boundary(bot):
    para = "x" * 3000
    text = f"{para}\n\n{para}\n\n{para}"
    chunks = bot._chunk_message(text, limit=4000)
    assert len(chunks) > 1
    assert all(len(c) <= 4000 for c in chunks)
    # Reassembling (paragraph breaks were the split points, so joined with
    # the separator that was consumed) recovers all the original content.
    assert "".join(chunks).replace("\n\n", "") == text.replace("\n\n", "")


def test_single_run_with_no_break_points_hard_cuts(bot):
    text = "x" * 5000  # no spaces/newlines anywhere
    chunks = bot._chunk_message(text, limit=4096)
    assert len(chunks) == 2
    assert len(chunks[0]) == 4096
    assert "".join(chunks) == text


def test_send_telegram_sends_each_chunk_and_succeeds(bot, monkeypatch):
    sent_payloads = []
    monkeypatch.setattr(bot, "_post_send_message", lambda payload: sent_payloads.append(payload))

    para = "y" * 3000
    message = f"{para}\n\n{para}\n\n{para}"  # > TELEGRAM_MAX_CHARS
    assert bot.send_telegram(message) is True
    assert len(sent_payloads) > 1
    assert all(p["parse_mode"] == "Markdown" for p in sent_payloads)
    assert all(len(p["text"]) <= bot.TELEGRAM_MAX_CHARS for p in sent_payloads)


def test_send_telegram_stops_and_fails_if_a_chunk_send_fails(bot, monkeypatch):
    calls = []

    def flaky_post(payload):
        calls.append(payload)
        if len(calls) == 2:
            raise RuntimeError("simulated Telegram failure")

    monkeypatch.setattr(bot, "_post_send_message", flaky_post)

    para = "z" * 3000
    message = f"{para}\n\n{para}\n\n{para}"
    assert bot.send_telegram(message) is False
    assert len(calls) == 2  # stopped at the failing chunk, didn't send the rest


def test_send_telegram_single_chunk_unchanged_behaviour(bot, monkeypatch):
    sent_payloads = []
    monkeypatch.setattr(bot, "_post_send_message", lambda payload: sent_payloads.append(payload))

    assert bot.send_telegram("short reply") is True
    assert len(sent_payloads) == 1
    assert sent_payloads[0]["text"] == "short reply"
