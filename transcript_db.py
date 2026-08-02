"""
Transcript logging — records every coaching conversation turn (user input,
tool calls, assistant output, and any uploaded images) to Postgres for local
history/analysis. Storage is provider-agnostic: the raw LLM response object
is stashed as JSONB alongside the extracted text, so Anthropic and Azure
Foundry (or any future provider) turns land in the same schema.

Never allowed to break the bot: every public function catches its own
exceptions and logs a warning instead of raising.
"""

import io
import json
import logging
import os

import psycopg2
import psycopg2.extras

log = logging.getLogger(__name__)

_DB_URL = os.environ.get("TRANSCRIPT_DB_URL")
_conn = None


def _get_conn():
    """Lazily open (or reopen, after a drop) a single connection — this bot
    processes one Telegram update at a time, so a connection pool would be
    pure overhead."""
    global _conn
    if _conn is not None and _conn.closed == 0:
        return _conn
    _conn = psycopg2.connect(_DB_URL)
    _conn.autocommit = True
    return _conn


def ensure_schema() -> None:
    """Create the transcript tables if they don't already exist. Safe to call
    on every startup."""
    if not _DB_URL:
        log.warning("TRANSCRIPT_DB_URL not set — transcript logging disabled")
        return
    try:
        with _get_conn().cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id                    SERIAL PRIMARY KEY,
                    chat_id               TEXT NOT NULL,
                    telegram_message_id   BIGINT,
                    created_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
                    user_text             TEXT,
                    has_image             BOOLEAN NOT NULL DEFAULT FALSE,
                    provider              TEXT,
                    model                 TEXT,
                    reply_text            TEXT,
                    raw_response          JSONB,
                    error                 TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_turns_chat_id    ON conversation_turns (chat_id);
                CREATE INDEX IF NOT EXISTS idx_turns_created_at ON conversation_turns (created_at);

                CREATE TABLE IF NOT EXISTS turn_images (
                    id           SERIAL PRIMARY KEY,
                    turn_id      INTEGER NOT NULL REFERENCES conversation_turns (id) ON DELETE CASCADE,
                    media_type   TEXT,
                    size_bytes   INTEGER,
                    width        INTEGER,
                    height       INTEGER,
                    data         BYTEA NOT NULL,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                );

                CREATE TABLE IF NOT EXISTS tool_calls (
                    id           SERIAL PRIMARY KEY,
                    turn_id      INTEGER NOT NULL REFERENCES conversation_turns (id) ON DELETE CASCADE,
                    sequence     INTEGER NOT NULL,
                    tool_name    TEXT NOT NULL,
                    arguments    JSONB,
                    output       TEXT,
                    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
                );
                CREATE INDEX IF NOT EXISTS idx_tool_calls_turn_id ON tool_calls (turn_id);
            """)
        log.info("Transcript DB schema ready")
    except Exception as e:
        log.error(f"Transcript DB schema setup failed: {e}")


def _image_dimensions(image_bytes: bytes) -> tuple[int | None, int | None]:
    try:
        from PIL import Image
        with Image.open(io.BytesIO(image_bytes)) as img:
            return img.width, img.height
    except Exception:
        return None, None


def start_turn(
    chat_id: str,
    telegram_message_id: int | None,
    user_text: str,
    image_bytes: bytes | None,
    media_type: str,
    provider: str,
    model: str,
) -> int | None:
    """Insert the user side of a turn (+ image, if any). Returns the turn id
    to thread through tool-call logging and the final reply, or None if
    logging is unavailable."""
    if not _DB_URL:
        return None
    try:
        conn = _get_conn()
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversation_turns
                       (chat_id, telegram_message_id, user_text, has_image, provider, model)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (chat_id, telegram_message_id, user_text, image_bytes is not None, provider, model),
            )
            turn_id = cur.fetchone()[0]

            if image_bytes is not None:
                width, height = _image_dimensions(image_bytes)
                cur.execute(
                    """INSERT INTO turn_images (turn_id, media_type, size_bytes, width, height, data)
                       VALUES (%s, %s, %s, %s, %s, %s)""",
                    (turn_id, media_type, len(image_bytes), width, height, psycopg2.Binary(image_bytes)),
                )
        return turn_id
    except Exception as e:
        log.error(f"Transcript log (start_turn) failed: {e}")
        return None


def log_tool_call(turn_id: int | None, sequence: int, tool_name: str, arguments: dict, output: str) -> None:
    if turn_id is None or not _DB_URL:
        return
    try:
        with _get_conn().cursor() as cur:
            cur.execute(
                """INSERT INTO tool_calls (turn_id, sequence, tool_name, arguments, output)
                   VALUES (%s, %s, %s, %s, %s)""",
                (turn_id, sequence, tool_name, json.dumps(arguments), output),
            )
    except Exception as e:
        log.error(f"Transcript log (log_tool_call) failed: {e}")


def log_reply(turn_id: int | None, reply_text: str, raw_response: dict | None, error: str | None = None) -> None:
    if turn_id is None or not _DB_URL:
        return
    try:
        with _get_conn().cursor() as cur:
            cur.execute(
                """UPDATE conversation_turns
                   SET reply_text = %s, raw_response = %s, error = %s
                   WHERE id = %s""",
                (reply_text, json.dumps(raw_response) if raw_response is not None else None, error, turn_id),
            )
    except Exception as e:
        log.error(f"Transcript log (log_reply) failed: {e}")
