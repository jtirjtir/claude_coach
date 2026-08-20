"""Test bootstrap.

`coach_bot` does real work at import time: it reads required env vars with
`os.environ[...]`, calls `load_dotenv()` on the real `.env`, constructs an LLM
client, and calls `transcript_db.ensure_schema()`. So the environment has to be
faked *before* the first import, which is why this lives at module scope rather
than in a fixture.

`load_dotenv` does not override variables that are already set, so setting them
here also stops the developer's real credentials leaking into a test run.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent

_TEST_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-token",
    "TELEGRAM_CHAT_ID": "123456",
    "INTERVALS_API_KEY": "test-key",
    "INTERVALS_ATHLETE_ID": "i00000",
    "ANTHROPIC_API_KEY": "test-anthropic-key",
    "LLM_PROVIDER": "anthropic",
    # Empty (not unset) — the documented "transcript logging disabled" path, so
    # ensure_schema() returns without trying to reach Postgres.
    "TRANSCRIPT_DB_URL": "",
    # Strava off by default; tests that need it monkeypatch STRAVA_ENABLED.
    "STRAVA_CLIENT_ID": "",
    "STRAVA_CLIENT_SECRET": "",
    "STRAVA_REFRESH_TOKEN": "",
}
for _k, _v in _TEST_ENV.items():
    os.environ[_k] = _v

if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import pytest  # noqa: E402

import coach_bot as _coach_bot  # noqa: E402


@pytest.fixture
def bot():
    """The imported coach_bot module."""
    return _coach_bot


@pytest.fixture
def state_file(tmp_path, monkeypatch):
    """Point STATE_FILE at a temp path so state tests never touch the real one."""
    path = tmp_path / "state.json"
    monkeypatch.setattr(_coach_bot, "STATE_FILE", str(path))
    return path


def set_goal_days_out(bot, monkeypatch, days: int, **overrides):
    """Pin the active goal `days` from today.

    Tests that reason about a lookahead window need the race-protection window
    (RACE_PROTECT_DAYS before goal day) kept well clear of it, otherwise which
    branch they exercise silently depends on when the suite runs. Patching the
    module-level active goal rather than calling set_active_goal() keeps the
    change scoped to the test."""
    date = (datetime.now(bot.AEST) + timedelta(days=days)).strftime("%Y-%m-%d")
    goal = bot.normalise_goal({**bot.DEFAULT_GOAL, "date": date, **overrides})
    monkeypatch.setattr(bot, "_active_goal", goal)
    return goal


@pytest.fixture
def no_telegram(monkeypatch):
    """Swallow all outbound Telegram traffic and record it."""
    sent = []
    monkeypatch.setattr(_coach_bot, "send_telegram", lambda msg: sent.append(msg) or True)
    monkeypatch.setattr(
        _coach_bot, "send_telegram_proposal",
        lambda summary, pid: sent.append((pid, summary)) or True,
    )
    return sent
