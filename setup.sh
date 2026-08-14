#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Virtual Coaching Bot — Ubuntu Setup Script
# Run as: bash setup.sh  (from anywhere — uses script's own directory)
# ─────────────────────────────────────────────────────────────────────────────
set -e

# Anything this script creates holds credentials (.env) or personal health data
# (state.json), so never let the caller's umask make them group/world readable.
umask 077

# Always resolve to the directory this script lives in
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_NAME="coaching-bot"
CURRENT_USER=$(whoami)
VENV_DIR="$INSTALL_DIR/.venv"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Virtual Coaching Bot — Setup"
echo "  Install dir: $INSTALL_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Install Python deps into a virtualenv
# A venv rather than `pip3 install --break-system-packages`: that bypasses PEP 668
# and can break system Python tooling, and left the deployed version set to
# whatever happened to be on the box. Versions are pinned in requirements.txt.
echo ""
echo "▶ Creating virtualenv at $VENV_DIR..."
python3 -m venv "$VENV_DIR"

echo ""
echo "▶ Installing Python dependencies..."
"$VENV_DIR/bin/pip" install --upgrade pip
"$VENV_DIR/bin/pip" install -r "$INSTALL_DIR/requirements.txt"

# 2. Copy env template if .env doesn't exist
if [ ! -f "$INSTALL_DIR/.env" ]; then
    if [ -f "$INSTALL_DIR/.env.template" ]; then
        echo ""
        echo "▶ Creating .env from template..."
        cp "$INSTALL_DIR/.env.template" "$INSTALL_DIR/.env"
    else
        echo ""
        echo "▶ Creating fresh .env file..."
        cat > "$INSTALL_DIR/.env" << 'ENVEOF'
# Telegram Bot (get token from @BotFather)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
# Your Telegram Chat ID — message your bot, then visit:
# https://api.telegram.org/bot<TOKEN>/getUpdates  and find "chat":{"id":...}
TELEGRAM_CHAT_ID=your_chat_id_here

# Anthropic Claude API — https://console.anthropic.com
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Intervals.icu — Settings → API
INTERVALS_API_KEY=your_intervals_api_key_here
# Athlete ID from your Intervals.icu profile URL (e.g. "i12345")
INTERVALS_ATHLETE_ID=i00000

# Garmin Connect — OPTIONAL, and read-only. Only used by the 05:00 freshness
# probe, which logs whether Garmin has last night's sleep/HRV at a time when
# Intervals.icu does not. Leave blank to skip the probe entirely.
GARMIN_EMAIL=
GARMIN_PASSWORD=

# Transcript DB — OPTIONAL Postgres URL. Stores every conversation turn, and is
# also what gives the coach its memory of the recent chat: leave it blank and
# each message is answered as a fresh conversation with no history.
# e.g. postgresql://user:pass@localhost:5432/coach
TRANSCRIPT_DB_URL=

# Conversation memory tuning — OPTIONAL, defaults shown. Only the question and
# final reply of each turn are replayed (never tool output or images).
# MEMORY_TURNS=6           # prior turn pairs replayed into each question
# MEMORY_WINDOW_HOURS=24   # ignore turns older than this
# MEMORY_MAX_CHARS=2000    # truncate each replayed message to this length
ENVEOF
    fi
    echo "  ⚠️  Edit $INSTALL_DIR/.env with your credentials before starting!"
else
    echo ""
    echo "▶ .env already exists — skipping (not overwriting credentials)"
fi

# 3. Initialise state file
if [ ! -f "$INSTALL_DIR/state.json" ]; then
    echo '{"last_activity_id": null, "last_briefing_date": null}' > "$INSTALL_DIR/state.json"
    echo "▶ Created state.json"
fi

# Enforce restrictive modes explicitly rather than relying on the umask above —
# these files may predate this script, or have been created by an editor.
# .env holds every API credential; state.json holds wellness/HRV history.
chmod 600 "$INSTALL_DIR/.env" "$INSTALL_DIR/state.json"
for _secret in foundry_apikey foundry_endpoint; do
    if [ -f "$INSTALL_DIR/$_secret" ]; then
        chmod 600 "$INSTALL_DIR/$_secret"
    fi
done

# 4. Install systemd service (updates path to actual install dir)
echo ""
echo "▶ Installing systemd service..."
sed -e "s|User=ubuntu|User=$CURRENT_USER|g" \
    -e "s|WorkingDirectory=.*|WorkingDirectory=$INSTALL_DIR|g" \
    -e "s|EnvironmentFile=.*|EnvironmentFile=$INSTALL_DIR/.env|g" \
    -e "s|ExecStart=.*|ExecStart=$VENV_DIR/bin/python $INSTALL_DIR/coach_bot.py|g" \
    "$INSTALL_DIR/coaching-bot.service" > /tmp/coaching-bot.service.tmp

sudo cp /tmp/coaching-bot.service.tmp /etc/systemd/system/coaching-bot.service
sudo systemctl daemon-reload

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo "  1. Fill in credentials:  nano $INSTALL_DIR/.env"
echo "  2. Test manually:        $VENV_DIR/bin/python $INSTALL_DIR/coach_bot.py"
echo "  3. Enable on boot:       sudo systemctl enable $SERVICE_NAME"
echo "  4. Start now:            sudo systemctl start $SERVICE_NAME"
echo "  5. Watch logs:           sudo journalctl -u $SERVICE_NAME -f"
echo ""
echo "  Quick test — send yourself a message:"
echo "  cd $INSTALL_DIR && python3 -c \""
echo "import os; from dotenv import load_dotenv; load_dotenv()"
echo "from coach_bot import send_telegram"
echo "send_telegram('🤖 Coaching bot online! City2Surf Aug 2026 — lets go!')\""
