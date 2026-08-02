#!/usr/bin/env python3
"""
Quick CLI to inspect recent coaching-bot transcript entries in Postgres.

Usage:
  python3 check_transcripts.py                  # last 10 turns (summary)
  python3 check_transcripts.py -n 25             # last 25 turns
  python3 check_transcripts.py --turn 42          # full detail for one turn
  python3 check_transcripts.py --errors           # only turns that errored
  python3 check_transcripts.py --dump-image 42 out.png   # save turn 42's image to a file
"""

import argparse
import os
import sys
from zoneinfo import ZoneInfo

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

DB_URL = os.environ.get("TRANSCRIPT_DB_URL")

# Timestamps are stored in UTC (Postgres TIMESTAMPTZ) — display them in the
# host's local zone (same zone coach_bot.py uses for scheduling: AEST/AEDT).
try:
    with open("/etc/timezone") as f:
        LOCAL_TZ = ZoneInfo(f.read().strip())
except OSError:
    LOCAL_TZ = ZoneInfo("Australia/Sydney")


def connect():
    if not DB_URL:
        sys.exit("TRANSCRIPT_DB_URL not set — check .env")
    return psycopg2.connect(DB_URL)


def truncate(s: str | None, n: int = 70) -> str:
    if not s:
        return ""
    s = s.replace("\n", " ")
    return s if len(s) <= n else s[: n - 1] + "…"


def print_summary(rows: list[dict]) -> None:
    for row in rows:
        flags = []
        if row["has_image"]:
            flags.append("📷")
        if row["error"]:
            flags.append("⚠️ERROR")
        flag_str = " ".join(flags)
        local_time = row["created_at"].astimezone(LOCAL_TZ)
        print(
            f"[{row['id']:>4}] {local_time:%Y-%m-%d %H:%M:%S %Z}  "
            f"{row['provider']}/{row['model']}  {flag_str}"
        )
        print(f"       U: {truncate(row['user_text'])}")
        if row["error"]:
            print(f"       E: {truncate(row['error'])}")
        else:
            print(f"       A: {truncate(row['reply_text'])}")
        print()


def print_turn_detail(conn, turn_id: int) -> None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM conversation_turns WHERE id = %s", (turn_id,))
        turn = cur.fetchone()
        if not turn:
            sys.exit(f"No turn with id {turn_id}")

        print(f"── Turn {turn_id} ──────────────────────────────────")
        print(f"chat_id:    {turn['chat_id']}  (telegram_message_id={turn['telegram_message_id']})")
        print(f"created_at: {turn['created_at'].astimezone(LOCAL_TZ):%Y-%m-%d %H:%M:%S %Z}")
        print(f"provider:   {turn['provider']} / {turn['model']}")
        print(f"has_image:  {turn['has_image']}")
        print(f"\nUser:\n{turn['user_text']}\n")

        cur.execute(
            "SELECT sequence, tool_name, arguments, output FROM tool_calls "
            "WHERE turn_id = %s ORDER BY sequence",
            (turn_id,),
        )
        tool_calls = cur.fetchall()
        if tool_calls:
            print("Tool calls:")
            for tc in tool_calls:
                print(f"  {tc['sequence']}. {tc['tool_name']}({tc['arguments']})")
                print(f"     → {truncate(tc['output'], 200)}")
            print()

        if turn["has_image"]:
            cur.execute(
                "SELECT media_type, size_bytes, width, height FROM turn_images WHERE turn_id = %s",
                (turn_id,),
            )
            img = cur.fetchone()
            if img:
                print(f"Image: {img['media_type']}, {img['size_bytes']} bytes, {img['width']}x{img['height']}\n")

        if turn["error"]:
            print(f"Error:\n{turn['error']}\n")
        else:
            print(f"Assistant:\n{turn['reply_text']}\n")


def dump_image(conn, turn_id: int, out_path: str) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT data FROM turn_images WHERE turn_id = %s", (turn_id,))
        row = cur.fetchone()
        if not row:
            sys.exit(f"No image found for turn {turn_id}")
        with open(out_path, "wb") as f:
            f.write(bytes(row[0]))
        print(f"Wrote {out_path} ({len(row[0])} bytes)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-n", type=int, default=10, help="Number of recent turns to show (default 10)")
    parser.add_argument("--turn", type=int, help="Show full detail for a single turn id")
    parser.add_argument("--errors", action="store_true", help="Only show turns that errored")
    parser.add_argument("--dump-image", nargs=2, metavar=("TURN_ID", "OUT_PATH"), help="Save a turn's image to a file")
    args = parser.parse_args()

    conn = connect()

    if args.dump_image:
        dump_image(conn, int(args.dump_image[0]), args.dump_image[1])
        return

    if args.turn is not None:
        print_turn_detail(conn, args.turn)
        return

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        where = "WHERE error IS NOT NULL" if args.errors else ""
        cur.execute(f"""
            SELECT id, created_at, has_image, provider, model, user_text, reply_text, error
            FROM conversation_turns
            {where}
            ORDER BY id DESC
            LIMIT %s
        """, (args.n,))
        rows = cur.fetchall()

    if not rows:
        print("No matching turns found.")
        return

    print_summary(list(reversed(rows)))


if __name__ == "__main__":
    main()
