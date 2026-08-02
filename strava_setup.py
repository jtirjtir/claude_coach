#!/usr/bin/env python3
"""
Strava OAuth setup — run once to get a refresh token and save it to .env
Usage: python3 strava_setup.py
"""

import os
import sys
import webbrowser
import requests
from dotenv import load_dotenv, set_key

ENV_FILE = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_FILE)

REDIRECT_URI = "http://localhost:8888/callback"
SCOPE        = "read,activity:read_all,profile:read_all"

def main():
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Strava OAuth Setup")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    print("You need a Strava API application. If you don't have one:")
    print("  1. Go to https://www.strava.com/settings/api")
    print("  2. Create an app (any name, set 'Authorization Callback Domain' to localhost)")
    print("  3. Copy the Client ID and Client Secret below\n")

    client_id = os.environ.get("STRAVA_CLIENT_ID", "").strip()
    if not client_id:
        client_id = input("Strava Client ID: ").strip()
    else:
        print(f"Using STRAVA_CLIENT_ID from .env: {client_id}")

    client_secret = os.environ.get("STRAVA_CLIENT_SECRET", "").strip()
    if not client_secret:
        client_secret = input("Strava Client Secret: ").strip()
    else:
        print("Using STRAVA_CLIENT_SECRET from .env")

    if not client_id or not client_secret:
        print("\n✗ Client ID and Secret are required.")
        sys.exit(1)

    auth_url = (
        f"https://www.strava.com/oauth/authorize"
        f"?client_id={client_id}"
        f"&response_type=code"
        f"&redirect_uri={REDIRECT_URI}"
        f"&approval_prompt=force"
        f"&scope={SCOPE}"
    )

    print(f"\nOpening browser to authorize…")
    print(f"If it doesn't open automatically, visit:\n  {auth_url}\n")
    webbrowser.open(auth_url)

    print("After authorizing, your browser will be redirected to localhost (which will fail — that's OK).")
    print("Copy the full URL from the browser address bar and paste it here.\n")
    redirect_url = input("Paste redirect URL: ").strip()

    # Extract the code from the redirect URL
    if "code=" not in redirect_url:
        print("\n✗ Could not find authorization code in URL.")
        sys.exit(1)

    code = redirect_url.split("code=")[1].split("&")[0]

    # Exchange code for tokens
    print("\nExchanging code for tokens…")
    r = requests.post("https://www.strava.com/oauth/token", json={
        "client_id":     client_id,
        "client_secret": client_secret,
        "code":          code,
        "grant_type":    "authorization_code",
    }, timeout=15)

    if r.status_code != 200:
        print(f"\n✗ Token exchange failed: {r.status_code} {r.text}")
        sys.exit(1)

    data          = r.json()
    refresh_token = data["refresh_token"]
    access_token  = data["access_token"]
    athlete       = data.get("athlete", {})
    athlete_name  = f"{athlete.get('firstname', '')} {athlete.get('lastname', '')}".strip()

    print(f"\n✓ Authorized as: {athlete_name} (ID: {athlete.get('id')})")

    # Save to .env
    set_key(ENV_FILE, "STRAVA_CLIENT_ID",     client_id)
    set_key(ENV_FILE, "STRAVA_CLIENT_SECRET", client_secret)
    set_key(ENV_FILE, "STRAVA_REFRESH_TOKEN", refresh_token)

    print(f"✓ Saved STRAVA_CLIENT_ID, STRAVA_CLIENT_SECRET, STRAVA_REFRESH_TOKEN to .env")
    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("  Setup complete! Restart the coaching-bot service:")
    print("  sudo systemctl restart coaching-bot")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")

if __name__ == "__main__":
    main()
