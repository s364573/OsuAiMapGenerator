"""
download_maps.py - Last ned ranked osu! beatmaps automatisk
Bruker osu! API v2 for søk + Chimu.moe mirror for nedlasting

Bruk:
    python download_maps.py --client_id DIN_ID --client_secret DIN_SECRET --count 3000

Skaff client_id og client_secret på:
    https://osu.ppy.sh/home/account/edit  →  OAuth  →  New OAuth Application
    Callback URL: http://localhost
"""

import os
import sys
import time
import json
import zipfile
import argparse
import requests
from pathlib import Path
from datetime import datetime


# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
OSU_API_BASE  = "https://osu.ppy.sh/api/v2"
OSU_TOKEN_URL = "https://osu.ppy.sh/oauth/token"
MIRRORS = [
    "https://api.chimu.moe/v1/download/{set_id}",
    "https://beatconnect.io/b/{set_id}",
    "https://api.nerinyan.moe/d/{set_id}",
    "https://catboy.best/d/{set_id}",
]

RATE_LIMIT_DELAY = 1.0   # sekunder mellom API-kall
DOWNLOAD_DELAY   = 2.0   # sekunder mellom nedlastinger

# Søk-filter
MIN_STAR    = 4.0
MAX_STAR    = 8.0
MIN_PLAYS   = 10000
GAME_MODE   = 0          # 0 = standard
STATUS      = "ranked"


# ─────────────────────────────────────────────
# AUTH
# ─────────────────────────────────────────────
def get_token(client_id: int, client_secret: str) -> str:
    """Hent OAuth token med client credentials flow."""
    resp = requests.post(OSU_TOKEN_URL, json={
        "client_id":     client_id,
        "client_secret": client_secret,
        "grant_type":    "client_credentials",
        "scope":         "public",
    }, timeout=30)

    if resp.status_code != 200:
        print(f"Token-feil: {resp.status_code} – {resp.text}")
        sys.exit(1)

    token = resp.json()["access_token"]
    print("OAuth token hentet!")
    return token


# ─────────────────────────────────────────────
# SØKEFUNKSJONER
# ─────────────────────────────────────────────
def search_beatmapsets(token: str, cursor_string: str = None,
                       sort: str = "plays_desc") -> dict:
    """Søk etter ranked beatmap sets."""
    headers = {"Authorization": f"Bearer {token}"}
    params  = {
        "m":      GAME_MODE,
        "s":      STATUS,
        "sort":   sort,
        "q":      "",
    }
    if cursor_string:
        params["cursor_string"] = cursor_string

    resp = requests.get(f"{OSU_API_BASE}/beatmapsets/search",
                        headers=headers, params=params, timeout=30)

    if resp.status_code == 401:
        print("Token utløpt – prøv igjen")
        sys.exit(1)
    if resp.status_code != 200:
        print(f"Søk-feil: {resp.status_code}")
        return {"beatmapsets": [], "cursor_string": None}

    return resp.json()


def filter_beatmapset(bms: dict) -> bool:
    """Returner True hvis settet er relevant for trening."""
    beatmaps = bms.get("beatmaps", [])

    # Sjekk at det finnes minst én hard map i riktig range
    has_hard = any(
        MIN_STAR <= bm.get("difficulty_rating", 0) <= MAX_STAR
        and bm.get("mode_int", -1) == GAME_MODE
        for bm in beatmaps
    )

    play_count = bms.get("play_count", 0)

    return has_hard and play_count >= MIN_PLAYS


# ─────────────────────────────────────────────
# NEDLASTING
# ─────────────────────────────────────────────
def download_osz(set_id: int, output_dir: Path,
                 session: requests.Session) -> bool:
    """Last ned .osz fil fra mirror. Returner True hvis vellykket."""

    output_path = output_dir / f"{set_id}.osz"
    if output_path.exists():
        return True  # Allerede lastet ned

    # Prøv Chimu.moe først
    mirrors = [m.format(set_id=set_id) for m in MIRRORS]

    for url in mirrors:
        try:
            resp = session.get(url, timeout=60, stream=True,
                               allow_redirects=True)

            if resp.status_code == 200:
                content_type = resp.headers.get("content-type", "")
                if "application/zip" in content_type or \
                   "application/octet-stream" in content_type or \
                   "application/x-zip" in content_type or \
                   len(resp.content) > 10000:

                    with open(output_path, "wb") as f:
                        for chunk in resp.iter_content(chunk_size=8192):
                            f.write(chunk)

                    # Valider at det er en gyldig zip-fil
                    try:
                        with zipfile.ZipFile(output_path, 'r') as z:
                            names = z.namelist()
                            if any(n.endswith('.osu') for n in names):
                                return True
                        # Zip er gyldig men mangler .osu
                        output_path.unlink()
                    except zipfile.BadZipFile:
                        output_path.unlink()

        except requests.exceptions.RequestException:
            continue
        except Exception:
            if output_path.exists():
                output_path.unlink()
            continue

    return False


def extract_osz(osz_path: Path, songs_dir: Path) -> bool:
    """Pakk ut .osz til Songs-mappe."""
    try:
        set_id    = osz_path.stem
        dest_dir  = songs_dir / set_id
        dest_dir.mkdir(exist_ok=True)

        with zipfile.ZipFile(osz_path, 'r') as z:
            z.extractall(dest_dir)

        osz_path.unlink()  # Slett .osz etter utpakking
        return True

    except Exception as e:
        print(f"  Utpakking feilet: {e}")
        return False


# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
def load_progress(progress_file: Path) -> dict:
    if progress_file.exists():
        with open(progress_file) as f:
            return json.load(f)
    return {"downloaded": [], "failed": [], "cursor": None, "count": 0}


def save_progress(progress_file: Path, progress: dict):
    with open(progress_file, "w") as f:
        json.dump(progress, f, indent=2)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Last ned osu! beatmaps")
    parser.add_argument("--client_id",     type=int, required=True)
    parser.add_argument("--client_secret",           required=True)
    parser.add_argument("--count",         type=int, default=3000,
                        help="Antall sets å laste ned")
    parser.add_argument("--output",        default="downloads",
                        help="Mappe for nedlastede .osz filer")
    parser.add_argument("--songs",         default=None,
                        help="Songs-mappe (pakk ut direkte). Utelat for bare .osz")
    parser.add_argument("--sort",          default="plays_desc",
                        choices=["plays_desc", "ranked_desc", "difficulty_desc"],
                        help="Sorteringsrekkefølge")
    parser.add_argument("--resume",        action="store_true",
                        help="Fortsett fra forrige kjøring")
    args = parser.parse_args()

    output_dir    = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    progress_file = output_dir / "progress.json"

    if args.songs:
        songs_dir = Path(args.songs)
        songs_dir.mkdir(exist_ok=True)
        print(f"Songs-mappe: {songs_dir}")
    else:
        songs_dir = None

    # Last inn progress
    progress = load_progress(progress_file) if args.resume else \
               {"downloaded": [], "failed": [], "cursor": None, "count": 0}

    downloaded_ids = set(progress["downloaded"])
    print(f"Starter (allerede lastet ned: {len(downloaded_ids)})")

    # OAuth token
    token   = get_token(args.client_id, args.client_secret)
    session = requests.Session()
    session.headers.update({
        "User-Agent": "osu-beatmap-downloader/1.0"
    })

    cursor_string = progress.get("cursor") if args.resume else None
    total_downloaded = progress["count"]
    consecutive_failures = 0

    print(f"\nMål: {args.count} beatmap sets")
    print(f"Filter: {MIN_STAR}★ – {MAX_STAR}★, ≥{MIN_PLAYS:,} plays")
    print(f"Sort: {args.sort}\n")

    try:
        while total_downloaded < args.count:
            # Hent neste side med søkeresultater
            result        = search_beatmapsets(token, cursor_string, args.sort)
            beatmapsets   = result.get("beatmapsets", [])
            cursor_string = result.get("cursor_string")

            if not beatmapsets:
                print("Ingen flere resultater.")
                break

            for bms in beatmapsets:
                if total_downloaded >= args.count:
                    break

                set_id = bms["id"]

                if set_id in downloaded_ids:
                    continue

                if not filter_beatmapset(bms):
                    continue

                # Info
                title     = bms.get("title", "?")
                artist    = bms.get("artist", "?")
                plays     = bms.get("play_count", 0)
                stars     = [b.get("difficulty_rating", 0)
                             for b in bms.get("beatmaps", [])
                             if b.get("mode_int") == GAME_MODE]
                max_star  = max(stars) if stars else 0

                print(f"[{total_downloaded+1}/{args.count}] "
                      f"{artist} - {title} "
                      f"({max_star:.1f}★, {plays:,} plays)", end=" ... ")

                success = download_osz(set_id, output_dir, session)

                if success:
                    if songs_dir:
                        osz_path = output_dir / f"{set_id}.osz"
                        if osz_path.exists():
                            extract_osz(osz_path, songs_dir)

                    downloaded_ids.add(set_id)
                    progress["downloaded"].append(set_id)
                    progress["count"] = total_downloaded + 1
                    total_downloaded += 1
                    consecutive_failures = 0
                    print("OK")
                else:
                    progress["failed"].append(set_id)
                    consecutive_failures += 1
                    print("FEILET")

                    if consecutive_failures >= 10:
                        print("\n10 failures på rad – pause 30 sek...")
                        time.sleep(30)
                        consecutive_failures = 0

                # Lagre progress etter hvert set
                progress["cursor"] = cursor_string
                save_progress(progress_file, progress)

                time.sleep(DOWNLOAD_DELAY)

            if not cursor_string:
                print("Nådd slutten av søkeresultatene.")
                break

            time.sleep(RATE_LIMIT_DELAY)

    except KeyboardInterrupt:
        print(f"\n\nAvbrutt! Progress lagret til {progress_file}")

    finally:
        save_progress(progress_file, progress)
        print(f"\n{'='*50}")
        print(f"Lastet ned: {total_downloaded} sets")
        print(f"Feilet:     {len(progress['failed'])}")
        print(f"Progress:   {progress_file}")
        if songs_dir:
            print(f"Songs:      {songs_dir}")


if __name__ == "__main__":
    main()