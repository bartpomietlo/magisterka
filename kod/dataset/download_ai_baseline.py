#!/usr/bin/env python3
"""
download_ai_baseline.py
Pobiera bazowy zbiór krótkich klipów AI do dataset/ai_baseline z YouTube
na podstawie ukierunkowanych zapytań (Runway, Luma, Kling, Pika, Sora).
"""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from yt_dlp import YoutubeDL
except ImportError:
    print("Brak pakietu yt-dlp. Zainstaluj: pip install yt-dlp", file=sys.stderr)
    sys.exit(1)

OUT_DIR  = Path("dataset/ai_baseline")
MANIFEST = OUT_DIR / "manifest.csv"

QUERIES = [
    ("runway",    "Runway Gen-3 AI short film"),
    ("runway",    "Runway AI generated video"),
    ("luma",      "Luma Dream Machine AI video"),
    ("kling",     "Kling AI generated video"),
    ("pika",      "Pika AI generated video"),
    ("sora",      "OpenAI Sora AI video demo"),
    ("synthetic", "AI generated cinematic video"),
    ("synthetic", "text to video AI demo"),
]


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {row["video_id"] for row in csv.DictReader(f) if row.get("video_id")}


def append_rows(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "video_id", "title", "uploader", "duration_s", "webpage_url",
        "query_tag", "query_text", "local_path", "downloaded_at_utc"
    ]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def search_entries(query: str, limit: int) -> list[dict]:
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist"}
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{limit}:{query}", download=False)
    return info.get("entries", []) if info else []


def download_video(url: str, out_dir: Path) -> tuple[dict, str | None]:
    outtmpl = str(out_dir / "%(id)s_%(title).120B.%(ext)s")
    opts = {
        "quiet": True, "no_warnings": True,
        "merge_output_format": "mp4",
        "format": "bv*[ext=mp4]+ba[ext=m4a]/b[ext=mp4]/b",
        "outtmpl": outtmpl,
        "noplaylist": True,
        "windowsfilenames": True,
        "overwrites": False,
        "ffmpeg_location": shutil.which("ffmpeg") or "ffmpeg",
    }
    with YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        path = ydl.prepare_filename(info)
        if not path.lower().endswith(".mp4"):
            merged = str(Path(path).with_suffix("")) + ".mp4"
            if Path(merged).exists():
                path = merged
        return info, path if Path(path).exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-query",    type=int, default=12)
    parser.add_argument("--max-duration", type=int, default=180)
    parser.add_argument("--min-duration", type=int, default=4)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    existing_ids = load_existing_ids(MANIFEST)
    seen_ids     = set(existing_ids)
    selected     = []

    print(f"[INFO] Istniejących ID w manifeście: {len(existing_ids)}")

    for tag, query in QUERIES:
        print(f"[SEARCH] {query}")
        try:
            entries = search_entries(query, args.per_query)
        except Exception as e:
            print(f"  [BŁĄD] search: {e}", file=sys.stderr)
            continue
        for e in entries:
            if not e:
                continue
            vid      = e.get("id")
            duration = e.get("duration")
            url      = e.get("url") or (f"https://www.youtube.com/watch?v={vid}" if vid else None)
            if not vid or not url:
                continue
            if vid in seen_ids:
                continue
            if duration is None or not (args.min_duration <= duration <= args.max_duration):
                continue
            selected.append({"video_id": vid, "title": e.get("title") or "untitled",
                              "duration_s": duration, "webpage_url": url,
                              "query_tag": tag, "query_text": query})
            seen_ids.add(vid)

    print(f"[INFO] Wybranych unikalnych filmów: {len(selected)}")
    added_rows = []

    for item in selected:
        print(f"[DOWNLOAD] {item['title']}")
        try:
            info, local_path = download_video(item["webpage_url"], OUT_DIR)
        except Exception as e:
            print(f"  [BŁĄD] {item['video_id']}: {e}", file=sys.stderr)
            continue
        added_rows.append({
            "video_id":         item["video_id"],
            "title":            info.get("title") or item["title"],
            "uploader":         info.get("uploader") or "",
            "duration_s":       info.get("duration") or item["duration_s"],
            "webpage_url":      info.get("webpage_url") or item["webpage_url"],
            "query_tag":        item["query_tag"],
            "query_text":       item["query_text"],
            "local_path":       local_path or "",
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        })

    if added_rows:
        append_rows(MANIFEST, added_rows)

    print(f"\n[MANIFEST] Zapisano: {MANIFEST}")
    print(f"[WYNIK] Nowo pobranych: {len(added_rows)}")
    print(f"[WYNIK] Łącznie: {len(existing_ids) + len(added_rows)}")


if __name__ == "__main__":
    main()