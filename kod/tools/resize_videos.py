#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Brak narzedzia w PATH: {name}")
    return binary


def probe_duration_seconds(ffprobe: str, video_path: Path) -> float:
    cmd = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def build_ffmpeg_cmd(
    ffmpeg: str,
    in_path: Path,
    out_path: Path,
    max_seconds: float,
    max_height: int,
    trim: bool,
) -> list[str]:
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(in_path),
    ]
    if trim:
        cmd += ["-t", str(max_seconds)]

    # Nie powieksza materialu ponizej max_height i zachowuje proporcje.
    # Przecinek w min() musi byc escapowany dla ffmpeg.
    scale_filter = f"scale=-2:min({max_height}\\,ih)"
    cmd += [
        "-vf",
        scale_filter,
        "-c:v",
        "libx264",
        "-crf",
        "23",
        "-c:a",
        "aac",
        str(out_path),
    ]
    return cmd


def main() -> int:
    # Windows cp1250 potrafi wysypac wypisywanie znakow z nazw plikow.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="Przytnij filmy do 15s i przeskaluj do 480p (max height)."
    )
    parser.add_argument(
        "--input-dir",
        default="input",
        help="Folder wejsciowy z plikami .mp4 (domyslnie: input)",
    )
    parser.add_argument(
        "--output-dir",
        default="output_resized",
        help="Folder wyjsciowy (domyslnie: output_resized)",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=15.0,
        help="Maksymalna dlugosc wyjsciowa w sekundach (domyslnie: 15)",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=480,
        help="Maksymalna wysokosc wyjsciowa (domyslnie: 480)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Brak folderu wejsciowego: {input_dir}", file=sys.stderr)
        return 1

    files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"
    )
    if not files:
        print(f"Brak plikow .mp4 w: {input_dir}")
        return 0

    output_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg = require_binary("ffmpeg")
    ffprobe = require_binary("ffprobe")

    total = len(files)
    print(f"Znaleziono {total} plikow .mp4. Start przetwarzania...")

    for idx, in_path in enumerate(files, 1):
        try:
            duration = probe_duration_seconds(ffprobe, in_path)
        except Exception as exc:  # noqa: BLE001
            print(f"[{idx}/{total}] {in_path.name} | blad ffprobe: {exc}")
            continue

        trim = duration > args.max_seconds
        out_path = output_dir / in_path.name
        trimmed_flag = "tak" if trim else "nie"
        print(
            f"[{idx}/{total}] {in_path.name} | dlugosc={duration:.2f}s | "
            f"przycieto={trimmed_flag}"
        )

        cmd = build_ffmpeg_cmd(
            ffmpeg=ffmpeg,
            in_path=in_path,
            out_path=out_path,
            max_seconds=args.max_seconds,
            max_height=args.max_height,
            trim=trim,
        )
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[{idx}/{total}] blad ffmpeg dla {in_path.name}")
            if result.stderr:
                print(result.stderr.strip())
            continue

    print("Gotowe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
