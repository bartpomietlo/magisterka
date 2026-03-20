#!/usr/bin/env python3
"""
generate_adversarial.py
Generuje wersje adv_compressed i adv_cropped z klipów ai_baseline.
Wejście:  dataset/ai_baseline/*.mp4
Wyjście:  dataset/adv_compressed/*.mp4
          dataset/adv_cropped/*.mp4
"""

import subprocess
import shutil
from pathlib import Path
import csv
import sys

# ── Konfiguracja ──────────────────────────────────────────────────────────────
SRC_DIR        = Path("dataset/ai_baseline")
COMP_DIR       = Path("dataset/adv_compressed")
CROP_DIR       = Path("dataset/adv_cropped")
MANIFEST_PATH  = Path("dataset/adversarial_manifest.csv")

# Kompresja: CRF 28 = wyraźna ale nie destrukcyjna (typowy YouTube re-encode)
COMPRESS_CRF   = 28
COMPRESS_PRESET = "fast"

# Kadrowanie: usuń 5% z każdej krawędzi (in/out = 0.90 rozmiaru)
CROP_RATIO     = 0.90   # zostaje 90% szerokości i wysokości

FFMPEG         = shutil.which("ffmpeg") or "ffmpeg"
# ──────────────────────────────────────────────────────────────────────────────


def run_ffmpeg(args: list[str], label: str) -> bool:
    cmd = [FFMPEG, "-y", "-hide_banner", "-loglevel", "error"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [BŁĄD] {label}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def get_video_size(path: Path) -> tuple[int, int] | None:
    """Zwraca (width, height) używając ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=p=0",
        str(path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    parts = result.stdout.strip().split(",")
    return int(parts[0]), int(parts[1])


def make_compressed(src: Path, dst: Path) -> bool:
    return run_ffmpeg([
        "-i", str(src),
        "-c:v", "libx264",
        "-crf", str(COMPRESS_CRF),
        "-preset", COMPRESS_PRESET,
        "-c:a", "copy",
        str(dst)
    ], label=f"compress {src.name}")


def make_cropped(src: Path, dst: Path) -> bool:
    size = get_video_size(src)
    if size is None:
        print(f"  [POMIŃ] nie można odczytać rozmiaru: {src.name}", file=sys.stderr)
        return False
    w, h = size
    nw = int(w * CROP_RATIO) & ~1   # musi być parzyste dla libx264
    nh = int(h * CROP_RATIO) & ~1
    x  = (w - nw) // 2
    y  = (h - nh) // 2
    crop_filter = f"crop={nw}:{nh}:{x}:{y}"
    return run_ffmpeg([
        "-i", str(src),
        "-vf", crop_filter,
        "-c:v", "libx264",
        "-crf", "18",           # wysoka jakość – zmiana tylko geometrii
        "-preset", COMPRESS_PRESET,
        "-c:a", "copy",
        str(dst)
    ], label=f"crop {src.name}")


def main():
    sources = sorted(SRC_DIR.glob("*.mp4"))
    if not sources:
        print(f"Brak plików MP4 w {SRC_DIR}. Uruchom najpierw pobieranie ai_baseline.")
        sys.exit(1)

    COMP_DIR.mkdir(parents=True, exist_ok=True)
    CROP_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for src in sources:
        stem = src.stem
        print(f"[{stem}]")

        # --- compressed ---
        dst_c = COMP_DIR / src.name
        if dst_c.exists():
            print("  compressed: już istnieje, pomijam")
            ok_c = True
        else:
            ok_c = make_compressed(src, dst_c)
            print(f"  compressed: {'OK' if ok_c else 'BŁĄD'}")

        # --- cropped ---
        dst_r = CROP_DIR / src.name
        if dst_r.exists():
            print("  cropped:    już istnieje, pomijam")
            ok_r = True
        else:
            ok_r = make_cropped(src, dst_r)
            print(f"  cropped:    {'OK' if ok_r else 'BŁĄD'}")

        rows.append({
            "source":          str(src),
            "compressed":      str(dst_c) if ok_c else "",
            "cropped":         str(dst_r) if ok_r else "",
            "compressed_ok":   ok_c,
            "cropped_ok":      ok_r,
        })

    # Manifest
    with open(MANIFEST_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    ok_c_n = sum(1 for r in rows if r["compressed_ok"])
    ok_r_n = sum(1 for r in rows if r["cropped_ok"])
    print(f"\n[MANIFEST] Zapisano: {MANIFEST_PATH}")
    print(f"[WYNIK] compressed: {ok_c_n}/{len(rows)}  |  cropped: {ok_r_n}/{len(rows)}")


if __name__ == "__main__":
    main()
