#!/usr/bin/env python3
from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = REPO_ROOT / "kod" / "dataset"
OUTPUT_DIR = REPO_ROOT / "output_undetectable"
OUTPUT_CSV = REPO_ROOT / "resize_full_comparison.csv"
SIZE_LIMIT_BYTES = int(4.5 * 1024 * 1024)  # 4.5 MB


@dataclass(frozen=True)
class SourceEntry:
    split: str
    path: Path


def require_binary(name: str) -> str:
    binary = shutil.which(name)
    if not binary:
        raise RuntimeError(f"Brak narzedzia w PATH: {name}")
    return binary


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.exists() else 0


def run_ffmpeg(
    ffmpeg: str,
    src: Path,
    dst: Path,
    *,
    seconds: int,
    height: int,
    crf: int,
) -> None:
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-t",
        str(seconds),
        "-vf",
        f"scale=-2:min({height}\\,ih)",
        "-c:v",
        "libx264",
        "-crf",
        str(crf),
        "-c:a",
        "aac",
        str(dst),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def prepare_video(ffmpeg: str, src: Path, dst: Path) -> tuple[int, int, int]:
    """
    Sekwencja zgodna z taskiem:
    1) 15s + 480p
    2) jeśli >4.5MB -> 15s + 360p
    3) jeśli >4.5MB -> 10s + 360p
    Dodatkowo awaryjnie (rzadko): podnoszenie CRF dla 10s/360p, aby wymusic limit.
    Zwraca (seconds, height, crf) finalnych parametrow.
    """
    attempts = [
        (15, 480, 23),
        (15, 360, 23),
        (10, 360, 23),
    ]
    for seconds, height, crf in attempts:
        run_ffmpeg(ffmpeg, src, dst, seconds=seconds, height=height, crf=crf)
        if file_size_bytes(dst) <= SIZE_LIMIT_BYTES:
            return seconds, height, crf

    # Fallback: mocniejsze kodowanie, gdyby nadal > 4.5MB.
    for crf in (26, 28, 30, 32):
        run_ffmpeg(ffmpeg, src, dst, seconds=10, height=360, crf=crf)
        if file_size_bytes(dst) <= SIZE_LIMIT_BYTES:
            return 10, 360, crf

    return 10, 360, 32


def collect_sources() -> list[SourceEntry]:
    sources: list[SourceEntry] = []
    for split in ("ai_baseline", "adv_fp_trap"):
        folder = DATASET_ROOT / split
        files = sorted(folder.glob("*.mp4"))
        for p in files:
            sources.append(SourceEntry(split=split, path=p))
    return sources


def run_detector(eval_module: Any, video_path: Path) -> tuple[int, float]:
    result, _elapsed = eval_module.scan_video(video_path)
    sig = eval_module.extract_signals(result)
    c2pa_sig = eval_module.detect_c2pa_signal(video_path)

    det, score, mode, ai_specific, broadcast_trap = eval_module.fuse(
        zv_count=sig["zv_count"],
        zv_lower_third_roi_count=sig["zv_lower_third_roi_count"],
        of_count=sig["of_count"],
        of_max_area=sig["of_max_area"],
        of_max_area_ratio=sig["of_max_area_ratio"],
        iw_similarity=sig["iw_best_similarity"],
        iw_matched=sig["iw_matched"],
        fft_score=sig["fft_score"],
        of_texture_variance_mean=sig["of_texture_variance_mean"],
        of_low_texture_roi_count=sig["of_low_texture_roi_count"],
        of_wide_lower_roi_count=sig["of_wide_lower_roi_count"],
        of_corner_compact_roi_count=sig["of_corner_compact_roi_count"],
        of_lower_third_roi_ratio=sig["of_lower_third_roi_ratio"],
        of_upper_third_roi_ratio=sig["of_upper_third_roi_ratio"],
        of_center_roi_ratio=sig["of_center_roi_ratio"],
        of_wide_top_bottom_count=sig["of_wide_top_bottom_count"],
        broadcast_scoreboard_trap=sig["broadcast_scoreboard_trap"],
        broadcast_billboard_trap=sig["broadcast_billboard_trap"],
        broadcast_pattern_trap=sig["broadcast_pattern_trap"],
        broadcast_lower_third_pattern=sig["broadcast_lower_third_pattern"],
        broadcast_scoreboard_pattern=sig["broadcast_scoreboard_pattern"],
        broadcast_billboard_pattern=sig["broadcast_billboard_pattern"],
        freq_hf_ratio_mean=sig["freq_hf_ratio_mean"],
        c2pa_ai=c2pa_sig["c2pa_ai"],
    )

    # Post-fuse overrides odwzorowane 1:1 z evaluate.main().
    c2pa_override = int(c2pa_sig.get("c2pa_ai", 0)) == 1
    if c2pa_override:
        det = 1
        mode = mode + ";c2pa_override=1"

    kling_static_ai = (
        int(sig.get("zv_count", 0)) == 4
        and float(sig.get("iw_best_similarity", 0.0)) >= 0.40
        and int(broadcast_trap) == 0
    )
    if kling_static_ai:
        det = 1
        mode = mode + ";kling_static_ai=1"

    sora_static_override = (
        int(sig.get("of_count", 0)) == 0
        and int(sig.get("zv_count", 0)) == 0
        and float(sig.get("freq_hf_ratio_mean", 1.0)) < 0.38
    )
    if sora_static_override:
        det = 1
        mode = mode + ";sora_static_override=1"

    guard_bypass = c2pa_override or kling_static_ai or sora_static_override
    rescue_guard_override = False
    if not guard_bypass:
        rescue_guard_override = (
            ai_specific == 0
            and det == 1
            and float(score) < float(eval_module.POINTS_THRESHOLD_DEFAULT)
            and int(sig.get("of_count", 0)) >= 1
            and int(sig.get("of_wide_lower_roi_count", 0)) == 0
            and float(sig.get("iw_best_similarity", 0.0)) >= 0.60
            and float(sig.get("freq_hf_ratio_mean", 1.0)) < eval_module.CLEAN_AI_HF_THRESHOLD
            and int(sig.get("broadcast_scoreboard_trap", 0)) == 0
            and int(sig.get("broadcast_billboard_trap", 0)) == 0
            and int(sig.get("broadcast_pattern_trap", 0)) == 0
        )
        if ai_specific == 0 and not rescue_guard_override:
            det = 0
            mode = mode + ";guard_no_ai_specific=1"
        if rescue_guard_override:
            mode = mode + ";guard_rescue_override=1"
        if (
            sig.get("of_lower_third_roi_ratio", 0.0) > eval_module.LOWER_THIRD_HARD_THRESHOLD
            and ai_specific == 0
        ):
            det = 0
            mode = mode + ";guard_lowerthird_without_ai=1"

    if (not c2pa_override) and (
        det == 0
        and float(score) >= float(eval_module.HIGH_SCORE_OVERRIDE_THRESHOLD)
        and int(broadcast_trap) == 1
    ):
        det = 1
        mode = mode + ";high_score_override=1"

    _ = mode  # zachowane dla czytelnosci; CSV wymaga tylko decision + score
    return int(det), float(score)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    ffmpeg = require_binary("ffmpeg")
    _ffprobe = require_binary("ffprobe")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sources = collect_sources()
    if not sources:
        print("Brak plikow .mp4 w zrodlowych splitach.")
        return 0

    sys.path.insert(0, str(DATASET_ROOT))
    import evaluate as ev  # type: ignore

    print("[INIT] Inicjalizacja invisible watermark...")
    ev.initialize_invisible_watermark()

    rows: list[dict[str, Any]] = []
    changed = 0

    total = len(sources)
    for idx, entry in enumerate(sources, 1):
        src = entry.path
        out = OUTPUT_DIR / src.name

        seconds, height, crf = prepare_video(ffmpeg, src, out)
        out_size = file_size_bytes(out)
        out_size_mb = out_size / (1024 * 1024)
        trimmed = "tak" if seconds < 15 or " -t 15 " in " ".join(["-t", str(seconds)]) else "nie"

        print(
            f"[{idx}/{total}] {entry.split}/{src.name} -> "
            f"{seconds}s,{height}p,crf={crf},size={out_size_mb:.2f}MB"
        )

        original_decision, original_score = run_detector(ev, src)
        resized_decision, resized_score = run_detector(ev, out)
        if original_decision != resized_decision:
            changed += 1

        rows.append(
            {
                "filename": src.name,
                "split": entry.split,
                "original_decision": original_decision,
                "original_score": f"{original_score:.4f}",
                "resized_decision": resized_decision,
                "resized_score": f"{resized_score:.4f}",
            }
        )

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "split",
                "original_decision",
                "original_score",
                "resized_decision",
                "resized_score",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Zapisano: {OUTPUT_CSV}")
    print(f"[OK] Pliki wyjsciowe: {OUTPUT_DIR}")
    print(f"[SUMMARY] Zmiana decyzji po resize: {changed}/{len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
