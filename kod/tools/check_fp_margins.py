#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ai_style_clip_detector import AIStyleCLIPDetector
from flux_fft_detector import FluxFFTDetector


FLUX_CLIP_HIGH_CONF_THRESHOLD = 0.60


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check safety margins for CLIP probabilities on adv_fp_trap."
    )
    parser.add_argument("--fp-dir", type=Path, default=Path("kod/dataset/adv_fp_trap"))
    parser.add_argument("--ai-dir", type=Path, default=Path("kod/dataset/ai_baseline"))
    parser.add_argument("--out-fp-csv", type=Path, default=Path("fp_trap_clip_margins.csv"))
    parser.add_argument("--out-dist-csv", type=Path, default=Path("clip_prob_distribution.csv"))
    parser.add_argument("--model-path", type=Path, default=Path("clip_classifier.pkl"))
    parser.add_argument("--fft-thresholds", type=Path, default=Path("flux_fft_thresholds.json"))
    return parser.parse_args()


def bucket_name(p: float) -> str:
    if p < 0.3:
        return "0.0-0.3"
    if p < 0.5:
        return "0.3-0.5"
    if p < 0.6:
        return "0.5-0.6"
    return "0.6+"


def main() -> int:
    args = parse_args()
    fp_files = sorted(args.fp_dir.glob("*.mp4"))
    ai_files = sorted(args.ai_dir.glob("*.mp4"))
    if not fp_files:
        print(f"[ERR] Brak plikow w {args.fp_dir}")
        return 1
    if not ai_files:
        print(f"[ERR] Brak plikow w {args.ai_dir}")
        return 1

    clip = AIStyleCLIPDetector(model_path=args.model_path)
    if not getattr(clip, "enabled", False):
        print(f"[ERR] AIStyleCLIPDetector disabled: {getattr(clip, 'load_error', 'unknown')}")
        return 2

    fft = FluxFFTDetector(thresholds_path=args.fft_thresholds)
    detector_threshold = float(getattr(clip, "threshold", 0.5))
    print(f"[INIT] AIStyle threshold={detector_threshold:.3f}")
    print(f"[INIT] Flux-combined high-conf threshold={FLUX_CLIP_HIGH_CONF_THRESHOLD:.2f}")

    args.out_fp_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_dist_csv.parent.mkdir(parents=True, exist_ok=True)

    fp_rows: list[dict[str, str | float | int]] = []
    dist_rows: list[dict[str, str | float]] = []

    for vp in fp_files:
        clip_res = clip.detect_video(vp)
        fft_res = fft.detect_video(vp)

        prob = float(clip_res.get("ai_style_prob", 0.0))
        detected = bool(clip_res.get("ai_style_detected", False))
        fft_score = int(fft_res.get("fft_score", 0))
        flux_combined = int(prob > FLUX_CLIP_HIGH_CONF_THRESHOLD)
        margin = detector_threshold - prob

        fp_rows.append(
            {
                "filename": vp.name,
                "ai_style_prob": round(prob, 6),
                "ai_style_detected": int(detected),
                "fft_score": fft_score,
                "flux_combined": flux_combined,
                "margin_to_threshold": round(margin, 6),
            }
        )
        dist_rows.append(
            {
                "filename": vp.name,
                "category": "adv_fp_trap",
                "ai_style_prob": round(prob, 6),
            }
        )

    for vp in ai_files:
        clip_res = clip.detect_video(vp)
        dist_rows.append(
            {
                "filename": vp.name,
                "category": "ai_baseline",
                "ai_style_prob": round(float(clip_res.get("ai_style_prob", 0.0)), 6),
            }
        )

    with args.out_fp_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "ai_style_prob",
                "ai_style_detected",
                "fft_score",
                "flux_combined",
                "margin_to_threshold",
            ],
        )
        writer.writeheader()
        writer.writerows(fp_rows)

    with args.out_dist_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "category", "ai_style_prob"],
        )
        writer.writeheader()
        writer.writerows(dist_rows)

    probs = [float(r["ai_style_prob"]) for r in fp_rows]
    gt_050 = sum(1 for p in probs if p > 0.50)
    gt_055 = sum(1 for p in probs if p > 0.55)
    high = max(fp_rows, key=lambda r: float(r["ai_style_prob"]))
    histogram = {"0.0-0.3": 0, "0.3-0.5": 0, "0.5-0.6": 0, "0.6+": 0}
    for p in probs:
        histogram[bucket_name(float(p))] += 1

    print(f"[OUT] {args.out_fp_csv.resolve()}")
    print(f"[OUT] {args.out_dist_csv.resolve()}")
    print("\n=== FP Trap Margins Summary ===")
    print(f"adv_fp_trap files: {len(fp_rows)}")
    print(f"prob > 0.50: {gt_050}")
    print(f"prob > 0.55: {gt_055}")
    print(
        "highest prob (most dangerous FP): "
        f"{high['filename']} prob={float(high['ai_style_prob']):.4f}"
    )
    print(
        "histogram: "
        f"0.0-0.3={histogram['0.0-0.3']} "
        f"0.3-0.5={histogram['0.3-0.5']} "
        f"0.5-0.6={histogram['0.5-0.6']} "
        f"0.6+={histogram['0.6+']}"
    )

    warnings = [r for r in fp_rows if float(r["ai_style_prob"]) > detector_threshold]
    if warnings:
        for row in warnings:
            print(
                f"WARNING: FP candidate: {row['filename']} "
                f"prob={float(row['ai_style_prob']):.2f}"
            )
    else:
        print("No FP candidates above current AIStyleCLIP threshold.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
