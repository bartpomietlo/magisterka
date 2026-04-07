#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"

sys.path.insert(0, str(ROOT))
from temporal_consistency_detector import TemporalConsistencyDetector  # type: ignore

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="kod/results/latest/evaluation_results.csv")
    parser.add_argument("--output", default="temporal_consistency_scores.csv")
    args = parser.parse_args()

    eval_rows = list(csv.DictReader(Path(args.eval).open("r", encoding="utf-8", newline="")))
    before_det = {(r.get("category", ""), r.get("filename", "")): _to_int(r.get("detected", 0)) for r in eval_rows}

    categories = [
        ("ai_baseline", 1),
        ("adv_compressed", 1),
        ("adv_fp_trap", 0),
    ]

    det = TemporalConsistencyDetector()
    out_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    score_frame_diff_inv: list[float] = []
    score_of_smooth: list[float] = []
    score_lum_inv: list[float] = []

    for cat, label in categories:
        files = sorted((DATASET / cat).glob("*.mp4"))
        for idx, vp in enumerate(files, 1):
            print(f"[{cat}] {idx}/{len(files)} {vp.name}")
            res = det.detect_video(vp)
            out_rows.append(
                {
                    "filename": vp.name,
                    "category": cat,
                    "detected_before": before_det.get((cat, vp.name), 0),
                    "tc_score": int(res["tc_score"]),
                    "tc_detected": int(bool(res["tc_detected"])),
                    "frame_diff_variance": float(res["frame_diff_variance"]),
                    "of_smoothness": float(res["of_smoothness"]),
                    "luminance_temporal_std": float(res["luminance_temporal_std"]),
                }
            )
            y_true.append(label)
            score_frame_diff_inv.append(-float(res["frame_diff_variance"]))
            score_of_smooth.append(float(res["of_smoothness"]))
            score_lum_inv.append(-float(res["luminance_temporal_std"]))

    out_path = Path(args.output)
    fields = [
        "filename",
        "category",
        "detected_before",
        "tc_score",
        "tc_detected",
        "frame_diff_variance",
        "of_smoothness",
        "luminance_temporal_std",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    auc_frame = roc_auc_score(y_true, score_frame_diff_inv)
    auc_of = roc_auc_score(y_true, score_of_smooth)
    auc_lum = roc_auc_score(y_true, score_lum_inv)

    compressed_fn = [
        r for r in out_rows
        if r["category"] == "adv_compressed"
        and int(r["detected_before"]) == 0
    ]
    fn_covered = sum(1 for r in compressed_fn if int(r["tc_detected"]) == 1)
    fp_added_risk = sum(
        1 for r in out_rows
        if r["category"] == "adv_fp_trap" and int(r["tc_detected"]) == 1
    )

    print("\n=== TEMPORAL CONSISTENCY SUMMARY ===")
    print(f"AUC frame_diff_variance (inverted): {auc_frame:.4f}")
    print(f"AUC of_smoothness: {auc_of:.4f}")
    print(f"AUC luminance_temporal_std (inverted): {auc_lum:.4f}")
    print(f"Compressed FN covered by tc_detected: {fn_covered}/{len(compressed_fn)}")
    print(f"FP risk in adv_fp_trap (tc_detected=1): {fp_added_risk}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

