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
from dct_artifact_detector import DCTArtifactDetector  # type: ignore

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
    parser.add_argument("--output", default="dct_scores.csv")
    args = parser.parse_args()

    eval_rows = list(csv.DictReader(Path(args.eval).open("r", encoding="utf-8", newline="")))
    before_det = {(r.get("category", ""), r.get("filename", "")): _to_int(r.get("detected", 0)) for r in eval_rows}

    categories = [
        ("ai_baseline", 1),
        ("adv_compressed", 1),
        ("adv_fp_trap", 0),
    ]

    det = DCTArtifactDetector()
    out_rows: list[dict[str, Any]] = []
    y_true: list[int] = []
    blockiness_scores: list[float] = []
    hf_scores_inv: list[float] = []

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
                    "dct_score": int(res["dct_score"]),
                    "dct_bonus": int(res["dct_bonus"]),
                    "blockiness": float(res["blockiness"]),
                    "hf_suppression": float(res["hf_suppression"]),
                }
            )
            y_true.append(label)
            blockiness_scores.append(float(res["blockiness"]))
            hf_scores_inv.append(-float(res["hf_suppression"]))

    out_path = Path(args.output)
    fields = ["filename", "category", "detected_before", "dct_score", "dct_bonus", "blockiness", "hf_suppression"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    auc_block = roc_auc_score(y_true, blockiness_scores)
    auc_hf = roc_auc_score(y_true, hf_scores_inv)

    compressed_fn = [
        r for r in out_rows
        if r["category"] == "adv_compressed"
        and int(r["detected_before"]) == 0
    ]
    fn_covered = sum(1 for r in compressed_fn if int(r["dct_score"]) >= 1)
    fp_added_risk = sum(
        1 for r in out_rows
        if r["category"] == "adv_fp_trap" and int(r["dct_score"]) >= 1
    )

    print("\n=== DCT ARTIFACT SUMMARY ===")
    print(f"AUC blockiness: {auc_block:.4f}")
    print(f"AUC hf_suppression (inverted): {auc_hf:.4f}")
    print(f"Compressed FN covered (dct_score>=1): {fn_covered}/{len(compressed_fn)}")
    print(f"FP risk in adv_fp_trap (dct_score>=1): {fp_added_risk}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

