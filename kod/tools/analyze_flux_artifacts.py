#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux_fft_detector import compute_video_flux_fft_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze FFT/gradient artifacts on AI vs authentic videos."
    )
    parser.add_argument("--ai-dir", type=Path, default=Path("kod/dataset/ai_baseline"))
    parser.add_argument("--auth-dir", type=Path, default=Path("kod/dataset/adv_fp_trap"))
    parser.add_argument("--frames-per-video", type=int, default=8)
    parser.add_argument("--out-csv", type=Path, default=Path("flux_artifact_scores.csv"))
    parser.add_argument("--out-thresholds", type=Path, default=Path("flux_fft_thresholds.json"))
    parser.add_argument("--out-report", type=Path, default=Path("flux_artifact_report.txt"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ai_files = sorted(args.ai_dir.glob("*.mp4"))
    auth_files = sorted(args.auth_dir.glob("*.mp4"))
    if not ai_files or not auth_files:
        print(f"[ERR] Brak plikow: ai={len(ai_files)} auth={len(auth_files)}")
        return 1

    rows: list[dict[str, Any]] = []
    total = len(ai_files) + len(auth_files)
    idx = 0
    for p in ai_files:
        idx += 1
        print(f"[AI ] ({idx}/{total}) {p.name}")
        m = compute_video_flux_fft_metrics(p, n_frames=args.frames_per_video)
        rows.append(
            {
                "filename": p.name,
                "split": "ai_baseline",
                "label": 1,
                **m,
            }
        )
    for p in auth_files:
        idx += 1
        print(f"[AUT] ({idx}/{total}) {p.name}")
        m = compute_video_flux_fft_metrics(p, n_frames=args.frames_per_video)
        rows.append(
            {
                "filename": p.name,
                "split": "adv_fp_trap",
                "label": 0,
                **m,
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "split",
                "label",
                "oversmoothing_ratio",
                "bimodality_coeff",
                "hf_noise_ratio",
                "ssim_variance",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {args.out_csv.resolve()}")

    y = np.array([int(r["label"]) for r in rows], dtype=np.int32)
    metric_names = [
        "oversmoothing_ratio",
        "bimodality_coeff",
        "hf_noise_ratio",
        "ssim_variance",
    ]

    metrics_cfg: dict[str, dict[str, Any]] = {}
    report_lines = ["AI-style FFT artifact analysis", ""]
    report_lines.append(f"AI videos: {len(ai_files)}")
    report_lines.append(f"Authentic videos: {len(auth_files)}")
    report_lines.append("")

    for name in metric_names:
        vals = np.array([float(r[name]) for r in rows], dtype=np.float32)
        vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
        ai_vals = vals[y == 1]
        au_vals = vals[y == 0]

        try:
            _u, p_val = mannwhitneyu(ai_vals, au_vals, alternative="two-sided")
            p_val_f = float(p_val)
        except Exception:
            p_val_f = 1.0

        # AUC w obu orientacjach i wybór lepszej.
        auc_high = float(roc_auc_score(y, vals))
        auc_low = float(roc_auc_score(y, -vals))
        if auc_low > auc_high:
            direction = "le"  # niższa wartość => bardziej AI
            auc = auc_low
        else:
            direction = "ge"  # wyższa wartość => bardziej AI
            auc = auc_high

        ai_p25 = float(np.percentile(ai_vals, 25))
        enabled = bool(auc > 0.65)
        metrics_cfg[name] = {
            "threshold": ai_p25,
            "direction": direction,
            "enabled": enabled,
            "auc": float(round(auc, 4)),
            "p_value": float(round(p_val_f, 6)),
        }

        report_lines.append(
            f"{name}: auc={auc:.4f} p={p_val_f:.6f} "
            f"direction={direction} threshold(p25_ai)={ai_p25:.6f} enabled={enabled}"
        )

    payload = {
        "min_active": 2,
        "metrics": metrics_cfg,
        "note": "threshold = percentile 25 of AI distribution per metric",
    }
    args.out_thresholds.parent.mkdir(parents=True, exist_ok=True)
    with args.out_thresholds.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[OUT] {args.out_thresholds.resolve()}")

    auc_good = [k for k, v in metrics_cfg.items() if float(v["auc"]) > 0.65]
    report_lines.append("")
    report_lines.append("Metrics with AUC > 0.65:")
    for m in auc_good:
        report_lines.append(f"- {m}")
    if not auc_good:
        report_lines.append("- none")

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"[OUT] {args.out_report.resolve()}")

    print("\n=== Metric summary (AUC / p-value) ===")
    for name in metric_names:
        cfg = metrics_cfg[name]
        print(
            f"{name:22s} auc={cfg['auc']:.4f} p={cfg['p_value']:.6f} "
            f"dir={cfg['direction']} thr={cfg['threshold']:.6f} enabled={cfg['enabled']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
