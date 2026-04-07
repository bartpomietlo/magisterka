#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from flux_fft_detector import FluxFFTDetector


METRICS = [
    "oversmoothing_ratio",
    "bimodality_coeff",
    "hf_noise_ratio",
    "ssim_variance",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug FFT detector metrics and score activity."
    )
    parser.add_argument("--ai-dir", type=Path, default=Path("kod/dataset/ai_baseline"))
    parser.add_argument("--auth-dir", type=Path, default=Path("kod/dataset/adv_fp_trap"))
    parser.add_argument("--thresholds", type=Path, default=Path("flux_fft_thresholds.json"))
    parser.add_argument("--out-csv", type=Path, default=Path("fft_debug_scores.csv"))
    return parser.parse_args()


def triggered(value: float, threshold: float, direction: str) -> bool:
    if direction == "le":
        return value <= threshold
    return value >= threshold


def diag_label(auc_best: float, all_triggered: int) -> str:
    if auc_best < 0.55:
        return "DEAD: AUC < 0.55 — metryka losowa, rozważ usunięcie"
    if auc_best <= 0.65:
        return "WEAK: AUC 0.55-0.65 — słaba ale nie martwa"
    if all_triggered == 0:
        return "THRESHOLD_BUG: metryka OK ale żaden film nie przekracza progu — próg za wysoki"
    return "OK: AUC > 0.65 — metryka działa, sprawdź próg"


def safe_auc(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    try:
        auc_hi = float(roc_auc_score(y, x))
    except Exception:
        auc_hi = 0.5
    try:
        auc_lo = float(roc_auc_score(y, -x))
    except Exception:
        auc_lo = 0.5
    return auc_hi, auc_lo


def main() -> int:
    args = parse_args()
    ai_files = sorted(args.ai_dir.glob("*.mp4"))
    auth_files = sorted(args.auth_dir.glob("*.mp4"))
    if not ai_files:
        print(f"[ERR] Brak plikow w {args.ai_dir}")
        return 1
    if not auth_files:
        print(f"[ERR] Brak plikow w {args.auth_dir}")
        return 1

    detector = FluxFFTDetector(thresholds_path=args.thresholds)
    rows: list[dict[str, str | int | float]] = []
    total = len(ai_files) + len(auth_files)
    idx = 0

    for p in ai_files:
        idx += 1
        print(f"[AI ] ({idx}/{total}) {p.name}")
        res = detector.detect_video(p)
        metrics = res.get("metrics", {})
        rows.append(
            {
                "filename": p.name,
                "category": "ai_baseline",
                "oversmoothing_ratio": float(metrics.get("oversmoothing_ratio", 0.0)),
                "bimodality_coeff": float(metrics.get("bimodality_coeff", 0.0)),
                "hf_noise_ratio": float(metrics.get("hf_noise_ratio", 0.0)),
                "ssim_variance": float(metrics.get("ssim_variance", 0.0)),
                "fft_score": int(res.get("fft_score", 0)),
                "fft_detected": int(int(res.get("fft_bonus", 0)) > 0),
            }
        )

    for p in auth_files:
        idx += 1
        print(f"[AUT] ({idx}/{total}) {p.name}")
        res = detector.detect_video(p)
        metrics = res.get("metrics", {})
        rows.append(
            {
                "filename": p.name,
                "category": "adv_fp_trap",
                "oversmoothing_ratio": float(metrics.get("oversmoothing_ratio", 0.0)),
                "bimodality_coeff": float(metrics.get("bimodality_coeff", 0.0)),
                "hf_noise_ratio": float(metrics.get("hf_noise_ratio", 0.0)),
                "ssim_variance": float(metrics.get("ssim_variance", 0.0)),
                "fft_score": int(res.get("fft_score", 0)),
                "fft_detected": int(int(res.get("fft_bonus", 0)) > 0),
            }
        )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "filename",
                "category",
                "oversmoothing_ratio",
                "bimodality_coeff",
                "hf_noise_ratio",
                "ssim_variance",
                "fft_score",
                "fft_detected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OUT] {args.out_csv.resolve()}")

    y = np.array([1 if r["category"] == "ai_baseline" else 0 for r in rows], dtype=np.int32)
    n_ok = 0
    n_dead_or_weak = 0

    print("\n=== FFT Metric Diagnostics ===")
    for metric in METRICS:
        vals = np.array([float(r[metric]) for r in rows], dtype=np.float32)
        ai_vals = vals[y == 1]
        au_vals = vals[y == 0]

        cfg = detector.metrics_cfg.get(metric, {})
        threshold = float(cfg.get("threshold", 0.0))
        direction = str(cfg.get("direction", "ge"))
        ai_tr = sum(1 for v in ai_vals if triggered(float(v), threshold, direction))
        au_tr = sum(1 for v in au_vals if triggered(float(v), threshold, direction))
        all_tr = ai_tr + au_tr

        auc_hi, auc_lo = safe_auc(y, vals)
        auc_best = max(auc_hi, auc_lo)
        label = diag_label(auc_best, all_tr)

        if label.startswith("OK"):
            n_ok += 1
        if label.startswith("DEAD") or label.startswith("WEAK"):
            n_dead_or_weak += 1

        print(f"\n[{metric}]")
        print(f"AI mean/std: {float(np.mean(ai_vals)):.6f} / {float(np.std(ai_vals)):.6f}")
        print(f"AUTH mean/std: {float(np.mean(au_vals)):.6f} / {float(np.std(au_vals)):.6f}")
        print(f"threshold: {threshold:.6f} (direction={direction}, enabled={bool(cfg.get('enabled', True))})")
        print(f"triggered AI: {ai_tr}/{len(ai_vals)} | triggered AUTH: {au_tr}/{len(au_vals)}")
        print(f"AUC(high): {auc_hi:.4f} | AUC(low): {auc_lo:.4f} | AUC(best): {auc_best:.4f}")
        print(f"Diagnosis: {label}")

    print("\n=== Recommendation ===")
    if n_dead_or_weak == len(METRICS):
        print(
            "Wszystkie metryki FFT są DEAD/WEAK. "
            "Rekomendacja: wyłączyć FFT jako osobny gate i zostawić tylko jako bonus punktów "
            "(np. score+=1, bez gatingu)."
        )
    elif n_ok >= 2:
        print(
            "Co najmniej 2 metryki są OK. "
            "Rekomendacja: wykonać tuning progów i kierunku aktywacji dla FFT."
        )
    else:
        print(
            "FFT jest częściowo użyteczne, ale niestabilne. "
            "Rekomendacja: ostrożny tuning progów i ponowna walidacja AUC."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
