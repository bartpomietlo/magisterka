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

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _to_int(v: Any, default: int = 0) -> int:
    try:
        return int(float(v))
    except Exception:
        return default


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _parse_mode(mode: str) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(mode, str):
        return out
    for tok in mode.split(";"):
        tok = tok.strip()
        if not tok:
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[tok] = "1"
    return out


def _load_rows(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    out: dict[tuple[str, str], dict[str, str]] = {}
    for r in rows:
        out[(r.get("category", ""), r.get("filename", ""))] = r
    return out


def _ai_prob(row: dict[str, str]) -> float:
    return _to_float(row.get("ai_style_prob", row.get("flux_clip_prob", 0.0)))


def _cause(before: dict[str, str], after: dict[str, str]) -> str:
    before_mode = str(before.get("fusion_mode", ""))
    after_mode = str(after.get("fusion_mode", ""))
    bm = _parse_mode(before_mode)
    am = _parse_mode(after_mode)
    before_score = _to_float(before.get("fusion_score", 0.0))
    after_score = _to_float(after.get("fusion_score", 0.0))
    after_prob = _ai_prob(after)

    if "soft_threshold=1" in before_mode and am.get("soft_threshold_min") == "4" and "soft_threshold=1" not in after_mode:
        return "Fix C (soft_threshold_tighten)"
    if am.get("guard_of_penalty") == "1":
        return "Fix B (optical_flow_penalty)"
    if (
        bm.get("lower_third_effective", bm.get("lower_third_ok", "1")) == "1"
        and am.get("lower_third_effective", am.get("lower_third_ok", "1")) == "0"
        and 0.45 <= after_prob <= 0.55
    ):
        return "Fix D (lower_third_quality_gate)"
    if (
        _to_int(after.get("broadcast_pattern_trap", 0)) == 1
        and before_score > after_score
        and after_score <= 3.0
    ):
        return "Fix A (broadcast_score_cap)"
    return "Unknown/mixed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--before-eval",
        default="kod/results/2026-03-31_3691f7f_after_clip_fft_tuned/evaluation_results.csv",
        help="Evaluation CSV before FP fixes (FFT-refactor baseline).",
    )
    parser.add_argument(
        "--after-eval",
        default="kod/results/latest/evaluation_results.csv",
        help="Evaluation CSV after fixes.",
    )
    parser.add_argument(
        "--baseline-report",
        default="fft_refactor_regression.txt",
        help="Baseline regression text report (for context only).",
    )
    parser.add_argument(
        "--after-metrics",
        default="kod/results/latest/metrics_summary.csv",
        help="Metrics summary after fixes (for context only).",
    )
    parser.add_argument(
        "--output",
        default="lost_tp_analysis.csv",
        help="Output CSV.",
    )
    args = parser.parse_args()

    before_eval = Path(args.before_eval)
    after_eval = Path(args.after_eval)
    out_path = Path(args.output)
    if not before_eval.exists():
        raise FileNotFoundError(f"Missing before-eval: {before_eval}")
    if not after_eval.exists():
        raise FileNotFoundError(f"Missing after-eval: {after_eval}")

    before = _load_rows(before_eval)
    after = _load_rows(after_eval)

    lost_rows: list[dict[str, Any]] = []
    for key, b in before.items():
        gt = _to_int(b.get("ground_truth", 0))
        b_det = _to_int(b.get("detected", 0))
        if gt != 1 or b_det != 1:
            continue
        a = after.get(key)
        if a is None:
            continue
        a_det = _to_int(a.get("detected", 0))
        if a_det != 0:
            continue

        lost_rows.append(
            {
                "filename": key[1],
                "category": key[0],
                "before_score": _to_float(b.get("fusion_score", 0.0)),
                "after_score": _to_float(a.get("fusion_score", 0.0)),
                "before_ai_style_prob": _ai_prob(b),
                "after_ai_style_prob": _ai_prob(a),
                "before_mode": str(b.get("fusion_mode", "")),
                "after_mode": str(a.get("fusion_mode", "")),
                "broadcast_pattern_trap_after": _to_int(a.get("broadcast_pattern_trap", 0)),
                "which_fix_caused_loss": _cause(b, a),
            }
        )

    fields = [
        "filename",
        "category",
        "before_score",
        "after_score",
        "before_ai_style_prob",
        "after_ai_style_prob",
        "before_mode",
        "after_mode",
        "broadcast_pattern_trap_after",
        "which_fix_caused_loss",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in lost_rows:
            w.writerow(row)

    print(f"Lost TP rows: {len(lost_rows)}")
    for row in lost_rows:
        print(
            f"- {row['category']}/{row['filename']}: "
            f"{row['before_score']} -> {row['after_score']} "
            f"(prob {row['before_ai_style_prob']:.3f}->{row['after_ai_style_prob']:.3f}) "
            f"=> {row['which_fix_caused_loss']}"
        )
    print(f"Saved: {out_path.resolve()}")

    baseline_text = Path(args.baseline_report)
    after_metrics = Path(args.after_metrics)
    if baseline_text.exists():
        print(f"Context baseline report: {baseline_text.resolve()}")
    if after_metrics.exists():
        print(f"Context after metrics: {after_metrics.resolve()}")


if __name__ == "__main__":
    main()
