#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Any


def _parse_mode(mode: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not isinstance(mode, str):
        return out
    for token in mode.split(";"):
        token = token.strip()
        if not token:
            continue
        if "=" in token:
            k, v = token.split("=", 1)
            out[k.strip()] = v.strip()
        else:
            out[token] = 1
    return out


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(float(value))
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _classify_failure(row: dict[str, Any]) -> str:
    score = _as_float(row.get("score"), 0.0)
    ai_specific = _as_int(row.get("ai_specific"), 0)
    lower_third_ok = _as_int(row.get("lower_third_ok"), 0)
    soft_threshold = _as_int(row.get("soft_threshold"), 0)
    of_count = _as_int(row.get("optical_flow_contours"), 0)
    c2pa_ai = _as_int(row.get("c2pa_ai"), 0)
    ai_style_prob = _as_float(row.get("ai_style_prob"), 0.0)

    if soft_threshold == 1:
        return "SOFT_THRESHOLD"
    if ai_specific == 0 and score >= 5.0:
        return "HIGH_SCORE"
    if lower_third_ok == 1 and ai_specific == 1 and 3.0 <= score <= 4.0:
        return "LOWER_THIRD"
    if of_count > 20 and ai_specific == 0 and c2pa_ai == 0 and ai_style_prob < 0.55:
        return "OPTICAL_FLOW"
    return "OTHER"


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(Path("kod/results/latest/evaluation_results.csv")),
        help="Path to evaluation_results.csv",
    )
    parser.add_argument(
        "--output",
        default="fp_trap_fp_analysis.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output)
    if not in_path.exists():
        raise FileNotFoundError(f"Missing input file: {in_path}")

    src_rows = _read_rows(in_path)
    rows: list[dict[str, Any]] = []
    for r in src_rows:
        if r.get("category") != "adv_fp_trap":
            continue
        if _as_int(r.get("ground_truth", 0)) != 0:
            continue
        if _as_int(r.get("detected", 0)) != 1:
            continue

        mode = _parse_mode(str(r.get("fusion_mode", "")))
        row = {
            "filename": str(r.get("filename", "")),
            "score": _as_float(r.get("fusion_score", 0.0)),
            "ai_specific": _as_int(r.get("ai_specific", 0)),
            "lower_third_ok": _as_int(mode.get("lower_third_ok", 1 - _as_int(r.get("broadcast_trap", 0)))),
            "c2pa_ai": _as_int(r.get("c2pa_ai", 0)),
            "ai_style_prob": _as_float(r.get("ai_style_prob", r.get("flux_clip_prob", 0.0)), 0.0),
            "fft_bonus": _as_int(r.get("fft_bonus", 0)),
            "optical_flow_contours": _as_int(r.get("of_count", 0)),
            "guard_no_ai_specific": _as_int(mode.get("guard_no_ai_specific", 0)),
            "guard_lowerthird_without_ai": _as_int(mode.get("guard_lowerthird_without_ai", 0)),
            "sora_static_override": _as_int(mode.get("sora_static_override", 0)),
            "c2pa_override": _as_int(mode.get("c2pa_override", 0)),
            "soft_threshold": _as_int(mode.get("soft_threshold", 0)),
        }
        row["failure_group"] = _classify_failure(row)
        rows.append(row)

    fieldnames = [
        "filename",
        "score",
        "ai_specific",
        "lower_third_ok",
        "c2pa_ai",
        "ai_style_prob",
        "fft_bonus",
        "optical_flow_contours",
        "guard_no_ai_specific",
        "guard_lowerthird_without_ai",
        "sora_static_override",
        "c2pa_override",
        "soft_threshold",
        "failure_group",
    ]
    _write_rows(out_path, fieldnames, rows)

    print(f"FP rows: {len(rows)}")
    counts = Counter(r["failure_group"] for r in rows)
    if counts:
        print("\nFP by group:")
        for key in sorted(counts):
            print(f"- {key}: {counts[key]}")
    print(f"\nSaved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

