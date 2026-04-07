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


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _video_id(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    return name.split("_", 1)[0].strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="kod/results/latest/evaluation_results.csv")
    parser.add_argument("--raw", default="kod/results/latest/raw_signals.csv")
    parser.add_argument("--output", default="compressed_fn_analysis.csv")
    args = parser.parse_args()

    eval_rows = _load_csv(Path(args.eval))
    raw_rows = _load_csv(Path(args.raw))

    eval_by_key = {(r.get("category", ""), r.get("filename", "")): r for r in eval_rows}
    raw_by_key = {(r.get("category", ""), r.get("filename", "")): r for r in raw_rows}
    ai_baseline_by_name = {r.get("filename", ""): r for r in eval_rows if r.get("category") == "ai_baseline"}
    ai_baseline_by_id = {
        _video_id(r.get("filename", "")): r
        for r in eval_rows
        if r.get("category") == "ai_baseline"
    }

    comp_fn = [
        r for r in eval_rows
        if r.get("category") == "adv_compressed"
        and _to_int(r.get("ground_truth", 0)) == 1
        and _to_int(r.get("detected", 0)) == 0
    ]

    out_rows: list[dict[str, Any]] = []
    for r in comp_fn:
        fn = r.get("filename", "")
        mode = str(r.get("fusion_mode", ""))
        md = _parse_mode(mode)
        raw = raw_by_key.get(("adv_compressed", fn), {})

        base = ai_baseline_by_name.get(fn, {})
        match_type = "exact"
        if not base:
            base = ai_baseline_by_id.get(_video_id(fn), {})
            match_type = "video_id" if base else "none"
        base_mode = str(base.get("fusion_mode", ""))

        row = {
            "filename": fn,
            "score": _to_float(r.get("fusion_score", 0.0)),
            "ai_specific": _to_int(r.get("ai_specific", 0)),
            "lower_third_ok": _to_int(md.get("lower_third_ok", 1 - _to_int(r.get("broadcast_trap", 0)))),
            "c2pa_ai": _to_int(r.get("c2pa_ai", raw.get("c2pa_ai", 0))),
            "ai_style_prob": _to_float(r.get("ai_style_prob", r.get("flux_clip_prob", 0.0))),
            "fft_bonus": _to_int(r.get("fft_bonus", 0)),
            "optical_flow_contours": _to_int(r.get("of_count", raw.get("of_count", 0))),
            "guard_no_ai_specific": _to_int(md.get("guard_no_ai_specific", 0)),
            "guard_lowerthird_without_ai": _to_int(md.get("guard_lowerthird_without_ai", 0)),
            "guard_broadcast_score_cap": _to_int(md.get("guard_broadcast_score_cap", 0)),
            "guard_of_penalty": _to_int(md.get("guard_of_penalty", 0)),
            "soft_threshold": _to_int(md.get("soft_threshold", 0)),
            "soft_threshold_min": _to_int(md.get("soft_threshold_min", 3)),
            "sora_static_override": _to_int(md.get("sora_static_override", 0)),
            "broadcast_pattern_trap": _to_int(r.get("broadcast_pattern_trap", raw.get("broadcast_pattern_trap", 0))),
            "mode": mode,
            "baseline_filename": base.get("filename", ""),
            "baseline_detected": _to_int(base.get("detected", 0)),
            "baseline_score": _to_float(base.get("fusion_score", 0.0)) if base else "",
            "baseline_ai_style_prob": _to_float(base.get("ai_style_prob", base.get("flux_clip_prob", 0.0))) if base else "",
            "baseline_mode": base_mode,
            "baseline_match_type": match_type,
        }
        if base:
            row["score_drop"] = _to_float(row["baseline_score"]) - _to_float(row["score"])
            row["clip_drop"] = _to_float(row["baseline_ai_style_prob"]) - _to_float(row["ai_style_prob"])
        else:
            row["score_drop"] = ""
            row["clip_drop"] = ""
        out_rows.append(row)

    fields = [
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
        "guard_broadcast_score_cap",
        "guard_of_penalty",
        "soft_threshold",
        "soft_threshold_min",
        "sora_static_override",
        "broadcast_pattern_trap",
        "mode",
        "baseline_filename",
        "baseline_detected",
        "baseline_score",
        "baseline_ai_style_prob",
        "baseline_mode",
        "baseline_match_type",
        "score_drop",
        "clip_drop",
    ]
    with Path(args.output).open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"Compressed FN rows: {len(out_rows)}")
    for row in out_rows:
        if row["baseline_match_type"] == "none":
            print(
                f"- {row['filename']}: score={row['score']:.1f}, ai_prob={row['ai_style_prob']:.3f}, "
                "no baseline pair"
            )
        else:
            print(
                f"- {row['filename']}: score={row['score']:.1f}, ai_prob={row['ai_style_prob']:.3f}, "
                f"score_drop={float(row['score_drop']):.1f}, clip_drop={float(row['clip_drop']):.3f}"
            )
    print(f"Saved: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
