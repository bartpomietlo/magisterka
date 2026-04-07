#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent  # .../magisterka/kod
PROJECT_ROOT = ROOT.parent                     # .../magisterka
DATASET_DIR = ROOT / "dataset"

sys.path.insert(0, str(ROOT))
from ai_style_clip_detector import AIStyleCLIPDetector  # type: ignore
from flux_fft_detector import FluxFFTDetector  # type: ignore

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


def _video_id(name: str) -> str:
    if not isinstance(name, str) or not name:
        return ""
    return name.split("_", 1)[0].strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="kod/results/latest/evaluation_results.csv")
    parser.add_argument("--raw", default="kod/results/latest/raw_signals.csv")
    parser.add_argument("--output", default="compression_signal_degradation.csv")
    args = parser.parse_args()

    eval_rows = _read_csv(Path(args.eval))
    raw_rows = _read_csv(Path(args.raw))
    eval_by_key = {(r.get("category", ""), r.get("filename", "")): r for r in eval_rows}
    raw_by_key = {(r.get("category", ""), r.get("filename", "")): r for r in raw_rows}
    base_by_name = {r.get("filename", ""): r for r in eval_rows if r.get("category") == "ai_baseline"}
    base_by_id = {
        _video_id(r.get("filename", "")): r
        for r in eval_rows
        if r.get("category") == "ai_baseline"
    }
    base_raw_by_name = {r.get("filename", ""): r for r in raw_rows if r.get("category") == "ai_baseline"}
    base_raw_by_id = {
        _video_id(r.get("filename", "")): r
        for r in raw_rows
        if r.get("category") == "ai_baseline"
    }

    clip_detector = AIStyleCLIPDetector(model_path=PROJECT_ROOT / "clip_classifier.pkl")
    fft_detector = FluxFFTDetector(thresholds_path=PROJECT_ROOT / "flux_fft_thresholds.json")

    comp_files = sorted((DATASET_DIR / "adv_compressed").glob("*.mp4"))
    out_rows: list[dict[str, Any]] = []

    for idx, vp in enumerate(comp_files, 1):
        print(f"[{idx}/{len(comp_files)}] {vp.name}")
        clip_res = clip_detector.detect_video(vp)
        fft_res = fft_detector.detect_video(vp)

        comp_eval = eval_by_key.get(("adv_compressed", vp.name), {})
        comp_raw = raw_by_key.get(("adv_compressed", vp.name), {})

        base_eval = base_by_name.get(vp.name, {})
        base_raw = base_raw_by_name.get(vp.name, {})
        match_type = "exact"
        if not base_eval:
            vid = _video_id(vp.name)
            base_eval = base_by_id.get(vid, {})
            base_raw = base_raw_by_id.get(vid, {})
            match_type = "video_id" if base_eval else "none"

        comp_prob = _to_float(clip_res.get("ai_style_prob", clip_res.get("clip_ai_prob", 0.0)))
        comp_fft_bonus = _to_int(fft_res.get("fft_bonus", 0))
        comp_fft_score = _to_int(fft_res.get("fft_score", 0))
        comp_score = _to_float(comp_eval.get("fusion_score", 0.0))
        comp_detected = _to_int(comp_eval.get("detected", 0))
        comp_of = _to_int(comp_raw.get("of_count", comp_eval.get("of_count", 0)))
        comp_ai_specific = _to_int(comp_eval.get("ai_specific", 0))
        comp_c2pa_ai = _to_int(comp_eval.get("c2pa_ai", comp_raw.get("c2pa_ai", 0)))

        if base_eval:
            orig_prob = _to_float(base_eval.get("ai_style_prob", base_eval.get("flux_clip_prob", 0.0)))
            orig_fft_bonus = _to_int(base_eval.get("fft_bonus", 0))
            orig_score = _to_float(base_eval.get("fusion_score", 0.0))
            orig_detected = _to_int(base_eval.get("detected", 0))
            orig_of = _to_int(base_raw.get("of_count", base_eval.get("of_count", 0)))
        else:
            orig_prob = ""
            orig_fft_bonus = ""
            orig_score = ""
            orig_detected = ""
            orig_of = ""

        row: dict[str, Any] = {
            "filename": vp.name,
            "match_type": match_type,
            "orig_ai_style_prob": orig_prob,
            "comp_ai_style_prob": comp_prob,
            "clip_drop": (_to_float(orig_prob) - comp_prob) if base_eval else "",
            "orig_fft_bonus": orig_fft_bonus,
            "comp_fft_bonus": comp_fft_bonus,
            "fft_drop": (_to_int(orig_fft_bonus) - comp_fft_bonus) if base_eval else "",
            "orig_score": orig_score,
            "comp_score": comp_score,
            "score_drop": (_to_float(orig_score) - comp_score) if base_eval else "",
            "detected_orig": orig_detected,
            "detected_comp": comp_detected,
            "orig_optical_flow_contours": orig_of,
            "comp_optical_flow_contours": comp_of,
            "of_drop": (_to_int(orig_of) - comp_of) if base_eval else "",
            "comp_ai_specific": comp_ai_specific,
            "comp_c2pa_ai": comp_c2pa_ai,
            "comp_fft_score_live": comp_fft_score,
            "comp_tc_placeholder": "",
        }
        out_rows.append(row)

    fields = [
        "filename",
        "match_type",
        "orig_ai_style_prob",
        "comp_ai_style_prob",
        "clip_drop",
        "orig_fft_bonus",
        "comp_fft_bonus",
        "fft_drop",
        "orig_score",
        "comp_score",
        "score_drop",
        "detected_orig",
        "detected_comp",
        "orig_optical_flow_contours",
        "comp_optical_flow_contours",
        "of_drop",
        "comp_ai_specific",
        "comp_c2pa_ai",
        "comp_fft_score_live",
        "comp_tc_placeholder",
    ]
    out_path = Path(args.output)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    # Summary
    fn_rows = [r for r in out_rows if _to_int(r.get("detected_comp", 0)) == 0]
    tp_rows = [r for r in out_rows if _to_int(r.get("detected_comp", 0)) == 1]
    fn_clip = [_to_float(r.get("clip_drop", np.nan), np.nan) for r in fn_rows if str(r.get("clip_drop", "")) != ""]
    tp_clip = [_to_float(r.get("clip_drop", np.nan), np.nan) for r in tp_rows if str(r.get("clip_drop", "")) != ""]
    fn_clip = [x for x in fn_clip if np.isfinite(x)]
    tp_clip = [x for x in tp_clip if np.isfinite(x)]

    fn_prob_hi = sum(1 for r in fn_rows if _to_float(r.get("comp_ai_style_prob", 0.0)) > 0.55)
    fn_prob_lo = sum(1 for r in fn_rows if _to_float(r.get("comp_ai_style_prob", 0.0)) < 0.45)

    def _avg(rows: list[dict[str, Any]], key: str) -> float:
        vals = [_to_float(r.get(key, np.nan), np.nan) for r in rows if str(r.get(key, "")) != ""]
        vals = [v for v in vals if np.isfinite(v)]
        if not vals:
            return 0.0
        return float(statistics.mean(vals))

    avg_clip_drop_fn = float(statistics.mean(fn_clip)) if fn_clip else 0.0
    avg_clip_drop_tp = float(statistics.mean(tp_clip)) if tp_clip else 0.0
    avg_fft_drop = _avg(out_rows, "fft_drop")
    avg_score_drop = _avg(out_rows, "score_drop")
    avg_of_drop = _avg(out_rows, "of_drop")

    print("\n=== SUMMARY ===")
    print(f"Total adv_compressed files: {len(out_rows)}")
    print(f"FN count: {len(fn_rows)}  TP count: {len(tp_rows)}")
    print(f"Mean clip_drop FN: {avg_clip_drop_fn:.4f}")
    print(f"Mean clip_drop TP: {avg_clip_drop_tp:.4f}")
    print(f"FN with comp_ai_style_prob > 0.55: {fn_prob_hi}")
    print(f"FN with comp_ai_style_prob < 0.45: {fn_prob_lo}")
    print(f"Mean fft_drop:   {avg_fft_drop:.4f}")
    print(f"Mean score_drop: {avg_score_drop:.4f}")
    print(f"Mean of_drop:    {avg_of_drop:.4f}")

    abs_means = {
        "CLIP": abs(avg_clip_drop_fn),
        "FFT": abs(avg_fft_drop),
        "SCORE": abs(avg_score_drop),
        "OPTICAL_FLOW": abs(avg_of_drop),
    }
    dominant = sorted(abs_means.items(), key=lambda x: x[1], reverse=True)[0][0]
    print(f"Most degraded signal (magnitude): {dominant}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()
