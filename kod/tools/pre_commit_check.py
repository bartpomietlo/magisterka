#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import importlib
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    kod_dir = repo_root / "kod"
    tools_dir = kod_dir / "tools"
    dataset_dir = kod_dir / "dataset" / "ai_baseline"

    sys.path.insert(0, str(kod_dir))
    sys.path.insert(0, str(tools_dir))

    errors: list[str] = []
    modules_to_import = [
        "ai_style_clip_detector",
        "flux_clip_detector",
        "flux_fft_detector",
        "check_fp_margins",
        "debug_fft_scores",
    ]

    imported = {}
    for mod_name in modules_to_import:
        try:
            imported[mod_name] = importlib.import_module(mod_name)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"import failed: {mod_name}: {exc}")

    test_file = None
    grok_files = sorted(dataset_dir.glob("grok-video-*.mp4"))
    if grok_files:
        test_file = grok_files[0]
    else:
        fallback = sorted(dataset_dir.glob("*.mp4"))
        if fallback:
            test_file = fallback[0]

    if test_file is None:
        errors.append("no test video found in kod/dataset/ai_baseline")
    else:
        print(f"[TEST] video: {test_file.name}")

    if "ai_style_clip_detector" in imported and test_file is not None:
        try:
            cls = imported["ai_style_clip_detector"].AIStyleCLIPDetector
            det = cls(model_path=repo_root / "clip_classifier.pkl")
            if not getattr(det, "enabled", False):
                errors.append(f"AIStyleCLIPDetector disabled: {getattr(det, 'load_error', 'unknown')}")
            else:
                res = det.detect_video(test_file)
                required = {
                    "ai_style_prob",
                    "ai_style_detected",
                    "ai_style_threshold",
                    "ai_style_top_dims",
                    "error",
                }
                missing = sorted(required - set(res.keys()))
                if missing:
                    errors.append(f"AIStyleCLIPDetector missing keys: {missing}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"AIStyleCLIPDetector run failed: {exc}")

    if "flux_fft_detector" in imported and test_file is not None:
        try:
            cls = imported["flux_fft_detector"].FluxFFTDetector
            det = cls(thresholds_path=repo_root / "flux_fft_thresholds.json")
            res = det.detect_video(test_file)
            required = {"fft_score", "fft_bonus", "metrics", "active_metrics"}
            missing = sorted(required - set(res.keys()))
            if missing:
                errors.append(f"FluxFFTDetector missing keys: {missing}")
        except Exception as exc:  # noqa: BLE001
            errors.append(f"FluxFFTDetector run failed: {exc}")

    if errors:
        print("PRE-COMMIT CHECK: FAIL")
        for e in errors:
            print(f"- {e}")
        return 1

    print("PRE-COMMIT CHECK: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
