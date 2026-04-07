#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _probe(path: Path) -> dict[str, Any]:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        str(path),
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        return json.loads(out)
    except Exception:
        return {}


def _fps_from_ratio(r: str) -> float:
    if not r or r == "0/0":
        return 0.0
    if "/" in r:
        a, b = r.split("/", 1)
        try:
            return float(a) / max(float(b), 1e-9)
        except Exception:
            return 0.0
    return _to_float(r, 0.0)


def _opencv_meta(path: Path) -> dict[str, Any]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"fps": 0.0, "width": 0, "height": 0, "duration_s": 0.0}
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    duration = float(frames / fps) if fps > 0 else 0.0
    return {"fps": float(fps), "width": width, "height": height, "duration_s": duration}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default="kod/results/latest/evaluation_results.csv")
    parser.add_argument("--output", default="compression_params.csv")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    rows = list(csv.DictReader(eval_path.open("r", encoding="utf-8", newline="")))
    fn = [
        r for r in rows
        if r.get("category") == "adv_compressed"
        and int(float(r.get("ground_truth", 0))) == 1
        and int(float(r.get("detected", 0))) == 0
    ]
    base = Path("kod/dataset/adv_compressed")

    out_rows: list[dict[str, Any]] = []
    for r in fn:
        fn_name = r.get("filename", "")
        path = base / fn_name
        probe = _probe(path)

        fmt = probe.get("format", {}) if isinstance(probe, dict) else {}
        streams = probe.get("streams", []) if isinstance(probe, dict) else []
        vstream = {}
        for s in streams:
            if s.get("codec_type") == "video":
                vstream = s
                break

        if vstream:
            codec = str(vstream.get("codec_name", ""))
            width = int(vstream.get("width", 0) or 0)
            height = int(vstream.get("height", 0) or 0)
            fps = _fps_from_ratio(str(vstream.get("avg_frame_rate", vstream.get("r_frame_rate", "0/0"))))
            stream_br = _to_float(vstream.get("bit_rate", 0.0))
        else:
            codec = ""
            width = 0
            height = 0
            fps = 0.0
            stream_br = 0.0

        format_br = _to_float(fmt.get("bit_rate", 0.0))
        duration = _to_float(fmt.get("duration", 0.0))
        if duration <= 0.0:
            meta = _opencv_meta(path)
            duration = meta["duration_s"]
            if width == 0:
                width = int(meta["width"])
            if height == 0:
                height = int(meta["height"])
            if fps == 0.0:
                fps = float(meta["fps"])

        bitrate = stream_br if stream_br > 0 else format_br
        if bitrate <= 0 and duration > 0 and path.exists():
            bitrate = float((path.stat().st_size * 8) / duration)

        tags = vstream.get("tags", {}) if isinstance(vstream, dict) else {}
        encoder = str(tags.get("encoder", "")) if isinstance(tags, dict) else ""
        qp_est = ""
        if "crf" in encoder.lower():
            qp_est = encoder

        out_rows.append(
            {
                "filename": fn_name,
                "bitrate_kbps": round(bitrate / 1000.0, 2),
                "codec": codec,
                "resolution": f"{width}x{height}",
                "frame_rate": round(fps, 3),
                "duration_s": round(duration, 3),
                "qp_crf_estimate": qp_est,
            }
        )

    out_rows.sort(key=lambda x: _to_float(x["bitrate_kbps"], 0.0))

    out_path = Path(args.output)
    fields = ["filename", "bitrate_kbps", "codec", "resolution", "frame_rate", "duration_s", "qp_crf_estimate"]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in out_rows:
            w.writerow(row)

    print(f"FN compressed files analyzed: {len(out_rows)}")
    if out_rows:
        print("Strongest compression candidates (lowest bitrate):")
        for row in out_rows[:3]:
            print(f"- {row['filename']}: {row['bitrate_kbps']} kbps, {row['codec']}, {row['resolution']}")
    print(f"Saved: {out_path.resolve()}")


if __name__ == "__main__":
    main()

