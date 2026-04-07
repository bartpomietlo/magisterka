#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Flux watermark signature from grok videos."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("kod/dataset/ai_baseline"),
        help="Folder with ai_baseline videos (default: kod/dataset/ai_baseline).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("kod/dataset/flux_signature.json"),
        help="Output JSON with extracted signature.",
    )
    parser.add_argument(
        "--output-heatmap",
        type=Path,
        default=Path("kod/dataset/flux_signature_heatmap.png"),
        help="Output heatmap image.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=10,
        help="Number of sampled frames per video (default: 10).",
    )
    return parser.parse_args()


def sample_frame_indices(total_frames: int, n_samples: int) -> np.ndarray:
    if total_frames <= 1:
        return np.zeros(n_samples, dtype=int)
    return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)


def ensure_min_size(frame_bgr: np.ndarray, min_size: int = 256) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    if h >= min_size and w >= min_size:
        return frame_bgr
    scale = max(min_size / max(h, 1), min_size / max(w, 1))
    new_w = max(min_size, int(round(w * scale)))
    new_h = max(min_size, int(round(h * scale)))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def decode_bits(frame_bgr: np.ndarray, method: str, bits_len: int = 64) -> np.ndarray | None:
    try:
        from imwatermark import WatermarkDecoder  # type: ignore
    except Exception:
        return None
    frame = ensure_min_size(frame_bgr)
    try:
        decoder = WatermarkDecoder("bits", bits_len)
        bits = decoder.decode(frame[:, :, :3], method)
        if bits is None:
            return None
        arr = np.array([int(b) for b in bits], dtype=np.int8)
        if arr.size < bits_len:
            return None
        return arr[:bits_len]
    except Exception:
        return None


def majority_signature(bit_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # bit_matrix: [n_samples, 64]
    ones_rate = bit_matrix.mean(axis=0)
    signature = (ones_rate >= 0.5).astype(np.int8)
    confidence = np.maximum(ones_rate, 1.0 - ones_rate)
    return signature, confidence


def save_heatmap(
    conf_svd: np.ndarray,
    conf_dwt: np.ndarray,
    output_path: Path,
) -> None:
    mat = np.vstack([conf_svd, conf_dwt])
    plt.figure(figsize=(16, 2.8))
    im = plt.imshow(mat, cmap="viridis", aspect="auto", vmin=0.5, vmax=1.0)
    plt.yticks([0, 1], ["dwtDctSvd", "dwtDct"])
    plt.xticks(np.arange(0, 64, 4))
    plt.xlabel("Bit index")
    plt.title("Flux signature bit confidence (majority vote)")
    cbar = plt.colorbar(im, fraction=0.02, pad=0.02)
    cbar.set_label("confidence")

    # Oznaczenia niestabilnych bitów.
    for row_idx, conf in enumerate([conf_svd, conf_dwt]):
        for bit_idx, c in enumerate(conf):
            if c < 0.6:
                plt.text(bit_idx, row_idx, "x", ha="center", va="center", color="white", fontsize=7)
            elif c >= 0.8:
                plt.text(bit_idx, row_idx, ".", ha="center", va="center", color="white", fontsize=7)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def main() -> int:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    out_json = args.output_json.resolve()
    out_heatmap = args.output_heatmap.resolve()

    grok_files = sorted(
        p for p in dataset_dir.glob("*.mp4") if "grok" in p.name.lower()
    )
    if not grok_files:
        print(f"[ERR] Brak plikow grok*.mp4 w: {dataset_dir}")
        return 1
    if len(grok_files) < 3:
        print("[WARN] n_films < 3, sygnatura moze byc niestabilna.")

    print(f"[INIT] Grok files: {len(grok_files)}")
    for p in grok_files:
        print(f"  - {p.name}")

    bits_svd_all: list[np.ndarray] = []
    bits_dwt_all: list[np.ndarray] = []
    n_frames_total = 0

    for vid_idx, video_path in enumerate(grok_files, 1):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[WARN] Nie mozna otworzyc: {video_path.name}")
            continue
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = sample_frame_indices(total_frames, args.frames_per_video)
        print(f"[VIDEO] ({vid_idx}/{len(grok_files)}) {video_path.name} frames={len(frame_indices)}")

        ok_frames = 0
        for fi in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            svd_bits = decode_bits(frame, method="dwtDctSvd", bits_len=64)
            dwt_bits = decode_bits(frame, method="dwtDct", bits_len=64)
            if svd_bits is not None:
                bits_svd_all.append(svd_bits)
            if dwt_bits is not None:
                bits_dwt_all.append(dwt_bits)
            ok_frames += 1
        cap.release()
        n_frames_total += ok_frames
        print(f"  decoded_frames={ok_frames}")

    if not bits_svd_all or not bits_dwt_all:
        print("[ERR] Nie udalo sie zdekodowac bitow dla wymaganych metod.")
        return 2

    svd_matrix = np.stack(bits_svd_all, axis=0).astype(np.int8)
    dwt_matrix = np.stack(bits_dwt_all, axis=0).astype(np.int8)

    sig_svd, conf_svd = majority_signature(svd_matrix)
    sig_dwt, conf_dwt = majority_signature(dwt_matrix)

    stable_svd = int(np.sum(conf_svd >= 0.8))
    stable_dwt = int(np.sum(conf_dwt >= 0.8))
    noisy_svd = int(np.sum(conf_svd < 0.6))
    noisy_dwt = int(np.sum(conf_dwt < 0.6))

    payload: dict[str, Any] = {
        "dwtDctSvd": "".join(str(int(b)) for b in sig_svd.tolist()),
        "dwtDct": "".join(str(int(b)) for b in sig_dwt.tolist()),
        "n_films": len(grok_files),
        "n_frames_total": int(n_frames_total),
        "confidence_per_bit": {
            "dwtDctSvd": [float(round(x, 4)) for x in conf_svd.tolist()],
            "dwtDct": [float(round(x, 4)) for x in conf_dwt.tolist()],
        },
        "stable_bits_gt_0_8": {
            "dwtDctSvd": stable_svd,
            "dwtDct": stable_dwt,
        },
        "noisy_bits_lt_0_6": {
            "dwtDctSvd": noisy_svd,
            "dwtDct": noisy_dwt,
        },
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    save_heatmap(conf_svd, conf_dwt, out_heatmap)
    print(f"[OUT] {out_json}")
    print(f"[OUT] {out_heatmap}")
    print(
        "[SUMMARY] stable bits >0.8 | "
        f"dwtDctSvd={stable_svd}/64, dwtDct={stable_dwt}/64"
    )
    print(
        "[SUMMARY] noisy bits <0.6 | "
        f"dwtDctSvd={noisy_svd}/64, dwtDct={noisy_dwt}/64"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
