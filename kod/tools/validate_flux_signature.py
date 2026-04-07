#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class VideoFluxScore:
    filename: str
    split: str
    label: int
    similarity: float
    similarity_std: float
    method: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Flux signature on AI and authentic datasets."
    )
    parser.add_argument(
        "--signature-json",
        type=Path,
        default=Path("kod/dataset/flux_signature.json"),
        help="Path to flux_signature.json",
    )
    parser.add_argument(
        "--ai-dir",
        type=Path,
        default=Path("kod/dataset/ai_baseline"),
        help="Path to ai_baseline folder",
    )
    parser.add_argument(
        "--auth-dir",
        type=Path,
        default=Path("kod/dataset/adv_fp_trap"),
        help="Path to authentic folder (default: adv_fp_trap)",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=10,
        help="Number of sampled frames for detection",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.45, 0.50, 0.55, 0.60],
        help="Thresholds for FPR/TPR report",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("kod/dataset/flux_validation_scores.csv"),
        help="Output CSV for per-video similarity",
    )
    parser.add_argument(
        "--write-threshold",
        action="store_true",
        help="Write selected threshold back to flux_signature.json (optimal_threshold).",
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


def hamming_similarity(bits: np.ndarray, sig_bits: np.ndarray) -> float:
    cmp_len = min(bits.size, sig_bits.size)
    if cmp_len <= 0:
        return 0.0
    return float(np.mean(bits[:cmp_len] == sig_bits[:cmp_len]))


def score_video(video_path: Path, sig_svd: np.ndarray, sig_dwt: np.ndarray, n_frames: int) -> VideoFluxScore:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return VideoFluxScore(video_path.name, "", -1, 0.0, 0.0, "none")
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = sample_frame_indices(total_frames, n_frames)

    sims_svd: list[float] = []
    sims_dwt: list[float] = []
    for fi in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        bits_svd = decode_bits(frame, "dwtDctSvd", bits_len=64)
        bits_dwt = decode_bits(frame, "dwtDct", bits_len=64)
        if bits_svd is not None:
            sims_svd.append(hamming_similarity(bits_svd, sig_svd))
        if bits_dwt is not None:
            sims_dwt.append(hamming_similarity(bits_dwt, sig_dwt))
    cap.release()

    med_svd = float(np.median(sims_svd)) if sims_svd else 0.0
    med_dwt = float(np.median(sims_dwt)) if sims_dwt else 0.0
    if med_svd >= med_dwt:
        all_sims = np.array(sims_svd, dtype=np.float32) if sims_svd else np.array([0.0], dtype=np.float32)
        return VideoFluxScore(
            filename=video_path.name,
            split="",
            label=-1,
            similarity=float(med_svd),
            similarity_std=float(np.std(all_sims)),
            method="dwtDctSvd",
        )
    all_sims = np.array(sims_dwt, dtype=np.float32) if sims_dwt else np.array([0.0], dtype=np.float32)
    return VideoFluxScore(
        filename=video_path.name,
        split="",
        label=-1,
        similarity=float(med_dwt),
        similarity_std=float(np.std(all_sims)),
        method="dwtDct",
    )


def main() -> int:
    args = parse_args()
    with args.signature_json.open("r", encoding="utf-8") as f:
        sig_payload = json.load(f)

    sig_svd = np.array([int(c) for c in str(sig_payload["dwtDctSvd"]).strip()], dtype=np.int8)
    sig_dwt = np.array([int(c) for c in str(sig_payload["dwtDct"]).strip()], dtype=np.int8)

    ai_files = sorted(args.ai_dir.glob("*.mp4"))
    auth_files = sorted(args.auth_dir.glob("*.mp4"))
    if not ai_files or not auth_files:
        print(f"[ERR] Brak plikow. ai={len(ai_files)} auth={len(auth_files)}")
        return 1

    rows: list[VideoFluxScore] = []
    total = len(ai_files) + len(auth_files)
    i = 0
    for p in ai_files:
        i += 1
        print(f"[AI ] ({i}/{total}) {p.name}")
        sc = score_video(p, sig_svd, sig_dwt, args.frames_per_video)
        sc.split = "ai_baseline"
        sc.label = 1
        rows.append(sc)
    for p in auth_files:
        i += 1
        print(f"[AUT] ({i}/{total}) {p.name}")
        sc = score_video(p, sig_svd, sig_dwt, args.frames_per_video)
        sc.split = "adv_fp_trap"
        sc.label = 0
        rows.append(sc)

    # Ranking AI by similarity
    ai_rank = sorted((r for r in rows if r.label == 1), key=lambda x: x.similarity, reverse=True)
    auth_rank = sorted((r for r in rows if r.label == 0), key=lambda x: x.similarity, reverse=True)
    print("\n=== Top AI similarity ranking ===")
    for r in ai_rank[:20]:
        print(f"{r.similarity:.3f}  {r.filename}  method={r.method}")
    print("\n=== Top AUTH similarity ranking ===")
    for r in auth_rank[:20]:
        print(f"{r.similarity:.3f}  {r.filename}  method={r.method}")

    # Threshold sweep
    ai_scores = np.array([r.similarity for r in rows if r.label == 1], dtype=np.float32)
    auth_scores = np.array([r.similarity for r in rows if r.label == 0], dtype=np.float32)

    print("\n=== Threshold sweep ===")
    best_threshold = None
    best_tpr = -1.0
    for thr in args.thresholds:
        tpr = float(np.mean(ai_scores >= thr)) if ai_scores.size else 0.0
        fpr = float(np.mean(auth_scores >= thr)) if auth_scores.size else 0.0
        print(f"thr={thr:.2f}  TPR={tpr:.4f}  FPR={fpr:.4f}")
        if fpr < 0.05 and tpr > best_tpr:
            best_tpr = tpr
            best_threshold = thr

    safe_for_integration = True
    if best_threshold is None:
        print("[OPT] Brak progu z FPR < 0.05 w zadanym zakresie.")
        # fallback: minimal FPR, then max TPR
        candidates = []
        for thr in np.arange(0.40, 0.91, 0.01):
            tpr = float(np.mean(ai_scores >= thr)) if ai_scores.size else 0.0
            fpr = float(np.mean(auth_scores >= thr)) if auth_scores.size else 0.0
            candidates.append((fpr, -tpr, thr, tpr))
        candidates.sort()
        fpr, _neg_tpr, thr, tpr = candidates[0]
        best_threshold = float(thr)
        safe_for_integration = False
        print(f"[OPT] Najlepszy kompromis: thr={best_threshold:.2f} TPR={tpr:.4f} FPR={fpr:.4f}")
    else:
        fpr = float(np.mean(auth_scores >= best_threshold))
        print(f"[OPT] Wybrany prog: thr={best_threshold:.2f} TPR={best_tpr:.4f} FPR={fpr:.4f}")

    if args.write_threshold:
        sig_payload["optimal_threshold"] = float(round(best_threshold, 4))
        sig_payload["safe_for_integration"] = bool(safe_for_integration)
        with args.signature_json.open("w", encoding="utf-8") as f:
            json.dump(sig_payload, f, ensure_ascii=False, indent=2)
        print(
            f"[OUT] Updated {args.signature_json.resolve()} with "
            f"optimal_threshold={float(best_threshold):.4f}, "
            f"safe_for_integration={bool(safe_for_integration)}"
        )

    # Save per-video scores
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split", "label", "flux_similarity", "flux_similarity_std", "flux_method"])
        for r in rows:
            writer.writerow(
                [
                    r.filename,
                    r.split,
                    r.label,
                    f"{r.similarity:.6f}",
                    f"{r.similarity_std:.6f}",
                    r.method,
                ]
            )
    print(f"[OUT] {args.out_csv.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
