from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from agh_watermark import AGHWatermark
from pot_watermark import POTWatermark, compute_psnr, compute_ssim

AttackName = Literal["clean", "h264_jpeg_q75", "crop_center_0.8", "resize_0.5_bicubic"]
WatermarkMethod = Literal["QIM", "AGH_NONEXP", "AGH_EXP", "AGH_SPARSE"]


def _sample_frame_indices(total_frames: int, n_samples: int) -> list[int]:
    if total_frames <= 0:
        return []
    if total_frames == 1:
        return [0] * n_samples
    idxs = np.linspace(0, total_frames - 1, num=n_samples, dtype=int)
    return [int(i) for i in idxs]


def _read_frame_by_index(cap: cv2.VideoCapture, idx: int) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def _ensure_uint8_bgr(frame_bgr: np.ndarray) -> np.ndarray:
    if frame_bgr.dtype == np.uint8:
        return frame_bgr
    return np.clip(np.round(frame_bgr), 0, 255).astype(np.uint8)


def attack_h264_like(frame_bgr: np.ndarray) -> np.ndarray:
    """
    Symulacja "H264": wymaganie zadania mówi, żeby zakodować klatkę do JPEG
    o quality=75 i z powrotem.
    """
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 75]
    ok, enc = cv2.imencode(".jpg", _ensure_uint8_bgr(frame_bgr), encode_param)
    if not ok:
        return frame_bgr
    dec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if dec is None:
        return frame_bgr
    return dec


def attack_crop_center(frame_bgr: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
    """
    Crop do centralnych 80% (crop center) + resize z powrotem do rozmiaru wejściowego.
    """
    h, w = frame_bgr.shape[:2]
    ch = max(1, int(round(h * crop_ratio)))
    cw = max(1, int(round(w * crop_ratio)))

    y1 = (h - ch) // 2
    x1 = (w - cw) // 2

    cropped = frame_bgr[y1 : y1 + ch, x1 : x1 + cw]
    resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized


def attack_resize_half(frame_bgr: np.ndarray) -> np.ndarray:
    """
    RESIZE: skaluje do 50% i z powrotem do oryginalnego rozmiaru (bicubic).
    """
    h, w = frame_bgr.shape[:2]
    nh = max(1, int(round(h * 0.5)))
    nw = max(1, int(round(w * 0.5)))
    small = cv2.resize(frame_bgr, (nw, nh), interpolation=cv2.INTER_CUBIC)
    resized = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
    return resized


def _aggregate_frame_results(frame_results: list[dict[str, Any]]) -> dict[str, Any]:
    psnr_vals = [float(r["psnr"]) for r in frame_results if "psnr" in r]
    ssim_vals = [float(r["ssim"]) for r in frame_results if "ssim" in r]
    ber_vals = [float(r["ber"]) for r in frame_results]

    magic_ok_rate = float(np.mean([1.0 if r["magic_ok"] else 0.0 for r in frame_results])) if frame_results else 0.0
    crc_ok_rate = float(np.mean([1.0 if r["crc_ok"] else 0.0 for r in frame_results])) if frame_results else 0.0
    payload_recovery_rate = float(np.mean([1.0 if r["detected"] else 0.0 for r in frame_results])) if frame_results else 0.0
    blocks_decoded_mean = float(np.mean([float(r["blocks_decoded"]) for r in frame_results])) if frame_results else 0.0

    return {
        "psnr_mean": float(np.mean(psnr_vals)) if psnr_vals else 0.0,
        "ssim_mean": float(np.mean(ssim_vals)) if ssim_vals else 0.0,
        "ber_mean": float(np.mean(ber_vals)) if ber_vals else 0.0,
        "magic_ok_rate": magic_ok_rate,
        "crc_ok_rate": crc_ok_rate,
        "payload_recovery_rate": payload_recovery_rate,
        "blocks_decoded_mean": blocks_decoded_mean,
    }


def evaluate_video(
    video_path: Path,
    watermark: POTWatermark | AGHWatermark,
    method: str,
    strength: float,
    n_frames: int,
    output_frames_dir: Path | None,
) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie mozna otworzyc wideo: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = _sample_frame_indices(total_frames, n_frames)
    if not idxs:
        cap.release()
        return pd.DataFrame()

    # Odczyt klatek
    frames_orig: list[np.ndarray] = []
    for idx in idxs:
        frame = _read_frame_by_index(cap, idx)
        if frame is None:
            continue
        frames_orig.append(frame)

    cap.release()

    if not frames_orig:
        return pd.DataFrame()

    out_rows: list[dict[str, Any]] = []

    # Define ataki
    attacks: list[tuple[AttackName, Any]] = [
        ("clean", lambda x: x),
        ("h264_jpeg_q75", attack_h264_like),
        ("crop_center_0.8", lambda x: attack_crop_center(x, crop_ratio=0.8)),
        ("resize_0.5_bicubic", attack_resize_half),
    ]

    # Osadzamy watermark raz dla każdej klatki (clean), a potem ataki od tej wersji.
    # Wymaganie: w pętli b) zapisać watermarked frame - robimy to przez save do plików png.
    frames_wm: list[np.ndarray] = []
    payload_embed_infos: list[dict[str, Any]] = []

    for frame_idx, frame in enumerate(frames_orig):
        frame_wm, info = watermark.embed(frame, frame_id=frame_idx % 256, method=method, strength=strength)
        frames_wm.append(frame_wm)
        payload_embed_infos.append(info)

        if output_frames_dir is not None:
            output_frames_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_frames_dir / f"frame_{frame_idx:03d}_wm.png"
            # cv2 ma BGR; zapis ok
            cv2.imwrite(str(out_file), frame_wm)

    for attack_name, attack_fn in attacks:
        frame_results: list[dict[str, Any]] = []

        for frame_idx, (frame_orig, frame_wm) in enumerate(zip(frames_orig, frames_wm)):
            attacked = attack_fn(frame_wm)

            # Metryki vs oryginał (wymaganie c)
            psnr_val = compute_psnr(frame_orig, attacked)
            ssim_val = compute_ssim(frame_orig, attacked)

            # Decode po ataku (wymaganie d,e)
            dec = watermark.decode(attacked, method=method, strength=strength)

            frame_results.append(
                {
                    "psnr": psnr_val,
                    "ssim": ssim_val,
                    "ber": float(dec.get("ber", 0.0)),
                    "magic_ok": bool(dec.get("magic_ok", False)),
                    "crc_ok": bool(dec.get("crc_ok", False)),
                    "detected": bool(dec.get("detected", False)),
                    "blocks_decoded": int(dec.get("blocks_decoded", 0)),
                }
            )

            if output_frames_dir is not None:
                # zapis wyników po atakach
                out_attack_file = output_frames_dir / f"frame_{frame_idx:03d}_{attack_name}.png"
                cv2.imwrite(str(out_attack_file), attacked)

        agg = _aggregate_frame_results(frame_results)
        row_base = {
            "filename": video_path.name,
            "method": method,
            "attack": attack_name,
        }
        row_base.update(agg)
        out_rows.append(row_base)

    return pd.DataFrame(out_rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="results_watermark.csv")
    parser.add_argument("--n_frames", type=int, default=30)
    parser.add_argument("--method", type=str, default="QIM")
    parser.add_argument(
        "--watermark_method",
        type=str,
        default="QIM",
        choices=["QIM", "AGH_NONEXP", "AGH_EXP", "AGH_SPARSE"],
        help="Wybierz obecny POT/QIM albo watermark AGH z transformata NONEXP/EXP/SPARSE.",
    )
    parser.add_argument("--strength", type=float, default=8.0)
    parser.add_argument("--save_frames", action="store_true",
                        help="Zapisz klatki PNG (domyslnie wylaczone)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"input_dir nie istnieje: {input_dir}")

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    watermark_method: WatermarkMethod = str(args.watermark_method).upper()  # type: ignore[assignment]
    if watermark_method == "QIM":
        eval_method = str(args.method).upper()
        watermark: POTWatermark | AGHWatermark = POTWatermark(method=eval_method)
    else:
        agh_transform_map = {
            "AGH_NONEXP": "NONEXP",
            "AGH_EXP": "EXP",
            "AGH_SPARSE": "SPARSE_NONEXP",
        }
        eval_method = agh_transform_map[watermark_method]
        watermark = AGHWatermark(transform_type=eval_method)

    # folder na zapisy klatek (opcjonalnie - zawsze tworzymy, bo wymaganie mówi "zapisz")
    frames_root = (out_csv.parent / "watermark_frames" / out_csv.stem) if args.save_frames else None

    video_exts = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".mpeg", ".mpg"}
    videos = [p for p in sorted(input_dir.iterdir()) if p.is_file() and p.suffix.lower() in video_exts]
    if not videos:
        print(f"Brak plików wideo w {input_dir}")
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    all_rows: list[pd.DataFrame] = []
    for vid_idx, video_path in enumerate(tqdm(videos, desc="Watermark eval")):
        per_video_frames_dir = (frames_root / f"video_{vid_idx:04d}") if frames_root else None
        df_video = evaluate_video(
            video_path=video_path,
            watermark=watermark,
            method=eval_method,
            strength=float(args.strength),
            n_frames=int(args.n_frames),
            output_frames_dir=per_video_frames_dir,
        )
        if not df_video.empty:
            all_rows.append(df_video)

    if not all_rows:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all.to_csv(out_csv, index=False)

    # Tabela zbiorcza per attack
    summary_cols = ["attack", "psnr_mean", "ssim_mean", "ber_mean", "magic_ok_rate", "crc_ok_rate", "payload_recovery_rate", "blocks_decoded_mean"]
    summary = (
        df_all.groupby("attack")[summary_cols[1:]]
        .mean()
        .reset_index()
    )

    print("\n=== Podsumowanie per attack (średnia po wideo) ===")
    # proste formatowanie
    with pd.option_context("display.max_columns", None):
        print(summary.sort_values("attack").to_string(index=False))


if __name__ == "__main__":
    main()
