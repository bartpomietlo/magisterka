#!/usr/bin/env python3
"""
evaluate.py
Ewaluacja detektora watermarków — używa wyłącznie run_advanced_scan (bez OCR).
Kategorie:
  ai_baseline    (filmy AI z watermarkiem)        → gt=1
  adv_compressed (filmy AI skompresowane)          → gt=1
  adv_cropped    (filmy AI przycięte)              → gt=1
  adv_fp_trap    (filmy ludzkie / pułapki FP)      → gt=0
"""

from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from advanced_detectors import run_advanced_scan

DATASET_ROOT = Path(__file__).parent
RESULTS_DIR  = DATASET_ROOT.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CSV    = RESULTS_DIR / "evaluation_results.csv"
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"

CATEGORIES = {
    "ai_baseline":    (DATASET_ROOT / "ai_baseline",    1),
    "adv_compressed": (DATASET_ROOT / "adv_compressed", 1),
    "adv_cropped":    (DATASET_ROOT / "adv_cropped",    1),
    "adv_fp_trap":    (DATASET_ROOT / "adv_fp_trap",    0),
}

EVAL_FIELDS = [
    "category", "filename", "ground_truth",
    "detected", "zero_variance_rois", "optical_flow_rois",
    "invisible_wm_found", "fft_found", "duration_s"
]


def load_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as f:
        return {f"{r['category']}::{r['filename']}" for r in csv.DictReader(f)}


def scan_video(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Nie można otworzyć: {video_path}")
    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    t0 = time.time()
    result = run_advanced_scan(
        cap, fps, total_frames,
        n_frames_median=30,
        check_invisible=True,
        check_fft=True,
        check_optical_flow=True,
        of_scale=0.5,
    )
    elapsed = time.time() - t0
    cap.release()
    return result, elapsed


def build_row(
    category: str, path: Path, ground_truth: int,
    result: dict, elapsed: float
) -> dict:
    zv  = len(result.get("zero_variance_rois", []))
    of  = len(result.get("optical_flow_rois",  []))

    iw_data   = result.get("invisible_wm", {})
    iw_found  = 1 if iw_data.get("matched") is not None else 0

    fft_data  = result.get("fft_artifacts", {})
    fft_found = 1 if fft_data.get("found", False) else 0

    # POPRAWKA: Nowa, bardziej odporna logika detekcji
    static_overlay = int(zv > 0 and of > 0)
    of_found = int(of >= 3)  # próg minimalny
    detected = int(of_found or static_overlay)


    return {
        "category":           category,
        "filename":           path.name,
        "ground_truth":       ground_truth,
        "detected":           detected,
        "zero_variance_rois": zv,
        "optical_flow_rois":  of,
        "invisible_wm_found": iw_found,
        "fft_found":          fft_found,
        "duration_s":         f"{elapsed:.2f}",
    }


def compute_metrics(rows: list[dict]) -> list[dict]:
    cats: dict[str, list] = {}
    for row in rows:
        cats.setdefault(row["category"], []).append(row)
    metric_rows = []
    for cat, cat_rows in cats.items():
        gt   = [int(r["ground_truth"]) for r in cat_rows]
        pred = [int(r["detected"])     for r in cat_rows]
        tp = sum(p == 1 and g == 1 for p, g in zip(pred, gt))
        tn = sum(p == 0 and g == 0 for p, g in zip(pred, gt))
        fp = sum(p == 1 and g == 0 for p, g in zip(pred, gt))
        fn = sum(p == 0 and g == 1 for p, g in zip(pred, gt))
        n  = len(gt)
        acc  = (tp + tn) / n if n else 0
        prec = tp / (tp + fp) if (tp + fp) else 0
        rec  = tp / (tp + fn) if (tp + fn) else 0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
        fpr  = fp / (fp + tn) if (fp + tn) else 0
        metric_rows.append({
            "category": cat, "n": n,
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "accuracy":  f"{acc:.4f}",
            "precision": f"{prec:.4f}",
            "recall":    f"{rec:.4f}",
            "f1":        f"{f1:.4f}",
            "FPR":       f"{fpr:.4f}",
        })
    return metric_rows


def main() -> None:
    # POPRAWKA: Usuń stare wyniki, aby zapewnić czystą ewaluację
    if EVAL_CSV.exists():
        print(f"[INFO] Usuwam stary plik wyników: {EVAL_CSV}")
        EVAL_CSV.unlink()
    if METRICS_CSV.exists():
        print(f"[INFO] Usuwam stary plik metryk: {METRICS_CSV}")
        METRICS_CSV.unlink()


    done     = load_done(EVAL_CSV)
    new_rows = []

    for category, (folder, gt) in CATEGORIES.items():
        videos = sorted(folder.glob("*.mp4"))
        if not videos:
            print(f"[WARN] Brak .mp4 w {folder}", file=sys.stderr)
            continue
        print(f"\n=== {category} ({len(videos)} filmów) ===")
        for vp in videos:
            key = f"{category}::{vp.name}"
            if key in done:
                print(f"  [SKIP] {vp.name}")
                continue
            print(f"  [SCAN] {vp.name} ... ", end="", flush=True)
            try:
                result, elapsed = scan_video(vp)
                row = build_row(category, vp, gt, result, elapsed)
                new_rows.append(row)
                det_str = "WYKRYTO" if int(row["detected"]) else "brak"
                print(f"{det_str}  ({elapsed:.1f}s)")
            except Exception as e:
                print(f"BŁĄD: {e}", file=sys.stderr)

    exists = EVAL_CSV.exists()
    with EVAL_CSV.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVAL_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerows(new_rows)

    all_rows: list[dict] = []
    with EVAL_CSV.open("r", encoding="utf-8", newline="") as f:
        all_rows = list(csv.DictReader(f))

    if not all_rows:
        print("[WARN] Brak danych do obliczenia metryk.", file=sys.stderr)
        return

    metrics = compute_metrics(all_rows)
    with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
        writer.writeheader()
        writer.writerows(metrics)

    print(f"\n[WYNIKI]  {EVAL_CSV}")
    print(f"[METRYKI] {METRICS_CSV}")
    print("\n--- Podsumowanie metryk ---")
    for m in metrics:
        print(
            f"  {m['category']:18s}: n={m['n']:3d} "
            f"acc={m['accuracy']}  f1={m['f1']}  "
            f"FPR={m['FPR']}  "
            f"TP={m['TP']} FP={m['FP']} TN={m['TN']} FN={m['FN']}"
        )


if __name__ == "__main__":
    main()
