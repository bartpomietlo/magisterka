#!/usr/bin/env python3
"""
evaluate.py
Ewaluacja detektora watermarków z uwzględnieniem pełnego pipeline'u OCR + Advanced.
"""

from __future__ import annotations
import csv
import sys
import time
from pathlib import Path
import cv2

sys.path.insert(0, str(Path(__file__).parent.parent))
from ocr_detector import scan_for_watermarks

DATASET_ROOT = Path(__file__).parent
RESULTS_DIR  = DATASET_ROOT.parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CSV    = RESULTS_DIR / "evaluation_results.csv"
METRICS_CSV = RESULTS_DIR / "metrics_summary.csv"

# POPRAWKA 1: Ground Truth dla kategorii AI ustawiony na 1 (oczekiwany znak wodny)
CATEGORIES = {
  "ai_baseline":    (DATASET_ROOT / "ai_baseline",    1),
  "adv_compressed": (DATASET_ROOT / "adv_compressed", 1),
  "adv_cropped":    (DATASET_ROOT / "adv_cropped",    1),
  "adv_fp_trap":    (DATASET_ROOT / "adv_fp_trap",    0),
}

EVAL_FIELDS = [
  "category", "filename", "ground_truth",
  "detected", "watermark_types", "invisible_wm_found", "fft_found",
  "optical_flow_contours", "duration_s"
]

def load_done(path: Path) -> set[str]:
  if not path.exists():
    return set()
  with path.open("r", encoding="utf-8", newline="") as f:
    return {f"{r['category']}::{r['filename']}" for r in csv.DictReader(f)}

def scan_video(video_path: Path):
  t0 = time.time()
  # POPRAWKA 2: Użycie pełnego silnika detekcji (OCR + Advanced)
  result = scan_for_watermarks(
    media_path=str(video_path),
    confidence=0.6,
    sample_rate=30,
    detailed_scan=False
  )
  elapsed = time.time() - t0
  return result, elapsed

def build_row(category: str, path: Path, ground_truth: int, result: dict, elapsed: float) -> dict:
  adv = result.get("advanced", {})

  # Strict check dla invisible WM - tylko pewne dopasowania
  iw_data = adv.get("invisible_wm", {})
  iw_found = 1 if iw_data.get("matched") is not None else 0

  fft = 1 if adv.get("fft_artifacts", {}).get("found", False) else 0
  of_count = len(adv.get("optical_flow_rois", []))

  # Detekcja jest pozytywna jeśli silnik główny (OCR) znalazł znak LUB pewne metody zaawansowane
  ocr_found = 1 if result.get("watermark_found", False) else 0
  final_detected = 1 if (ocr_found or iw_found or fft) else 0

  wm_types = "|".join(result.get("watermark_types", []))

  return {
    "category":              category,
    "filename":              path.name,
    "ground_truth":          ground_truth,
    "detected":              final_detected,
    "watermark_types":       wm_types,
    "invisible_wm_found":    iw_found,
    "fft_found":             fft,
    "optical_flow_contours": of_count,
    "duration_s":            f"{elapsed:.2f}",
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
  done = load_done(EVAL_CSV)
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

  all_rows = []
  with EVAL_CSV.open("r", encoding="utf-8", newline="") as f:
    all_rows = list(csv.DictReader(f))

  metrics = compute_metrics(all_rows)
  with METRICS_CSV.open("w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(metrics[0].keys()))
    writer.writeheader()
    writer.writerows(metrics)

  print(f"\n[WYNIKI]  {EVAL_CSV}")
  print(f"[METRYKI] {METRICS_CSV}")
  print("\n--- Podsumowanie metryk ---")
  for m in metrics:
    print(f"  {m['category']}: n={m['n']}, acc={m['accuracy']}, "
          f"FPR={m['FPR']}, TP={m['TP']}, FP={m['FP']}, "
          f"TN={m['TN']}, FN={m['FN']}")

if __name__ == "__main__":
  main()
