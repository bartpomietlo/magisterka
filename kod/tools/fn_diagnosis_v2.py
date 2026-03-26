#!/usr/bin/env python3
"""
fn_diagnosis_v2.py

More thesis-ready false-negative analysis.

What it adds over fn_diagnosis.py:
- reports the scale of persistent false negatives numerically,
- separates "extreme low-motion cases" from broader heuristic failures,
- prints a more accurate thesis wording based on the observed count,
- keeps the per-video dump for manual discussion in the thesis.
"""

from __future__ import annotations

import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

RESULTS_ROOT = Path(__file__).parent.parent / "results" / "latest"
EVAL_CSV = RESULTS_ROOT / "evaluation_results.csv"
RAW_CSV = RESULTS_ROOT / "raw_signals.csv"
POSITIVE_SPLITS = ["ai_baseline", "adv_compressed", "adv_cropped"]


def _int(x):
    return int(float(x)) if x not in ("", None) else 0


def _float(x):
    return float(x) if x not in ("", None) else 0.0


def load_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_failure(rows: list[dict]) -> tuple[str, list[str]]:
    mean_of_count = mean(_int(r.get("of_count", 0)) for r in rows)
    mean_zv_count = mean(_int(r.get("zv_count", 0)) for r in rows)
    mean_motion = mean(_float(r.get("of_global_motion", 0.0)) for r in rows)
    mean_area = mean(_float(r.get("of_max_area", 0.0)) for r in rows)

    tags: list[str] = []
    if mean_of_count <= 1.0:
        tags.append("very few optical-flow contours")
    if mean_motion < 1.0:
        tags.append("low global motion")
    if mean_zv_count == 0:
        tags.append("no zero-variance ROIs")
    if mean_area < 50000:
        tags.append("small static contour area")
    if not tags:
        tags.append("weak heuristic evidence in all AI-positive splits")
    return ", ".join(tags), tags


def main() -> None:
    if not EVAL_CSV.exists():
        print(f"[ERROR] Missing {EVAL_CSV}. Run evaluate.py first.")
        sys.exit(1)

    eval_rows = load_csv(EVAL_CSV)
    raw_rows = load_csv(RAW_CSV) if RAW_CSV.exists() else []
    raw_by_key = {(r["category"], r["filename"]): r for r in raw_rows}

    positive_eval_rows = [r for r in eval_rows if r["category"] in POSITIVE_SPLITS]
    unique_positive_videos = sorted({r["filename"] for r in positive_eval_rows})

    by_filename = defaultdict(list)
    for row in positive_eval_rows:
        by_filename[row["filename"]].append(row)

    always_fn = []
    for filename, rows in by_filename.items():
        present_splits = {r["category"] for r in rows}
        if set(POSITIVE_SPLITS).issubset(present_splits) and all(_int(r["detected"]) == 0 for r in rows):
            enriched = []
            for r in rows:
                key = (r["category"], r["filename"])
                merged = dict(r)
                merged.update(raw_by_key.get(key, {}))
                enriched.append(merged)
            always_fn.append((filename, enriched))

    print("Persistent false negatives in every AI-positive split")
    print("=" * 76)
    if not always_fn:
        print("No such videos found.")
        return

    total_positive = len(unique_positive_videos)
    persistent_fn_count = len(always_fn)
    persistent_fn_ratio = persistent_fn_count / total_positive if total_positive else 0.0
    print(f"AI-positive source videos analysed: {total_positive}")
    print(f"Persistent FN across baseline/compressed/cropped: {persistent_fn_count} ({persistent_fn_ratio:.1%})")

    tag_counter = Counter()
    extreme_low_motion = 0
    for _, rows in always_fn:
        _, tags = summarize_failure(rows)
        tag_counter.update(tags)
        if "low global motion" in tags and "very few optical-flow contours" in tags:
            extreme_low_motion += 1

    print("\nFailure pattern counts:")
    for tag, count in tag_counter.most_common():
        print(f"  - {tag}: {count}")
    print(f"  - extreme low-motion + sparse-OF subset: {extreme_low_motion}")

    for filename, rows in sorted(always_fn):
        reason, _ = summarize_failure(rows)
        print(f"\n{filename}")
        print(f"  reason: {reason}")
        for r in sorted(rows, key=lambda x: x["category"]):
            print(
                f"  - {r['category']:15s} detected={_int(r.get('detected', 0))} "
                f"of_count={_int(r.get('of_count', 0))} "
                f"of_global_motion={_float(r.get('of_global_motion', 0.0)):.3f} "
                f"of_max_area={_float(r.get('of_max_area', 0.0)):.1f} "
                f"zv_count={_int(r.get('zv_count', 0))}"
            )

    print("\nSuggested thesis wording:")
    print(
        f"A substantial subset of AI videos ({persistent_fn_count}/{total_positive}, {persistent_fn_ratio:.1%}) "
        f"remained false negative in all AI-positive splits (baseline, compressed, cropped). "
        "Most of these failures were associated with the absence of zero-variance ROIs and, in many cases, "
        "small or weak static contour regions, indicating that the implemented Optical Flow and Zero-Variance "
        "heuristics are not reliable positive indicators of AI generation in this dataset. "
        "A smaller extreme subset additionally exhibited almost no measurable motion and almost no optical-flow contours, "
        "which further suppresses heuristic evidence."
    )


if __name__ == "__main__":
    main()
