# Codex Task: Improve AI Video Detection Precision

## Context

This is a master's thesis project for detecting AI-generated video using heuristic signals:
- **Optical Flow (OF)** — detects regions of motion between frames
- **Zero-Variance (ZV)** — detects pixel regions that never change (static overlays)
- **C2PA / Invisible watermark** — cryptographic metadata (currently always returns 0)

The main detector logic is in `kod/advanced_detectors.py`.
The evaluation pipeline is in `kod/dataset/evaluate.py`.
Raw signal output is in `kod/results/latest/raw_signals.csv`.

## Problem (from fn_diagnosis_v2.py output)

**100% of AI source videos are persistent False Negatives** — detected=0 across all splits.

Dominant failure patterns:
- `no zero-variance ROIs` — 96.3% of FN videos (zv_count=0 everywhere)
- `small static contour area` — 59.3% of FN videos (of_max_area too small)
- `very few optical-flow contours` — 7.4% of FN videos

Key insight from data:
- Most AI videos have **of_count > 0** (e.g. 17–49 contours) but are still FN
- Most AI videos have **of_global_motion > 10** — there IS motion, but it is not being
  used to classify as AI
- zv_count is 0 for 26/27 videos — the ZV detector is not triggering at all
- Current decision rule: `detected = 1` only if `optical_flow_rois > 0` AND threshold
  conditions met — but this is the SAME signal that also fires on TV overlays (FP trap)

## Root Cause

The current heuristic is **symmetric**: it cannot distinguish AI-generated motion patterns
from real-camera or TV-overlay motion patterns. The OF/ZV signals fire on both or neither.

## What Codex Should Implement

### 1. New feature: `of_texture_variance` (most important)

In `kod/advanced_detectors.py`, inside the Optical Flow analysis function:
- For each detected motion contour (ROI), compute the **local texture variance** of the
  pixels inside that contour using `cv2.Laplacian` or pixel std.
- AI-generated video tends to have **lower local texture variance** inside motion regions
  (smoother, more uniform pixel patches) compared to real camera footage.
- Add a new output field: `of_texture_variance_mean` (mean over all ROI patches)
- Add: `of_low_texture_roi_count` (number of ROIs with variance below threshold=50)

### 2. New feature: `freq_domain_score` (second most important)

In `kod/advanced_detectors.py`:
- Sample N=5 frames evenly from the video
- For each frame, compute 2D FFT and analyze the **high-frequency energy ratio**:
  `hf_ratio = energy(f > 0.3 * nyquist) / total_energy`
- AI-generated video typically has **lower high-frequency energy** (smoother textures)
  than real camera footage (which has natural grain/noise)
- Add output field: `freq_hf_ratio_mean` (mean over sampled frames)

### 3. New decision rule in `evaluate.py`

Replace the current binary `detected` column logic with a **multi-signal score**:

```python
def compute_ai_score(row):
    score = 0
    # Signal 1: many OF contours (but not huge area — that's TV overlay)
    if row['of_count'] >= 5 and row['of_max_area'] < 500_000:
        score += 1
    # Signal 2: low texture variance inside motion regions (AI smoothness)
    if row.get('of_low_texture_roi_count', 0) >= 2:
        score += 2
    # Signal 3: low high-frequency energy (AI smoothness in frequency domain)
    if row.get('freq_hf_ratio_mean', 1.0) < 0.15:
        score += 2
    # Signal 4: zero-variance regions present (static AI watermark/logo)
    if row['zv_count'] >= 1:
        score += 1
    return score

# Threshold: detected=1 if score >= 3
row['detected'] = int(compute_ai_score(row) >= 3)
```

### 4. Update `raw_signals.csv` schema

Add the two new columns to the CSV output in `evaluate.py`:
- `of_texture_variance_mean`
- `of_low_texture_roi_count`  
- `freq_hf_ratio_mean`

### 5. Update `kod/dataset/evaluate.py`

- Call the new feature extraction functions from `advanced_detectors.py`
- Pass results to `build_row()` function
- Keep backward compatibility: if new features fail (exception), default to 0

## What NOT to change

- Do NOT change `plot_results.py` or `plot_results_v2.py`
- Do NOT change `fn_diagnosis_v2.py` or `compare_external_apps.py`
- Do NOT change the dataset download scripts
- Do NOT change `gui.py` (UI is separate concern)
- Keep existing CSV columns intact — only ADD new columns

## Expected outcome

After these changes, re-running `evaluate.py` should produce a `raw_signals.csv` where:
- AI videos have higher `ai_score` due to low texture variance + low HF energy
- TV/FP trap videos have lower `ai_score` because their motion regions have high texture
  variance (real camera footage, natural grain)
- Persistent FN rate should drop from 100% toward 60-70%
- FPR on `adv_fp_trap` should stay below 20%

## Files to modify

1. `kod/advanced_detectors.py` — add `of_texture_variance_mean`, `of_low_texture_roi_count`, `freq_hf_ratio_mean`
2. `kod/dataset/evaluate.py` — integrate new features, update `build_row()`, new `compute_ai_score()`

## Files to create

None required — all changes go into existing files.

## Testing

After implementing:
```bash
python -m py_compile kod/advanced_detectors.py
python -m py_compile kod/dataset/evaluate.py
python kod/dataset/evaluate.py  # should complete without error
```

Check that `raw_signals.csv` has the 3 new columns.
