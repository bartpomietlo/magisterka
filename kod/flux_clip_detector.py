# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: DEPRECATED

from __future__ import annotations

import warnings

from ai_style_clip_detector import AIStyleCLIPDetector as FluxCLIPDetector

warnings.warn(
    "FluxCLIPDetector renamed to AIStyleCLIPDetector",
    DeprecationWarning,
    stacklevel=2,
)

