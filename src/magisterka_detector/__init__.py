"""Magisterka Detector - AI/Deepfake Detection System"""

__version__ = "0.2.0"

from .api import analyze_media, analyze_batch, begin_run
from .types import (
    AnalyzeOptions,
    AnalysisFeatures,
    AnalysisResult,
    DetectMode,
    MediaKind,
    PolicyName,
)

__all__ = [
    "analyze_media",
    "analyze_batch",
    "begin_run",
    "AnalyzeOptions",
    "AnalysisFeatures",
    "AnalysisResult",
    "DetectMode",
    "MediaKind",
    "PolicyName",
]
