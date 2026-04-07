from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class FluxDetectionResult:
    flux_detected: bool
    similarity: float
    similarity_std: float
    method: str


class FluxWatermark:
    """
    Detektor sygnatury watermark Flux (Grok/xAI Aurora) oparty o imwatermark.

    - laduje sygnature 64-bit dla dwtDctSvd i dwtDct z flux_signature.json
    - detect(frame) -> max similarity (0..1)
    - detect_video(video_path) -> agregacja na 10 klatkach
    """

    def __init__(
        self,
        signature_path: str | Path | None = None,
        threshold: float | None = None,
        frames_per_video: int = 10,
        min_frame_size: int = 256,
    ) -> None:
        self.signature_path = (
            Path(signature_path)
            if signature_path is not None
            else (Path(__file__).parent / "dataset" / "flux_signature.json")
        )
        self.frames_per_video = int(frames_per_video)
        self.min_frame_size = int(min_frame_size)

        self.sig_svd = np.zeros(64, dtype=np.int8)
        self.sig_dwt = np.zeros(64, dtype=np.int8)
        self.loaded = False
        self.safe_for_integration = True
        self.load_error = ""
        self.threshold = 0.55 if threshold is None else float(threshold)
        self._load_signature()

    def _load_signature(self) -> None:
        try:
            with self.signature_path.open("r", encoding="utf-8") as f:
                payload = json.load(f)
            self.sig_svd = np.array([int(c) for c in str(payload["dwtDctSvd"])[:64]], dtype=np.int8)
            self.sig_dwt = np.array([int(c) for c in str(payload["dwtDct"])[:64]], dtype=np.int8)
            if self.sig_svd.size < 64 or self.sig_dwt.size < 64:
                raise ValueError("sygnatura ma mniej niz 64 bity")
            if "optimal_threshold" in payload:
                self.threshold = float(payload["optimal_threshold"])
            elif "recommended_threshold" in payload:
                self.threshold = float(payload["recommended_threshold"])
            self.safe_for_integration = bool(payload.get("safe_for_integration", True))
            self.loaded = True
        except Exception as exc:  # noqa: BLE001
            self.loaded = False
            self.load_error = str(exc)

    @staticmethod
    def _sample_indices(total_frames: int, n_samples: int) -> np.ndarray:
        if total_frames <= 1:
            return np.zeros(n_samples, dtype=int)
        return np.linspace(0, total_frames - 1, num=n_samples, dtype=int)

    def _ensure_min_size(self, frame_bgr: np.ndarray) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if h >= self.min_frame_size and w >= self.min_frame_size:
            return frame_bgr
        scale = max(self.min_frame_size / max(h, 1), self.min_frame_size / max(w, 1))
        new_w = max(self.min_frame_size, int(round(w * scale)))
        new_h = max(self.min_frame_size, int(round(h * scale)))
        return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _hamming_similarity(bits: np.ndarray, signature_bits: np.ndarray) -> float:
        cmp_len = min(bits.size, signature_bits.size)
        if cmp_len <= 0:
            return 0.0
        return float(np.mean(bits[:cmp_len] == signature_bits[:cmp_len]))

    def _decode_bits(self, frame_bgr: np.ndarray, method: str) -> np.ndarray | None:
        if not self.loaded:
            return None
        try:
            from imwatermark import WatermarkDecoder  # type: ignore
        except Exception:
            return None
        frame = self._ensure_min_size(frame_bgr)
        try:
            decoder = WatermarkDecoder("bits", 64)
            bits = decoder.decode(frame[:, :, :3], method)
            if bits is None:
                return None
            arr = np.array([int(b) for b in bits], dtype=np.int8)
            if arr.size < 64:
                return None
            return arr[:64]
        except Exception:
            return None

    def _frame_similarity(self, frame_bgr: np.ndarray) -> tuple[float, float]:
        bits_svd = self._decode_bits(frame_bgr, "dwtDctSvd")
        bits_dwt = self._decode_bits(frame_bgr, "dwtDct")
        sim_svd = (
            self._hamming_similarity(bits_svd, self.sig_svd)
            if bits_svd is not None
            else 0.0
        )
        sim_dwt = (
            self._hamming_similarity(bits_dwt, self.sig_dwt)
            if bits_dwt is not None
            else 0.0
        )
        return float(sim_svd), float(sim_dwt)

    def detect(self, frame_bgr: np.ndarray) -> float:
        sim_svd, sim_dwt = self._frame_similarity(frame_bgr)
        return max(sim_svd, sim_dwt)

    def detect_frames(self, frames_bgr: list[np.ndarray]) -> FluxDetectionResult:
        if not self.loaded or not frames_bgr:
            return FluxDetectionResult(False, 0.0, 0.0, "none")

        svd_scores: list[float] = []
        dwt_scores: list[float] = []
        for frame in frames_bgr:
            sim_svd, sim_dwt = self._frame_similarity(frame)
            svd_scores.append(sim_svd)
            dwt_scores.append(sim_dwt)

        svd_arr = np.array(svd_scores, dtype=np.float32) if svd_scores else np.array([0.0], dtype=np.float32)
        dwt_arr = np.array(dwt_scores, dtype=np.float32) if dwt_scores else np.array([0.0], dtype=np.float32)
        med_svd = float(np.median(svd_arr))
        med_dwt = float(np.median(dwt_arr))

        if med_svd >= med_dwt:
            similarity = med_svd
            similarity_std = float(np.std(svd_arr))
            method = "dwtDctSvd"
        else:
            similarity = med_dwt
            similarity_std = float(np.std(dwt_arr))
            method = "dwtDct"

        detected = bool(self.safe_for_integration and similarity >= self.threshold)
        return FluxDetectionResult(detected, similarity, similarity_std, method)

    def detect_video(self, video_path: str | Path) -> dict[str, Any]:
        if not self.loaded:
            return {
                "flux_detected": False,
                "similarity": 0.0,
                "similarity_std": 0.0,
                "method": "none",
                "error": self.load_error or "signature_not_loaded",
                "threshold": self.threshold,
                "safe_for_integration": bool(self.safe_for_integration),
            }
        vp = Path(video_path)
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return {
                "flux_detected": False,
                "similarity": 0.0,
                "similarity_std": 0.0,
                "method": "none",
                "error": "video_open_failed",
                "threshold": self.threshold,
                "safe_for_integration": bool(self.safe_for_integration),
            }
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = self._sample_indices(total_frames, self.frames_per_video)
        frames: list[np.ndarray] = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if ok and frame is not None:
                frames.append(frame)
        cap.release()

        det = self.detect_frames(frames)
        return {
            "flux_detected": bool(det.flux_detected),
            "similarity": float(det.similarity),
            "similarity_std": float(det.similarity_std),
            "method": det.method,
            "error": "",
            "threshold": self.threshold,
            "safe_for_integration": bool(self.safe_for_integration),
        }
