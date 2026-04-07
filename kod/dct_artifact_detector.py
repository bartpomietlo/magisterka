#!/usr/bin/env python3
# Created: 2026-03-31
# Part of: AI Video Detector v2 (CLIP+FFT integration)
# Status: ACTIVE

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


class DCTArtifactDetector:
    def __init__(
        self,
        blockiness_thr: float = 1.4,
        hf_suppression_thr: float = 0.12,
        n_frames: int = 10,
    ) -> None:
        self.blockiness_thr = float(blockiness_thr)
        self.hf_suppression_thr = float(hf_suppression_thr)
        self.n_frames = int(n_frames)

    def _sample_frames(self, path: str | Path) -> list[np.ndarray]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            return []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 1:
            idxs = [0] * self.n_frames
        else:
            idxs = np.linspace(0, total - 1, num=self.n_frames, dtype=int).tolist()
        frames: list[np.ndarray] = []
        for idx in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, fr = cap.read()
            if ok and fr is not None:
                frames.append(fr)
        cap.release()
        return frames

    @staticmethod
    def _blockiness(gray: np.ndarray) -> float:
        g = gray.astype(np.float32)
        gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)

        h, w = grad.shape
        yy, xx = np.indices((h, w))
        boundary = ((xx % 8) == 0) | ((yy % 8) == 0)
        interior = ~boundary
        b_mean = float(np.mean(grad[boundary])) if np.any(boundary) else 0.0
        i_mean = float(np.mean(grad[interior])) if np.any(interior) else 1e-6
        return b_mean / max(i_mean, 1e-6)

    @staticmethod
    def _hf_suppression(gray: np.ndarray) -> float:
        fr = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA).astype(np.float32)
        fr = fr - np.mean(fr)
        fft = np.fft.fft2(fr)
        mag = np.abs(np.fft.fftshift(fft))

        h, w = mag.shape
        cy, cx = h // 2, w // 2
        yy, xx = np.ogrid[:h, :w]
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr /= max(1e-6, 0.5 * min(h, w))

        low = mag[rr < 0.30]
        high = mag[(rr >= 0.30) & (rr <= 0.50)]
        low_e = float(np.mean(low)) if low.size else 1e-6
        high_e = float(np.mean(high)) if high.size else 0.0
        return high_e / max(low_e, 1e-6)

    def detect_video(self, path: str | Path) -> dict[str, Any]:
        frames = self._sample_frames(path)
        if not frames:
            return {
                "dct_score": 0,
                "dct_bonus": 0,
                "blockiness": 0.0,
                "hf_suppression": 0.0,
            }

        blk = []
        hf = []
        for fr in frames:
            gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            blk.append(self._blockiness(gray))
            hf.append(self._hf_suppression(gray))

        blockiness = float(np.mean(blk)) if blk else 0.0
        hf_suppression = float(np.mean(hf)) if hf else 0.0

        score = 0
        if blockiness > self.blockiness_thr:
            score += 1
        if hf_suppression < self.hf_suppression_thr:
            score += 1

        return {
            "dct_score": int(score),
            "dct_bonus": int(score),
            "blockiness": blockiness,
            "hf_suppression": hf_suppression,
        }

