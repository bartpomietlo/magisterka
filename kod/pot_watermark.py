from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

EmbeddingMethod = Literal["QIM", "M3"]


def compute_psnr(original_bgr: np.ndarray, watermarked_bgr: np.ndarray) -> float:
    """
    Oblicza PSNR pomiędzy oryginalnym i watermarkowanym kadrem (dla obrazu kolorowego BGR).

    Args:
        original_bgr: obraz wejściowy uint8, kształt (H,W,3)
        watermarked_bgr: obraz wynikowy uint8, kształt (H,W,3)

    Returns:
        PSNR w dB.
    """
    orig = original_bgr.astype(np.uint8)
    wm = watermarked_bgr.astype(np.uint8)
    return float(peak_signal_noise_ratio(orig, wm, data_range=255))


def compute_ssim(original_bgr: np.ndarray, watermarked_bgr: np.ndarray) -> float:
    """
    Oblicza SSIM pomiędzy oryginalnym i watermarkowanym kadrem (dla obrazu kolorowego BGR).

    Returns:
        SSIM w zakresie 0..1 (dla typowych obrazów naturalnych).
    """
    orig = original_bgr.astype(np.uint8)
    wm = watermarked_bgr.astype(np.uint8)
    return float(structural_similarity(orig, wm, data_range=255, channel_axis=2))


@dataclass(frozen=True)
class POTPositions:
    """
    Pozycje współczynników mid-frequency dla embed/decode w dziedzinie POT/F.
    """
    pos: tuple[tuple[int, int], ...] = ((2, 1), (1, 2), (3, 1), (1, 3))


class POTWatermark:
    """
    POT (Parametric Orthogonal Transform) watermarking dla kanału Y (luminancja w YCbCr / YCrCb).

    Wymagania implementacyjne:
    - Bloki 8x8 na kanale Y (pomijamy resztę na krawędziach)
    - Transformacja: F = M @ block @ M.T
    - Odwrotna: block_rec = M.T @ F_modified @ M
    - Współczynniki mid-frequency: (2,1), (1,2), (3,1), (1,3) – 4 współczynniki na bit
      (łącznie 32 bity => 8 bloków * 4 współczynniki = 32 bity w cyklu block_idx mod 32).
    - Payload: 32 bity = 16 magic + 8 frame_id + 8 CRC.
    - Metody osadzania: QIM lub M3.
    """

    MAGIC = 0xABCD
    ALPHA = 0.5
    BLOCK = 8
    PAYLOAD_BITS = 32
    QIM_BASE_STEP = 16

    def __init__(self, method: EmbeddingMethod = "QIM") -> None:
        """
        Args:
            method: domyślna metoda embed/decode.
        """
        self.default_method: EmbeddingMethod = method
        self.positions = POTPositions()
        self.M: np.ndarray = self._build_orthogonal_M()

    def _build_orthogonal_M(self) -> np.ndarray:
        """
        Buduje macierz M (8x8) z elementów exp(-alpha*|i-j|) oraz wymusza ortogonalność przez SVD.

        Returns:
            Ortogonalna macierz M o wymiarach (8,8).
        """
        n = self.BLOCK
        M_raw = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(n):
                M_raw[i, j] = math.exp(-self.ALPHA * abs(i - j))

        # Normalizacja (wariant "norm" jako norma Frobeniusa całości).
        norm = float(np.linalg.norm(M_raw) + 1e-12)
        M_raw = M_raw / norm

        # Ortogonalizacja: M = U @ Vt.
        U, _, Vt = np.linalg.svd(M_raw, full_matrices=False)
        M = U @ Vt
        return M.astype(np.float64)

    @staticmethod
    def _u16_to_bits_be(val: int) -> list[int]:
        """Zamienia uint16 na listę 16 bitów (MSB..LSB)."""
        return [int((val >> (15 - i)) & 1) for i in range(16)]

    @staticmethod
    def _u8_to_bits_be(val: int) -> list[int]:
        """Zamienia uint8 na listę 8 bitów (MSB..LSB)."""
        return [int((val >> (7 - i)) & 1) for i in range(8)]

    def _build_payload_bits(self, frame_id: int) -> list[int]:
        """
        Buduje 32-bitowy payload:
            16 magic (0xABCD),
            8 frame_id (mod 256),
            8 CRC = XOR bajtów magic_hi ^ magic_lo ^ frame_id.

        Returns:
            Lista 32 bitów (0/1), wewnątrz każdego pola MSB..LSB.
        """
        frame_id_u8 = int(frame_id) & 0xFF
        magic_bits = self._u16_to_bits_be(self.MAGIC)
        frame_bits = self._u8_to_bits_be(frame_id_u8)

        crc8 = ((self.MAGIC >> 8) & 0xFF) ^ (self.MAGIC & 0xFF) ^ frame_id_u8
        crc_bits = self._u8_to_bits_be(int(crc8))

        return magic_bits + frame_bits + crc_bits

    def _qim_step(self, strength: float) -> float:
        """
        Step size dla QIM.
        Domyślnie strength=8 => step=16 (zgodnie z wymaganiami).
        """
        base = float(self.QIM_BASE_STEP)
        scale = float(strength) / 8.0
        step = base * scale
        return max(1.0, step)

    def _modify_coeff(
        self,
        coeff: float,
        bit: int,
        method: EmbeddingMethod,
        strength: float,
    ) -> float:
        """
        Modyfikuje pojedynczy współczynnik F zgodnie z metodą embed.
        """
        bit_i = int(bit) & 1

        if method == "QIM":
            step = self._qim_step(strength)
            # Wymaganie:
            # q0 = round(coeff/step)*step
            # q1 = round((coeff - step/2)/step)*step + step/2
            q0 = float(np.round(coeff / step) * step)
            q1 = float(np.round((coeff - step / 2.0) / step) * step + step / 2.0)
            return q1 if bit_i == 1 else q0

        # method == "M3": adaptacyjne LSB mid-frequency współczynników.
        orig_int = int(np.round(coeff))
        new_int = orig_int
        if (orig_int & 1) != bit_i:
            new_int = orig_int + 1 if bit_i == 1 else orig_int - 1

        delta_int = new_int - orig_int  # zwykle -1/0/+1
        return float(coeff + float(strength) * float(delta_int))

    def _decode_coeff(
        self,
        coeff: float,
        method: EmbeddingMethod,
        strength: float,
    ) -> int:
        """
        Odczytuje bit z pojedynczego współczynnika F na podstawie metody.
        """
        if method == "QIM":
            step = self._qim_step(strength)
            q0 = float(np.round(coeff / step) * step)
            q1 = float(np.round((coeff - step / 2.0) / step) * step + step / 2.0)
            return 0 if abs(coeff - q0) <= abs(coeff - q1) else 1

        # method == "M3"
        coeff_int = int(np.round(coeff))
        return coeff_int & 1

    def embed(
        self,
        frame_bgr: np.ndarray,
        frame_id: int = 0,
        method: EmbeddingMethod | str | None = "QIM",
        strength: float = 8.0,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Osadza payload w obrazie przez POT na kanale Y.

        Args:
            frame_bgr: obraz BGR uint8.
            frame_id: 8-bitowy identyfikator ramki (mod 256).
            method: "QIM" albo "M3".
            strength: siła embed.

        Returns:
            (frame_bgr_watermarked, info_dict) gdzie info_dict zawiera:
                - payload_bits: list[int]
                - blocks_used: int
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr musi być obrazem BGR o kształcie (H,W,3+)")

        embed_method: EmbeddingMethod = (
            self.default_method
            if method is None
            else (str(method).upper())  # type: ignore[assignment]
        )  # type: ignore[arg-type]
        if embed_method not in {"QIM", "M3"}:
            raise ValueError("method musi być 'QIM' albo 'M3'")

        h, w = frame_bgr.shape[:2]
        if h < self.BLOCK or w < self.BLOCK:
            # Obsługa przypadku, gdy nie ma nawet jednego bloku 8x8.
            payload_bits = self._build_payload_bits(frame_id)
            return frame_bgr, {
                "payload_bits": payload_bits,
                "blocks_used": 0,
                "error": f"Klatka za mała: {h}x{w} (wymagane >= {self.BLOCK}x{self.BLOCK})",
            }

        payload_bits = self._build_payload_bits(frame_id)

        # BGR -> YCrCb
        ycrcb = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)

        h_blocks = h // self.BLOCK
        w_blocks = w // self.BLOCK
        y_out = y.copy()

        block_idx = 0
        blocks_used = 0

        for by in range(h_blocks):
            for bx in range(w_blocks):
                block = y_out[by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8]
                # F = M @ block @ M.T
                F = self.M @ block @ self.M.T

                bit_pos = block_idx % self.PAYLOAD_BITS
                bit = payload_bits[bit_pos]

                # Embed w 4 mid-frequency pozycjach:
                for (i, j) in self.positions.pos:
                    F[i, j] = self._modify_coeff(
                        float(F[i, j]),
                        bit=bit,
                        method=embed_method,
                        strength=float(strength),
                    )

                # inverse: block_rec = M.T @ F_modified @ M
                block_rec = self.M.T @ F @ self.M
                y_out[by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8] = block_rec

                block_idx += 1
                blocks_used += 1

        y_out_u8 = np.clip(np.round(y_out), 0, 255).astype(np.uint8)
        ycrcb[:, :, 0] = y_out_u8
        frame_bgr_watermarked = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return frame_bgr_watermarked, {
            "payload_bits": payload_bits,
            "blocks_used": blocks_used,
        }

    def decode(
        self,
        frame_bgr: np.ndarray,
        method: EmbeddingMethod | str | None = "QIM",
        strength: float = 8.0,
    ) -> dict[str, Any]:
        """
        Dekoduje payload z watermarkowanego kadru przez majority voting.

        Args:
            frame_bgr: obraz BGR uint8.
            method: "QIM" albo "M3".
            strength: musi być spójny z embed (dla QIM).

        Returns:
            dict:
                - detected: bool
                - magic_ok: bool
                - crc_ok: bool
                - ber: float
                - payload_bits: list[int] (32 bity)
                - confidence: float (0..1)
                - blocks_decoded: int
                - score: float  (confidence gdy detected=True, else 0.0)
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr musi być obrazem BGR o kształcie (H,W,3+)")

        dec_method: EmbeddingMethod = (
            self.default_method
            if method is None
            else (str(method).upper())  # type: ignore[assignment]
        )  # type: ignore[arg-type]
        if dec_method not in {"QIM", "M3"}:
            raise ValueError("method musi być 'QIM' albo 'M3'")

        h, w = frame_bgr.shape[:2]
        if h < self.BLOCK or w < self.BLOCK:
            payload_bits = [0] * self.PAYLOAD_BITS
            return {
                "detected": False,
                "magic_ok": False,
                "crc_ok": False,
                "ber": 0.0,
                "payload_bits": payload_bits,
                "confidence": 0.0,
                "blocks_decoded": 0,
                "score": 0.0,
            }

        ycrcb = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)

        h_blocks = h // self.BLOCK
        w_blocks = w // self.BLOCK
        votes: list[list[int]] = [[] for _ in range(self.PAYLOAD_BITS)]
        blocks_decoded = 0

        block_idx = 0
        for by in range(h_blocks):
            for bx in range(w_blocks):
                block = y[by * 8 : by * 8 + 8, bx * 8 : bx * 8 + 8]
                F = self.M @ block @ self.M.T

                bit_pos = block_idx % self.PAYLOAD_BITS

                # 4 pozycje kodują ten sam bit => zbieramy 4 głosy na bit.
                for (i, j) in self.positions.pos:
                    b = self._decode_coeff(
                        float(F[i, j]),
                        method=dec_method,
                        strength=float(strength),
                    )
                    votes[bit_pos].append(b)

                block_idx += 1
                blocks_decoded += 1

        payload_out: list[int] = []
        confidence_parts: list[float] = []
        ber_parts: list[float] = []

        for bit_pos in range(self.PAYLOAD_BITS):
            v = votes[bit_pos]
            if not v:
                payload_out.append(0)
                confidence_parts.append(0.0)
                ber_parts.append(1.0)
                continue

            ones = int(sum(v))
            zeros = len(v) - ones
            majority_bit = 1 if ones > zeros else 0
            payload_out.append(majority_bit)

            total = float(len(v))
            agree = float(max(ones, zeros)) / total
            confidence_parts.append(agree)

            # BER: średnia frakcja niezgody wobec większości.
            disagree = float(min(ones, zeros)) / total
            ber_parts.append(disagree)

        confidence = float(np.mean(confidence_parts)) if confidence_parts else 0.0
        ber = float(np.mean(ber_parts)) if ber_parts else 0.0

        # magic + crc na podstawie 32 bitów
        magic_bits = payload_out[:16]
        frame_id_bits = payload_out[16:24]
        crc_bits = payload_out[24:32]

        magic_dec = 0
        for b in magic_bits:
            magic_dec = (magic_dec << 1) | int(b)

        frame_id_dec = 0
        for b in frame_id_bits:
            frame_id_dec = (frame_id_dec << 1) | int(b)

        crc_dec = 0
        for b in crc_bits:
            crc_dec = (crc_dec << 1) | int(b)

        crc_expected = ((self.MAGIC >> 8) & 0xFF) ^ (self.MAGIC & 0xFF) ^ (frame_id_dec & 0xFF)

        magic_ok = bool(magic_dec == self.MAGIC)
        crc_ok = bool(crc_dec == (crc_expected & 0xFF))
        detected = bool(magic_ok and crc_ok)

        score = float(confidence) if detected else 0.0

        return {
            "detected": detected,
            "magic_ok": magic_ok,
            "crc_ok": crc_ok,
            "ber": ber,
            "payload_bits": payload_out,
            "confidence": confidence,
            "blocks_decoded": blocks_decoded,
            "score": score,
        }


if __name__ == "__main__":
    # Prosty test jednostkowy "clean" na losowym obrazie.
    rng = np.random.default_rng(12345)
    test_frame = rng.integers(low=0, high=256, size=(480, 640, 3), dtype=np.uint8)

    wm = POTWatermark(method="QIM")
    frame_wm, info = wm.embed(test_frame, frame_id=0, method="QIM", strength=8.0)
    dec = wm.decode(frame_wm, method="QIM", strength=8.0)

    payload_true = info["payload_bits"]
    payload_hat = dec["payload_bits"]
    ber_direct = float(np.mean(np.array(payload_true, dtype=np.int8) != np.array(payload_hat, dtype=np.int8)))

    psnr = compute_psnr(test_frame, frame_wm)
    ssim = compute_ssim(test_frame, frame_wm)

    print("[POT] clean test")
    print(f"  magic_ok={dec['magic_ok']} crc_ok={dec['crc_ok']} detected={dec['detected']}")
    print(f"  confidence={dec['confidence']:.4f} blocks_decoded={dec['blocks_decoded']}")
    print(f"  PSNR={psnr:.3f} dB  SSIM={ssim:.5f}")
    print(f"  BER(direct)={ber_direct:.6f} BER(estimated)={dec['ber']:.6f}")

    if not (dec["magic_ok"] and dec["crc_ok"]):
        raise SystemExit("POT watermark clean test FAILED (magic_ok/crc_ok).")

    if psnr <= 30.0:
        raise SystemExit(f"POT watermark clean test FAILED: PSNR={psnr:.3f} <= 30dB")

    if ber_direct >= 0.1:
        raise SystemExit(f"POT watermark clean test FAILED: BER={ber_direct:.6f} >= 0.1")

    print("[POT] clean test PASSED")
