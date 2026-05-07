from __future__ import annotations

from typing import Any, Literal

import cv2
import numpy as np

try:
    from .pot_watermark import compute_psnr
except ImportError:
    from pot_watermark import compute_psnr

TransformType = Literal["NONEXP", "EXP", "SPARSE_NONEXP"]


class AGHWatermark:
    """
    Watermarking in new orthogonal transform domains from Bogacki & Dziech (2022).

    The implementation uses 8x8 luminance blocks, grouped spectral channels, and
    adaptive QIM channel selection corresponding to Method 5 from Section 3.5.
    """

    MAGIC = 0xABCD
    BLOCK = 8
    PAYLOAD_BITS = 32
    QIM_BASE_STEP = 16.0
    ROBUST_CHANNELS: dict[str, tuple[tuple[int, int], ...]] = {
        # Adaptive channel ranking is not stable after resize/JPEG. DC is less
        # elegant than the middle-channel M5 choice, but it keeps embed/extract
        # synchronized under low-pass attacks.
        "NONEXP": ((0, 0),),
        "EXP": ((0, 0),),
        "SPARSE_NONEXP": ((0, 0),),
    }
    STEP_MULTIPLIERS: dict[str, float] = {
        # AGH coefficients live on a different scale than the DCT QIM baseline.
        # These defaults keep strength=8 useful after resize_0.5. SPARSE_NONEXP
        # needs a stronger step because its DC quantization is less stable after
        # low-pass resampling.
        "NONEXP": 12.0,
        "EXP": 6.0,
        "SPARSE_NONEXP": 18.0,
    }
    SLOT_SEEDS: dict[str, int] = {
        "NONEXP": 104729,
        "EXP": 130363,
        "SPARSE_NONEXP": 155921,
    }

    def __init__(self, transform_type: TransformType | str = "NONEXP") -> None:
        self.default_transform_type = self._normalize_transform_type(transform_type)
        self._matrix_cache: dict[str, tuple[np.ndarray, float]] = {}

    @staticmethod
    def _normalize_transform_type(transform_type: TransformType | str | None) -> TransformType:
        if transform_type is None:
            return "NONEXP"
        normalized = str(transform_type).upper()
        if normalized == "AGH_NONEXP":
            normalized = "NONEXP"
        elif normalized == "AGH_EXP":
            normalized = "EXP"
        elif normalized in {"AGH_SPARSE", "NONEXP_SPARSE"}:
            normalized = "SPARSE_NONEXP"
        if normalized not in {"NONEXP", "EXP", "SPARSE_NONEXP"}:
            raise ValueError("transform_type musi byc NONEXP, EXP albo SPARSE_NONEXP")
        return normalized  # type: ignore[return-value]

    @staticmethod
    def _u16_to_bits_be(val: int) -> list[int]:
        return [int((val >> (15 - i)) & 1) for i in range(16)]

    @staticmethod
    def _u8_to_bits_be(val: int) -> list[int]:
        return [int((val >> (7 - i)) & 1) for i in range(8)]

    @staticmethod
    def _bytes_to_bits(payload: bytes) -> list[int]:
        return [int((byte >> (7 - i)) & 1) for byte in payload for i in range(8)]

    @staticmethod
    def _bits_to_bytes(bits: list[int]) -> bytes:
        out = bytearray()
        for start in range(0, len(bits), 8):
            byte = 0
            chunk = bits[start : start + 8]
            for bit in chunk:
                byte = (byte << 1) | (int(bit) & 1)
            if len(chunk) < 8:
                byte <<= 8 - len(chunk)
            out.append(byte)
        return bytes(out)

    def _build_payload_bits(self, frame_id: int) -> list[int]:
        frame_id_u8 = int(frame_id) & 0xFF
        crc8 = ((self.MAGIC >> 8) & 0xFF) ^ (self.MAGIC & 0xFF) ^ frame_id_u8
        return (
            self._u16_to_bits_be(self.MAGIC)
            + self._u8_to_bits_be(frame_id_u8)
            + self._u8_to_bits_be(crc8)
        )

    @staticmethod
    def _nonexp_basis(size: int, first_three: tuple[float, float, float] = (1.0, 2.0, 3.0)) -> np.ndarray:
        """
        Builds the non-exponential basis sequence.

        Eq. (1)-(4) from Bogacki & Dziech 2022: after choosing the first
        three elements, subsequent elements satisfy a_i * a_{i+3} =
        a_{i+1} * a_{i+2}.
        """
        basis = np.zeros(size, dtype=np.float64)
        basis[:3] = np.asarray(first_three, dtype=np.float64)
        for i in range(size - 3):
            denom = basis[i] if abs(basis[i]) > 1e-12 else 1e-12
            basis[i + 3] = basis[i + 1] * basis[i + 2] / denom
        return basis

    @staticmethod
    def _exp_basis(size: int, a: float = 1.0) -> np.ndarray:
        """Eq. (10) from Bogacki & Dziech 2022: exponential basis a^k."""
        return np.asarray([float(a) ** k for k in range(size)], dtype=np.float64)

    @classmethod
    def _signed_xor_matrix(cls, basis: np.ndarray) -> np.ndarray:
        """
        Builds a symmetric generalized Hadamard-like matrix from the basis.

        This is the recursive structure of Eq. (1)-(4): entries use the basis
        element indexed by i xor j and the same sign pattern as the recursive
        orthogonal construction.
        """
        size = int(len(basis))
        matrix = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                sign = cls._recursive_sign(i, j)
                matrix[i, j] = sign * basis[i ^ j]
        return matrix

    @staticmethod
    def _recursive_sign(i: int, j: int) -> float:
        sign = 1
        while i or j:
            if (i & 1) and (j & 1):
                sign *= -1
            i >>= 1
            j >>= 1
        return float(sign)

    @classmethod
    def _build_nonexp_matrix(cls, n: int = 3) -> np.ndarray:
        size = 2**n
        return cls._signed_xor_matrix(cls._nonexp_basis(size))

    @classmethod
    def _build_exp_matrix(cls, n: int = 3, a: float = 1.0) -> np.ndarray:
        size = 2**n
        return cls._signed_xor_matrix(cls._exp_basis(size, a=a))

    @classmethod
    def _build_sparse_nonexp_matrix(cls, n: int = 3, m: int = 2) -> np.ndarray:
        """
        Eq. (12)-(13) from Bogacki & Dziech 2022: sparse matrix with smaller
        NONEXP transform matrices positioned on the diagonal.
        """
        size = 2**n
        sub_size = 2**m
        matrix = np.zeros((size, size), dtype=np.float64)
        sub = cls._signed_xor_matrix(cls._nonexp_basis(sub_size))
        for start in range(0, size, sub_size):
            matrix[start : start + sub_size, start : start + sub_size] = sub
        return matrix

    @classmethod
    def build_transform_matrix(cls, transform_type: TransformType | str) -> np.ndarray:
        """
        Creates one of the three matrices selected in Section 4.2 and applies
        preprocessing from Section 4.3: divide all elements by the first-row sum.
        """
        normalized = cls._normalize_transform_type(transform_type)
        if normalized == "NONEXP":
            matrix = cls._build_nonexp_matrix(n=3)
        elif normalized == "EXP":
            matrix = cls._build_exp_matrix(n=3, a=1.0)
        else:
            matrix = cls._build_sparse_nonexp_matrix(n=3, m=2)

        row_sum = float(np.sum(matrix[0]))
        if abs(row_sum) < 1e-12:
            raise ValueError("Nie mozna znormalizowac macierzy: suma pierwszego wiersza wynosi 0")
        return (matrix / row_sum).astype(np.float64)

    def _get_transform(self, transform_type: TransformType | str | None) -> tuple[np.ndarray, float]:
        normalized = self._normalize_transform_type(transform_type or self.default_transform_type)
        if normalized not in self._matrix_cache:
            matrix = self.build_transform_matrix(normalized)
            c_norm = float(np.sum(matrix[0] ** 2))
            self._matrix_cache[normalized] = (matrix, c_norm)
        return self._matrix_cache[normalized]

    def _qim_step(self, strength: float, transform_type: TransformType | str | None = None) -> float:
        transform_name = self._normalize_transform_type(transform_type or self.default_transform_type)
        multiplier = self.STEP_MULTIPLIERS[transform_name]
        return max(1.0, self.QIM_BASE_STEP * multiplier * float(strength) / 8.0)

    def _modify_coeff(
        self,
        coeff: float,
        bit: int,
        strength: float,
        transform_type: TransformType | str | None = None,
    ) -> float:
        step = self._qim_step(strength, transform_type)
        q0 = float(np.round(coeff / step) * step)
        q1 = float(np.round((coeff - step / 2.0) / step) * step + step / 2.0)
        return q1 if (int(bit) & 1) else q0

    def _decode_coeff(
        self,
        coeff: float,
        strength: float,
        transform_type: TransformType | str | None = None,
    ) -> int:
        step = self._qim_step(strength, transform_type)
        q0 = float(np.round(coeff / step) * step)
        q1 = float(np.round((coeff - step / 2.0) / step) * step + step / 2.0)
        return 0 if abs(coeff - q0) <= abs(coeff - q1) else 1

    def _forward(self, matrix: np.ndarray, block: np.ndarray, c_norm: float) -> np.ndarray:
        """Forward 2D transform, Eq. (25), normalized by C."""
        return (matrix @ block @ matrix.T) / c_norm

    def _inverse(self, matrix: np.ndarray, spectrum: np.ndarray, c_norm: float) -> np.ndarray:
        """Inverse 2D transform using the same symmetric matrix and coefficient C."""
        return (matrix.T @ spectrum @ matrix) / c_norm

    def _block_view(self, y: np.ndarray) -> tuple[np.ndarray, int, int]:
        h, w = y.shape[:2]
        h_blocks = h // self.BLOCK
        w_blocks = w // self.BLOCK
        blocks = np.empty((h_blocks * w_blocks, self.BLOCK, self.BLOCK), dtype=np.float64)
        idx = 0
        for by in range(h_blocks):
            for bx in range(w_blocks):
                blocks[idx] = y[by * self.BLOCK : by * self.BLOCK + self.BLOCK, bx * self.BLOCK : bx * self.BLOCK + self.BLOCK]
                idx += 1
        return blocks, h_blocks, w_blocks

    def _merge_blocks(self, y_base: np.ndarray, blocks: np.ndarray, h_blocks: int, w_blocks: int) -> np.ndarray:
        y_out = y_base.copy()
        idx = 0
        for by in range(h_blocks):
            for bx in range(w_blocks):
                y_out[by * self.BLOCK : by * self.BLOCK + self.BLOCK, bx * self.BLOCK : bx * self.BLOCK + self.BLOCK] = blocks[idx]
                idx += 1
        return y_out

    @staticmethod
    def _select_channels(spectra: np.ndarray) -> list[tuple[int, int]]:
        """
        Groups 64 spectral channels as in Section 3.2 and selects middle channels
        after sorting by ascending mean absolute coefficient value.
        """
        means = np.mean(np.abs(spectra), axis=0).reshape(-1)
        order = np.argsort(means)
        margin = max(1, int(round(0.10 * len(order))))
        selected = order[margin : len(order) - margin]
        return [(int(idx // 8), int(idx % 8)) for idx in selected]

    def _robust_channels(self, transform_type: TransformType | str | None) -> tuple[tuple[int, int], ...]:
        transform_name = self._normalize_transform_type(transform_type or self.default_transform_type)
        return self.ROBUST_CHANNELS[transform_name]

    def _slot_order(self, n_blocks: int, n_channels: int, transform_type: TransformType | str | None) -> np.ndarray:
        transform_name = self._normalize_transform_type(transform_type or self.default_transform_type)
        slots = np.arange(int(n_blocks) * int(n_channels), dtype=np.int64)
        rng = np.random.default_rng(self.SLOT_SEEDS[transform_name])
        rng.shuffle(slots)
        return slots

    def embed(
        self,
        frame_bgr: np.ndarray,
        frame_id: int | bytes = 0,
        method: str | None = None,
        strength: float = 8.0,
        bit_payload: bytes | None = None,
        transform_type: TransformType | str | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Embeds bytes using Method 5 adaptive QIM.

        The first four parameters intentionally mirror POTWatermark.embed().
        For AGH, method/transform_type selects NONEXP, EXP, or SPARSE_NONEXP.
        """
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr musi byc obrazem BGR o ksztalcie (H,W,3+)")

        transform_name = self._normalize_transform_type(transform_type or method or self.default_transform_type)
        if isinstance(frame_id, (bytes, bytearray)) and bit_payload is None:
            bit_payload = bytes(frame_id)
            frame_id = 0
        payload_bits = self._bytes_to_bits(bit_payload) if bit_payload is not None else self._build_payload_bits(int(frame_id))

        h, w = frame_bgr.shape[:2]
        if h < self.BLOCK or w < self.BLOCK or not payload_bits:
            return frame_bgr.copy(), {
                "psnr": float("inf"),
                "transform_type": transform_name,
                "bits_embedded": 0,
                "payload_bits": payload_bits,
                "blocks_used": 0,
            }

        matrix, c_norm = self._get_transform(transform_name)
        ycrcb = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)
        blocks, h_blocks, w_blocks = self._block_view(y)

        spectra = np.empty_like(blocks)
        for idx, block in enumerate(blocks):
            spectra[idx] = self._forward(matrix, block, c_norm)

        channels = self._robust_channels(transform_name)
        slot_order = self._slot_order(spectra.shape[0], len(channels), transform_name)
        capacity = int(spectra.shape[0] * len(channels))
        write_limit = min(capacity, len(payload_bits) * 32)
        write_count = 0
        for slot in slot_order[:write_limit]:
            block_idx = int(slot) // len(channels)
            channel_idx = int(slot) % len(channels)
            row, col = channels[channel_idx]
            bit = payload_bits[write_count % len(payload_bits)]
            spectra[block_idx, row, col] = self._modify_coeff(
                float(spectra[block_idx, row, col]),
                bit,
                strength,
                transform_name,
            )
            write_count += 1

        rec_blocks = np.empty_like(blocks)
        for idx, spectrum in enumerate(spectra):
            rec_blocks[idx] = self._inverse(matrix, spectrum, c_norm)

        y_out = self._merge_blocks(y, rec_blocks, h_blocks, w_blocks)
        ycrcb[:, :, 0] = np.clip(np.round(y_out), 0, 255).astype(np.uint8)
        watermarked = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        return watermarked, {
            "psnr": compute_psnr(frame_bgr[:, :, :3], watermarked),
            "transform_type": transform_name,
            "bits_embedded": min(len(payload_bits), capacity),
            "coefficients_modified": write_count,
            "qim_step": self._qim_step(strength, transform_name),
            "channels": channels,
            "payload_bits": payload_bits,
            "blocks_used": int(blocks.shape[0]),
        }

    def extract(
        self,
        frame_bgr: np.ndarray,
        n_bits: int,
        transform_type: TransformType | str | None = None,
        strength: float = 8.0,
        method: str | None = None,
    ) -> bytes:
        """Extracts n_bits embedded by Method 5 adaptive QIM and returns bytes."""
        bits = self._extract_bits(
            frame_bgr=frame_bgr,
            n_bits=int(n_bits),
            transform_type=transform_type or method or self.default_transform_type,
            strength=float(strength),
        )
        return self._bits_to_bytes(bits)

    def _extract_bits(
        self,
        frame_bgr: np.ndarray,
        n_bits: int,
        transform_type: TransformType | str | None,
        strength: float,
    ) -> list[int]:
        if frame_bgr is None or frame_bgr.ndim != 3 or frame_bgr.shape[2] < 3:
            raise ValueError("frame_bgr musi byc obrazem BGR o ksztalcie (H,W,3+)")
        if n_bits <= 0:
            return []

        transform_name = self._normalize_transform_type(transform_type or self.default_transform_type)
        matrix, c_norm = self._get_transform(transform_name)
        h, w = frame_bgr.shape[:2]
        if h < self.BLOCK or w < self.BLOCK:
            return [0] * n_bits

        ycrcb = cv2.cvtColor(frame_bgr[:, :, :3], cv2.COLOR_BGR2YCrCb)
        y = ycrcb[:, :, 0].astype(np.float64)
        blocks, _, _ = self._block_view(y)

        spectra = np.empty_like(blocks)
        for idx, block in enumerate(blocks):
            spectra[idx] = self._forward(matrix, block, c_norm)

        channels = self._robust_channels(transform_name)
        slot_order = self._slot_order(spectra.shape[0], len(channels), transform_name)
        votes: list[list[int]] = [[] for _ in range(n_bits)]
        read_limit = min(int(spectra.shape[0] * len(channels)), n_bits * 32)
        for read_count, slot in enumerate(slot_order[:read_limit]):
            block_idx = int(slot) // len(channels)
            channel_idx = int(slot) % len(channels)
            row, col = channels[channel_idx]
            bit = self._decode_coeff(float(spectra[block_idx, row, col]), strength, transform_name)
            votes[read_count % n_bits].append(bit)

        out_bits: list[int] = []
        for bit_votes in votes:
            if not bit_votes:
                out_bits.append(0)
                continue
            ones = int(sum(bit_votes))
            zeros = len(bit_votes) - ones
            out_bits.append(1 if ones > zeros else 0)
        return out_bits

    def decode(
        self,
        frame_bgr: np.ndarray,
        method: str | None = None,
        strength: float = 8.0,
    ) -> dict[str, Any]:
        """POTWatermark-compatible detector wrapper for the default 32-bit payload."""
        payload_out = self._extract_bits(
            frame_bgr=frame_bgr,
            n_bits=self.PAYLOAD_BITS,
            transform_type=method or self.default_transform_type,
            strength=float(strength),
        )

        magic_dec = 0
        for b in payload_out[:16]:
            magic_dec = (magic_dec << 1) | int(b)

        frame_id_dec = 0
        for b in payload_out[16:24]:
            frame_id_dec = (frame_id_dec << 1) | int(b)

        crc_dec = 0
        for b in payload_out[24:32]:
            crc_dec = (crc_dec << 1) | int(b)

        crc_expected = ((self.MAGIC >> 8) & 0xFF) ^ (self.MAGIC & 0xFF) ^ (frame_id_dec & 0xFF)
        magic_ok = bool(magic_dec == self.MAGIC)
        crc_ok = bool(crc_dec == (crc_expected & 0xFF))
        detected = bool(magic_ok and crc_ok)

        h, w = frame_bgr.shape[:2]
        blocks_decoded = int((h // self.BLOCK) * (w // self.BLOCK)) if frame_bgr is not None else 0
        return {
            "detected": detected,
            "magic_ok": magic_ok,
            "crc_ok": crc_ok,
            "ber": 0.0 if detected else 1.0,
            "payload_bits": payload_out,
            "confidence": 1.0 if detected else 0.0,
            "blocks_decoded": blocks_decoded,
            "score": 1.0 if detected else 0.0,
        }
