"""
advanced_detectors.py

Zaawansowane metody detekcji znakow wodnych:
1. Temporal Median Filtering  – wydobywa statyczny znak wodny z sekwencji klatek
2. Invisible Watermark        – dekoduje ukryty token DWT/DWT-DCT/RivaGAN (imwatermark)
3. Noise Residual / FFT       – wykrywa periodyczne artefakty upsamplingu AI
4. Zero-Variance ROI          – wykrywa regiony bez zmian w czasie (statyczny overlay)

Kazda metoda zwraca slownik z kluczami:
  found   : bool
  method  : str
  details : str       (opis znaleziska)
  score   : float     (0-1, pewnosc)
  image   : np.ndarray | None   (obraz diagnostyczny, opcjonalny)
"""

from __future__ import annotations

import sys
from typing import List, Optional, Dict, Any

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# 1. TEMPORAL MEDIAN FILTERING
# ---------------------------------------------------------------------------

def build_temporal_median(
    frames: List[np.ndarray],
    max_frames: int = 50
) -> np.ndarray:
    """
    Oblicza mediane po osi czasu dla listy klatek BGR.
    Ruchome obiekty 'znikaja', statyczny znak wodny pozostaje ostry.
    """
    if not frames:
        raise ValueError("Pusta lista klatek")

    if len(frames) > max_frames:
        step = len(frames) // max_frames
        frames = frames[::step][:max_frames]

    stack = np.stack([f.astype(np.float32) for f in frames], axis=0)
    median_frame = np.median(stack, axis=0).astype(np.uint8)
    return median_frame


def extract_static_overlay(
    median_frame: np.ndarray,
    reference_frame: np.ndarray,
    amp: float = 4.0
) -> np.ndarray:
    """
    Odejmuje medianę od klatki referencyjnej i wzmacnia różnice.
    Efekt: statyczny overlay (znak wodny) staje się widoczny.
    """
    diff = cv2.absdiff(reference_frame.astype(np.float32),
                       median_frame.astype(np.float32))
    diff = np.clip(diff * amp, 0, 255).astype(np.uint8)
    return diff


def detect_zero_variance_rois(
    frames: List[np.ndarray],
    corner_ratio: float = 0.20,
    variance_threshold: float = 8.0,
    min_fraction: float = 0.30
) -> List[Dict[str, Any]]:
    """
    Szuka narożników kadru z warianacją pixelow bliskiej 0 –
    silna heurystyka nalożonego statycznego elementu.
    """
    if len(frames) < 5:
        return []

    h, w = frames[0].shape[:2]
    ch = int(h * corner_ratio)
    cw = int(w * corner_ratio)

    corners = [
        ("CORNER-TL", (0,      0,      cw,     ch)),
        ("CORNER-TR", (w - cw, 0,      w,      ch)),
        ("CORNER-BL", (0,      h - ch, cw,     h)),
        ("CORNER-BR", (w - cw, h - ch, w,      h)),
    ]

    gray_stack = np.stack(
        [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) for f in frames],
        axis=0
    )
    variance_map = np.var(gray_stack, axis=0)

    results = []
    for name, (x1, y1, x2, y2) in corners:
        roi_var = variance_map[y1:y2, x1:x2]
        if roi_var.size == 0:
            continue
        low_var_fraction = float(np.mean(roi_var < variance_threshold))
        if low_var_fraction >= min_fraction:
            results.append({
                "name": name,
                "bbox": (x1, y1, x2, y2),
                "score": low_var_fraction,
                "variance_map": roi_var
            })
    return results


# ---------------------------------------------------------------------------
# 2. INVISIBLE WATERMARK (imwatermark / DWT / RivaGAN)
# ---------------------------------------------------------------------------

_KNOWN_SIGNATURES: Dict[str, str] = {
    "STABILITY_AI": "110100110100101000001111111011010101010001000111",
    "RUNWAY_WATERMARK": "101010101010101010101010",
}

_INVISIBLE_WM_AVAILABLE = None


def _check_imwatermark() -> bool:
    global _INVISIBLE_WM_AVAILABLE
    if _INVISIBLE_WM_AVAILABLE is not None:
        return _INVISIBLE_WM_AVAILABLE
    try:
        from imwatermark import WatermarkDecoder  # type: ignore  # noqa
        _INVISIBLE_WM_AVAILABLE = True
    except ImportError:
        _INVISIBLE_WM_AVAILABLE = False
        print("[INVIS-WM] imwatermark niedostepny. pip install invisible-watermark",
              file=sys.stderr)
    return _INVISIBLE_WM_AVAILABLE


def _torch_available() -> bool:
    """
    Sprawdza czy torch jest zaimportowany BEZ faktycznego importowania go teraz.
    Unika bledu DLL przy starcie na maszynach bez CUDA / z uszkodzonym torch.
    Jesli torch byl juz zaladowany wczesniej przez inny modul - uzywa go.
    Jesli nie - proba cichego importu w try/except.
    """
    # Jesli torch zostal zaladowany wczesniej przez inny modul
    if 'torch' in sys.modules:
        return True
    # Proba cichego importu - ignorujemy blad DLL i inne
    try:
        import importlib
        importlib.import_module('torch')
        return True
    except Exception:
        return False


def detect_invisible_watermark(
    frame_bgr: np.ndarray,
    methods: Optional[List[str]] = None,
    watermark_length: int = 48
) -> Dict[str, Any]:
    """
    Proba zdekodowania ukrytego znaku wodnego z pojedynczej klatki.
    """
    result = {"found": False, "method": "invisible_watermark",
              "bits": "", "matched": None, "score": 0.0, "details": ""}

    if not _check_imwatermark():
        result["details"] = "imwatermark niedostepny"
        return result

    if methods is None:
        # POPRAWKA: sprawdzamy torch przez _torch_available() zamiast bezposrednio
        # importowac - unikamy bledu WinError 1114 przy ladowaniu DLL
        methods = ["dwtDct", "dwtDctSvd"]
        if _torch_available():
            methods.append("rivaGan")

    try:
        from imwatermark import WatermarkDecoder  # type: ignore
    except ImportError:
        result["details"] = "import error"
        return result

    bgr = frame_bgr[:, :, :3] if frame_bgr.ndim == 3 and frame_bgr.shape[2] != 3 else frame_bgr

    for method in methods:
        try:
            decoder = WatermarkDecoder('bits', watermark_length)
            watermark_bits = decoder.decode(bgr, method)
            bits_str = ''.join(str(int(b)) for b in watermark_bits)

            matched = None
            best_similarity = 0.0
            for sig_name, sig_bits in _KNOWN_SIGNATURES.items():
                if len(sig_bits) <= len(bits_str):
                    sub = bits_str[:len(sig_bits)]
                    matches = sum(a == b for a, b in zip(sub, sig_bits))
                    sim = matches / len(sig_bits)
                    if sim > best_similarity:
                        best_similarity = sim
                        if sim >= 0.85:
                            matched = sig_name

            ones_ratio = bits_str.count('1') / max(len(bits_str), 1)
            is_nontrivial = 0.15 < ones_ratio < 0.85

            max_run = max(
                (sum(1 for _ in g) for _, g in
                 __import__('itertools').groupby(bits_str)),
                default=0
            )
            has_structure = max_run < len(bits_str) * 0.6

            if matched or (is_nontrivial and has_structure):
                result["found"] = True
                result["method"] = f"invisible_watermark:{method}"
                result["bits"] = bits_str
                result["matched"] = matched
                result["score"] = best_similarity if matched else 0.5
                result["details"] = (
                    f"Metoda={method}, bits={bits_str[:32]}..., "
                    f"pasuje_do={matched or 'nieznany'}, podobienstwo={best_similarity:.2f}"
                )
                return result

        except Exception as e:
            result["details"] = f"blad {method}: {e}"
            continue

    return result


# ---------------------------------------------------------------------------
# 3. NOISE RESIDUAL + FFT – artefakty upsamplingu AI
# ---------------------------------------------------------------------------

def detect_ai_noise_artifacts(
    frame_bgr: np.ndarray,
    fft_peak_threshold: float = 0.35,
    wiener_ksize: int = 5
) -> Dict[str, Any]:
    """
    Wykrywa periodyczne artefakty upsamplingu przez FFT na residual noise.

    POPRAWKA: cv2.circle wymaga macierzy uint8 (C-contiguous) jako pierwszego
    argumentu. Wczesniej magnitude_no_center bylo float64 z np.fft, co
    powodowalo blad 'Layout incompatible with cv::Mat'. Teraz robimy kopie
    uint8 do wizualizacji, a maska float jest obliczana na float64.
    """
    result = {"found": False, "method": "noise_residual_fft",
              "score": 0.0, "details": "", "fft_image": None}

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

    blurred = cv2.GaussianBlur(gray, (wiener_ksize, wiener_ksize), 0)
    noise = gray - blurred

    fft = np.fft.fft2(noise)
    fft_shifted = np.fft.fftshift(fft)
    magnitude = np.log1p(np.abs(fft_shifted))   # float64, niemutowalne przez FFT

    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8

    # POPRAWKA: maska jako osobna tablica float64 (nie rysujemy po magnitude)
    mask = np.ones((h, w), dtype=np.float64)
    # cv2.circle wymaga uint8 lub float32 C-contiguous - uzyj osobnej maski uint8
    mask_u8 = np.ones((h, w), dtype=np.uint8)
    cv2.circle(mask_u8, (cx, cy), r, 0, -1)     # rysuj po uint8, nie po float
    mask = mask_u8.astype(np.float64)            # z powrotem float do mnozenia

    magnitude_no_center = magnitude * mask

    nonzero = magnitude_no_center[magnitude_no_center > 0]
    mean_bg = float(np.mean(nonzero)) if nonzero.size > 0 else 1e-6
    max_peak = float(np.max(magnitude_no_center))
    ratio = max_peak / (mean_bg + 1e-6)

    vis = cv2.normalize(
        magnitude_no_center.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX
    ).astype(np.uint8)
    vis_color = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
    result["fft_image"] = vis_color

    if ratio > fft_peak_threshold * 10:
        result["found"] = True
        result["score"] = min(1.0, (ratio - fft_peak_threshold * 10) / 50.0)
        result["details"] = (
            f"FFT peak ratio={ratio:.2f}, mean_bg={mean_bg:.3f}, "
            f"max_peak={max_peak:.3f} – mozliwe artefakty AI upsamplingu"
        )
    else:
        result["details"] = f"FFT ratio={ratio:.2f} – brak anomalii"

    return result


# ---------------------------------------------------------------------------
# 4. FASADA: skan zaawansowany dla jednego pliku wideo
# ---------------------------------------------------------------------------

def run_advanced_scan(
    cap: cv2.VideoCapture,
    fps: float,
    total_frames: int,
    n_frames_median: int = 40,
    check_invisible: bool = True,
    check_fft: bool = True,
    log_fn=None
) -> Dict[str, Any]:
    """
    Zbiera klatki z otwartego VideoCapture i uruchamia wszystkie zaawansowane metody.
    """
    def _log(msg):
        print(msg, file=sys.stderr)
        if log_fn:
            try:
                log_fn(msg)
            except Exception:
                pass

    result: Dict[str, Any] = {
        "temporal_median_frame": None,
        "overlay_diff": None,
        "zero_variance_rois": [],
        "invisible_wm": {"found": False},
        "fft_artifacts": {"found": False},
        "summary": ""
    }

    _log("[ADV] Pobieram klatki do analizy temporalnej...")
    step = max(1, total_frames // n_frames_median)
    frames: List[np.ndarray] = []
    pos = 0
    while len(frames) < n_frames_median and pos < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
        pos += step

    if len(frames) < 3:
        result["summary"] = "Za malo klatek do analizy zaawansowanej."
        return result

    _log(f"[ADV] Zebrano {len(frames)} klatek. Licze mediane temporalna...")

    # 1. Temporal median
    try:
        median_frame = build_temporal_median(frames)
        result["temporal_median_frame"] = median_frame
        mid = frames[len(frames) // 2]
        result["overlay_diff"] = extract_static_overlay(median_frame, mid, amp=5.0)
        _log("[ADV] Mediana temporalna obliczona.")
    except Exception as e:
        _log(f"[ADV] Blad mediany temporalnej: {e}")

    # 2. Zero variance ROIs
    try:
        zv_rois = detect_zero_variance_rois(frames)
        result["zero_variance_rois"] = zv_rois
        if zv_rois:
            names = [r['name'] for r in zv_rois]
            _log(f"[ADV] Zerowa wariancja ROI: {names} – mozliwy statyczny overlay")
    except Exception as e:
        _log(f"[ADV] Blad zero-variance: {e}")

    # 3. Invisible watermark
    if check_invisible and result["temporal_median_frame"] is not None:
        _log("[ADV] Sprawdzam invisible watermark (imwatermark)...")
        try:
            iw = detect_invisible_watermark(result["temporal_median_frame"])
            result["invisible_wm"] = iw
            if iw["found"]:
                _log(f"[ADV] INVISIBLE WM ZNALEZIONY: {iw['details']}")
            else:
                _log(f"[ADV] Invisible WM: brak / {iw['details']}")
        except Exception as e:
            _log(f"[ADV] Blad invisible WM: {e}")

    # 4. FFT noise
    if check_fft:
        _log("[ADV] Sprawdzam artefakty FFT noise...")
        try:
            fft_res = detect_ai_noise_artifacts(frames[len(frames) // 2])
            result["fft_artifacts"] = fft_res
            if fft_res["found"]:
                _log(f"[ADV] FFT artefakty: {fft_res['details']}")
            else:
                _log(f"[ADV] FFT: {fft_res['details']}")
        except Exception as e:
            _log(f"[ADV] Blad FFT: {e}")

    # Podsumowanie
    findings = []
    if result["zero_variance_rois"]:
        findings.append(f"statyczny_overlay({len(result['zero_variance_rois'])} ROI)")
    if result["invisible_wm"].get("found"):
        findings.append(f"invisible_wm({result['invisible_wm'].get('matched', 'nieznany')})")
    if result["fft_artifacts"].get("found"):
        findings.append(f"fft_artefakty(score={result['fft_artifacts'].get('score', 0):.2f})")
    result["summary"] = "ZNALEZIONO: " + ", ".join(findings) if findings else "Brak wynikow zaawansowanych."
    _log(f"[ADV] Podsumowanie: {result['summary']}")

    return result
