"""
ocr_detector.py

Detekcja znakow wodnych / napisow AI w obrazach i wideo.
- Zapis CSV z detekcjami.
- Konfiguracja progu pewnosci (confidence) oraz probkowania (sample_rate).
- Opcjonalne drugie przejscie (szczegolowa analiza dwufazowa).
- Zapisywanie na dysk wersji oryginalnej i przefiltrowanej.
- Template Matching dla graficznych znakow wodnych.
- Corner ROI scanning.
"""

from __future__ import annotations

import os
import csv
import math
import threading
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Callable

import cv2
import numpy as np

import config

_OCR_READER = None
_OCR_ENGINE_TYPE = None
_YOLO_MODEL = None
_OCR_LOCK = threading.Lock()  # zapobiega rownoleglem inicjalizacji modelu

CORNER_RATIO = 0.25
CORNER_SCALE = 3.0


class TextTracker:
    def __init__(self):
        self.history = {}

    def update(self, frame_idx, type_id, bbox):
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        if type_id not in self.history:
            self.history[type_id] = []
            status = "NOWY"
        else:
            self.history[type_id].sort(key=lambda x: x['frame'])
            closest = min(self.history[type_id], key=lambda x: abs(x['frame'] - frame_idx))
            dist = math.hypot(cx - closest['centroid'][0], cy - closest['centroid'][1])
            frames_diff = abs(frame_idx - closest['frame'])
            if frames_diff > 30:
                status = "POJAWIENIE"
            elif dist < 25:
                status = "STATYCZNY"
            else:
                status = "RUCHOMY"

        self.history[type_id].append({"frame": frame_idx, "centroid": (cx, cy), "bbox": bbox})
        return status


def reset_reader():
    """
    Resetuje singleton OCR readera.
    Wywolaj PRZED uruchomieniem skanu w nowym watku (np. QThread),
    aby wymusic ponowna inicjalizacje modelu w kontekscie tego watku.
    """
    global _OCR_READER, _OCR_ENGINE_TYPE
    with _OCR_LOCK:
        _OCR_READER = None
        _OCR_ENGINE_TYPE = None


def _get_reader():
    """Zwraca singleton OCR reader. Thread-safe, lazy-load."""
    global _OCR_READER, _OCR_ENGINE_TYPE
    if _OCR_READER is not None:
        return _OCR_READER

    with _OCR_LOCK:
        # podwojne sprawdzenie po wejsciu do locka
        if _OCR_READER is not None:
            return _OCR_READER

        try:
            from paddleocr import PaddleOCR  # type: ignore
            _OCR_READER = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
            _OCR_ENGINE_TYPE = "paddle"
            return _OCR_READER
        except ImportError:
            pass
        except Exception:
            pass

        try:
            import easyocr  # type: ignore
            _OCR_READER = easyocr.Reader(["en", "pl"], gpu=False, verbose=False)
            _OCR_ENGINE_TYPE = "easyocr"
            return _OCR_READER
        except Exception as e:
            _OCR_READER = None
            _OCR_ENGINE_TYPE = None

    return _OCR_READER


def _get_yolo():
    global _YOLO_MODEL
    if _YOLO_MODEL is not None:
        return _YOLO_MODEL

    path = getattr(config, "SORA_YOLO_MODEL_PATH", "") or ""
    model_id = getattr(config, "WATERMARK_YOLO_MODEL_ID", "") or ""
    if not path and not model_id:
        return None

    try:
        from ultralytics import YOLO  # type: ignore
        if path and os.path.exists(path):
            _YOLO_MODEL = YOLO(path)
        elif model_id:
            _YOLO_MODEL = YOLO(model_id)
        return _YOLO_MODEL
    except Exception:
        _YOLO_MODEL = None
        return None


def _detect_yolo_watermark(frame_bgr, min_conf: float) -> List[Tuple]:
    model = _get_yolo()
    if model is None:
        return []
    try:
        results = model(frame_bgr, verbose=False)
    except Exception:
        return []
    detections = []
    class_map = {0: "WATERMARK", 1: "SORA", 2: "OPENAI"}
    for r in results:
        if not hasattr(r, "boxes"):
            continue
        for b in r.boxes:
            try:
                cls_id = int(b.cls[0])
                conf = float(b.conf[0])
                if conf < min_conf:
                    continue
                label = class_map.get(cls_id, "WATERMARK")
                if hasattr(r, "names") and cls_id in r.names:
                    label = r.names[cls_id].upper()
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                detections.append((x1, y1, x2, y2, conf, label))
            except Exception:
                continue
    return detections


def _detect_template_watermarks(image_to_scan: np.ndarray, confidence: float) -> List[dict]:
    templates_dir = getattr(config, "TEMPLATES_DIR", "watermark_templates")
    if not os.path.exists(templates_dir):
        try:
            os.makedirs(templates_dir, exist_ok=True)
        except Exception:
            return []
    detections = []
    gray_frame = cv2.cvtColor(image_to_scan, cv2.COLOR_BGR2GRAY) if len(image_to_scan.shape) == 3 else image_to_scan
    for fname in os.listdir(templates_dir):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        tpl = cv2.imread(os.path.join(templates_dir, fname), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        try:
            res = cv2.matchTemplate(gray_frame, tpl, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= confidence)
            h, w = tpl.shape
            label = os.path.splitext(fname)[0].upper()
            for pt in zip(*loc[::-1]):
                x1, y1 = int(pt[0]), int(pt[1])
                detections.append({
                    "type": f"LOGO-{label}",
                    "confidence": float(res[y1, x1]),
                    "text": f"[IMG: {label}]",
                    "bbox": (x1, y1, x1 + w, y1 + h)
                })
        except Exception:
            pass
    return detections


def _preprocess_for_ocr(roi_bgr: np.ndarray) -> np.ndarray:
    try:
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return roi_bgr


def _get_advanced_filters(frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray]]:
    filters = []
    try:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5)
        filters.append(("AGGR-ADAPT", cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        tophat = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-TOPHAT", cv2.cvtColor(tophat, cv2.COLOR_GRAY2BGR)))
        blackhat = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-BLACKHAT", cv2.cvtColor(blackhat, cv2.COLOR_GRAY2BGR)))
        gaussian = cv2.GaussianBlur(frame_bgr, (9, 9), 10.0)
        filters.append(("AGGR-SHARPEN", cv2.addWeighted(frame_bgr, 1.5, gaussian, -0.5, 0)))
        table = np.array([((i / 255.0) ** (1.0 / 0.4)) * 255 for i in range(256)]).astype("uint8")
        dark_inv_clahe = _preprocess_for_ocr(cv2.bitwise_not(cv2.LUT(frame_bgr, table)))
        filters.append(("AGGR-EXTREME-WHITE", dark_inv_clahe))
        bg = cv2.medianBlur(gray, 51)
        diff = cv2.normalize(cv2.absdiff(gray, bg), None, 0, 255, cv2.NORM_MINMAX)
        filters.append(("AGGR-BGSUB", cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)))
    except Exception:
        pass
    return filters


def _make_session_dir(input_path: str) -> str:
    filename_clean = os.path.basename(input_path).replace(".", "_")
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = getattr(config, "REPORTS_BASE_DIR", "reports")
    out_dir = os.path.join(base, "watermarks", f"{filename_clean}_{timestamp_str}")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def _extract_corner_rois(frame_bgr: np.ndarray) -> List[Tuple[str, np.ndarray, int, int]]:
    h, w = frame_bgr.shape[:2]
    ch = int(h * CORNER_RATIO)
    cw = int(w * CORNER_RATIO)
    corners = [
        ("CORNER-TL", frame_bgr[0:ch, 0:cw],        0,      0),
        ("CORNER-TR", frame_bgr[0:ch, w - cw:w],     w - cw, 0),
        ("CORNER-BL", frame_bgr[h - ch:h, 0:cw],     0,      h - ch),
        ("CORNER-BR", frame_bgr[h - ch:h, w - cw:w], w - cw, h - ch),
    ]
    result = []
    for name, roi, ox, oy in corners:
        if roi.size == 0:
            continue
        upscaled = cv2.resize(roi, None, fx=CORNER_SCALE, fy=CORNER_SCALE, interpolation=cv2.INTER_CUBIC)
        result.append((name, upscaled, ox, oy))
    return result


def _ocr_on_image(image: np.ndarray, confidence: float, keywords: List[str]) -> List[Tuple]:
    reader = _get_reader()
    if reader is None:
        return []

    results = []
    try:
        if _OCR_ENGINE_TYPE == "paddle":
            raw = reader.ocr(image, cls=False)
            if raw and raw[0]:
                for line in raw[0]:
                    results.append((line[0], line[1][0], line[1][1]))
        else:
            results = reader.readtext(image)
    except Exception:
        return []

    matches = []
    for (bbox, text, prob) in results:
        if float(prob) < confidence:
            continue
        t_clean = str(text).upper().replace(" ", "")
        for k in keywords:
            if k.replace(" ", "") in t_clean:
                matches.append((bbox, text, float(prob), k))
                break
    return matches


def _perform_scan(
    frame_original: np.ndarray,
    confidence: float,
    keywords: List[str],
    versions_to_scan: List[Tuple[str, np.ndarray]],
    scale_factor: float = 1.0
) -> List[dict]:
    frame_detections = []
    found_keys: set = set()

    for (x1, y1, x2, y2, conf, label) in _detect_yolo_watermark(frame_original, confidence):
        frame_detections.append({"type": label, "confidence": conf, "text": f"[{label}]",
                                  "bbox": (x1, y1, x2, y2), "source": "YOLO"})
        found_keys.add(label)

    for source_name, base_image in versions_to_scan:
        for det in _detect_template_watermarks(base_image, confidence):
            if det["type"] not in found_keys:
                det["source"] = f"TEMPLATE-{source_name}"
                frame_detections.append(det)
                found_keys.add(det["type"])

        img = cv2.resize(base_image, None, fx=scale_factor, fy=scale_factor,
                         interpolation=cv2.INTER_CUBIC) if scale_factor != 1.0 else base_image

        for (bbox, text, prob, kw) in _ocr_on_image(img, confidence, keywords):
            if kw in found_keys:
                continue
            x1 = int(bbox[0][0] / scale_factor)
            y1 = int(bbox[0][1] / scale_factor)
            x2 = int(bbox[2][0] / scale_factor)
            y2 = int(bbox[2][1] / scale_factor)
            frame_detections.append({"type": kw, "confidence": prob, "text": str(text).upper(),
                                      "bbox": (x1, y1, x2, y2), "source": source_name})
            found_keys.add(kw)

    # FIX: corner ROI uzywa osobnego found_keys zeby nie blokowac detekcji
    # ktore moglby pominac glowny skan (np. watermark w rogu nie znaleziony w pelnej klatce)
    h_orig, w_orig = frame_original.shape[:2]
    for (corner_name, roi_upscaled, ox, oy) in _extract_corner_rois(frame_original):
        roi_versions = [
            roi_upscaled,
            _preprocess_for_ocr(roi_upscaled),
            cv2.bitwise_not(roi_upscaled),
        ]
        corner_found: set = set()  # izolowany per-corner, nie blokuje przez found_keys
        for rv in roi_versions:
            for (bbox, text, prob, kw) in _ocr_on_image(rv, confidence, keywords):
                if kw in corner_found:
                    continue
                x1 = ox + int(bbox[0][0] / CORNER_SCALE)
                y1 = oy + int(bbox[0][1] / CORNER_SCALE)
                x2 = ox + int(bbox[2][0] / CORNER_SCALE)
                y2 = oy + int(bbox[2][1] / CORNER_SCALE)
                x1, x2 = max(0, x1), min(w_orig, x2)
                y1, y2 = max(0, y1), min(h_orig, y2)
                frame_detections.append({"type": kw, "confidence": prob, "text": str(text).upper(),
                                          "bbox": (x1, y1, x2, y2), "source": corner_name})
                corner_found.add(kw)
                found_keys.add(kw)

    return frame_detections


def scan_for_watermarks(
    media_path: str,
    check_stop=None,
    progress_callback=None,
    confidence: float = 0.6,
    sample_rate: int = 30,
    detailed_scan: bool = False,
    preview_callback: Optional[Callable[[np.ndarray, list], None]] = None
) -> Dict[str, Any]:

    # FIX: guard na nieprawidlowy sample_rate
    if not isinstance(sample_rate, int) or sample_rate <= 0:
        sample_rate = 1

    is_video = os.path.splitext(media_path)[1].lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    cap = cv2.VideoCapture(os.path.abspath(media_path))
    if not cap.isOpened():
        return {"status": "ERROR", "error": "Nie mozna otworzyc pliku."}

    default_keywords = [
        "SORA", "OPENAI", "GENERATED", "AI VIDEO", "MADE WITH", "AI GENERATED",
        "RUNWAY", "PIKA", "LUMA", "GEN-2", "GEN-3", "GEN-4", "GEN-5",
        "GEN3", "GEN4", "GEN5", "GEN 3", "GEN 4", "GEN 5",
        "TIKTOK", "KWAI", "CAPCUT", "STABLE VIDEO",
        "KLING", "VEED", "INVIDEO", "KAPWING", "SYNTHID", "MINIMAX", "HAIPER", "DREAMLUX"
    ]
    keywords = [str(k).upper() for k in getattr(config, "WATERMARK_KEYWORDS", default_keywords)]

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if is_video else 1

    out_dir = _make_session_dir(media_path)
    csv_path = os.path.join(out_dir, "report.csv")

    saved_paths: List[str] = []
    missed_frames: List[int] = []

    frame_idx = 0
    detections_count = 0
    found_types: set = set()
    tracker = TextTracker()

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Plik", "Typ", "Klatka", "Timestamp", "Typ watermarku",
                         "Confidence", "Tekst", "Ruch", "Zrodlo", "Zapisany plik"])

        # ============ FAZA 1 ============
        while True:
            if check_stop and check_stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            if progress_callback and is_video and frame_idx % 10 == 0:
                progress_callback(frame_idx, total_frames)

            if is_video and frame_idx % sample_rate != 0 and frame_idx != 1:
                continue

            now_sec = frame_idx / float(fps) if is_video else 0.0

            table = np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)]).astype("uint8")
            darkened = cv2.LUT(frame, table)

            versions_to_scan = [
                ("OCR-RAW",   frame),
                ("OCR-CLAHE", _preprocess_for_ocr(frame)),
                ("OCR-INV",   cv2.bitwise_not(frame)),
                ("OCR-DARK",  darkened),
            ]

            frame_detections = _perform_scan(frame, confidence, keywords, versions_to_scan, scale_factor=1.0)

            if not frame_detections:
                missed_frames.append(frame_idx)

            frame_to_draw = frame.copy()
            for det in frame_detections:
                x1, y1, x2, y2 = det["bbox"]
                motion = tracker.update(frame_idx, det['type'], det['bbox'])
                cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(frame_to_draw,
                            f"{det['type']} [{motion}] ({int(det['confidence']*100)}%)",
                            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                found_types.add(det['type'])
                detections_count += 1
                fname = f"frame_{frame_idx}_t_{now_sec:.2f}s.jpg"
                save_path = os.path.join(out_dir, fname)
                writer.writerow([
                    os.path.basename(media_path), "Video" if is_video else "Image",
                    frame_idx, f"{now_sec:.2f}", det['type'],
                    f"{det['confidence']:.2f}", det['text'], motion, det.get('source', ''), save_path
                ])
                try:
                    cv2.imwrite(save_path, frame_to_draw)
                    saved_paths.append(save_path)
                except Exception:
                    pass

            if preview_callback:
                if not frame_detections:
                    cv2.putText(frame_to_draw, f"Brak detekcji (klatka {frame_idx})",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                preview_callback(frame_to_draw, frame_detections)

        # ============ FAZA 2 (Szczegolowa) ============
        if detailed_scan and missed_frames and not (check_stop and check_stop()):
            for i, m_idx in enumerate(missed_frames):
                if check_stop and check_stop():
                    break
                cap.set(cv2.CAP_PROP_POS_FRAMES, m_idx - 1)
                ok, frame = cap.read()
                if not ok:
                    continue

                if progress_callback:
                    progress_callback(i + 1, len(missed_frames))

                now_sec = m_idx / float(fps)
                aggr_versions = _get_advanced_filters(frame)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                _, bw = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                aggr_versions.append(("AGGR-BW",   cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)))
                aggr_versions.append(("AGGR-ORIG", frame))

                frame_detections = _perform_scan(frame, confidence, keywords, aggr_versions, scale_factor=2.0)

                if frame_detections:
                    frame_to_draw = frame.copy()
                    for det in frame_detections:
                        x1, y1, x2, y2 = det["bbox"]
                        motion = tracker.update(m_idx, det['type'], det['bbox'])
                        cv2.rectangle(frame_to_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)
                        cv2.putText(frame_to_draw,
                                    f"{det['type']} [AGGR:{motion}]",
                                    (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        found_types.add(det['type'])
                        detections_count += 1
                        writer.writerow([
                            os.path.basename(media_path), "Video (Aggr)" if is_video else "Image (Aggr)",
                            m_idx, f"{now_sec:.2f}", det['type'],
                            f"{det['confidence']:.2f}", det['text'], motion, det.get('source', ''), ""
                        ])

                    save_path = os.path.join(out_dir, f"frame_{m_idx}_aggr_t_{now_sec:.2f}s.jpg")
                    try:
                        cv2.imwrite(save_path, frame_to_draw)
                        saved_paths.append(save_path)
                    except Exception:
                        pass

                    if preview_callback:
                        preview_callback(frame_to_draw, frame_detections)
                else:
                    if preview_callback:
                        tmp = frame.copy()
                        cv2.putText(tmp, f"Brak detekcji (AGGR klatka {m_idx})",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                        preview_callback(tmp, [])

    cap.release()

    return {
        "status": "OK",
        "watermark_found": detections_count > 0,
        "watermark_types": list(found_types),
        "watermark_count": detections_count,
        "watermark_folder": out_dir,
        "csv_path": csv_path,
        "watermark_frames": saved_paths,
    }
