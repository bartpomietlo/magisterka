"""
diag_ocr.py - Diagnostyka OCR na pojedynczym pliku wideo.

Uruchom:
    python kod/diag_ocr.py sciezka/do/pliku.mp4

Wypisuje BEZ filtrowania keywords:
 - wszystkie teksty wykryte przez OCR w pelnej klatce
 - wszystkie teksty wykryte w kazdym z 4 rogow (wiele wersji preprocessingu)
 - zapisuje klatke diagnostyczna do diag_frame.jpg
"""

import sys
import os
import cv2
import numpy as np

# Dodaj folder kod/ do path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))


def preprocess_for_ocr(roi_bgr):
    try:
        import cv2
        lab = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2LAB)
        l_channel, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_channel)
        return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        return roi_bgr


def get_corner_versions(roi):
    versions = [("RAW", roi)]
    try:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        versions.append(("CLAHE", preprocess_for_ocr(roi)))
        versions.append(("INV", cv2.bitwise_not(roi)))
        versions.append(("CLAHE+INV", cv2.bitwise_not(preprocess_for_ocr(roi))))
        norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        versions.append(("NORM", cv2.cvtColor(norm, cv2.COLOR_GRAY2BGR)))
        versions.append(("NORM+INV", cv2.cvtColor(cv2.bitwise_not(norm), cv2.COLOR_GRAY2BGR)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 5))
        bh = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        versions.append(("BLACKHAT", cv2.cvtColor(bh, cv2.COLOR_GRAY2BGR)))
        th = cv2.normalize(cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel), None, 0, 255, cv2.NORM_MINMAX)
        versions.append(("TOPHAT", cv2.cvtColor(th, cv2.COLOR_GRAY2BGR)))
        adapt = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        versions.append(("ADAPT", cv2.cvtColor(adapt, cv2.COLOR_GRAY2BGR)))
        versions.append(("ADAPT+INV", cv2.cvtColor(cv2.bitwise_not(adapt), cv2.COLOR_GRAY2BGR)))
        blur = cv2.GaussianBlur(roi, (3, 3), 0)
        versions.append(("SHARP", cv2.addWeighted(roi, 2.0, blur, -1.0, 0)))
    except Exception as e:
        print(f"  [preprocessing error] {e}")
    return versions


def run_ocr(image, confidence_threshold=0.0):
    """Uruchamia OCR i zwraca wszystkie wyniki bez filtrowania keywords."""
    try:
        from paddleocr import PaddleOCR
        reader = PaddleOCR(use_angle_cls=False, lang='en', show_log=False)
        engine = "PaddleOCR"
        raw = reader.ocr(image, cls=False)
        results = []
        if raw and raw[0]:
            for line in raw[0]:
                bbox = line[0]
                text = line[1][0]
                prob = line[1][1]
                results.append((bbox, text, float(prob)))
        return engine, results
    except ImportError:
        pass
    except Exception as e:
        print(f"  [PaddleOCR error] {e}")

    try:
        import easyocr
        reader = easyocr.Reader(["en", "pl"], gpu=False, verbose=False)
        engine = "EasyOCR"
        results = reader.readtext(image)
        return engine, [(r[0], r[1], float(r[2])) for r in results]
    except Exception as e:
        print(f"  [EasyOCR error] {e}")

    return "BRAK", []


def main():
    if len(sys.argv) < 2:
        print("Uzycie: python kod/diag_ocr.py sciezka/do/pliku.mp4")
        print("Przyklad: python kod/diag_ocr.py nagrania/test.mp4")
        sys.exit(1)

    video_path = sys.argv[1]
    if not os.path.exists(video_path):
        print(f"BLAD: Plik nie istnieje: {video_path}")
        sys.exit(1)

    print(f"\n=== DIAGNOSTYKA OCR ===")
    print(f"Plik: {video_path}")

    cap = cv2.VideoCapture(os.path.abspath(video_path))
    if not cap.isOpened():
        print("BLAD: Nie mozna otworzyc pliku.")
        sys.exit(1)

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Klatki: {total}, FPS: {fps:.1f}, Czas: {total/fps:.1f}s")

    # Wyciagnij 3 klatki: pierwsza, srodkowa, ostatnia
    test_frames = [1, total // 2, max(1, total - 5)]
    frame_cache = {}
    for fidx in test_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx - 1)
        ok, frame = cap.read()
        if ok:
            frame_cache[fidx] = frame
    cap.release()

    if not frame_cache:
        print("BLAD: Nie udalo sie odczytac zadnej klatki.")
        sys.exit(1)

    print(f"\n=== INICJALIZACJA OCR ===")
    # Uzyj pierwszej dostepnej klatki do testu
    test_frame = next(iter(frame_cache.values()))
    h, w = test_frame.shape[:2]
    print(f"Rozmiar klatki: {w}x{h}")

    engine, results = run_ocr(test_frame)
    print(f"Engine: {engine}")

    if engine == "BRAK":
        print("BLAD KRYTYCZNY: Brak dzialajacego silnika OCR!")
        print("Zainstaluj: pip install easyocr lub pip install paddlepaddle paddleocr")
        sys.exit(1)

    print(f"\n=== PELNA KLATKA (klatka {list(frame_cache.keys())[0]}) ===")
    print(f"Wykrytych tekstow: {len(results)}")
    for bbox, text, prob in sorted(results, key=lambda x: -x[2]):
        print(f"  [{prob:.3f}] '{text}'")

    if not results:
        print("  (brak wynikow na pelnej klatce)")

    # Zapisz klatke diagnostyczna z zaznaczonymi obszarami
    diag_frame = test_frame.copy()
    for bbox, text, prob in results:
        try:
            pts = np.array(bbox, dtype=np.int32)
            cv2.polylines(diag_frame, [pts], True, (0, 255, 0), 2)
            x, y = pts[0]
            cv2.putText(diag_frame, f"{text[:20]} ({prob:.2f})",
                        (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        except Exception:
            pass

    # Test cornerow na tej samej klatce
    CORNER_RATIO = 0.15
    CORNER_SCALE = 5.0
    ch = int(h * CORNER_RATIO)
    cw = int(w * CORNER_RATIO)
    corners = [
        ("CORNER-TL", test_frame[0:ch, 0:cw],              0,      0),
        ("CORNER-TR", test_frame[0:ch, w - cw:w],           w - cw, 0),
        ("CORNER-BL", test_frame[h - ch:h, 0:cw],           0,      h - ch),
        ("CORNER-BR", test_frame[h - ch:h, w - cw:w],       w - cw, h - ch),
    ]

    print(f"\n=== CORNER ROI (CORNER_RATIO={CORNER_RATIO}, CORNER_SCALE={CORNER_SCALE}x) ===")
    print(f"Rozmiar cornera: {cw}x{ch}px, po upscalu: {int(cw*CORNER_SCALE)}x{int(ch*CORNER_SCALE)}px")

    for corner_name, roi, ox, oy in corners:
        if roi.size == 0:
            continue
        upscaled = cv2.resize(roi, None, fx=CORNER_SCALE, fy=CORNER_SCALE, interpolation=cv2.INTER_CUBIC)

        # Zaznacz corner na klatce diagnostycznej
        cv2.rectangle(diag_frame, (ox, oy), (ox + cw, oy + ch),
                      (255, 165, 0) if "BR" in corner_name else (100, 100, 255), 2)
        cv2.putText(diag_frame, corner_name, (ox + 3, oy + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)

        any_found = False
        for ver_name, ver_img in get_corner_versions(upscaled):
            _, ver_results = run_ocr(ver_img)
            if ver_results:
                any_found = True
                print(f"  {corner_name} [{ver_name}]: {len(ver_results)} tekstow")
                for bbox, text, prob in sorted(ver_results, key=lambda x: -x[2])[:5]:
                    print(f"    [{prob:.3f}] '{text}'")

        if not any_found:
            print(f"  {corner_name}: brak wynikow we wszystkich wersjach")

    diag_path = "diag_frame.jpg"
    cv2.imwrite(diag_path, diag_frame)
    print(f"\nZapisano klatke diagnostyczna: {diag_path}")
    print("Otworz diag_frame.jpg zeby zobaczyc co OCR wykryl na klatce.")
    print("\nSzukane slowa kluczowe to m.in.: RUNWAY, SORA, OPENAI, PIKA, LUMA, GEN-3 itp.")


if __name__ == "__main__":
    main()
