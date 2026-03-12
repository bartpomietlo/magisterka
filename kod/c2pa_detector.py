# c2pa_detector.py
# Detekcja metadanych C2PA (Coalition for Content Provenance and Authenticity)
# w plikach wideo MP4.
#
# Wymaga: pip install c2pa-python
# Dokumentacja: https://github.com/contentauth/c2pa-python

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import c2pa
    C2PA_AVAILABLE = True
except ImportError:
    C2PA_AVAILABLE = False
    print("[C2PADetector] UWAGA: biblioteka c2pa-python nie jest zainstalowana.")
    print("[C2PADetector] Zainstaluj: pip install c2pa-python")


# =============================================================================
# Struktury danych
# =============================================================================

@dataclass
class C2PAAssertion:
    """Pojedyncze twierdzenie (assertion) z manifestu C2PA."""
    label: str
    data: Any


@dataclass
class C2PAResult:
    """Wynik detekcji C2PA dla jednego pliku wideo."""
    video_path: str
    found: bool
    generator: Optional[str] = None
    generator_version: Optional[str] = None
    producer: Optional[str] = None
    created_at: Optional[str] = None
    assertions: List[C2PAAssertion] = field(default_factory=list)
    raw_manifest: Optional[Dict] = None
    error: Optional[str] = None

    @property
    def is_ai_generated(self) -> bool:
        if not self.found:
            return False
        for assertion in self.assertions:
            if assertion.label == "c2pa.actions":
                actions = assertion.data if isinstance(assertion.data, list) else []
                for action in actions:
                    if isinstance(action, dict) and action.get("action") == "c2pa.created":
                        return True
        return self.generator is not None

    def summary(self) -> Dict[str, Any]:
        return {
            "file": os.path.basename(self.video_path),
            "c2pa_found": self.found,
            "is_ai_generated": self.is_ai_generated,
            "generator": self.generator or "N/A",
            "generator_version": self.generator_version or "N/A",
            "producer": self.producer or "N/A",
            "created_at": self.created_at or "N/A",
            "assertions_count": len(self.assertions),
            "error": self.error or "brak",
        }


# =============================================================================
# Glowna klasa detektora
# =============================================================================

class C2PADetector:
    """
    Detektor metadanych C2PA w plikach wideo.
    Uzywa c2pa.Reader (API >= 0.11) zamiast przestarzalego read_file.
    """

    KNOWN_AI_GENERATORS = {
        "sora": "OpenAI Sora",
        "runway": "Runway",
        "adobe firefly": "Adobe Firefly",
        "lumaai": "LumaAI",
        "luma": "LumaAI",
        "pika": "Pika Labs",
        "stable video": "Stable Video Diffusion",
        "cogvideo": "CogVideo",
        "kling": "Kling AI",
        "hailuo": "Hailuo AI",
    }

    # Mapowanie rozszerzen na MIME types
    MIME_TYPES = {
        ".mp4": "video/mp4",
        ".mov": "video/quicktime",
        ".avi": "video/avi",
        ".mkv": "video/x-matroska",
        ".webm": "video/webm",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
    }

    def __init__(self):
        if not C2PA_AVAILABLE:
            print("[C2PADetector] Biblioteka c2pa-python niedostepna.")

    def _get_mime(self, path: str) -> str:
        ext = os.path.splitext(path)[1].lower()
        return self.MIME_TYPES.get(ext, "video/mp4")

    def detect(self, video_path: str) -> C2PAResult:
        """
        Odczytuje manifest C2PA z pliku wideo uzywajac c2pa.Reader.
        """
        if not os.path.exists(video_path):
            return C2PAResult(video_path=video_path, found=False,
                              error=f"Plik nie istnieje: {video_path}")

        if not C2PA_AVAILABLE:
            return C2PAResult(video_path=video_path, found=False,
                              error="Biblioteka c2pa-python nie jest zainstalowana")

        try:
            mime = self._get_mime(video_path)
            with open(video_path, "rb") as f:
                reader = c2pa.Reader(mime, f)
                manifest_json = reader.json()

            if not manifest_json:
                return C2PAResult(video_path=video_path, found=False)

            manifest = json.loads(manifest_json)

            # Jesli nie ma zadnych manifestow = brak C2PA
            if not manifest.get("manifests"):
                return C2PAResult(video_path=video_path, found=False)

            return self._parse_manifest(video_path, manifest)

        except Exception as e:
            error_msg = str(e)
            # C2PA nie znaleziono - normalny przypadek
            no_c2pa_keywords = ["not found", "no manifest", "jumbf", "missing",
                                 "no active manifest", "c2pa not found", "no c2pa"]
            if any(kw in error_msg.lower() for kw in no_c2pa_keywords):
                return C2PAResult(video_path=video_path, found=False)
            return C2PAResult(video_path=video_path, found=False, error=error_msg)

    def _parse_manifest(self, video_path: str, manifest: Dict) -> C2PAResult:
        assertions = []
        generator = None
        generator_version = None
        producer = None
        created_at = None

        active_label = manifest.get("active_manifest")
        manifests = manifest.get("manifests", {})
        active = manifests.get(active_label, {}) if active_label else next(iter(manifests.values()), {})

        for assertion_data in active.get("assertions", []):
            label = assertion_data.get("label", "")
            data = assertion_data.get("data", {})
            assertions.append(C2PAAssertion(label=label, data=data))

            if label in ("c2pa.generator.info", "org.c2pa.generator"):
                if isinstance(data, dict):
                    generator = data.get("name") or data.get("product")
                    generator_version = data.get("version")

            if label == "c2pa.actions":
                for action in (data if isinstance(data, list) else []):
                    if isinstance(action, dict) and action.get("when"):
                        created_at = action["when"]
                        break

        claim = active.get("claim", {})
        if isinstance(claim, dict):
            producer = claim.get("dc:publisher") or claim.get("producer")

        for cred in active.get("credentials", []):
            if isinstance(cred, dict):
                subj = cred.get("credentialSubject", {})
                if isinstance(subj, dict):
                    producer = producer or subj.get("name") or subj.get("organization")

        # Sprawdz signature_info jako dodatkowe zrodlo producenta
        sig_info = active.get("signature_info", {})
        if isinstance(sig_info, dict):
            producer = producer or sig_info.get("issuer")

        if generator:
            generator = self._normalize_generator(generator)

        return C2PAResult(
            video_path=video_path,
            found=True,
            generator=generator,
            generator_version=generator_version,
            producer=producer,
            created_at=created_at,
            assertions=assertions,
            raw_manifest=manifest,
        )

    def _normalize_generator(self, raw: str) -> str:
        lower = raw.lower()
        for key, name in self.KNOWN_AI_GENERATORS.items():
            if key in lower:
                return name
        return raw

    def detect_batch(self, video_paths: List[str]) -> List[C2PAResult]:
        results = []
        for i, path in enumerate(video_paths):
            print(f"[C2PADetector] {i + 1}/{len(video_paths)}: {os.path.basename(path)}")
            results.append(self.detect(path))
        return results


# =============================================================================
# Funkcje pomocnicze
# =============================================================================

def detect_c2pa(video_path: str) -> C2PAResult:
    return C2PADetector().detect(video_path)


def print_c2pa_summary(result: C2PAResult):
    s = result.summary()
    print(f"\n=== C2PA: {s['file']} ===")
    print(f"  Manifest C2PA:   {'TAK v' if s['c2pa_found'] else 'NIE x'}")
    print(f"  Generacja AI:    {'TAK' if s['is_ai_generated'] else 'NIE'}")
    print(f"  Generator:       {s['generator']}")
    print(f"  Wersja:          {s['generator_version']}")
    print(f"  Producent:       {s['producer']}")
    print(f"  Data:            {s['created_at']}")
    print(f"  Assertions:      {s['assertions_count']}")
    if s['error'] != 'brak':
        print(f"  Blad:            {s['error']}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Uzycie: python c2pa_detector.py <plik.mp4> [plik2.mp4 ...]")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])

    if not paths:
        print("Nie znaleziono plikow.")
        sys.exit(1)

    detector = C2PADetector()
    results = detector.detect_batch(paths)

    print("\n" + "=" * 60)
    print("PODSUMOWANIE ZBIORCZE")
    print("=" * 60)
    found_count = sum(1 for r in results if r.found)
    print(f"Pliki z C2PA:    {found_count}/{len(results)}")
    print(f"Pliki bez C2PA:  {len(results) - found_count}/{len(results)}")

    for result in results:
        print_c2pa_summary(result)
