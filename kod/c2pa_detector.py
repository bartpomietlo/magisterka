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
    label: str          # np. "c2pa.actions", "c2pa.generator.info"
    data: Any           # Dane twierdzenia (dict lub prymityw)


@dataclass
class C2PAResult:
    """Wynik detekcji C2PA dla jednego pliku wideo."""
    video_path: str
    found: bool                                  # Czy manifest C2PA istnieje
    generator: Optional[str] = None             # Nazwa generatora (np. "Sora", "Runway")
    generator_version: Optional[str] = None     # Wersja generatora
    producer: Optional[str] = None              # Producent/firma
    created_at: Optional[str] = None            # Data utworzenia wg C2PA
    assertions: List[C2PAAssertion] = field(default_factory=list)
    raw_manifest: Optional[Dict] = None         # Surowy manifest JSON
    error: Optional[str] = None                 # Błąd parsowania (jeśli był)

    @property
    def is_ai_generated(self) -> bool:
        """Zwraca True jeśli manifest wskazuje na generację AI."""
        if not self.found:
            return False
        # Szukaj assertion c2pa.actions z akcją "c2pa.created"
        for assertion in self.assertions:
            if assertion.label == "c2pa.actions":
                actions = assertion.data if isinstance(assertion.data, list) else []
                for action in actions:
                    if isinstance(action, dict) and action.get("action") == "c2pa.created":
                        return True
        # Alternatywnie: jeśli jest generator info, uznajemy za AI
        return self.generator is not None

    def summary(self) -> Dict[str, Any]:
        """Zwraca czytelne podsumowanie wyniku."""
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
# Główna klasa detektora
# =============================================================================

class C2PADetector:
    """
    Detektor metadanych C2PA w plikach wideo.

    C2PA (Coalition for Content Provenance and Authenticity) to otwarty standard
    do oznaczania treści danymi o jej pochodzeniu. Pliki wideo wygenerowane przez
    Sora, Runway, Adobe Firefly i inne narzędzia AI mogą zawierać manifest C2PA
    osadzony w metadanych pliku MP4/MOV.

    Użycie:
        detector = C2PADetector()
        result = detector.detect("video.mp4")
        print(result.summary())
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

    def __init__(self):
        if not C2PA_AVAILABLE:
            print("[C2PADetector] Biblioteka c2pa-python niedostępna — detekcja będzie zwracać found=False.")

    def detect(self, video_path: str) -> C2PAResult:
        """
        Odczytuje manifest C2PA z pliku wideo.

        Args:
            video_path: Ścieżka do pliku MP4/MOV.

        Returns:
            C2PAResult z informacjami o proweniencji.
        """
        if not os.path.exists(video_path):
            return C2PAResult(
                video_path=video_path,
                found=False,
                error=f"Plik nie istnieje: {video_path}"
            )

        if not C2PA_AVAILABLE:
            return C2PAResult(
                video_path=video_path,
                found=False,
                error="Biblioteka c2pa-python nie jest zainstalowana"
            )

        try:
            # Odczytaj manifest C2PA z pliku
            manifest_json = c2pa.read_file(video_path)

            if manifest_json is None:
                return C2PAResult(video_path=video_path, found=False)

            # Parsuj manifest
            manifest = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json

            return self._parse_manifest(video_path, manifest)

        except Exception as e:
            error_msg = str(e)
            # C2PA nie znaleziono w pliku — to normalny przypadek dla filmów bez C2PA
            if any(kw in error_msg.lower() for kw in ["not found", "no manifest", "jumbf", "missing"]):
                return C2PAResult(video_path=video_path, found=False)
            # Inny błąd
            return C2PAResult(video_path=video_path, found=False, error=error_msg)

    def _parse_manifest(self, video_path: str, manifest: Dict) -> C2PAResult:
        """Parsuje surowy manifest C2PA do struktury C2PAResult."""
        assertions = []
        generator = None
        generator_version = None
        producer = None
        created_at = None

        # Aktywny manifest (najnowszy)
        active_label = manifest.get("active_manifest")
        manifests = manifest.get("manifests", {})
        active = manifests.get(active_label, {}) if active_label else next(iter(manifests.values()), {})

        # --- Assertions ---
        for assertion_data in active.get("assertions", []):
            label = assertion_data.get("label", "")
            data = assertion_data.get("data", {})
            assertions.append(C2PAAssertion(label=label, data=data))

            # Wyciągnij generator info
            if label in ("c2pa.generator.info", "org.c2pa.generator"):
                if isinstance(data, dict):
                    generator = data.get("name") or data.get("product")
                    generator_version = data.get("version")

            # Data stworzenia z akcji
            if label == "c2pa.actions":
                for action in (data if isinstance(data, list) else []):
                    if isinstance(action, dict) and action.get("when"):
                        created_at = action["when"]
                        break

        # --- Claim ---
        claim = active.get("claim", {})
        if isinstance(claim, dict):
            # Producent może być w claim lub w credentials
            producer = claim.get("dc:publisher") or claim.get("producer")

        # --- Credentials / author ---
        for cred in active.get("credentials", []):
            if isinstance(cred, dict):
                subj = cred.get("credentialSubject", {})
                if isinstance(subj, dict):
                    producer = producer or subj.get("name") or subj.get("organization")

        # --- Normalizacja generatora ---
        if generator:
            normalized = self._normalize_generator(generator)
            if normalized:
                generator = normalized

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

    def _normalize_generator(self, raw: str) -> Optional[str]:
        """Normalizuje nazwę generatora do czytelnej formy."""
        lower = raw.lower()
        for key, name in self.KNOWN_AI_GENERATORS.items():
            if key in lower:
                return name
        return raw  # Zwróć oryginał jeśli nie pasuje do żadnego

    def detect_batch(self, video_paths: List[str]) -> List[C2PAResult]:
        """
        Analizuje listę plików wideo.

        Args:
            video_paths: Lista ścieżek do plików MP4/MOV.

        Returns:
            Lista wyników C2PAResult.
        """
        results = []
        for i, path in enumerate(video_paths):
            print(f"[C2PADetector] {i + 1}/{len(video_paths)}: {os.path.basename(path)}")
            results.append(self.detect(path))
        return results


# =============================================================================
# Funkcje pomocnicze
# =============================================================================

def detect_c2pa(video_path: str) -> C2PAResult:
    """
    Skrót do jednorazowej detekcji C2PA.

    Args:
        video_path: Ścieżka do pliku MP4.

    Returns:
        C2PAResult.
    """
    return C2PADetector().detect(video_path)


def print_c2pa_summary(result: C2PAResult):
    """Wypisuje czytelne podsumowanie wyniku C2PA."""
    s = result.summary()
    print(f"\n=== C2PA: {s['file']} ===")
    print(f"  Manifest C2PA:   {'TAK ✓' if s['c2pa_found'] else 'NIE ✗'}")
    print(f"  Generacja AI:    {'TAK' if s['is_ai_generated'] else 'NIE'}")
    print(f"  Generator:       {s['generator']}")
    print(f"  Wersja:          {s['generator_version']}")
    print(f"  Producent:       {s['producer']}")
    print(f"  Data:            {s['created_at']}")
    print(f"  Assertions:      {s['assertions_count']}")
    if s['error'] != 'brak':
        print(f"  Błąd:            {s['error']}")


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Użycie: python c2pa_detector.py <plik.mp4> [plik2.mp4 ...]")
        print("Przykład: python c2pa_detector.py ../nagrania/*.mp4")
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        expanded = glob.glob(arg)
        paths.extend(expanded if expanded else [arg])

    if not paths:
        print("Nie znaleziono plików.")
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
