"""
download_fp_traps.py

Pobiera filmy z YouTube na podstawie zapytań tekstowych,
idealne do tworzenia bazy 'Adversarial FP Trap' do ewaluacji systemu.

Wymagania:
    pip install yt-dlp

Użycie:
    python download_fp_traps.py
    python download_fp_traps.py --output dataset/adv_fp_trap --per-query 20 --max-duration 180
"""

import os
import csv
import argparse
import yt_dlp
from yt_dlp.utils import match_filter_func


DEFAULT_OUTPUT_DIR = "dataset/adv_fp_trap"
DEFAULT_PER_QUERY  = 15
DEFAULT_MAX_DURATION = 300  # sekund, max 5 minut

# Zapytania skrojone pod testowanie detekcji napisów i Optical Flow.
# Celowo unikamy słów 'AI', 'generated', 'cinematic' - zwracają AI content.
# Chcemy: prawdziwe nagrania z napisami w tle (billboardy, overlay’e TV, tablice).
QUERIES = [
    "Times Square walking tour 1080p",
    "Tokyo street walk dashcam night",
    "live TV news broadcast lower thirds 2023",
    "billboard advertisement street real footage",
    "sports broadcast score overlay real game",   # statyczny wynik w rogu = FP trap
    "subway metro station signs walking",         # tablice informacyjne = FP trap
]


def download_fp_traps(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    num_per_query: int = DEFAULT_PER_QUERY,
    max_duration: int = DEFAULT_MAX_DURATION,
) -> None:
    """
    Pobiera filmy i zapisuje manifest.csv z ground truth label=REAL_FP_TRAP.

    Args:
        output_dir:    Folder docelowy dla filmów i manifestu.
        num_per_query: Ile filmów pobrać na zapytanie.
        max_duration:  Maksymalny czas trwania filmu w sekundach (filtr).
    """
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    downloaded_ids: set = set()

    queries = [f"ytsearch{num_per_query}:{q}" for q in QUERIES]

    ydl_opts = {
        # Max 1080p MP4 - nie zapychamy dysku plikami 4K/8K
        'format': (
            'bestvideo[height<=1080][ext=mp4]'
            '+bestaudio[ext=m4a]'
            '/best[height<=1080][ext=mp4]'
            '/best'
        ),
        # Nazwa pliku: ID_tytul (tytul uciety do 50 znakow)
        'outtmpl': os.path.join(output_dir, '%(id)s_%(title).50s.%(ext)s'),
        'ignoreerrors':   True,   # pomin niedostępne/prywatne filmy
        'no_warnings':    True,
        'quiet':          False,
        'nooverwrites':   True,   # nie pobieraj jesli plik juz istnieje
        'windowsfilenames': True, # bezpieczne nazwy plikow na Windows
        # Odfiltruj filmy dluzsze niz max_duration sekund
        'match_filter': match_filter_func(f'duration < {max_duration}'),
    }

    print(f"[CONFIG] Output:       {output_dir}")
    print(f"[CONFIG] Per query:    {num_per_query}")
    print(f"[CONFIG] Max duration: {max_duration}s ({max_duration // 60}m {max_duration % 60}s)")
    print(f"[CONFIG] Queries:      {len(queries)}")
    print(f"[CONFIG] Max clips:    ~{len(queries) * num_per_query} (przed filtrowaniem)")

    with open(manifest_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'title', 'duration_s', 'url', 'query', 'label'])

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for query in queries:
                print(f"\n[YT-DLP] Zapytanie: {query}")
                try:
                    info = ydl.extract_info(query, download=True)
                    entries = (info or {}).get('entries') or []
                    for entry in entries:
                        if not entry:
                            continue
                        vid_id = entry.get('id', '')
                        if vid_id in downloaded_ids:
                            print(f"  [SKIP] Duplikat: {vid_id}")
                            continue
                        downloaded_ids.add(vid_id)
                        writer.writerow([
                            vid_id,
                            entry.get('title', ''),
                            entry.get('duration', ''),
                            entry.get('webpage_url', ''),
                            query,
                            'REAL_FP_TRAP',  # ground truth: brak watermarku AI
                        ])
                except Exception as e:
                    print(f"[BŁĄD] Zapytanie '{query}': {e}")

    print(f"\n[MANIFEST] Zapisano: {manifest_path}")
    print(f"[MANIFEST] Unikalnych filmów: {len(downloaded_ids)}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pobiera bazę FP Trap z YouTube do ewaluacji detektora watermarków."
    )
    parser.add_argument(
        '--output', '-o',
        default=DEFAULT_OUTPUT_DIR,
        help=f'Folder docelowy (domyślnie: {DEFAULT_OUTPUT_DIR})'
    )
    parser.add_argument(
        '--per-query', '-n',
        type=int, default=DEFAULT_PER_QUERY,
        help=f'Liczba filmów na zapytanie (domyślnie: {DEFAULT_PER_QUERY})'
    )
    parser.add_argument(
        '--max-duration', '-d',
        type=int, default=DEFAULT_MAX_DURATION,
        help=f'Maks. czas trwania w sekundach (domyślnie: {DEFAULT_MAX_DURATION})'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(f"Pobieranie bazy FP Trap → {args.output}")
    download_fp_traps(
        output_dir=args.output,
        num_per_query=args.per_query,
        max_duration=args.max_duration,
    )
    print("\nPobieranie zakończone.")
