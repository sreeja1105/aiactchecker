"""
Download EU AI Act and related compliance documents into docs/.

Run from project root:
    python scripts/download_docs.py
"""

import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)

DOCUMENTS = {
    "eu_ai_act.pdf": "https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=OJ:L_202401689",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MIN_VALID_SIZE_BYTES = 100_000


def download_file(url: str, dest: Path) -> bool:
    print(f"[download] {dest.name}")
    print(f"  url: {url}")
    try:
        response = requests.get(url, stream=True, timeout=120, allow_redirects=True, headers=HEADERS)
        response.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        size = dest.stat().st_size
        if size < MIN_VALID_SIZE_BYTES:
            print(f"  failed: response too small ({size} bytes)")
            dest.unlink()
            return False

        print(f"  saved ({size / (1024 * 1024):.2f} MB)")
        return True
    except Exception as e:
        print(f"  failed: {e}")
        if dest.exists():
            dest.unlink()
        return False


def main():
    print(f"Downloading {len(DOCUMENTS)} document(s) into {DOCS_DIR}")
    print("-" * 78)

    success = 0
    for filename, url in DOCUMENTS.items():
        dest = DOCS_DIR / filename
        if dest.exists() and dest.stat().st_size > MIN_VALID_SIZE_BYTES:
            mb = dest.stat().st_size / (1024 * 1024)
            print(f"[skip] {filename} already exists ({mb:.2f} MB)")
            success += 1
            continue
        if download_file(url, dest):
            success += 1

    print()
    print(f"Done. {success}/{len(DOCUMENTS)} documents available.")
    if success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
