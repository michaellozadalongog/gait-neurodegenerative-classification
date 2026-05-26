"""
Download the PhysioNet Gait in Neurodegenerative Disease Database.
Run this ONCE. ~18 MB total.

Usage:
    python src/01_download_data.py
"""
import os
import urllib.request
from pathlib import Path

BASE_URL = "https://physionet.org/files/gaitndd/1.0.0"
DATA_DIR = Path(__file__).parent.parent / "data" / "gaitndd"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Subject groups and counts (from PhysioNet docs)
GROUPS = {
    "control": 16,
    "park": 15,
    "hunt": 20,
    "als": 13,
}

# Files to grab for each subject
SUFFIXES = [".ts", ".hea"]  # .ts = time series, .hea = header

# Files at the dataset root
ROOT_FILES = ["subject-description.txt", "RECORDS"]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  fetching: {dest.name}")
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    print(f"Downloading to {DATA_DIR}")
    print("Root files:")
    for fname in ROOT_FILES:
        download(f"{BASE_URL}/{fname}", DATA_DIR / fname)

    print("\nSubject files:")
    for prefix, count in GROUPS.items():
        for i in range(1, count + 1):
            for suffix in SUFFIXES:
                fname = f"{prefix}{i}{suffix}"
                download(f"{BASE_URL}/{fname}", DATA_DIR / fname)

    print(f"\nDone. {sum(GROUPS.values())} subjects total.")


if __name__ == "__main__":
    main()
