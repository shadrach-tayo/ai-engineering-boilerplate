"""Download the Amazon Fine Food Reviews dataset into ./data (gitignored)."""

from __future__ import annotations

import shutil
from pathlib import Path

import kagglehub  # type: ignore[import-untyped]

DATASET = "snap/amazon-fine-food-reviews"
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "amazon-fine-food-reviews"


def main() -> None:
    """Fetch the Kaggle dataset and copy it into the project data directory."""
    cache_path = Path(kagglehub.dataset_download(DATASET))
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for item in cache_path.iterdir():
        dest = DATA_DIR / item.name
        if item.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    print(f"Dataset ready at: {DATA_DIR}")  # noqa: T201
    for path in sorted(DATA_DIR.rglob("*")):
        if path.is_file():
            print(  # noqa: T201
                f"  - {path.relative_to(DATA_DIR)} ({path.stat().st_size:,} bytes)"
            )


if __name__ == "__main__":
    main()
