"""Скрипт автогенерації еталонних бінарних масок для 3D-рендерів Kaggle (70–80 фото).

Зона реалізації (Етап 1): Учасник 2 (member2).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def generate_kaggle_masks(
    kaggle_dir: Path,
    masks_dir: Path,
    bg_tolerance: int = 12,
) -> int:
    """Формує 1-канальні бінарні маски (0/255) для чистих top-down рендерів Kaggle.

    Args:
        kaggle_dir: Каталог ``dataset/kaggle_topdown/``.
        masks_dir: Цільовий каталог ``dataset/kaggle_masks/``.
        bg_tolerance: Допуск відхилення інтенсивності від однорідного фону рендера.

    Returns:
        Кількість згенерованих масок.
    """
    raise NotImplementedError("TODO (T1.10, Учасник 2): реалізувати generate_kaggle_masks.")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу CLI для генерації еталонних масок Kaggle."""
    parser = argparse.ArgumentParser(description="Генератор еталонних масок Kaggle.")
    parser.add_argument("--input", type=Path, default=Path("dataset/kaggle_topdown"))
    parser.add_argument("--output", type=Path, default=Path("dataset/kaggle_masks"))
    parser.add_argument("--tolerance", type=int, default=12)
    args = parser.parse_args(argv)
    generate_kaggle_masks(args.input, args.output, bg_tolerance=args.tolerance)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
