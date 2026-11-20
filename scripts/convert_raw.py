"""Офлайн-скрипт конвертації сирих фото (.nef / .dng) у 8-бітні PNG (1600 px).

Зона реалізації (Етап 1): Учасник 2 (member2).
Примітка: використовує опційну залежність ``rawpy`` (встановлення: ``pip install -e '.[prep]'``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def convert_raw_directory(
    input_dir: Path,
    output_dir: Path,
    max_side_px: int = 1600,
) -> int:
    """Конвертує всі ``.nef``/``.dng`` файли в 8-бітні PNG без автопокращень.

    Args:
        input_dir: Каталог із сирими RAW-файлами.
        output_dir: Цільовий каталог для збереження PNG.
        max_side_px: Максимальний розмір довгої сторони після ресайзу.

    Returns:
        Кількість успішно сконвертованих файлів.
    """
    raise NotImplementedError("TODO (T1.10, Учасник 2): реалізувати convert_raw_directory.")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу CLI для конвертації RAW-файлів."""
    parser = argparse.ArgumentParser(description="Конвертер RAW (.nef/.dng) -> 8-bit PNG.")
    parser.add_argument("--input", type=Path, required=True, help="Папка з RAW-файлами.")
    parser.add_argument("--output", type=Path, required=True, help="Папка для PNG.")
    parser.add_argument("--max-side", type=int, default=1600, help="Довга сторона у px.")
    args = parser.parse_args(argv)
    convert_raw_directory(args.input, args.output, max_side_px=args.max_side)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
