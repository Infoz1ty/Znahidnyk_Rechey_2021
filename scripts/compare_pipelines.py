"""Скрипт зведення CSV-метрик усіх 4 пайплайнів та побудови графіків для звіту (B2, C5).

Зона реалізації (Етап 3): Учасник 2 (member2) спільно з Учасником 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_comparison_report(
    metrics_csv_paths: list[Path],
    output_dir: Path,
) -> Path:
    """Агрегує метрики ``PSNR/MSE/SSIM/IoU/Accuracy/runtime_ms`` для П1–П4.

    Генерує зведені таблиці (``P1 vs P2``, ``P3 vs P4``, ``All P1..P4`` у розрізі сценаріїв)
    та діаграми ``matplotlib`` у ``docs/report/``.

    Args:
        metrics_csv_paths: Список шляхів до ``metrics.csv`` із прогонів пайплайнів.
        output_dir: Каталог збереження таблиць та графіків.

    Returns:
        Шлях до згенерованої зведеної CSV/Markdown-таблиці.
    """
    raise NotImplementedError("TODO (T3.1, Команда): реалізувати build_comparison_report.")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу CLI для порівняльного аналізу результатів П1–П4."""
    parser = argparse.ArgumentParser(description="Порівняння метрик пайплайнів П1–П4.")
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/report"))
    args = parser.parse_args(argv)
    build_comparison_report(args.metrics, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
