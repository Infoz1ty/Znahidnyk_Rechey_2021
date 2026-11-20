"""Оркестратор пакетного запуску пайплайнів, збереження стадій та експорту метрик.

Зона реалізації (Етап 1): Учасник 2 (member2).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.core.types import RunMetrics


@dataclass(slots=True)
class RunConfig:
    """Параметри одного пакетного або одиничного запуску системи.

    Attributes:
        input_path: Файл або каталог із вхідними зображеннями.
        output_dir: Коренева папка для збереження результатів.
        mode: Режим візуалізації (``all`` або ``find``).
        find_query: Запит пошуку для режиму ``find`` (наприклад, ``blue_2x4``).
        save_stages: Чи зберігати проміжні зображення ``stage_1..3``.
        eval_denoise: Чи накладати детермінований шум перед Стадією 1 для ``PSNR/MSE/SSIM``.
        masks_dir: Каталог еталонних масок для обчислення ``IoU``.
        labels_csv: Шлях до ``labels.csv`` для обчислення ``Accuracy`` і ``F1``.
    """

    input_path: Path
    output_dir: Path
    mode: str = "all"
    find_query: str | None = None
    save_stages: bool = True
    eval_denoise: bool = False
    masks_dir: Path | None = None
    labels_csv: Path | None = None


def run_batch(
    pipeline: BasePipeline,
    config: dict[str, Any],
    run_config: RunConfig,
) -> list[RunMetrics]:
    """Запускає пайплайн на всіх зображеннях, зберігає стадії та метрики у CSV/JSON.

    Повинно забезпечувати:
    * стійку обробку пошкоджених файлів (критерій C2) без аварійного падіння;
    * збереження проміжних результатів кожної стадії (``stage_1..3`` PNG);
    * запуск ``shape_classifier``, ``color_classifier`` та ``visualizer``;
    * експорт ``metrics.csv`` та ``summary.json``.

    Args:
        pipeline: Екземпляр пайплайну (``p1``..``p4``).
        config: Об'єднана JSON-конфігурація.
        run_config: Параметри запуску з CLI.

    Returns:
        Список обчислених ``RunMetrics`` для кожного успішно обробленого кадру.
    """
    raise NotImplementedError("TODO (T1.8, Учасник 2): реалізувати run_batch.")
