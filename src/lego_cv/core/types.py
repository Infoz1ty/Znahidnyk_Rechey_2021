"""Контракти даних (dataclasses) для обміну між стадіями пайплайнів та ядром."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


@dataclass(slots=True)
class StageOutput:
    """Результат виконання однієї стадії пайплайну.

    Attributes:
        name: Машинна назва стадії для імені файлу (наприклад, ``stage_1_median``).
        category: Категорія методу за методичкою (наприклад, ``Фільтрація зображень``).
        image: Проміжне зображення стадії у вигляді масиву NumPy.
        duration_ms: Тривалість виконання стадії в мілісекундах.
    """

    name: str
    category: str
    image: np.ndarray
    duration_ms: float


@dataclass(slots=True)
class PipelineResult:
    """Повний вихід одного пайплайну обробки зображення.

    Attributes:
        pipeline_id: Ідентифікатор пайплайну (``p1``, ``p2``, ``p3`` або ``p4``).
        mask: Підсумкова бінарна (0/1) або інстанс-маска (0, 1..K) типу ``int32``.
        denoised: Зображення після Стадії 1 для оцінки ``PSNR``/``MSE``/``SSIM``.
        stages: Упорядкований перелік проміжних результатів кожної стадії.
        metadata: Додаткові дані пайплайну (наприклад, ``stud_counts`` та ``glcm`` з П4).
    """

    pipeline_id: str
    mask: np.ndarray
    denoised: np.ndarray
    stages: list[StageOutput] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BrickDetection:
    """Опис однієї знайденої та класифікованої деталі LEGO.

    Attributes:
        brick_id: Локальний ідентифікатор деталі на кадрі (починаючи з 1).
        shape_class: Геометричний клас (наприклад, ``2x4``, ``1x1_round``).
        surface_type: Тип поверхні (``brick``, ``tile``, ``round`` або ``unknown``).
        color: Визначений колір (``Red``, ``Blue``, ``Yellow`` тощо або ``Unknown``).
        stud_count: Кількість виявлених шпильок (якщо обчислювалась).
        area_px: Площа деталі в пікселях.
        aspect_ratio: Співвідношення сторін орієнтованого прямокутника ``max/min``.
        circularity: Коефіцієнт окружності ``4 * pi * S / P^2``.
        bbox: Обмежувальний прямокутник ``(x_min, y_min, x_max, y_max)``.
        center: Координати центра мас ``(cx, cy)``.
    """

    brick_id: int
    shape_class: str
    surface_type: str
    color: str
    stud_count: int
    area_px: float
    aspect_ratio: float
    circularity: float
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass(slots=True)
class RunMetrics:
    """Метрики якості та продуктивності одного запуску на одному зображенні.

    Attributes:
        image_name: Ім'я або відносний шлях обробленого зображення.
        pipeline_id: Ідентифікатор пайплайну.
        psnr: Пікове відношення сигнал/шум (дБ) для Стадії 1.
        mse: Середньоквадратична помилка для Стадії 1.
        ssim: Індекс структурної подібності для Стадії 1.
        iou: Intersection over Union фінальної маски проти еталонної.
        accuracy: Точність класифікації знайдених деталей (за наявності ``labels.csv``).
        f1_score: F1-міра класифікації (за наявності ``labels.csv``).
        total_ms: Сумарний час виконання пайплайну в мілісекундах.
        detected_count: Кількість виявлених деталей у кадрі.
    """

    image_name: str
    pipeline_id: str
    psnr: float | None = None
    mse: float | None = None
    ssim: float | None = None
    iou: float | None = None
    accuracy: float | None = None
    f1_score: float | None = None
    total_ms: float = 0.0
    detected_count: int = 0
