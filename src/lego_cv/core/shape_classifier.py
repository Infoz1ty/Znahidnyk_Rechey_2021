"""Модуль А: Спільний геометричний класифікатор форми LEGO (фінал Кроку 1).

Зона реалізації (Етап 1): Учасник 1 (member1).
Усі алгоритми реалізуються вручну на NumPy без ``cv2.connectedComponents`` та ``cv2.minAreaRect``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.types import BrickDetection

if TYPE_CHECKING:
    import numpy as np


def connected_components_label(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Ручне маркування 8-зв'язних компонентів (2-pass Union-Find або BFS).

    Args:
        binary_mask: Бінарна маска форми ``(H, W)`` (``0`` — фон, ``> 0`` — об'єкт).

    Returns:
        Кортеж ``(labels, num_labels)``, де ``labels`` має форму ``(H, W)`` та тип ``int32``
        (``0`` — фон, ``1..num_labels`` — ID компонентів).
    """
    raise NotImplementedError(
        "TODO (T1.3, Учасник 1): реалізувати ручний connected_components_label."
    )


def compute_component_geometry(component_mask: np.ndarray) -> dict[str, Any]:
    """Обчислює геометричні ознаки одного зв'язного компонента вручну.

    Ознаки:
    * площа ``area_px``;
    * периметр ``perimeter_px`` (обхід межі компонента);
    * коефіцієнт окружності ``circularity = 4 * pi * S / P^2``;
    * орієнтований прямокутник через PCA головних осей -> ``aspect_ratio``;
    * обмежувальний прямокутник ``bbox`` та центр мас ``center``.

    Args:
        component_mask: Булева маска ``(H, W)`` однієї деталі.

    Returns:
        Словник геометричних ознак.
    """
    raise NotImplementedError(
        "TODO (T1.3, Учасник 1): реалізувати compute_component_geometry."
    )


def classify_shapes(
    mask: np.ndarray,
    config: dict[str, Any],
    pipeline_metadata: dict[str, Any] | None = None,
) -> list[BrickDetection]:
    """Класифікує геометричну форму й тип поверхні всіх деталей на масці.

    Відкидає сміття з площею ``S < min_area_ratio * stud_size_px^2`` та заповнює
    поля форми ``BrickDetection`` (поле ``color`` заповнюється на Кроці 2).

    Args:
        mask: Бінарна (0/1) або інстанс-маска (0, 1..K) типу ``int32``.
        config: Повний словник конфігурації (секції ``preprocessing`` та ``shape_classifier``).
        pipeline_metadata: Необов'язкові метадані пайплайну (напр., ``stud_counts`` із П4).

    Returns:
        Список попередньо заповнених ``BrickDetection``.
    """
    raise NotImplementedError("TODO (T1.3, Учасник 1): реалізувати classify_shapes.")
