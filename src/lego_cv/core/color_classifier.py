"""Модуль Б: Спільний класифікатор кольору деталей LEGO (Крок 2).

Зона реалізації (Етап 1): Учасник 2 (member2).
Усі операції (ерозія 3x3, RGB->HSV, медіанна класифікація) реалізуються вручну на NumPy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.types import BrickDetection

if TYPE_CHECKING:
    import numpy as np


def binary_erode_manual(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Ручна бінарна морфологічна ерозія квадратним ядром ``kernel_size x kernel_size``.

    Використовується для звуження маски на 1–3 пікселі, щоб не захоплювати тінь по краях.

    Args:
        mask: Булева або бінарна маска ``(H, W)``.
        kernel_size: Непарний розмір ядра (за замовчуванням 3).

    Returns:
        Еродована булева маска ``(H, W)``.
    """
    raise NotImplementedError("TODO (T1.7, Учасник 2): реалізувати binary_erode_manual.")


def rgb_to_hsv_manual(image_rgb: np.ndarray) -> np.ndarray:
    """Ручне аналітичне перетворення ``RGB [0..255]`` у ``HSV`` (``H: 0..360, S/V: 0..255``).

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.

    Returns:
        Масив ``float32`` форми ``(H, W, 3)`` у просторі HSV.
    """
    raise NotImplementedError("TODO (T1.7, Учасник 2): реалізувати rgb_to_hsv_manual.")


def classify_colors(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray,
    detections: list[BrickDetection],
    config: dict[str, Any],
) -> list[BrickDetection]:
    """Визначає колір кожної знайденої деталі за медіаною HSV усередині еродованої маски.

    Відкидає темні пікселі заглиблень (``V < shadow_v_min``) та відблиски
    (``S < glare_s_max`` при ``V > glare_v_min``). Для чорно-білих кадрів
    (``R == G == B``, наприклад, 3D-рендерів Kaggle) повертає ``"Grayscale"``,
    не перериваючи Крок 1.

    Args:
        image_rgb: Початкове кольорове RGB-зображення ``uint8``.
        instance_mask: Інстанс-маска деталей (``0`` — фон, ``brick_id`` — пікселі деталі).
        detections: Список ``BrickDetection``, отриманий на Кроці 1.
        config: Повна конфігурація (секція ``color_classifier``).

    Returns:
        Оновлений список ``BrickDetection`` із заповненим полем ``color``.
    """
    raise NotImplementedError("TODO (T1.7, Учасник 2): реалізувати classify_colors.")
