"""Модуль В: Візуалізація результатів розпізнавання та пошуку (``--mode all`` / ``--mode find``).

Зона реалізації (Етап 1): Учасник 2 (member2).
Дозволені бібліотеки: ``Pillow`` (``ImageDraw``, ``ImageFont``) та ``NumPy``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.types import BrickDetection

if TYPE_CHECKING:
    import numpy as np


def render_all_detections(
    image_rgb: np.ndarray,
    detections: list[BrickDetection],
    config: dict[str, Any],
) -> np.ndarray:
    """Малює обмежувальні рамки та підписи класу і кольору для всіх знайдених деталей.

    Args:
        image_rgb: Вихідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        detections: Повний список класифікованих деталей.
        config: Конфігурація (секція ``visualizer``).

    Returns:
        Анотоване RGB-зображення ``uint8``.
    """
    raise NotImplementedError("TODO (T1.8, Учасник 2): реалізувати render_all_detections.")


def render_search_highlight(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray,
    detections: list[BrickDetection],
    query: str,
    config: dict[str, Any],
) -> np.ndarray:
    """Режим пошуку ``--find``: затемнює фон на 70% і підсвічує лише шукані деталі.

    Args:
        image_rgb: Вихідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        instance_mask: Інстанс-маска деталей ``(H, W)`` типу ``int32``.
        detections: Список класифікованих деталей.
        query: Пошуковий запит (наприклад, ``blue_2x4`` або ``yellow_1x1_round``).
        config: Конфігурація (секція ``visualizer``).

    Returns:
        RGB-зображення ``uint8`` із підсвіченими збігами.
    """
    raise NotImplementedError(
        "TODO (T1.8, Учасник 2): реалізувати render_search_highlight."
    )
