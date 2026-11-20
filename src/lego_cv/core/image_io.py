"""Введення/виведення зображень та обробка крайніх випадків (критерій C2).

Зона реалізації (Етап 1): Учасник 1 (member1).
Дозволені бібліотеки: виключно ``Pillow`` для декодування/кодування PNG та ``NumPy``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class ImageLoadError(ValueError):
    """Виняток для порожніх, відсутніх, пошкоджених або непідтримуваних файлів."""


class InvalidImageError(ValueError):
    """Виняток для некоректних розмірів, типів даних або каналів масиву зображення."""


def load_image_rgb(
    path: Path,
    *,
    max_side_px: int | None = 1600,
    min_side_px: int = 3,
    max_allowed_side_px: int = 8000,
) -> np.ndarray:
    """Завантажує зображення у форматі ``RGB uint8`` з перевіркою крайніх випадків.

    Повинно коректно обробляти:
    * відсутній або порожній (0 байт) файл -> ``ImageLoadError``;
    * непідтримуване розширення або битий заголовок -> ``ImageLoadError``;
    * 1-канальні монохромні (``L``) та 4-канальні (``RGBA``) зображення -> ``RGB``;
    * замалі (``< min_side_px``) або надто великі розміри -> ``InvalidImageError``.

    Args:
        path: Шлях до файлу зображення.
        max_side_px: Якщо задано — пропорційно зменшити довгу сторону до цього розміру.
        min_side_px: Мінімально допустима ширина/висота кадру.
        max_allowed_side_px: Гранично допустима сторона вхідного файлу.

    Returns:
        Масив NumPy форми ``(H, W, 3)`` типу ``uint8``.
    """
    raise NotImplementedError("TODO (T1.1, Учасник 1): реалізувати load_image_rgb.")


def load_mask_binary(path: Path) -> np.ndarray:
    """Завантажує еталонну бінарну маску для обчислення ``IoU``.

    Args:
        path: Шлях до 1-канального PNG-файлу еталонної маски.

    Returns:
        Бінарний масив NumPy форми ``(H, W)`` типу ``uint8`` зі значеннями ``0`` або ``1``.
    """
    raise NotImplementedError("TODO (T1.1, Учасник 1): реалізувати load_mask_binary.")


def save_image(image: np.ndarray, path: Path) -> None:
    """Зберігає масив NumPy (grayscale або RGB) у файл PNG, створюючи батьківські каталоги.

    Args:
        image: Зображення форми ``(H, W)`` або ``(H, W, 3)``.
        path: Цільовий шлях для збереження файлу.
    """
    raise NotImplementedError("TODO (T1.1, Учасник 1): реалізувати save_image.")


def discover_images(input_path: Path) -> list[Path]:
    """Повертає відсортований перелік зображень із файлу або каталогу.

    Args:
        input_path: Шлях до одного файлу або папки із вибіркою.

    Returns:
        Список шляхів до файлів із підтримуваними розширеннями.
    """
    raise NotImplementedError("TODO (T1.1, Учасник 1): реалізувати discover_images.")
