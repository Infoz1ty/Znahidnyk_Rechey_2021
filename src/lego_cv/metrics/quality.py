"""Ручна реалізація кількісних метрик якості обробки зображень (PSNR, MSE, SSIM, IoU, F1).

Зона реалізації (Етап 1): Учасник 2 (member2).
Формули відповідають розділу «Аналіз результатів обробки зображень» методички.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """Обчислює середньоквадратичну помилку ``MSE = (1/N) * sum((I - I_proc)^2)``.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.

    Returns:
        Невід'ємне значення MSE.
    """
    raise NotImplementedError("TODO (T1.6, Учасник 2): реалізувати compute_mse.")


def compute_psnr(
    reference: np.ndarray,
    processed: np.ndarray,
    max_pixel_value: float = 255.0,
) -> float:
    """Обчислює пікове відношення сигнал/шум ``PSNR = 10 * log10(MAX^2 / MSE)`` (дБ).

    При ``MSE == 0`` повинно повертати ``float('inf')`` без ділення на нуль.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.
        max_pixel_value: Максимальна інтенсивність пікселя (255 для 8-біт).

    Returns:
        Значення PSNR у децибелах.
    """
    raise NotImplementedError("TODO (T1.6, Учасник 2): реалізувати compute_psnr.")


def compute_ssim(
    reference: np.ndarray,
    processed: np.ndarray,
    max_pixel_value: float = 255.0,
) -> float:
    """Обчислює індекс структурної подібності ``SSIM`` у діапазоні ``[-1, 1]``.

    Формула:
        ``((2*mu_x*mu_y + C1)*(2*sigma_xy + C2))`` /
        ``((mu_x^2 + mu_y^2 + C1)*(sigma_x^2 + sigma_y^2 + C2))``,
    де ``C1 = (0.01 * L)^2``, ``C2 = (0.03 * L)^2``.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.
        max_pixel_value: Динамічний діапазон ``L`` (255 для 8-біт).

    Returns:
        Значення SSIM (1.0 — повна ідентичність).
    """
    raise NotImplementedError("TODO (T1.6, Учасник 2): реалізувати compute_ssim.")


def compute_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Обчислює Intersection over Union (``IoU``) між бінарними масками.

    Args:
        predicted_mask: Передбачена маска (``> 0`` трактується як передній план).
        ground_truth_mask: Еталонна маска (``> 0`` трактується як передній план).

    Returns:
        Значення IoU в діапазоні ``[0.0, 1.0]`` (1.0 якщо обидві маски порожні).
    """
    raise NotImplementedError("TODO (T1.6, Учасник 2): реалізувати compute_iou.")


def compute_classification_metrics(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
) -> dict[str, float]:
    """Обчислює мультимножинні ``accuracy``, ``precision``, ``recall`` та ``f1_score``.

    Порівнює набір знайдених міток ``color_shape`` (або ``shape``) на кадрі зі
    списком еталонних міток із ``dataset/labels.csv``.

    Args:
        predicted_labels: Список передбачених міток на зображенні.
        ground_truth_labels: Список еталонних міток на зображенні.

    Returns:
        Словник із ключами ``accuracy``, ``precision``, ``recall``, ``f1_score``.
    """
    raise NotImplementedError(
        "TODO (T1.6, Учасник 2): реалізувати compute_classification_metrics."
    )
