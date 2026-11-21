"""Ручна реалізація кількісних метрик якості обробки зображень (PSNR, MSE, SSIM, IoU, F1).

Зона реалізації (Етап 1): Учасник 2 (member2).
Формули відповідають розділу «Аналіз результатів обробки зображень» методички.
"""

from __future__ import annotations

from collections import Counter
import math

import numpy as np


def _as_matched_float_arrays(
    reference: np.ndarray,
    processed: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Приводить два масиви до ``float64`` та перевіряє збіг форм і непорожність."""
    ref = np.asarray(reference, dtype=np.float64)
    proc = np.asarray(processed, dtype=np.float64)
    if ref.shape != proc.shape:
        raise ValueError(
            f"Форми зображень мають збігатися: отримано {ref.shape} та {proc.shape}."
        )
    if ref.size == 0:
        raise ValueError("Неможливо обчислити метрику для порожнього зображення.")
    return ref, proc


def compute_mse(reference: np.ndarray, processed: np.ndarray) -> float:
    """Обчислює середньоквадратичну помилку ``MSE = (1/N) * sum((I - I_proc)^2)``.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.

    Returns:
        Невід'ємне значення MSE.

    Raises:
        ValueError: Якщо форми зображень не збігаються або масиви порожні.
    """
    ref, proc = _as_matched_float_arrays(reference, processed)
    diff = ref - proc
    return float(np.mean(diff * diff))


def compute_psnr(
    reference: np.ndarray,
    processed: np.ndarray,
    max_pixel_value: float = 255.0,
) -> float:
    """Обчислює пікове відношення сигнал/шум ``PSNR = 10 * log10(MAX^2 / MSE)`` (дБ).

    При ``MSE == 0`` повертає ``float('inf')`` без ділення на нуль.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.
        max_pixel_value: Максимальна інтенсивність пікселя (255 для 8-біт).

    Returns:
        Значення PSNR у децибелах.

    Raises:
        ValueError: Якщо ``max_pixel_value <= 0`` або форми зображень не збігаються.
    """
    if max_pixel_value <= 0.0:
        raise ValueError(
            f"max_pixel_value має бути додатним, отримано {max_pixel_value}."
        )

    mse = compute_mse(reference, processed)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * math.log10((max_pixel_value * max_pixel_value) / mse))


def _ssim_single_plane(
    plane_x: np.ndarray,
    plane_y: np.ndarray,
    c1: float,
    c2: float,
) -> float:
    """Обчислює глобальний SSIM для одного 2D-каналу за формулою методички."""
    mu_x = float(np.mean(plane_x))
    mu_y = float(np.mean(plane_y))
    centered_x = plane_x - mu_x
    centered_y = plane_y - mu_y

    var_x = float(np.mean(centered_x * centered_x))
    var_y = float(np.mean(centered_y * centered_y))
    cov_xy = float(np.mean(centered_x * centered_y))

    numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2)
    denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2)
    if denominator == 0.0:
        return 1.0
    value = numerator / denominator
    return float(max(-1.0, min(1.0, value)))


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
    Для багатоканальних RGB-зображень повертає середнє SSIM по каналах.

    Args:
        reference: Еталонне зображення.
        processed: Оброблене зображення тієї самої форми.
        max_pixel_value: Динамічний діапазон ``L`` (255 для 8-біт).

    Returns:
        Значення SSIM (1.0 — повна ідентичність).

    Raises:
        ValueError: Якщо ``max_pixel_value <= 0`` або форми зображень не збігаються.
    """
    if max_pixel_value <= 0.0:
        raise ValueError(
            f"max_pixel_value має бути додатним, отримано {max_pixel_value}."
        )

    ref, proc = _as_matched_float_arrays(reference, processed)
    c1 = (0.01 * max_pixel_value) ** 2
    c2 = (0.03 * max_pixel_value) ** 2

    if ref.ndim == 3:
        channel_scores = [
            _ssim_single_plane(ref[..., ch], proc[..., ch], c1, c2)
            for ch in range(ref.shape[2])
        ]
        return float(sum(channel_scores) / len(channel_scores))

    return _ssim_single_plane(ref, proc, c1, c2)


def compute_iou(predicted_mask: np.ndarray, ground_truth_mask: np.ndarray) -> float:
    """Обчислює Intersection over Union (``IoU``) між бінарними або інстанс-масками.

    Підтримує булеві, бінарні (``0/1``, ``0/255``) та інстанс-маски (``0, 1..K``),
    де будь-яке значення ``> 0`` трактується як передній план.

    Args:
        predicted_mask: Передбачена маска (``> 0`` трактується як передній план).
        ground_truth_mask: Еталонна маска (``> 0`` трактується як передній план).

    Returns:
        Значення IoU в діапазоні ``[0.0, 1.0]`` (1.0 якщо обидві маски порожні).

    Raises:
        ValueError: Якщо форми масок не збігаються.
    """
    pred_arr = np.asarray(predicted_mask)
    gt_arr = np.asarray(ground_truth_mask)
    if pred_arr.shape != gt_arr.shape:
        raise ValueError(
            f"Форми масок мають збігатися: отримано {pred_arr.shape} та {gt_arr.shape}."
        )

    pred_fg = pred_arr > 0
    gt_fg = gt_arr > 0
    intersection = int(np.count_nonzero(pred_fg & gt_fg))
    union = int(np.count_nonzero(pred_fg | gt_fg))
    if union == 0:
        return 1.0
    return float(intersection / union)


def compute_classification_metrics(
    predicted_labels: list[str],
    ground_truth_labels: list[str],
) -> dict[str, float]:
    """Обчислює мультимножинні ``accuracy``, ``precision``, ``recall`` та ``f1_score``.

    Порівнює набір знайдених міток ``color_shape`` (або ``shape``) на кадрі зі
    списком еталонних міток із ``dataset/labels.csv`` без урахування регістру.

    Args:
        predicted_labels: Список передбачених міток на зображенні.
        ground_truth_labels: Список еталонних міток на зображенні.

    Returns:
        Словник із ключами ``accuracy``, ``precision``, ``recall``, ``f1_score``.
    """
    pred_counts = Counter(label.strip().lower() for label in predicted_labels)
    gt_counts = Counter(label.strip().lower() for label in ground_truth_labels)

    true_positives = sum((pred_counts & gt_counts).values())
    false_positives = sum(pred_counts.values()) - true_positives
    false_negatives = sum(gt_counts.values()) - true_positives

    total = true_positives + false_positives + false_negatives
    if total == 0:
        return {
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1_score": 1.0,
        }

    precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0.0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0.0
    )
    f1_score = (
        (2.0 * precision * recall) / (precision + recall)
        if (precision + recall) > 0.0
        else 0.0
    )
    accuracy = true_positives / total

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1_score),
    }
