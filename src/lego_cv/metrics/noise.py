"""Детерміновані генератори тестового шуму (із фіксованим ``seed``) для оцінки фільтрів.

Зона реалізації (Етап 1): Учасник 2 (member2).
"""

from __future__ import annotations

import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 15.0,
    seed: int = 42,
) -> np.ndarray:
    """Накладає адитивний гаусів шум ``N(0, sigma^2)`` із фіксованим ``seed``.

    Args:
        image: Вхідне зображення ``uint8`` (2D grayscale або 3D RGB).
        sigma: Стандартне відхилення шуму (має бути ``>= 0``).
        seed: Зерно генератора ``np.random.default_rng(seed)`` для відтворюваності.

    Returns:
        Зашумлене зображення ``uint8`` тієї самої форми з кліпінгом у ``[0, 255]``.

    Raises:
        ValueError: Якщо ``sigma < 0``.
    """
    if sigma < 0.0:
        raise ValueError(f"Параметр sigma має бути невід'ємним, отримано {sigma}.")

    arr = np.asarray(image, dtype=np.float64)
    if arr.size == 0 or sigma == 0.0:
        return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)

    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=float(sigma), size=arr.shape)
    noisy = np.clip(np.rint(arr + noise), 0.0, 255.0)
    return noisy.astype(np.uint8)


def add_salt_pepper_noise(
    image: np.ndarray,
    prob: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Накладає імпульсний шум «сіль і перець» із фіксованим ``seed``.

    Для багатоканальних зображень імпульс застосовується до всього пікселя ``(y, x)``,
    переводячи його в ``0`` (перець) або ``255`` (сіль) із рівною ймовірністю ``prob / 2``.

    Args:
        image: Вхідне зображення ``uint8``.
        prob: Сумарна частка зашумлених пікселів у діапазоні ``[0.0, 1.0]``.
        seed: Зерно генератора ``np.random.default_rng(seed)``.

    Returns:
        Зашумлене зображення ``uint8`` тієї самої форми.

    Raises:
        ValueError: Якщо ``prob`` виходить за межі ``[0.0, 1.0]``.
    """
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Параметр prob має належати [0.0, 1.0], отримано {prob}.")

    noisy = np.clip(np.asarray(image), 0, 255).astype(np.uint8, copy=True)
    if noisy.size == 0 or prob == 0.0:
        return noisy

    rng = np.random.default_rng(seed)
    spatial_shape = noisy.shape[:2] if noisy.ndim >= 2 else noisy.shape
    samples = rng.random(size=spatial_shape)

    half_prob = float(prob) / 2.0
    pepper_mask = samples < half_prob
    salt_mask = (samples >= half_prob) & (samples < float(prob))

    noisy[pepper_mask] = 0
    noisy[salt_mask] = 255
    return noisy
