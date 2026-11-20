"""Детерміновані генератори тестового шуму (із фіксованим ``seed``) для оцінки фільтрів.

Зона реалізації (Етап 1): Учасник 2 (member2).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def add_gaussian_noise(
    image: np.ndarray,
    sigma: float = 15.0,
    seed: int = 42,
) -> np.ndarray:
    """Накладає адитивний гаусів шум ``N(0, sigma^2)`` із фіксованим ``seed``.

    Args:
        image: Вхідне зображення ``uint8``.
        sigma: Стандартне відхилення шуму.
        seed: Зерно генератора ``np.random.default_rng(seed)`` для відтворюваності.

    Returns:
        Зашумлене зображення ``uint8`` тієї самої форми.
    """
    raise NotImplementedError("TODO (T1.6, Учасник 2): реалізувати add_gaussian_noise.")


def add_salt_pepper_noise(
    image: np.ndarray,
    prob: float = 0.02,
    seed: int = 42,
) -> np.ndarray:
    """Накладає імпульсний шум «сіль і перець» із фіксованим ``seed``.

    Args:
        image: Вхідне зображення ``uint8``.
        prob: Сумарна частка зашумлених пікселів (половина -> 0, половина -> 255).
        seed: Зерно генератора ``np.random.default_rng(seed)``.

    Returns:
        Зашумлене зображення ``uint8`` тієї самої форми.
    """
    raise NotImplementedError(
        "TODO (T1.6, Учасник 2): реалізувати add_salt_pepper_noise."
    )
