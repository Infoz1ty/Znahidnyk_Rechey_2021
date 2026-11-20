"""Автотести для особистої зони Учасника 2: пайплайни П3 та П4 (критерій C3).

Інструкція (Методичка, критерій C3):
Для кожної стадії та кожного пайплайну зніміть ``@pytest.mark.skip`` після реалізації
й перевірте збіг виходу на малій тестовій матриці (3x3 / 5x5 / 8x8) із ручним розрахунком.
"""

from __future__ import annotations

import pytest

from lego_cv.pipelines.member2 import FftTextureHoughPipeline, SobelWatershedPipeline


def test_member2_pipeline_metadata() -> None:
    """Контрактний тест: перевірка ідентифікаторів та унікальності категорій П3 і П4."""
    p3 = SobelWatershedPipeline()
    p4 = FftTextureHoughPipeline()
    p3.validate_contract()
    p4.validate_contract()
    assert p3.owner == "member2"
    assert p4.owner == "member2"
    assert set(p3.stage_categories) != set(p4.stage_categories)


@pytest.mark.skip(reason="TODO (T2.15, Учасник 2): реалізувати П3 — оператор Собеля 3x3.")
def test_p3_sobel_gradient_vertical_edge() -> None:
    """Перевіряє точний ручний градієнт Собеля на вертикальному перепаді 3x3."""


@pytest.mark.skip(reason="TODO (T2.15, Учасник 2): реалізувати П3 — білатеральний фільтр.")
def test_p3_bilateral_preserves_sharp_step() -> None:
    """Перевіряє збереження різкої межі при згладжуванні плоских ділянок 5x5."""


@pytest.mark.skip(reason="TODO (T2.15, Учасник 2): реалізувати П3 — Watershed розділення.")
def test_p3_watershed_splits_two_touching_squares() -> None:
    """Перевіряє отримання 2 окремих міток для двох дотичних квадратів."""


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — 2D FFT/IFFT 4x4.")
def test_p4_fft2_roundtrip_exact_reconstruction() -> None:
    """Перевіряє точне відновлення матриці 4x4 після ручного 2D FFT -> 2D IFFT."""


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — LBP 3x3 та GLCM 4x4.")
def test_p4_lbp_and_glcm_manual_values() -> None:
    """Перевіряє 8-бітний код LBP у центрі 3x3 та нульовий контраст GLCM на константі."""


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — Circle Hough + П3/П4 e2e.")
def test_member2_end_to_end_pipelines() -> None:
    """Перевіряє детекцію кола Хафа та автоматичний прогін усіх 3 стадій П3 і П4."""
