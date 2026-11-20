"""Автотести для особистої зони Учасника 1: пайплайни П1 та П2 (критерій C3).

Інструкція (Методичка, критерій C3):
Для кожної стадії та кожного пайплайну зніміть ``@pytest.mark.skip`` після реалізації
й перевірте збіг виходу на малій тестовій матриці (3x3 або 5x5) із ручним розрахунком.
"""

from __future__ import annotations

import pytest

from lego_cv.pipelines.member1 import ColorMorphPipeline, ContrastCannyPipeline


def test_member1_pipeline_metadata() -> None:
    """Контрактний тест: перевірка ідентифікаторів та унікальності категорій П1 і П2."""
    p1 = ColorMorphPipeline()
    p2 = ContrastCannyPipeline()
    p1.validate_contract()
    p2.validate_contract()
    assert p1.owner == "member1"
    assert p2.owner == "member1"
    assert set(p1.stage_categories) != set(p2.stage_categories)


@pytest.mark.skip(reason="TODO (T2.5, Учасник 1): реалізувати П1 — медіанний фільтр 3x3.")
def test_p1_median_filter_removes_impulse_spike() -> None:
    """Перевіряє видалення одиничного імпульсного викиду 255 у центрі матриці 3x3."""


@pytest.mark.skip(reason="TODO (T2.5, Учасник 1): реалізувати П1 — HSV-сегментація фону.")
def test_p1_hsv_background_segment_separates_colored_brick() -> None:
    """Перевіряє відділення яскравої деталі на сірому фоні 5x5 у просторі HSV."""


@pytest.mark.skip(reason="TODO (T2.5, Учасник 1): реалізувати П1 — закриття і flood-fill.")
def test_p1_morphological_close_fills_single_pixel_hole() -> None:
    """Перевіряє заповнення 1-піксельної дірки всередині квадрата 5x5."""


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати П2 — метод Оцу на 4x4.")
def test_p2_otsu_threshold_bimodal_matrix() -> None:
    """Перевіряє ручний поріг Оцу на бімодальній матриці (значення 20 і 200)."""


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати П2 — Кенні на 5x5.")
def test_p2_canny_detects_vertical_step_edge() -> None:
    """Перевіряє виділення вертикальної границі на ступінчастій матриці 5x5."""


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати наскрізні тести П1 та П2.")
def test_member1_end_to_end_pipelines() -> None:
    """Перевіряє автоматичний прогін усіх 3 стадій П1 і П2 зі збереженням StageOutput."""
