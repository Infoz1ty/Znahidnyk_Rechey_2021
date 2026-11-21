"""Автотести для особистої зони Учасника 1: пайплайни П1 та П2 (критерій C3).

Інструкція (Методичка, критерій C3):
Для кожної стадії та кожного пайплайну зніміть ``@pytest.mark.skip`` після реалізації
й перевірте збіг виходу на малій тестовій матриці (3x3 або 5x5) із ручним розрахунком.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lego_cv.config import load_pipeline_config
from lego_cv.pipelines.member1 import ColorMorphPipeline, ContrastCannyPipeline
from lego_cv.pipelines.member1.p1_color_morph import (
    fill_holes_manual,
    hsv_background_segment,
    median_filter_manual,
    morphological_close_and_fill,
    rgb_to_hsv_manual_p1,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_member1_pipeline_metadata() -> None:
    """Контрактний тест: перевірка ідентифікаторів та унікальності категорій П1 і П2."""
    p1 = ColorMorphPipeline()
    p2 = ContrastCannyPipeline()
    p1.validate_contract()
    p2.validate_contract()
    assert p1.owner == "member1"
    assert p2.owner == "member1"
    assert set(p1.stage_categories) != set(p2.stage_categories)


def test_p1_median_filter_removes_impulse_spike() -> None:
    """Перевіряє видалення одиничного імпульсного викиду 255 у центрі матриці 3x3."""
    gray = np.full((3, 3), 10, dtype=np.uint8)
    gray[1, 1] = 255
    filtered_gray = median_filter_manual(gray, kernel_size=3)
    assert filtered_gray.dtype == np.uint8
    assert np.array_equal(filtered_gray, np.full((3, 3), 10, dtype=np.uint8))

    rgb = np.zeros((3, 3, 3), dtype=np.uint8)
    rgb[:, :, 0] = 10
    rgb[:, :, 1] = 20
    rgb[:, :, 2] = 30
    rgb[1, 1] = [255, 255, 255]
    filtered_rgb = median_filter_manual(rgb, kernel_size=3)
    expected_rgb = np.broadcast_to(np.array([10, 20, 30], dtype=np.uint8), (3, 3, 3))
    assert np.array_equal(filtered_rgb, expected_rgb)

    assert np.array_equal(median_filter_manual(gray, kernel_size=1), gray)
    with pytest.raises(ValueError):
        median_filter_manual(gray, kernel_size=4)


def test_p1_hsv_background_segment_separates_colored_brick() -> None:
    """Перевіряє аналітичне RGB->HSV та відділення яскравої деталі на сірому фоні 5x5."""
    prim = np.array([[[255, 0, 0], [0, 255, 0], [0, 0, 255], [180, 180, 180]]], dtype=np.uint8)
    hsv = rgb_to_hsv_manual_p1(prim)
    assert np.allclose(hsv[0, 0], [0.0, 255.0, 255.0], atol=1e-3)
    assert np.allclose(hsv[0, 1], [120.0, 255.0, 255.0], atol=1e-3)
    assert np.allclose(hsv[0, 2], [240.0, 255.0, 255.0], atol=1e-3)
    assert np.allclose(hsv[0, 3], [0.0, 0.0, 180.0], atol=1e-3)

    img = np.full((5, 5, 3), 180, dtype=np.uint8)
    img[1:4, 1:4] = [220, 30, 30]
    expected = np.zeros((5, 5), dtype=np.uint8)
    expected[1:4, 1:4] = 255

    # Фіксований поріг фону
    mask_fixed = hsv_background_segment(
        img,
        {
            "use_border_bg_estimation": False,
            "bg_s_max": 45,
            "bg_v_min": 140,
            "bg_v_max": 245,
        },
    )
    assert np.array_equal(mask_fixed, expected)

    # Адаптивна оцінка фону за рамкою кадру
    mask_adaptive = hsv_background_segment(
        img,
        {
            "use_border_bg_estimation": True,
            "border_margin_px": 1,
            "color_distance_threshold": 30.0,
        },
    )
    assert np.array_equal(mask_adaptive, expected)

    # Однорідний кадр -> порожня нульова маска без падінь
    uniform = np.full((5, 5, 3), 150, dtype=np.uint8)
    assert np.all(
        hsv_background_segment(uniform, {"use_border_bg_estimation": True}) == 0
    )


def test_p1_morphological_close_fills_single_pixel_hole() -> None:
    """Перевіряє заповнення 1-піксельної дірки всередині квадрата 5x5 та видалення шуму."""
    raw = np.zeros((7, 7), dtype=np.uint8)
    raw[1:6, 1:6] = 255
    raw[3, 3] = 0  # внутрішня 1-піксельна дірка від відблиску
    raw[0, 0] = 255  # одиничний піксель шуму на фоні

    cleaned = morphological_close_and_fill(
        raw,
        {"close_kernel_size": 3, "open_kernel_size": 3, "fill_holes": True},
    )
    expected = np.zeros((7, 7), dtype=np.int32)
    expected[1:6, 1:6] = 1
    assert cleaned.dtype == np.int32
    assert np.array_equal(cleaned, expected)

    # Перевірка fill_holes_manual безпосередньо на квадраті 5x5 з діркою в центрі
    ring = np.ones((5, 5), dtype=np.uint8)
    ring[2, 2] = 0
    assert np.all(fill_holes_manual(ring))


def test_p1_end_to_end_pipeline() -> None:
    """Перевіряє автоматичний 3-стадійний прогін ColorMorphPipeline на синтетичному кадрі."""
    img = np.full((24, 24, 3), 185, dtype=np.uint8)
    img[7:17, 7:17] = [20, 45, 220]
    img[11, 11] = [185, 185, 185]  # внутрішній відблиск (має заповнитись)
    img[2, 2] = [255, 0, 0]  # одиничний імпульсний викид (має зникнути в медіані)

    cfg = load_pipeline_config(CONFIGS_DIR / "default.json", CONFIGS_DIR / "pipeline_p1.json")
    cfg["stage1_median"]["kernel_size"] = 3
    cfg["stage2_hsv_segmentation"]["border_margin_px"] = 2

    res = ColorMorphPipeline().run(img, cfg)
    assert res.pipeline_id == "p1"
    assert len(res.stages) == 3
    assert res.mask.dtype == np.int32
    assert res.denoised.shape == img.shape
    assert res.mask[11, 11] == 1
    assert res.mask[2, 2] == 0
    assert int(np.count_nonzero(res.mask)) >= 90


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати П2 — метод Оцу на 4x4.")
def test_p2_otsu_threshold_bimodal_matrix() -> None:
    """Перевіряє ручний поріг Оцу на бімодальній матриці (значення 20 і 200)."""


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати П2 — Кенні на 5x5.")
def test_p2_canny_detects_vertical_step_edge() -> None:
    """Перевіряє виділення вертикальної границі на ступінчастій матриці 5x5."""


@pytest.mark.skip(reason="TODO (T2.10, Учасник 1): реалізувати наскрізні тести П1 та П2.")
def test_member1_end_to_end_pipelines() -> None:
    """Перевіряє автоматичний прогін усіх 3 стадій П1 і П2 зі збереженням StageOutput."""
