"""Автотести для особистої зони Учасника 2: пайплайни П3 та П4 (критерій C3).

Інструкція (Методичка, критерій C3):
Для кожної стадії та кожного пайплайну зніміть ``@pytest.mark.skip`` після реалізації
й перевірте збіг виходу на малій тестовій матриці (3x3 / 5x5 / 8x8) із ручним розрахунком.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from lego_cv.pipelines.member2 import FftTextureHoughPipeline, SobelWatershedPipeline
from lego_cv.pipelines.member2.p3_sobel_watershed import (
    bilateral_filter_manual,
    distance_transform_manual,
    extract_watershed_markers,
    marker_watershed_manual,
    sobel_gradient_manual,
    unsharp_mask_manual,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_member2_pipeline_metadata() -> None:
    """Контрактний тест: перевірка ідентифікаторів та унікальності категорій П3 і П4."""
    p3 = SobelWatershedPipeline()
    p4 = FftTextureHoughPipeline()
    p3.validate_contract()
    p4.validate_contract()
    assert p3.owner == "member2"
    assert p4.owner == "member2"
    assert set(p3.stage_categories) != set(p4.stage_categories)


def test_p3_sobel_gradient_vertical_edge() -> None:
    """Перевіряє точний ручний градієнт Собеля на вертикальному та горизонтальному краї 3x3."""
    vert_edge = np.array(
        [[0, 0, 100], [0, 0, 100], [0, 0, 100]],
        dtype=np.uint8,
    )
    gx_v, gy_v, mag_v = sobel_gradient_manual(vert_edge)
    assert gx_v[1, 1] == pytest.approx(400.0, abs=1e-9)
    assert gy_v[1, 1] == pytest.approx(0.0, abs=1e-9)
    assert mag_v[1, 1] == pytest.approx(400.0, abs=1e-9)

    horiz_edge = np.array(
        [[0, 0, 0], [0, 0, 0], [100, 100, 100]],
        dtype=np.uint8,
    )
    gx_h, gy_h, mag_h = sobel_gradient_manual(horiz_edge)
    assert gx_h[1, 1] == pytest.approx(0.0, abs=1e-9)
    assert gy_h[1, 1] == pytest.approx(400.0, abs=1e-9)
    assert mag_h[1, 1] == pytest.approx(400.0, abs=1e-9)


def test_p3_bilateral_preserves_sharp_step() -> None:
    """Перевіряє збереження різкої межі білатеральним фільтром і роботу Unsharp Mask на 5x5."""
    step = np.full((5, 5), 20, dtype=np.uint8)
    step[:, 3:] = 220
    noisy_step = step.copy()
    noisy_step[2, 1] = 32  # локальне збурення на плоскій ділянці

    filtered = bilateral_filter_manual(
        noisy_step, radius=1, sigma_spatial=2.0, sigma_range=20.0
    )
    assert filtered.dtype == np.uint8 and filtered.shape == (5, 5)
    # Збурення в плоскій області згладилося ближче до 20
    assert int(filtered[2, 1]) < 30
    # Різкий перепад між стовпцями 2 і 3 зберігся
    assert int(filtered[2, 3]) - int(filtered[2, 2]) > 180

    # Unsharp Mask не змінює константну матрицю й підсилює перепад
    const_mat = np.full((5, 5), 100, dtype=np.uint8)
    assert np.array_equal(unsharp_mask_manual(const_mat, sigma=1.0, amount=1.0), const_mat)

    soft_step = np.full((5, 5), 80, dtype=np.uint8)
    soft_step[:, 3:] = 160
    sharp = unsharp_mask_manual(soft_step, sigma=1.0, amount=1.0)
    assert int(sharp[2, 3]) > int(soft_step[2, 3])
    assert int(sharp[2, 2]) < int(soft_step[2, 2])


def test_p3_watershed_splits_two_touching_squares() -> None:
    """Перевіряє Chamfer 3-4 на 5x5 та розділення двох злиплих квадратів Watershed-ом."""
    # Точна карта відстаней Chamfer 3-4 для квадрата 3x3 у матриці 5x5
    sq_mask = np.zeros((5, 5), dtype=np.bool_)
    sq_mask[1:4, 1:4] = True
    dist = distance_transform_manual(sq_mask)
    assert dist[0, 0] == pytest.approx(0.0, abs=1e-6)
    assert dist[1, 1] == pytest.approx(1.0, abs=1e-6)
    assert dist[2, 2] == pytest.approx(2.0, abs=1e-6)

    # Два квадрати 5x5, з'єднані вузьким перешийком -> 2 маркери
    pinched = np.zeros((7, 13), dtype=np.bool_)
    pinched[1:6, 1:6] = True
    pinched[1:6, 7:12] = True
    pinched[3, 6] = True  # перешийок товщиною 1 піксель
    pinched_dist = distance_transform_manual(pinched)
    auto_markers, num_markers = extract_watershed_markers(
        pinched_dist,
        pinched,
        peak_min_ratio=0.75,
        peak_min_distance_px=3,
        min_marker_area_px=5,
    )
    assert num_markers == 2
    assert auto_markers[3, 3] != auto_markers[3, 9]

    # Маркерний Watershed розділяє два злиплі квадрати вздовж градієнтного хребта
    fg_touching = np.ones((5, 9), dtype=np.bool_)
    grad = np.zeros((5, 9), dtype=np.float32)
    grad[:, 4] = 200.0  # високий градієнтний хребет на стику
    markers = np.zeros((5, 9), dtype=np.int32)
    markers[2, 1] = 1
    markers[2, 7] = 2

    ws_labels = marker_watershed_manual(grad, markers, fg_touching)
    assert ws_labels.dtype == np.int32
    assert set(np.unique(ws_labels).tolist()) == {1, 2}
    assert np.all(ws_labels[:, :4] == 1)
    assert np.all(ws_labels[:, 5:] == 2)


def test_p3_pipeline_end_to_end_and_edge_cases() -> None:
    """Інтеграційний тест SobelWatershedPipeline на малому кадрі та крайніх випадках (C2)."""
    with (REPO_ROOT / "configs" / "pipeline_p3.json").open("r", encoding="utf-8") as fh:
        p3_cfg = json.load(fh)
    p3_cfg["stage3_watershed"]["min_marker_area_px"] = 9
    p3_cfg["stage3_watershed"]["peak_min_distance_px"] = 4

    pipeline = SobelWatershedPipeline()

    # Малий синтетичний RGB-кадр 15x15 із яскравим квадратом у центрі
    img = np.full((15, 15, 3), 10, dtype=np.uint8)
    img[4:11, 4:11] = [220, 60, 40]
    res = pipeline.run(img, p3_cfg)

    assert res.pipeline_id == "p3"
    assert len(res.stages) == 3
    assert res.denoised.shape == img.shape and res.denoised.dtype == np.uint8
    assert res.mask.dtype == np.int32 and res.mask.shape == (15, 15)
    assert res.metadata["is_instance_mask"] is True
    assert res.metadata["num_markers"] == 1
    assert int(res.mask[7, 7]) == 1
    assert int(res.mask[0, 0]) == 0

    # Крайні випадки C2: однорідний кадр 5x5 та крихітний ч/б RGB 3x3 без падінь
    flat_res = pipeline.run(np.full((5, 5, 3), 80, dtype=np.uint8), p3_cfg)
    assert flat_res.mask.dtype == np.int32
    assert int(np.max(flat_res.mask)) == 0
    assert flat_res.metadata["num_markers"] == 0

    tiny_gray_rgb = np.full((3, 3, 3), 120, dtype=np.uint8)
    tiny_gray_rgb[1, 1] = 200
    tiny_res = pipeline.run(tiny_gray_rgb, p3_cfg)
    assert tiny_res.mask.shape == (3, 3) and tiny_res.mask.dtype == np.int32


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — 2D FFT/IFFT 4x4.")
def test_p4_fft2_roundtrip_exact_reconstruction() -> None:
    """Перевіряє точне відновлення матриці 4x4 після ручного 2D FFT -> 2D IFFT."""


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — LBP 3x3 та GLCM 4x4.")
def test_p4_lbp_and_glcm_manual_values() -> None:
    """Перевіряє 8-бітний код LBP у центрі 3x3 та нульовий контраст GLCM на константі."""


@pytest.mark.skip(reason="TODO (T2.20, Учасник 2): реалізувати П4 — Circle Hough + П3/П4 e2e.")
def test_member2_end_to_end_pipelines() -> None:
    """Перевіряє детекцію кола Хафа та автоматичний прогін усіх 3 стадій П3 і П4."""
