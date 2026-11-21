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
from lego_cv.pipelines.member2.p4_fft_texture_hough import (
    compute_glcm_features,
    compute_lbp_map,
    fft1d_manual,
    fft2_manual,
    hough_circles_manual,
    ifft2_manual,
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


def test_p4_fft2_roundtrip_exact_reconstruction() -> None:
    """Перевіряє точне відновлення матриці 4x4 після ручного 2D FFT -> 2D IFFT та спектр."""
    # Ручний спектр константи 4x4: F[0, 0] = 5 * 16 = 80, решта = 0
    const_mat = np.full((4, 4), 5.0, dtype=np.float64)
    const_spec = fft2_manual(const_mat)
    assert const_spec[0, 0] == pytest.approx(80.0 + 0.0j, abs=1e-9)
    const_spec_no_dc = const_spec.copy()
    const_spec_no_dc[0, 0] = 0.0
    assert float(np.max(np.abs(const_spec_no_dc))) < 1e-9

    # 1D знакозмінний сигнал [1, -1, 1, -1]: енергія лише на частоті Найквіста k=2 -> 4.0
    alt_spec = fft1d_manual(np.array([1.0, -1.0, 1.0, -1.0]))
    assert alt_spec[2] == pytest.approx(4.0 + 0.0j, abs=1e-9)
    assert abs(alt_spec[0]) < 1e-9

    # Точне відновлення довільної матриці 4x4: |ifft2(fft2(M)) - M| < 1e-5
    mat = np.array(
        [
            [10.0, 20.0, 30.0, 40.0],
            [50.0, 60.0, 70.0, 80.0],
            [15.0, 25.0, 35.0, 45.0],
            [5.0, 12.0, 18.0, 22.0],
        ],
        dtype=np.float64,
    )
    reconstructed = np.real(ifft2_manual(fft2_manual(mat)))
    assert float(np.max(np.abs(reconstructed - mat))) < 1e-5


def test_p4_lbp_and_glcm_manual_values() -> None:
    """Перевіряє 8-бітний код LBP у центрі 3x3 та GLCM на однорідній і шаховій 4x4 матрицях."""
    # Ручний LBP у центрі (1, 1) = 15: сусіди >= 15 це біти 1 (20), 2 (30), 3 (25), 4 (20) -> 30
    patch_3x3 = np.array(
        [[10, 20, 30], [5, 15, 25], [0, 10, 20]],
        dtype=np.uint8,
    )
    lbp = compute_lbp_map(patch_3x3)
    assert lbp.dtype == np.uint8 and lbp.shape == (3, 3)
    assert int(lbp[1, 1]) == 30

    # GLCM на однорідній 4x4 матриці: contrast == 0, homogeneity == 1, energy == 1
    uniform_4x4 = np.full((4, 4), 128, dtype=np.uint8)
    uni_feats = compute_glcm_features(uniform_4x4, levels=16, distances=(1,))
    assert uni_feats["contrast"] == pytest.approx(0.0, abs=1e-9)
    assert uni_feats["homogeneity"] == pytest.approx(1.0, abs=1e-9)
    assert uni_feats["energy"] == pytest.approx(1.0, abs=1e-9)

    # GLCM на шаховій 4x4 матриці: високий contrast, нижчі homogeneity та energy
    checker_4x4 = np.array(
        [
            [0, 255, 0, 255],
            [255, 0, 255, 0],
            [0, 255, 0, 255],
            [255, 0, 255, 0],
        ],
        dtype=np.uint8,
    )
    chk_feats = compute_glcm_features(checker_4x4, levels=16, distances=(1,))
    assert chk_feats["contrast"] > 100.0
    assert chk_feats["homogeneity"] < 0.6
    # Аналітично для шахової 4x4: (0.5 + 41/81 + 0.5 + 41/81) / 4 = 81.5/162 ≈ 0.503086
    assert chk_feats["energy"] == pytest.approx(0.50308642, abs=1e-6)
    assert chk_feats["energy"] < uni_feats["energy"]


def test_member2_end_to_end_pipelines() -> None:
    """Перевіряє ручний Circle Hough на синтетичному колі та повний прогін П4 (+ C2)."""
    # Синтетичне коло відомого радіуса r=10 у центрі (20, 20) на кадрі 41x41
    yy, xx = np.ogrid[:41, :41]
    circle_img = np.where((xx - 20) ** 2 + (yy - 20) ** 2 <= 10**2, 220, 20).astype(np.uint8)
    circles = hough_circles_manual(
        circle_img,
        radius_min=7,
        radius_max=13,
        gradient_threshold=40.0,
        accumulator_threshold_ratio=0.7,
        min_center_distance_px=15,
    )
    assert len(circles) == 1
    cx, cy, rad, _ = circles[0]
    assert abs(cx - 20) <= 1 and abs(cy - 20) <= 1 and abs(rad - 10) <= 1

    # Інтеграційний тест FftTextureHoughPipeline на малому синтетичному RGB-кадрі 44x44
    with (REPO_ROOT / "configs" / "pipeline_p4.json").open("r", encoding="utf-8") as fh:
        p4_cfg = json.load(fh)
    p4_cfg["stage2_lbp_glcm"]["min_component_area_px"] = 25
    p4_cfg["stage3_hough_circles"]["radius_min_px"] = 5
    p4_cfg["stage3_hough_circles"]["radius_max_px"] = 9
    p4_cfg["stage3_hough_circles"]["min_center_distance_px"] = 12

    frame = np.full((44, 44, 3), 10, dtype=np.uint8)
    frame[8:36, 8:36] = 160
    fy, fx = np.ogrid[:44, :44]
    frame[(fx - 22) ** 2 + (fy - 22) ** 2 <= 7**2] = 240

    p4 = FftTextureHoughPipeline()
    res = p4.run(frame, p4_cfg)
    assert res.pipeline_id == "p4"
    assert len(res.stages) == 3
    assert res.denoised.shape == (44, 44) and res.denoised.dtype == np.uint8
    assert res.mask.dtype == np.int32 and res.mask.shape == (44, 44)
    assert 1 in res.metadata["stud_counts"]
    assert res.metadata["stud_counts"][1] >= 1
    assert 1 in res.metadata["texture_features"]
    assert "homogeneity" in res.metadata["texture_features"][1]

    # Крайні випадки C2: однорідний кадр 5x5 та крихітний 4x4 grayscale-RGB
    flat_res = p4.run(np.full((5, 5, 3), 90, dtype=np.uint8), p4_cfg)
    assert flat_res.mask.dtype == np.int32 and int(np.max(flat_res.mask)) == 0
    assert flat_res.metadata["stud_counts"] == {}
    assert flat_res.metadata["texture_features"] == {}

    tiny_rgb = np.full((4, 4, 3), 50, dtype=np.uint8)
    tiny_rgb[1:3, 1:3] = 190
    tiny_res = p4.run(tiny_rgb, p4_cfg)
    assert tiny_res.mask.shape == (4, 4) and tiny_res.mask.dtype == np.int32
