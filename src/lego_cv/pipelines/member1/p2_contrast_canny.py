"""Пайплайн 2 (Учасник 1): Контрастно-контурний конвеєр обробки.

Категорії стадій (3 унікальні категорії за методичкою):
1. Гістограмна корекція — ручне RGB->LAB, адаптивний CLAHE на каналі L та Гаус 3x3.
2. Сегментація зображень — ручна порогова бінаризація за методом Оцу.
3. Контурний аналіз — ручний детектор границь Кенні та злиття границь із маскою.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.pipelines.member1.p1_color_morph import pad_reflect_safe


def rgb_to_lab_l_manual(image_rgb: np.ndarray) -> np.ndarray:
    """Ручне перетворення ``sRGB -> XYZ (D65) -> CIE L*`` (масштабоване в ``[0, 255]``).

    Підтримує як 3-канальні RGB-кадри ``(H, W, 3)``, так і 1-канальні ``(H, W)``.

    Args:
        image_rgb: Вхідне зображення ``uint8``.

    Returns:
        Одноканальний масив яскравості ``L`` форми ``(H, W)`` типу ``uint8``.

    Raises:
        ValueError: Якщо форма або кількість каналів зображення некоректні.
    """
    if image_rgb.size == 0 or image_rgb.ndim not in (2, 3):
        raise ValueError(f"Очікувався непорожній 2D/3D масив, отримано форму {image_rgb.shape}")
    if image_rgb.ndim == 3 and image_rgb.shape[2] != 3:
        raise ValueError(f"Очікувалось 3 канали RGB, отримано {image_rgb.shape[2]}")

    srgb = image_rgb.astype(np.float64) / 255.0
    # Лінеаризація sRGB за стандартом IEC 61966-2-1
    linear = np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)
    if linear.ndim == 3:
        y_xyz = (
            0.2126729 * linear[:, :, 0]
            + 0.7151522 * linear[:, :, 1]
            + 0.0721750 * linear[:, :, 2]
        )
    else:
        y_xyz = linear

    eps = (6.0 / 29.0) ** 3
    slope = 1.0 / (3.0 * (6.0 / 29.0) ** 2)
    f_y = np.where(y_xyz > eps, np.cbrt(np.maximum(y_xyz, 0.0)), slope * y_xyz + 4.0 / 29.0)
    l_star = 116.0 * f_y - 16.0
    return np.clip(np.rint(l_star * 2.55), 0, 255).astype(np.uint8)


def _compute_tile_lut(tile: np.ndarray, clip_limit: float) -> np.ndarray:
    """Обчислює обмежену за контрастом CDF-таблицю відображення ``[0..255]`` для одного тайлу."""
    hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
    abs_clip = max(1.0, clip_limit * (tile.size / 256.0))

    # Обрізання піків понад поріг та рівномірний перерозподіл надлишку по 256 бінах
    excess = float(np.sum(np.maximum(hist - abs_clip, 0.0)))
    hist = np.minimum(hist, abs_clip) + (excess / 256.0)

    cdf = np.cumsum(hist)
    nonzero = np.flatnonzero(hist > 0.0)
    cdf_min = float(cdf[nonzero[0]]) if nonzero.size > 0 else 0.0
    denom = float(cdf[-1]) - cdf_min
    if denom <= 1e-9:
        return np.arange(256, dtype=np.float32)
    return np.clip(((cdf - cdf_min) / denom) * 255.0, 0.0, 255.0).astype(np.float32)


def _axis_interp_indices(
    length: int,
    centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Повертає індекси двох сусідніх тайлів ``(i0, i1)`` і вагу інтерполяції ``w``."""
    coords = np.arange(length, dtype=np.float32)
    n_tiles = centers.size
    if n_tiles == 1:
        zeros = np.zeros(length, dtype=np.int32)
        return zeros, zeros, np.zeros(length, dtype=np.float32)

    right = np.searchsorted(centers, coords, side="right").astype(np.int32)
    idx0 = np.clip(right - 1, 0, n_tiles - 1)
    idx1 = np.clip(right, 0, n_tiles - 1)
    span = centers[idx1] - centers[idx0]
    weight = np.where(span > 1e-6, (coords - centers[idx0]) / np.where(span > 1e-6, span, 1.0), 0.0)
    return idx0, idx1, np.clip(weight, 0.0, 1.0).astype(np.float32)


def _build_clahe_luts(
    gray: np.ndarray,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
    clip_limit: float,
) -> np.ndarray:
    """Формує масив таблиць відображення ``LUT[r, c, 0..255]`` для всіх тайлів CLAHE."""
    eff_rows = row_edges.size - 1
    eff_cols = col_edges.size - 1
    luts = np.empty((eff_rows, eff_cols, 256), dtype=np.float32)
    for r_idx in range(eff_rows):
        y0, y1 = int(row_edges[r_idx]), int(row_edges[r_idx + 1])
        for c_idx in range(eff_cols):
            x0, x1 = int(col_edges[c_idx]), int(col_edges[c_idx + 1])
            luts[r_idx, c_idx] = _compute_tile_lut(gray[y0:y1, x0:x1], clip_limit)
    return luts


def _interpolate_clahe_tiles(
    gray: np.ndarray,
    luts: np.ndarray,
    row_edges: np.ndarray,
    col_edges: np.ndarray,
) -> np.ndarray:
    """Виконує векторизовану білінійну інтерполяцію значень між 4 сусідніми тайлами."""
    row_centers = (0.5 * (row_edges[:-1] + row_edges[1:] - 1.0)).astype(np.float32)
    col_centers = (0.5 * (col_edges[:-1] + col_edges[1:] - 1.0)).astype(np.float32)
    r0, r1, wy = _axis_interp_indices(gray.shape[0], row_centers)
    c0, c1, wx = _axis_interp_indices(gray.shape[1], col_centers)

    wy_2d = wy[:, None]
    wx_2d = wx[None, :]
    top = (1.0 - wx_2d) * luts[r0[:, None], c0[None, :], gray] + wx_2d * luts[
        r0[:, None], c1[None, :], gray
    ]
    bottom = (1.0 - wx_2d) * luts[r1[:, None], c0[None, :], gray] + wx_2d * luts[
        r1[:, None], c1[None, :], gray
    ]
    blended = (1.0 - wy_2d) * top + wy_2d * bottom
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def clahe_manual(
    channel: np.ndarray,
    tile_grid_rows: int = 8,
    tile_grid_cols: int = 8,
    clip_limit: float = 2.0,
) -> np.ndarray:
    """Ручна реалізація адаптивного гістограмного вирівнювання з обмеженням контрасту (CLAHE).

    Розбиває 1-канальний кадр на сітку тайлів, обрізає піки гістограми понад
    ``clip_limit``, будує локальні CDF-таблиці та виконує білінійну інтерполяцію
    між 4 сусідніми центрами тайлів для усунення блокових швів.

    Args:
        channel: Одноканальне зображення ``(H, W)`` типу ``uint8``.
        tile_grid_rows: Кількість тайлів по вертикалі.
        tile_grid_cols: Кількість тайлів по горизонталі.
        clip_limit: Множник порога обрізання гістограми (відносно рівномірної висоти біна).

    Returns:
        Контрастно вирівняний канал ``(H, W)`` типу ``uint8``.

    Raises:
        ValueError: Якщо параметри сітки, ``clip_limit`` або форма каналу некоректні.
    """
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError(f"Очікувався непорожній 2D-канал, отримано форму {channel.shape}")
    if tile_grid_rows < 1 or tile_grid_cols < 1 or clip_limit <= 0.0:
        raise ValueError("Параметри CLAHE мають задовольняти grid >= 1 і clip_limit > 0.")

    gray = np.ascontiguousarray(channel, dtype=np.uint8)
    if int(gray.min()) == int(gray.max()):
        return gray.copy()

    height, width = gray.shape
    row_edges = np.linspace(0, height, min(tile_grid_rows, height) + 1, dtype=np.int32)
    col_edges = np.linspace(0, width, min(tile_grid_cols, width) + 1, dtype=np.int32)
    luts = _build_clahe_luts(gray, row_edges, col_edges, clip_limit)
    return _interpolate_clahe_tiles(gray, luts, row_edges, col_edges)


def gaussian_blur_manual(
    channel: np.ndarray,
    kernel_size: int = 3,
    sigma: float = 0.8,
) -> np.ndarray:
    """Ручне роздільне 2D гаусове згладжування ``k x k`` (1D по X, потім 1D по Y).

    Args:
        channel: Одноканальне зображення ``(H, W)`` типу ``uint8``.
        kernel_size: Непарний розмір ядра (за замовчуванням 3).
        sigma: Стандартне відхилення розподілу Гауса (``> 0``).

    Returns:
        Згладжене зображення ``(H, W)`` типу ``uint8``.

    Raises:
        ValueError: Якщо ``kernel_size`` або ``sigma`` некоректні.
    """
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError(f"kernel_size має бути цілим числом, отримано {kernel_size!r}")
    if kernel_size < 1 or kernel_size % 2 == 0 or sigma <= 0.0:
        raise ValueError(
            f"Очікується непарний kernel_size >= 1 і sigma > 0, отримано {kernel_size}"
        )
    if channel.ndim != 2 or channel.size == 0:
        raise ValueError(f"Очікувався непорожній 2D-канал, отримано форму {channel.shape}")
    if kernel_size == 1:
        return np.ascontiguousarray(channel, dtype=np.uint8).copy()

    pad = kernel_size // 2
    coords = np.arange(-pad, pad + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= float(np.sum(kernel))

    height, width = channel.shape
    padded_x = pad_reflect_safe(channel.astype(np.float64), pad)
    mid_slice = padded_x[pad : pad + height, :]
    horiz = np.zeros((height, width), dtype=np.float64)
    for idx, weight in enumerate(kernel):
        horiz += weight * mid_slice[:, idx : idx + width]

    padded_y = pad_reflect_safe(horiz, pad)
    mid_cols = padded_y[:, pad : pad + width]
    result = np.zeros((height, width), dtype=np.float64)
    for idx, weight in enumerate(kernel):
        result += weight * mid_cols[idx : idx + height, :]
    return np.clip(np.rint(result), 0, 255).astype(np.uint8)


def lab_clahe_enhance_manual(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Переводить зображення у ``CIE LAB``, застосовує ручний CLAHE до ``L`` і Гаус 3x3.

    Вирівнює локальний контраст за нерівномірного освітлення та віньєтування.

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        stage_cfg: Секція ``stage1_lab_clahe`` із ``configs/pipeline_p2.json``.

    Returns:
        Покращений 1-канальний кадр яскравості ``(H, W)`` типу ``uint8``.
    """
    l_channel = rgb_to_lab_l_manual(image_rgb)
    enhanced = clahe_manual(
        l_channel,
        tile_grid_rows=int(stage_cfg.get("tile_grid_rows", 8)),
        tile_grid_cols=int(stage_cfg.get("tile_grid_cols", 8)),
        clip_limit=float(stage_cfg.get("clip_limit", 2.0)),
    )
    return gaussian_blur_manual(
        enhanced,
        kernel_size=int(stage_cfg.get("gaussian_kernel_size", 3)),
        sigma=float(stage_cfg.get("gaussian_sigma", 0.8)),
    )


def otsu_threshold_manual(
    gray_image: np.ndarray,
    invert_foreground: bool = True,
) -> tuple[np.ndarray, int]:
    """Ручна автоматична бінаризація за методом Оцу (максимізація міжкласової дисперсії).

    Args:
        gray_image: Одноканальне зображення ``(H, W)`` типу ``uint8``.
        invert_foreground: Якщо ``True``, темніші за поріг пікселі вважаються об'єктами.

    Returns:
        Кортеж ``(binary_mask_uint8, optimal_threshold)``.
    """
    raise NotImplementedError("TODO (T2.7, Учасник 1): реалізувати otsu_threshold_manual.")


def canny_edge_merge_manual(
    gray_image: np.ndarray,
    otsu_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Ручний детектор границь Кенні (градієнт -> NMS -> подвійний поріг -> гістерезис).

    Об'єднує виділені границі Кенні з пороговою маскою Оцу та замикає дрібні розриви.

    Args:
        gray_image: Вирівняний канал яскравості зі Стадії 1.
        otsu_mask: Бінарна маска зі Стадії 2.
        stage_cfg: Секція ``stage3_canny`` із ``configs/pipeline_p2.json``.

    Returns:
        Кортеж ``(final_mask_int32, edges_visual_uint8)``.
    """
    raise NotImplementedError(
        "TODO (T2.8, Учасник 1): реалізувати canny_edge_merge_manual."
    )


class ContrastCannyPipeline(BasePipeline):
    """Пайплайн П2 (Учасник 1): LAB+CLAHE -> Бінаризація Оцу -> Детектор Кенні."""

    pipeline_id = "p2"
    name = "Контрастно-контурний (LAB/CLAHE -> Оцу -> Кенні)"
    owner = "member1"
    stage_categories = (
        "Гістограмна корекція",
        "Сегментація зображень",
        "Контурний аналіз",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Гістограмна корекція): LAB + ручний CLAHE + Гаус 3x3."""
        stage_cfg = config.get("stage1_lab_clahe", {})
        enhanced_l = lab_clahe_enhance_manual(image_rgb, stage_cfg)
        return enhanced_l, {"enhanced_l": enhanced_l}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Сегментація): автоматична бінаризація методом Оцу."""
        stage_cfg = config.get("stage2_otsu", {})
        invert = bool(stage_cfg.get("invert_foreground", True))
        otsu_mask, threshold = otsu_threshold_manual(stage1_output, invert_foreground=invert)
        next_state = dict(state)
        next_state["otsu_threshold"] = threshold
        return otsu_mask, next_state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Контурний аналіз): детектор Кенні та уточнення маски."""
        stage_cfg = config.get("stage3_canny", {})
        final_mask, edges_vis = canny_edge_merge_manual(
            gray_image=state["enhanced_l"],
            otsu_mask=stage2_output,
            stage_cfg=stage_cfg,
        )
        return final_mask, edges_vis, {"otsu_threshold": state.get("otsu_threshold")}
