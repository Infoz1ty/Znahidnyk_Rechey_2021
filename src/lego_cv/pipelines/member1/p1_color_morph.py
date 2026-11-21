"""Пайплайн 1 (Учасник 1): Колірно-морфологічний конвеєр обробки.

Категорії стадій (3 унікальні категорії за методичкою):
1. Фільтрація зображень — ручний медіанний фільтр (5x5).
2. Колірна корекція — ручне RGB->HSV та порогова сегментація фону.
3. Морфологічна обробка — ручне морфологічне закриття та flood-fill дірок.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lego_cv.core.base_pipeline import BasePipeline


def _pad_reflect_safe(channel: np.ndarray, pad: int) -> np.ndarray:
    """Симетрично доповнює 2D-канал (reflect, або edge для осей коротших за ``pad``)."""
    if pad <= 0:
        return channel
    height, width = channel.shape
    mode_y = "reflect" if height > pad else "edge"
    padded_y = np.pad(channel, ((pad, pad), (0, 0)), mode=mode_y)
    mode_x = "reflect" if width > pad else "edge"
    return np.pad(padded_y, ((0, 0), (pad, pad)), mode=mode_x)


def _median_filter_2d(channel: np.ndarray, kernel_size: int) -> np.ndarray:
    """Застосовує ручний медіанний фільтр ``k x k`` до одного 2D-каналу чанками по рядках."""
    height, width = channel.shape
    pad = kernel_size // 2
    mid = (kernel_size * kernel_size) // 2
    padded = _pad_reflect_safe(channel, pad)

    result = np.empty((height, width), dtype=np.uint8)
    chunk_rows = 64
    for row_start in range(0, height, chunk_rows):
        row_end = min(row_start + chunk_rows, height)
        strip = padded[row_start : row_end + 2 * pad, :]
        windows = np.lib.stride_tricks.sliding_window_view(strip, (kernel_size, kernel_size))
        flat = windows.reshape(row_end - row_start, width, kernel_size * kernel_size)
        # Часткове сортування O(k^2) знаходить точну медіану без повного sort
        result[row_start:row_end] = np.partition(flat, mid, axis=-1)[..., mid]
    return result


def median_filter_manual(image_rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Ручний просторовий медіанний фільтр ``k x k`` із відзеркаленням меж.

    Придушує імпульсний шум та пил без розмиття прямих кутів деталей LEGO.

    Args:
        image_rgb: Вхідне зображення ``(H, W, 3)`` або ``(H, W)`` типу ``uint8``.
        kernel_size: Непарний розмір квадратного вікна (за замовчуванням 5).

    Returns:
        Відфільтроване зображення тієї самої форми типу ``uint8``.

    Raises:
        ValueError: Якщо ``kernel_size`` не є додатним непарним цілим або форма масиву
            некоректна.
    """
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise ValueError(f"kernel_size має бути цілим числом, отримано {kernel_size!r}")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(f"kernel_size має бути додатним непарним, отримано {kernel_size}")
    if image_rgb.ndim not in (2, 3) or image_rgb.size == 0:
        raise ValueError(f"Очікувався непорожній 2D/3D масив, отримано форму {image_rgb.shape}")

    arr_u8 = np.ascontiguousarray(image_rgb, dtype=np.uint8)
    if kernel_size == 1:
        return arr_u8.copy()
    if arr_u8.ndim == 2:
        return _median_filter_2d(arr_u8, kernel_size)

    output = np.empty_like(arr_u8)
    for ch_idx in range(arr_u8.shape[2]):
        output[:, :, ch_idx] = _median_filter_2d(arr_u8[:, :, ch_idx], kernel_size)
    return output


def rgb_to_hsv_manual_p1(image_rgb: np.ndarray) -> np.ndarray:
    """Ручне аналітичне перетворення ``RGB [0..255]`` у ``HSV`` (``H: 0..360, S/V: 0..255``).

    Args:
        image_rgb: Вхідне зображення ``(H, W, 3)`` або ``(H, W)`` типу ``uint8``.

    Returns:
        Масив ``float32`` форми ``(H, W, 3)`` у просторі HSV.

    Raises:
        ValueError: Якщо вхідне зображення порожнє або має некоректну форму.
    """
    if image_rgb.size == 0 or image_rgb.ndim not in (2, 3):
        raise ValueError(f"Очікувався непорожній 2D/3D масив, отримано форму {image_rgb.shape}")
    if image_rgb.ndim == 2:
        zeros = np.zeros(image_rgb.shape, dtype=np.float32)
        return np.stack((zeros, zeros, image_rgb.astype(np.float32)), axis=-1)
    if image_rgb.shape[2] != 3:
        raise ValueError(f"Очікувалось 3 канали RGB, отримано {image_rgb.shape[2]}")

    rgb_f = image_rgb.astype(np.float32) / 255.0
    r_ch = rgb_f[:, :, 0]
    g_ch = rgb_f[:, :, 1]
    b_ch = rgb_f[:, :, 2]

    cmax = np.maximum(np.maximum(r_ch, g_ch), b_ch)
    cmin = np.minimum(np.minimum(r_ch, g_ch), b_ch)
    delta = cmax - cmin

    v_ch = cmax * 255.0
    s_ch = np.zeros_like(cmax)
    nonzero_v = cmax > 0.0
    s_ch[nonzero_v] = (delta[nonzero_v] / cmax[nonzero_v]) * 255.0

    h_ch = np.zeros_like(cmax)
    nonzero_d = delta > 0.0
    mask_r = nonzero_d & (cmax == r_ch)
    mask_g = nonzero_d & (cmax == g_ch) & (~mask_r)
    mask_b = nonzero_d & (~mask_r) & (~mask_g)

    h_ch[mask_r] = 60.0 * (((g_ch[mask_r] - b_ch[mask_r]) / delta[mask_r]) % 6.0)
    h_ch[mask_g] = 60.0 * (((b_ch[mask_g] - r_ch[mask_g]) / delta[mask_g]) + 2.0)
    h_ch[mask_b] = 60.0 * (((r_ch[mask_b] - g_ch[mask_b]) / delta[mask_b]) + 4.0)
    h_ch = np.mod(h_ch, 360.0)

    return np.stack((h_ch, s_ch, v_ch), axis=-1).astype(np.float32)


def _extract_border_mask(height: int, width: int, margin_px: int) -> np.ndarray:
    """Формує булеву маску пікселів зовнішньої рамки кадру для оцінки фону."""
    margin = max(1, min(margin_px, max(1, height // 4), max(1, width // 4)))
    border = np.zeros((height, width), dtype=np.bool_)
    border[:margin, :] = True
    border[-margin:, :] = True
    border[:, :margin] = True
    border[:, -margin:] = True
    return border


def _adaptive_border_segment(hsv: np.ndarray, stage_cfg: dict[str, Any]) -> np.ndarray:
    """Адаптивно оцінює фон за рамкою кадру в циліндричних координатах HSV."""
    h_rad = np.deg2rad(hsv[:, :, 0])
    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    chroma_x = s_ch * np.cos(h_rad)
    chroma_y = s_ch * np.sin(h_rad)

    border = _extract_border_mask(
        hsv.shape[0], hsv.shape[1], int(stage_cfg.get("border_margin_px", 20))
    )
    bg_cx = float(np.median(chroma_x[border]))
    bg_cy = float(np.median(chroma_y[border]))
    bg_v = float(np.median(v_ch[border]))

    dist = np.sqrt((chroma_x - bg_cx) ** 2 + (chroma_y - bg_cy) ** 2 + (v_ch - bg_v) ** 2)
    dist_thresh = float(stage_cfg.get("color_distance_threshold", 30.0))
    return (dist > dist_thresh).astype(np.uint8) * 255


def hsv_background_segment(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне перетворення ``RGB -> HSV`` та виділення маски переднього плану.

    Відсікає нейтральний фон столу за низькою насиченістю ``S <= bg_s_max`` і
    яскравістю ``V`` у межах фону (або адаптивно за медіаною рамки кадру в
    циліндричних координатах HSV ``(S*cos H, S*sin H, V)``), повертаючи
    інвертовану бінарну маску деталей LEGO.

    Args:
        image_rgb: Відфільтроване RGB-зображення ``uint8``.
        stage_cfg: Секція ``stage2_hsv_segmentation`` із ``configs/pipeline_p1.json``.

    Returns:
        Бінарна маска переднього плану ``(H, W)`` типу ``uint8`` (0 або 255).
    """
    hsv = rgb_to_hsv_manual_p1(image_rgb)
    if bool(stage_cfg.get("use_border_bg_estimation", True)):
        return _adaptive_border_segment(hsv, stage_cfg)

    s_ch = hsv[:, :, 1]
    v_ch = hsv[:, :, 2]
    bg_s_max = float(stage_cfg.get("bg_s_max", 45.0))
    bg_v_min = float(stage_cfg.get("bg_v_min", 140.0))
    bg_v_max = float(stage_cfg.get("bg_v_max", 245.0))
    is_bg = (s_ch <= bg_s_max) & (v_ch >= bg_v_min) & (v_ch <= bg_v_max)
    return (~is_bg).astype(np.uint8) * 255


def morphological_close_and_fill(
    binary_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне морфологічне закриття (дилатація -> ерозія) та заповнення внутрішніх дірок.

    Усуває «дірки» від світлових відблисків на верхівках шпильок методом BFS flood-fill
    від меж зображення.

    Args:
        binary_mask: Бінарна маска ``(H, W)`` зі Стадії 2.
        stage_cfg: Секція ``stage3_morphology`` із ``configs/pipeline_p1.json``.

    Returns:
        Очищена бінарна маска ``(H, W)`` типу ``int32`` (0 — фон, 1 — деталі).
    """
    raise NotImplementedError(
        "TODO (T2.3, Учасник 1): реалізувати morphological_close_and_fill."
    )


class ColorMorphPipeline(BasePipeline):
    """Пайплайн П1 (Учасник 1): Медіанний фільтр -> HSV-сегментація -> Морфологія."""

    pipeline_id = "p1"
    name = "Колірно-морфологічний (Медіанний -> HSV -> Морфологія)"
    owner = "member1"
    stage_categories = (
        "Фільтрація зображень",
        "Колірна корекція",
        "Морфологічна обробка",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Фільтрація): медіанне шумоподавлення вхідного кадру."""
        stage_cfg = config.get("stage1_median", {})
        kernel_size = int(stage_cfg.get("kernel_size", 5))
        filtered = median_filter_manual(image_rgb, kernel_size=kernel_size)
        return filtered, {}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Колірна корекція): виділення об'єктів у просторі HSV."""
        stage_cfg = config.get("stage2_hsv_segmentation", {})
        raw_mask = hsv_background_segment(stage1_output, stage_cfg)
        return raw_mask, state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Морфологія): закриття контурів та заповнення відблисків."""
        stage_cfg = config.get("stage3_morphology", {})
        final_mask = morphological_close_and_fill(stage2_output, stage_cfg)
        visual = (final_mask > 0).astype("uint8") * 255
        return final_mask, visual, {"source_state": bool(state)}
