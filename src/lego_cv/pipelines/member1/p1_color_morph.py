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


def hsv_background_segment(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне перетворення ``RGB -> HSV`` та виділення маски переднього плану.

    Відсікає нейтральний фон столу за низькою насиченістю ``S < bg_s_max`` і
    яскравістю ``V`` у межах фону (або адаптивно за рамкою кадру), повертаючи
    інвертовану бінарну маску деталей LEGO.

    Args:
        image_rgb: Відфільтроване RGB-зображення ``uint8``.
        stage_cfg: Секція ``stage2_hsv_segmentation`` із ``configs/pipeline_p1.json``.

    Returns:
        Бінарна маска переднього плану ``(H, W)`` типу ``uint8`` (0 або 255).
    """
    raise NotImplementedError(
        "TODO (T2.2, Учасник 1): реалізувати hsv_background_segment."
    )


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
