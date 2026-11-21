"""Модуль Б: Спільний класифікатор кольору деталей LEGO (Крок 2).

Зона реалізації (Етап 1): Учасник 2 (member2).
Усі операції (ерозія 3x3, RGB->HSV, медіанна класифікація) реалізуються вручну на NumPy.
"""

from __future__ import annotations

from dataclasses import replace
import math
from typing import Any

import numpy as np

from lego_cv.core.types import BrickDetection

_ACHROMATIC_NAMES = frozenset({"white", "gray", "grey", "black"})


def binary_erode_manual(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Ручна бінарна морфологічна ерозія квадратним ядром ``kernel_size x kernel_size``.

    Використовується для звуження маски на 1–3 пікселі, щоб не захоплювати тінь по краях.
    За межами зображення застосовується доповнення нулями (``False``).

    Args:
        mask: Булева або бінарна маска ``(H, W)``.
        kernel_size: Непарний додатний розмір ядра (за замовчуванням 3).

    Returns:
        Еродована булева маска ``(H, W)``.

    Raises:
        ValueError: Якщо ``mask`` не 2D або ``kernel_size`` не є непарним додатним числом.
    """
    bool_mask = np.asarray(mask, dtype=np.bool_)
    if bool_mask.ndim != 2:
        raise ValueError(f"Очікується 2D-маска, отримано форму {bool_mask.shape}.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"kernel_size має бути непарним додатним числом, отримано {kernel_size}."
        )
    if kernel_size == 1 or bool_mask.size == 0:
        return bool_mask.copy()

    height, width = bool_mask.shape
    pad = kernel_size // 2
    padded = np.pad(bool_mask, pad_width=pad, mode="constant", constant_values=False)

    eroded = np.ones((height, width), dtype=np.bool_)
    for dy in range(kernel_size):
        for dx in range(kernel_size):
            eroded &= padded[dy : dy + height, dx : dx + width]
    return eroded


def rgb_to_hsv_manual(image_rgb: np.ndarray) -> np.ndarray:
    """Ручне аналітичне перетворення ``RGB [0..255]`` у ``HSV`` (``H: 0..360, S/V: 0..255``).

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.

    Returns:
        Масив ``float32`` форми ``(H, W, 3)`` у просторі HSV.

    Raises:
        ValueError: Якщо вхідний масив не має форму ``(H, W, 3)``.
    """
    rgb_arr = np.asarray(image_rgb, dtype=np.float32)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(
            f"Очікується RGB-зображення форми (H, W, 3), отримано {rgb_arr.shape}."
        )

    r_norm = rgb_arr[..., 0] / 255.0
    g_norm = rgb_arr[..., 1] / 255.0
    b_norm = rgb_arr[..., 2] / 255.0

    c_max = np.maximum(np.maximum(r_norm, g_norm), b_norm)
    c_min = np.minimum(np.minimum(r_norm, g_norm), b_norm)
    delta = c_max - c_min

    # Яскравість V та насиченість S у шкалі [0..255]
    v_chan = c_max * 255.0
    safe_c_max = np.where(c_max > 0.0, c_max, 1.0)
    s_chan = np.where(c_max > 0.0, (delta / safe_c_max) * 255.0, 0.0)

    # Тон H у градусах [0..360) із захистом від ділення на нуль при delta == 0
    safe_delta = np.where(delta > 0.0, delta, 1.0)
    h_chan = np.zeros_like(c_max, dtype=np.float32)

    nonzero = delta > 0.0
    is_red_max = nonzero & (c_max == r_norm)
    is_green_max = nonzero & (~is_red_max) & (c_max == g_norm)
    is_blue_max = nonzero & (~is_red_max) & (~is_green_max)

    h_chan[is_red_max] = 60.0 * (
        ((g_norm[is_red_max] - b_norm[is_red_max]) / safe_delta[is_red_max]) % 6.0
    )
    h_chan[is_green_max] = 60.0 * (
        ((b_norm[is_green_max] - r_norm[is_green_max]) / safe_delta[is_green_max]) + 2.0
    )
    h_chan[is_blue_max] = 60.0 * (
        ((r_norm[is_blue_max] - g_norm[is_blue_max]) / safe_delta[is_blue_max]) + 4.0
    )
    h_chan = np.mod(h_chan, 360.0)

    return np.stack((h_chan, s_chan, v_chan), axis=-1).astype(np.float32)


def _cyclic_hue_median(h_values: np.ndarray) -> float:
    """Обчислює циклічну медіану кута тону ``H`` у градусах ``[0, 360)``.

    Коректно обробляє перехід через ``0° / 360°`` (червоний сектор): центрує кути
    відносно кругового середнього вектора на одиничному колі, бере медіану
    знакових відхилень у ``[-180, 180)`` та повертає результат за модулем ``360``.
    """
    radians = np.deg2rad(h_values.astype(np.float64))
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))

    if math.hypot(mean_sin, mean_cos) < 1e-9:
        return float(np.median(h_values)) % 360.0

    mean_angle = math.degrees(math.atan2(mean_sin, mean_cos)) % 360.0
    signed_offsets = ((h_values.astype(np.float64) - mean_angle + 180.0) % 360.0) - 180.0
    return float((mean_angle + float(np.median(signed_offsets))) % 360.0)


def _hue_in_range(hue: float, h_min: float, h_max: float) -> bool:
    """Перевіряє потрапляння ``hue`` у діапазон із підтримкою циклічного переходу."""
    if h_min <= h_max:
        return h_min <= hue <= h_max
    return hue >= h_min or hue <= h_max


def _match_hsv_palette(
    hue: float,
    sat: float,
    val: float,
    palette_hsv: dict[str, dict[str, list[float]]],
) -> str:
    """Зіставляє медіанні ``(H, S, V)`` з палітрою, надаючи пріоритет хроматичним кольорам."""
    chromatic_items = [
        (name, bounds)
        for name, bounds in palette_hsv.items()
        if name.lower() not in _ACHROMATIC_NAMES
    ]
    achromatic_items = [
        (name, bounds)
        for name, bounds in palette_hsv.items()
        if name.lower() in _ACHROMATIC_NAMES
    ]

    for color_name, bounds in chromatic_items + achromatic_items:
        h_min, h_max = bounds["h"]
        s_min, s_max = bounds["s"]
        v_min, v_max = bounds["v"]
        if (
            _hue_in_range(hue, float(h_min), float(h_max))
            and float(s_min) <= sat <= float(s_max)
            and float(v_min) <= val <= float(v_max)
        ):
            return color_name
    return "Unknown"


def _is_grayscale_image(image_rgb: np.ndarray) -> bool:
    """Перевіряє, чи є RGB-зображення фактично чорно-білим (``R == G == B``)."""
    return bool(
        np.all(image_rgb[..., 0] == image_rgb[..., 1])
        and np.all(image_rgb[..., 1] == image_rgb[..., 2])
    )


def _classify_single_brick_color(
    hsv_image: np.ndarray,
    brick_mask: np.ndarray,
    color_cfg: dict[str, Any],
) -> str:
    """Визначає колір однієї деталі за еродованою маскою без тіней та відблисків."""
    if not np.any(brick_mask):
        return "Unknown"

    kernel_size = int(color_cfg.get("erosion_kernel", 3))
    eroded_mask = binary_erode_manual(brick_mask, kernel_size=kernel_size)
    # Фолбек на вихідну маску, якщо деталь надто тонка й ерозія знищила всі пікселі
    active_mask = eroded_mask if np.any(eroded_mask) else brick_mask

    pixels_hsv = hsv_image[active_mask]
    shadow_v_min = float(color_cfg.get("shadow_v_min", 40))
    glare_s_max = float(color_cfg.get("glare_s_max", 25))
    glare_v_min = float(color_cfg.get("glare_v_min", 235))

    not_shadow = pixels_hsv[:, 2] >= shadow_v_min
    not_glare = ~((pixels_hsv[:, 1] < glare_s_max) & (pixels_hsv[:, 2] > glare_v_min))
    filtered_hsv = pixels_hsv[not_shadow & not_glare]

    # Якщо після відсіювання тіні/відблисків пікселів не лишилось (напр., чорна/біла деталь),
    # повертаємось до усіх пікселів активної маски без аварійного падіння.
    valid_hsv = filtered_hsv if len(filtered_hsv) > 0 else pixels_hsv
    if len(valid_hsv) == 0:
        return "Unknown"

    median_h = _cyclic_hue_median(valid_hsv[:, 0])
    median_s = float(np.median(valid_hsv[:, 1]))
    median_v = float(np.median(valid_hsv[:, 2]))

    palette_hsv = color_cfg.get("palette_hsv", {})
    return _match_hsv_palette(median_h, median_s, median_v, palette_hsv)


def classify_colors(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray,
    detections: list[BrickDetection],
    config: dict[str, Any],
) -> list[BrickDetection]:
    """Визначає колір кожної знайденої деталі за медіаною HSV усередині еродованої маски.

    Відкидає темні пікселі заглиблень (``V < shadow_v_min``) та відблиски
    (``S < glare_s_max`` при ``V > glare_v_min``). Для чорно-білих кадрів
    (``R == G == B``, наприклад, 3D-рендерів Kaggle) повертає ``"Grayscale"``,
    не перериваючи Крок 1.

    Args:
        image_rgb: Початкове кольорове RGB-зображення ``uint8`` форми ``(H, W, 3)``.
        instance_mask: Інстанс-маска деталей (``0`` — фон, ``brick_id`` — пікселі деталі).
        detections: Список ``BrickDetection``, отриманий на Кроці 1.
        config: Повна конфігурація (секція ``color_classifier``).

    Returns:
        Оновлений список ``BrickDetection`` із заповненим полем ``color``.

    Raises:
        ValueError: Якщо просторові розміри ``image_rgb`` та ``instance_mask`` не збігаються.
    """
    rgb_arr = np.asarray(image_rgb, dtype=np.uint8)
    mask_arr = np.asarray(instance_mask)
    if rgb_arr.ndim != 3 or rgb_arr.shape[2] != 3:
        raise ValueError(
            f"Очікується RGB-зображення форми (H, W, 3), отримано {rgb_arr.shape}."
        )
    if mask_arr.shape != rgb_arr.shape[:2]:
        raise ValueError(
            f"Розмір маски {mask_arr.shape} не збігається з кадром {rgb_arr.shape[:2]}."
        )

    if not detections:
        return []

    if _is_grayscale_image(rgb_arr):
        return [replace(det, color="Grayscale") for det in detections]

    hsv_image = rgb_to_hsv_manual(rgb_arr)
    color_cfg = config.get("color_classifier", {})

    updated: list[BrickDetection] = []
    for det in detections:
        brick_mask = mask_arr == det.brick_id
        color_name = _classify_single_brick_color(hsv_image, brick_mask, color_cfg)
        updated.append(replace(det, color=color_name))
    return updated
