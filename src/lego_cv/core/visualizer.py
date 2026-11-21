"""Модуль В: Візуалізація результатів розпізнавання та пошуку (``--mode all`` / ``--mode find``).

Зона реалізації (Етап 1): Учасник 2 (member2).
Дозволені бібліотеки: ``Pillow`` (``Image``, ``ImageDraw``, ``ImageFont``) та ``NumPy``.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from lego_cv.core.types import BrickDetection

_DIMENSION_PATTERN = re.compile(r"^\d+x\d+$")


def _format_detection_label(detection: BrickDetection) -> str:
    """Формує лаконічний підпис деталі (напр., ``Red 2x4 brick``, ``Grayscale 1x1_round``)."""
    shape = detection.shape_class.strip() or "unknown"
    surface = detection.surface_type.strip().lower()
    color = detection.color.strip() or "Unknown"

    if surface in ("", "unknown") or shape.lower().endswith(f"_{surface}"):
        return f"{color} {shape}"
    return f"{color} {shape} {surface}"


def parse_find_query(query: str) -> list[str]:
    """Розбиває пошуковий запит на нормалізовані токени (зберігаючи складені форми).

    Наприклад:
    * ``"blue_2x4"`` -> ``["blue", "2x4"]``;
    * ``"red_2x2_tile"`` -> ``["red", "2x2", "tile"]``;
    * ``"yellow_1x1_round"`` -> ``["yellow", "1x1_round"]``.

    Args:
        query: Рядок запиту користувача з прапорця ``--find``.

    Returns:
        Список нормалізованих токенів у нижньому регістрі.
    """
    raw_tokens = [tok for tok in re.split(r"[\s_\-]+", query.strip().lower()) if tok]
    merged: list[str] = []
    idx = 0
    while idx < len(raw_tokens):
        current = raw_tokens[idx]
        if (
            idx + 1 < len(raw_tokens)
            and _DIMENSION_PATTERN.match(current)
            and raw_tokens[idx + 1] in ("round", "corner")
        ):
            merged.append(f"{current}_{raw_tokens[idx + 1]}")
            idx += 2
        else:
            merged.append(current)
            idx += 1
    return merged


def matches_find_query(detection: BrickDetection, query: str) -> bool:
    """Перевіряє, чи відповідає знайдена деталь пошуковому запиту ``query``.

    Args:
        detection: Класифікована деталь ``BrickDetection``.
        query: Пошуковий запит (наприклад, ``blue_2x4``, ``red_2x2_tile``, ``2x4``).

    Returns:
        ``True``, якщо всі токени запиту збігаються з атрибутами деталі.
    """
    query_tokens = parse_find_query(query)
    if not query_tokens:
        return False

    color_tok = detection.color.strip().lower()
    shape_tok = detection.shape_class.strip().lower()
    surface_tok = detection.surface_type.strip().lower()
    exact_attrs = {color_tok, shape_tok, surface_tok}

    for token in query_tokens:
        if token in exact_attrs:
            continue
        if token == "round" and "round" in shape_tok:
            continue
        return False
    return True


def _compute_label_rect(
    anchor_x: int,
    anchor_y: int,
    text_size: tuple[int, int],
    image_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    """Обчислює прямокутник фону підпису над рамкою (або всередині біля верхнього краю)."""
    width, height = image_size
    text_w, text_h = text_size
    pad = 2
    x0 = anchor_x
    x1 = min(width - 1, x0 + text_w + 2 * pad)
    y1 = (
        anchor_y - 1
        if anchor_y >= text_h + 2 * pad + 1
        else min(height - 1, anchor_y + text_h + 2 * pad)
    )
    y0 = max(0, y1 - text_h - 2 * pad)
    return (x0, y0, x1, y1)


def _draw_detection_box(
    draw: ImageDraw.ImageDraw,
    detection: BrickDetection,
    box_color: tuple[int, int, int],
    thickness: int,
    image_size: tuple[int, int],
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> None:
    """Малює обмежувальну рамку та контрастний текстовий підпис однієї деталі."""
    width, height = image_size
    x0 = max(0, min(int(detection.bbox[0]), width - 1))
    y0 = max(0, min(int(detection.bbox[1]), height - 1))
    x1 = max(x0, min(int(detection.bbox[2]), width - 1))
    y1 = max(y0, min(int(detection.bbox[3]), height - 1))

    draw.rectangle([x0, y0, x1, y1], outline=box_color, width=max(1, thickness))

    label = _format_detection_label(detection)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_size = (int(text_bbox[2] - text_bbox[0]), int(text_bbox[3] - text_bbox[1]))
    bg_x0, bg_y0, bg_x1, bg_y1 = _compute_label_rect(x0, y0, text_size, image_size)

    draw.rectangle([bg_x0, bg_y0, bg_x1, bg_y1], fill=box_color)
    draw.text((bg_x0 + 2, bg_y0 + 2), label, fill=(0, 0, 0), font=font)


def _as_rgb_tuple(raw_color: Any, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    """Перетворює RGB-колір із конфігу у кортеж трьох цілих ``[0..255]``."""
    if isinstance(raw_color, (list, tuple)) and len(raw_color) == 3:
        return (int(raw_color[0]), int(raw_color[1]), int(raw_color[2]))
    return fallback


def _validate_rgb_image(image_rgb: np.ndarray) -> np.ndarray:
    """Перевіряє, що масив є коректним RGB-зображенням ``(H, W, 3)`` типу ``uint8``."""
    arr = np.asarray(image_rgb, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError(
            f"Очікується RGB-зображення форми (H, W, 3), отримано {arr.shape}."
        )
    return arr


def render_all_detections(
    image_rgb: np.ndarray,
    detections: list[BrickDetection],
    config: dict[str, Any],
) -> np.ndarray:
    """Малює обмежувальні рамки та підписи класу і кольору для всіх знайдених деталей.

    Args:
        image_rgb: Вихідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        detections: Повний список класифікованих деталей.
        config: Конфігурація (секція ``visualizer``).

    Returns:
        Анотоване RGB-зображення ``uint8`` тієї самої форми ``(H, W, 3)``.
    """
    arr = _validate_rgb_image(image_rgb)
    if not detections:
        return arr.copy()

    vis_cfg = config.get("visualizer", {})
    box_thickness = int(vis_cfg.get("box_thickness", 3))
    box_color = _as_rgb_tuple(vis_cfg.get("default_box_color_rgb"), (255, 215, 0))

    pil_img = Image.fromarray(arr.copy(), mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    for detection in detections:
        _draw_detection_box(
            draw=draw,
            detection=detection,
            box_color=box_color,
            thickness=box_thickness,
            image_size=pil_img.size,
            font=font,
        )

    return np.asarray(pil_img, dtype=np.uint8)


def _restore_matched_regions(
    dimmed: np.ndarray,
    source_rgb: np.ndarray,
    mask_arr: np.ndarray,
    matched: list[BrickDetection],
) -> None:
    """Відновлює оригінальну яскравість усередині масок знайдених деталей."""
    height, width = source_rgb.shape[:2]
    for det in matched:
        brick_mask = mask_arr == det.brick_id
        if np.any(brick_mask):
            dimmed[brick_mask] = source_rgb[brick_mask]
            continue

        # Фолбек для бінарної маски (0/1), де brick_id > 1: відновлюємо область у межах bbox
        rx0 = max(0, int(det.bbox[0]))
        ry0 = max(0, int(det.bbox[1]))
        rx1 = min(width, int(det.bbox[2]) + 1)
        ry1 = min(height, int(det.bbox[3]) + 1)
        local_fg = mask_arr[ry0:ry1, rx0:rx1] > 0
        if np.any(local_fg):
            dimmed[ry0:ry1, rx0:rx1][local_fg] = source_rgb[ry0:ry1, rx0:rx1][local_fg]


def render_search_highlight(
    image_rgb: np.ndarray,
    instance_mask: np.ndarray,
    detections: list[BrickDetection],
    query: str,
    config: dict[str, Any],
) -> np.ndarray:
    """Режим пошуку ``--find``: затемнює фон на ``1 - dim_factor`` і підсвічує цільові деталі.

    Усе зображення затемнюється множником ``dim_factor`` (типово ``0.3``, тобто -70% яскравості),
    усередині масок знайдених за запитом деталей відновлюється оригінальна яскравість,
    а навколо кожної цільової деталі малюється яскрава рамка підсвітки з підписом.

    Args:
        image_rgb: Вихідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        instance_mask: Інстанс-маска (або бінарна маска) деталей ``(H, W)`` типу ``int32``.
        detections: Список класифікованих деталей.
        query: Пошуковий запит (наприклад, ``blue_2x4``, ``red_2x2_tile``, ``2x4``).
        config: Конфігурація (секція ``visualizer``).

    Returns:
        RGB-зображення ``uint8`` із підсвіченими збігами (або затемнений кадр, якщо збігів немає).
    """
    arr = _validate_rgb_image(image_rgb)
    mask_arr = np.asarray(instance_mask)
    if mask_arr.shape != arr.shape[:2]:
        raise ValueError(
            f"Розмір маски {mask_arr.shape} не збігається з кадром {arr.shape[:2]}."
        )

    vis_cfg = config.get("visualizer", {})
    dim_factor = float(vis_cfg.get("dim_factor", 0.3))
    box_thickness = int(vis_cfg.get("box_thickness", 3))
    highlight_color = _as_rgb_tuple(vis_cfg.get("highlight_color_rgb"), (0, 255, 128))

    dimmed = np.clip(np.rint(arr.astype(np.float32) * dim_factor), 0.0, 255.0).astype(
        np.uint8
    )
    matched = [det for det in detections if matches_find_query(det, query)]
    if not matched:
        return dimmed

    _restore_matched_regions(dimmed, arr, mask_arr, matched)

    pil_img = Image.fromarray(dimmed, mode="RGB")
    draw = ImageDraw.Draw(pil_img)
    font = ImageFont.load_default()

    for det in matched:
        _draw_detection_box(
            draw=draw,
            detection=det,
            box_color=highlight_color,
            thickness=box_thickness,
            image_size=pil_img.size,
            font=font,
        )

    return np.asarray(pil_img, dtype=np.uint8)
