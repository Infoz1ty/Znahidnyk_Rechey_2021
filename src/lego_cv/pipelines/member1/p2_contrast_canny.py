"""Пайплайн 2 (Учасник 1): Контрастно-контурний конвеєр обробки.

Категорії стадій (3 унікальні категорії за методичкою):
1. Гістограмна корекція — ручне RGB->LAB, адаптивний CLAHE на каналі L та Гаус 3x3.
2. Сегментація зображень — ручна порогова бінаризація за методом Оцу.
3. Контурний аналіз — ручний детектор границь Кенні та злиття границь із маскою.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.base_pipeline import BasePipeline

if TYPE_CHECKING:
    import numpy as np


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
    raise NotImplementedError(
        "TODO (T2.6, Учасник 1): реалізувати lab_clahe_enhance_manual."
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
