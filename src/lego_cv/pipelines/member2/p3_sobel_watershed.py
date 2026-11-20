"""Пайплайн 3 (Учасник 2): Градієнтний конвеєр із розділенням злиплих деталей.

Категорії стадій (3 унікальні категорії за методичкою):
1. Ретушування та покращення — ручний білатеральний фільтр + Unsharp Masking.
2. Контурний аналіз — ручний градієнтний оператор Собеля.
3. Сегментація зображень — ручна карта відстаней + маркерний алгоритм вододілу (Watershed).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.base_pipeline import BasePipeline

if TYPE_CHECKING:
    import numpy as np


def bilateral_unsharp_manual(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручний білатеральний фільтр зі збереженням ребер + підвищення різкості Unsharp Mask.

    Згладжує текстуру паперу й підкреслює тонкі щілини між сусідніми кубиками LEGO.

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        stage_cfg: Секція ``stage1_bilateral_unsharp`` із ``configs/pipeline_p3.json``.

    Returns:
        Покращене зображення ``(H, W, 3)`` або ``(H, W)`` типу ``uint8``.
    """
    raise NotImplementedError(
        "TODO (T2.11, Учасник 2): реалізувати bilateral_unsharp_manual."
    )


def sobel_gradient_manual(
    image: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Ручний оператор Собеля (``G_x``, ``G_y``, модуль градієнта) + груба маска об'єктів.

    Args:
        image: Покращене зображення зі Стадії 1.
        stage_cfg: Секція ``stage2_sobel_gradient`` із ``configs/pipeline_p3.json``.

    Returns:
        Кортеж ``(gradient_magnitude_uint8, rough_fg_mask_uint8)``.
    """
    raise NotImplementedError("TODO (T2.12, Учасник 2): реалізувати sobel_gradient_manual.")


def distance_watershed_manual(
    gradient_mag: np.ndarray,
    rough_fg_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Ручна карта відстаней (Chamfer/Euclidean) + маркерний Watershed на ``heapq``.

    Розрізає злиплі бортами деталі LEGO по хребтах градієнтного рельєфу Собеля та
    повертає інстанс-маску (``0`` — фон, ``1..K`` — окремі деталі).

    Args:
        gradient_mag: Градієнтний рельєф Собеля зі Стадії 2.
        rough_fg_mask: Попередня бінарна маска переднього плану.
        stage_cfg: Секція ``stage3_watershed`` із ``configs/pipeline_p3.json``.

    Returns:
        Кортеж ``(instance_mask_int32, watershed_visual_uint8)``.
    """
    raise NotImplementedError(
        "TODO (T2.13, Учасник 2): реалізувати distance_watershed_manual."
    )


class SobelWatershedPipeline(BasePipeline):
    """Пайплайн П3 (Учасник 2): Білатеральний+Unsharp -> Собель -> Watershed."""

    pipeline_id = "p3"
    name = "Градієнтний з розділенням злиплих (Білатеральний -> Собель -> Watershed)"
    owner = "member2"
    stage_categories = (
        "Ретушування та покращення зображень",
        "Контурний аналіз",
        "Сегментація зображень",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Покращення): білатеральне згладжування та Unsharp Masking."""
        stage_cfg = config.get("stage1_bilateral_unsharp", {})
        sharpened = bilateral_unsharp_manual(image_rgb, stage_cfg)
        return sharpened, {}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Контурний аналіз): градієнтний рельєф Собеля та груба маска."""
        stage_cfg = config.get("stage2_sobel_gradient", {})
        grad_mag, rough_fg = sobel_gradient_manual(stage1_output, stage_cfg)
        next_state = dict(state)
        next_state["rough_fg_mask"] = rough_fg
        return grad_mag, next_state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Сегментація): карта відстаней і маркерний вододіл."""
        stage_cfg = config.get("stage3_watershed", {})
        instance_mask, ws_vis = distance_watershed_manual(
            gradient_mag=stage2_output,
            rough_fg_mask=state["rough_fg_mask"],
            stage_cfg=stage_cfg,
        )
        return instance_mask, ws_vis, {"is_instance_mask": True}
