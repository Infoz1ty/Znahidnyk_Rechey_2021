"""Пайплайн 4 (Учасник 2): Частотно-текстурний конвеєр аналізу поверхні та шпильок.

Категорії стадій (3 унікальні категорії за методичкою):
1. Перетворення Фур'є — ручне 2D FFT (Кулі-Тьюкі) та смугова фільтрація Баттерворта.
2. Аналіз текстур — ручна карта LBP та ознаки матриці суміжності GLCM.
3. Перетворення Хафа — ручний Circle Hough Transform для детекції та підрахунку шпильок.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.base_pipeline import BasePipeline

if TYPE_CHECKING:
    import numpy as np


def fft2_bandpass_manual(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне 2D швидке перетворення Фур'є (Radix-2) та смугова фільтрація у частотній області.

    Придушення наднизьких частот вирівнює освітлення фону, а надвисоких — усуває дрібний шум.

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        stage_cfg: Секція ``stage1_fft_bandpass`` із ``configs/pipeline_p4.json``.

    Returns:
        Відновлений після 2D IFFT кадр яскравості ``(H, W)`` типу ``uint8``.
    """
    raise NotImplementedError("TODO (T2.16, Учасник 2): реалізувати fft2_bandpass_manual.")


def lbp_glcm_texture_segment(
    gray_image: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Будує ручну карту LBP для сегментації та обчислює ознаки GLCM (контраст, однорідність).

    Дозволяє відрізнити рельєфні брики зі шпильками від гладких тайлів.

    Args:
        gray_image: Відфільтроване зображення яскравості зі Стадії 1.
        stage_cfg: Секція ``stage2_lbp_glcm`` із ``configs/pipeline_p4.json``.

    Returns:
        Кортеж ``(texture_mask_uint8, texture_features_dict)``.
    """
    raise NotImplementedError(
        "TODO (T2.17, Учасник 2): реалізувати lbp_glcm_texture_segment."
    )


def hough_circles_studs_manual(
    gray_image: np.ndarray,
    texture_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Ручне градієнтне перетворення Хафа для кіл усередині знайдених масок деталей.

    Підраховує кількість шпильок ``K_studs`` для кожного компонента та повертає
    маску, візуалізацію знайдених кіл і метадані для геометричного класифікатора.

    Args:
        gray_image: Кадр яскравості зі Стадії 1.
        texture_mask: Маска кандидатів зі Стадії 2.
        stage_cfg: Секція ``stage3_hough_circles`` із ``configs/pipeline_p4.json``.

    Returns:
        Кортеж ``(final_mask_int32, circles_visual_uint8, stud_metadata)``.
    """
    raise NotImplementedError(
        "TODO (T2.18, Учасник 2): реалізувати hough_circles_studs_manual."
    )


class FftTextureHoughPipeline(BasePipeline):
    """Пайплайн П4 (Учасник 2): 2D FFT -> LBP/GLCM -> Кола Хафа."""

    pipeline_id = "p4"
    name = "Частотно-текстурний (2D FFT -> LBP/GLCM -> Кола Хафа)"
    owner = "member2"
    stage_categories = (
        "Перетворення Фур'є",
        "Аналіз текстур",
        "Перетворення Хафа",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Перетворення Фур'є): 2D FFT смугова фільтрація."""
        stage_cfg = config.get("stage1_fft_bandpass", {})
        filtered_gray = fft2_bandpass_manual(image_rgb, stage_cfg)
        return filtered_gray, {"filtered_gray": filtered_gray}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Аналіз текстур): LBP-карта та GLCM-ознаки поверхні."""
        stage_cfg = config.get("stage2_lbp_glcm", {})
        texture_mask, texture_features = lbp_glcm_texture_segment(stage1_output, stage_cfg)
        next_state = dict(state)
        next_state["texture_features"] = texture_features
        return texture_mask, next_state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Перетворення Хафа): пошук кіл-шпильок усередині деталей."""
        stage_cfg = config.get("stage3_hough_circles", {})
        final_mask, circles_vis, stud_meta = hough_circles_studs_manual(
            gray_image=state["filtered_gray"],
            texture_mask=stage2_output,
            stage_cfg=stage_cfg,
        )
        metadata = dict(stud_meta)
        metadata["texture_features"] = state.get("texture_features", {})
        return final_mask, circles_vis, metadata
