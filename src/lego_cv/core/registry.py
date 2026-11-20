"""Реєстр усіх зареєстрованих пайплайнів обробки (П1–П4)."""

from __future__ import annotations

from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.pipelines.member1.p1_color_morph import ColorMorphPipeline
from lego_cv.pipelines.member1.p2_contrast_canny import ContrastCannyPipeline
from lego_cv.pipelines.member2.p3_sobel_watershed import SobelWatershedPipeline
from lego_cv.pipelines.member2.p4_fft_texture_hough import FftTextureHoughPipeline

_PIPELINE_CLASSES: dict[str, type[BasePipeline]] = {
    "p1": ColorMorphPipeline,
    "p2": ContrastCannyPipeline,
    "p3": SobelWatershedPipeline,
    "p4": FftTextureHoughPipeline,
}


def get_pipeline(pipeline_id: str) -> BasePipeline:
    """Повертає екземпляр пайплайну за його ідентифікатором.

    Args:
        pipeline_id: Один із ключів ``p1``, ``p2``, ``p3``, ``p4``.

    Returns:
        Екземпляр відповідного класу ``BasePipeline``.

    Raises:
        KeyError: Якщо передано невідомий ідентифікатор.
    """
    normalized = pipeline_id.strip().lower()
    if normalized not in _PIPELINE_CLASSES:
        available = ", ".join(sorted(_PIPELINE_CLASSES))
        raise KeyError(
            f"Невідомий пайплайн '{pipeline_id}'. Доступні: {available}"
        )
    pipeline = _PIPELINE_CLASSES[normalized]()
    pipeline.validate_contract()
    return pipeline


def list_pipelines() -> list[BasePipeline]:
    """Повертає відсортований список усіх 4 пайплайнів команди.

    Returns:
        Список екземплярів ``BasePipeline`` у порядку ``p1..p4``.
    """
    return [get_pipeline(key) for key in sorted(_PIPELINE_CLASSES)]
