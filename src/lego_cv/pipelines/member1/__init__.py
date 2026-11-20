"""Особиста зона відповідальності Учасника 1 (member1): пайплайни П1 та П2."""

from lego_cv.pipelines.member1.p1_color_morph import ColorMorphPipeline
from lego_cv.pipelines.member1.p2_contrast_canny import ContrastCannyPipeline

__all__ = ["ColorMorphPipeline", "ContrastCannyPipeline"]
