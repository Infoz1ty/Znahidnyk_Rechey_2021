"""Пакет класичних тристадійних пайплайнів обробки зображень (member1 та member2)."""

from lego_cv.pipelines.member1 import ColorMorphPipeline, ContrastCannyPipeline
from lego_cv.pipelines.member2 import FftTextureHoughPipeline, SobelWatershedPipeline

__all__ = [
    "ColorMorphPipeline",
    "ContrastCannyPipeline",
    "FftTextureHoughPipeline",
    "SobelWatershedPipeline",
]
