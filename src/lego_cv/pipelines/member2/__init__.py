"""Особиста зона відповідальності Учасника 2 (member2): пайплайни П3 та П4."""

from lego_cv.pipelines.member2.p3_sobel_watershed import SobelWatershedPipeline
from lego_cv.pipelines.member2.p4_fft_texture_hough import FftTextureHoughPipeline

__all__ = ["FftTextureHoughPipeline", "SobelWatershedPipeline"]
