"""Спільне ядро системи: типи даних, I/O, базовий пайплайн, класифікатори та runner."""

from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.core.types import BrickDetection, PipelineResult, RunMetrics, StageOutput

__all__ = [
    "BasePipeline",
    "BrickDetection",
    "PipelineResult",
    "RunMetrics",
    "StageOutput",
]
