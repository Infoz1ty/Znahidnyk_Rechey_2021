"""Пайплайн 1 (Учасник 1): Колірно-морфологічний конвеєр обробки.

Категорії стадій (3 унікальні категорії за методичкою):
1. Фільтрація зображень — ручний медіанний фільтр (5x5).
2. Колірна корекція — ручне RGB->HSV та порогова сегментація фону.
3. Морфологічна обробка — ручне морфологічне закриття та flood-fill дірок.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from lego_cv.core.base_pipeline import BasePipeline

if TYPE_CHECKING:
    import numpy as np


def median_filter_manual(image_rgb: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Ручний просторовий медіанний фільтр ``k x k`` із відзеркаленням меж.

    Придушує імпульсний шум та пил без розмиття прямих кутів деталей LEGO.

    Args:
        image_rgb: Вхідне зображення ``(H, W, 3)`` або ``(H, W)`` типу ``uint8``.
        kernel_size: Непарний розмір квадратного вікна (за замовчуванням 5).

    Returns:
        Відфільтроване зображення тієї самої форми типу ``uint8``.
    """
    raise NotImplementedError("TODO (T2.1, Учасник 1): реалізувати median_filter_manual.")


def hsv_background_segment(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне перетворення ``RGB -> HSV`` та виділення маски переднього плану.

    Відсікає нейтральний фон столу за низькою насиченістю ``S < bg_s_max`` і
    яскравістю ``V`` у межах фону (або адаптивно за рамкою кадру), повертаючи
    інвертовану бінарну маску деталей LEGO.

    Args:
        image_rgb: Відфільтроване RGB-зображення ``uint8``.
        stage_cfg: Секція ``stage2_hsv_segmentation`` із ``configs/pipeline_p1.json``.

    Returns:
        Бінарна маска переднього плану ``(H, W)`` типу ``uint8`` (0 або 255).
    """
    raise NotImplementedError(
        "TODO (T2.2, Учасник 1): реалізувати hsv_background_segment."
    )


def morphological_close_and_fill(
    binary_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Ручне морфологічне закриття (дилатація -> ерозія) та заповнення внутрішніх дірок.

    Усуває «дірки» від світлових відблисків на верхівках шпильок методом BFS flood-fill
    від меж зображення.

    Args:
        binary_mask: Бінарна маска ``(H, W)`` зі Стадії 2.
        stage_cfg: Секція ``stage3_morphology`` із ``configs/pipeline_p1.json``.

    Returns:
        Очищена бінарна маска ``(H, W)`` типу ``int32`` (0 — фон, 1 — деталі).
    """
    raise NotImplementedError(
        "TODO (T2.3, Учасник 1): реалізувати morphological_close_and_fill."
    )


class ColorMorphPipeline(BasePipeline):
    """Пайплайн П1 (Учасник 1): Медіанний фільтр -> HSV-сегментація -> Морфологія."""

    pipeline_id = "p1"
    name = "Колірно-морфологічний (Медіанний -> HSV -> Морфологія)"
    owner = "member1"
    stage_categories = (
        "Фільтрація зображень",
        "Колірна корекція",
        "Морфологічна обробка",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Фільтрація): медіанне шумоподавлення вхідного кадру."""
        stage_cfg = config.get("stage1_median", {})
        kernel_size = int(stage_cfg.get("kernel_size", 5))
        filtered = median_filter_manual(image_rgb, kernel_size=kernel_size)
        return filtered, {}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Колірна корекція): виділення об'єктів у просторі HSV."""
        stage_cfg = config.get("stage2_hsv_segmentation", {})
        raw_mask = hsv_background_segment(stage1_output, stage_cfg)
        return raw_mask, state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Морфологія): закриття контурів та заповнення відблисків."""
        stage_cfg = config.get("stage3_morphology", {})
        final_mask = morphological_close_and_fill(stage2_output, stage_cfg)
        visual = (final_mask > 0).astype("uint8") * 255
        return final_mask, visual, {"source_state": bool(state)}
