"""Абстрактний базовий клас тристадійного пайплайну обробки зображень."""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import TYPE_CHECKING, Any

from lego_cv.core.types import PipelineResult, StageOutput

if TYPE_CHECKING:
    import numpy as np


class BasePipeline(ABC):
    """Єдиний контракт для всіх 4 класичних пайплайнів (П1–П4).

    Кожен нащадок зобов'язаний визначити:
    * ``pipeline_id`` (``p1``..``p4``);
    * ``name`` (людинозрозуміла назва);
    * ``owner`` (``member1`` або ``member2``);
    * ``stage_categories`` — кортеж щонайменше з 3 **унікальних** категорій методички.
    """

    pipeline_id: str = ""
    name: str = ""
    owner: str = ""
    stage_categories: tuple[str, str, str] = ("", "", "")

    def describe(self) -> dict[str, Any]:
        """Повертає метадані пайплайну для CLI, звітів та контрактних тестів.

        Returns:
            Словник із полями ``pipeline_id``, ``name``, ``owner`` та ``stage_categories``.
        """
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "owner": self.owner,
            "stage_categories": list(self.stage_categories),
        }

    def validate_contract(self) -> None:
        """Перевіряє відповідність пайплайну вимогам методички (3+ унікальні категорії).

        Raises:
            ValueError: Якщо метадані неповні або категорії стадій дублюються.
        """
        if not self.pipeline_id or not self.name or not self.owner:
            raise ValueError(f"Неповні метадані пайплайну: {self.__class__.__name__}")
        cleaned = [cat.strip() for cat in self.stage_categories if cat.strip()]
        if len(cleaned) < 3 or len(set(cleaned)) < 3:
            raise ValueError(
                f"Пайплайн '{self.pipeline_id}' повинен містити щонайменше "
                f"3 стадії з 3 різних категорій, отримано: {self.stage_categories}"
            )

    @abstractmethod
    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Виконує Стадію 1 (фільтрація / покращення / частотна обробка).

        Args:
            image_rgb: Вхідне зображення ``(H, W, 3)`` типу ``uint8``.
            config: Об'єднаний словник конфігурації.

        Returns:
            Кортеж ``(stage1_image, state)``, де ``state`` передається на Стадію 2.
        """

    @abstractmethod
    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Виконує Стадію 2 (сегментація / градієнт / текстурний аналіз).

        Args:
            stage1_output: Вихідне зображення Стадії 1.
            state: Проміжний словник стану від Стадії 1.
            config: Об'єднаний словник конфігурації.

        Returns:
            Кортеж ``(stage2_image, state)``, де ``state`` передається на Стадію 3.
        """

    @abstractmethod
    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Виконує Стадію 3 (морфологія / контури / Watershed / Хаф).

        Args:
            stage2_output: Вихідне зображення Стадії 2.
            state: Проміжний словник стану від Стадії 2.
            config: Об'єднаний словник конфігурації.

        Returns:
            Кортеж ``(final_mask, stage3_visual, metadata)``, де ``final_mask`` —
            бінарна або інстанс-маска типу ``int32``, ``stage3_visual`` — зображення
            для збереження, ``metadata`` — додаткові ознаки (напр. шпильки Хафа).
        """

    def run(self, image_rgb: np.ndarray, config: dict[str, Any]) -> PipelineResult:
        """Послідовно й автоматично виконує всі 3 стадії пайплайну з таймінгом.

        Args:
            image_rgb: Вхідне RGB-зображення ``uint8``.
            config: Словник параметрів запуску.

        Returns:
            Об'єкт ``PipelineResult`` із фінальною маскою та проміжними стадіями.
        """
        self.validate_contract()

        t0 = time.perf_counter()
        stage1_img, state1 = self.stage1_preprocess(image_rgb, config)
        dt1 = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        stage2_img, state2 = self.stage2_segment(stage1_img, state1, config)
        dt2 = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        final_mask, stage3_vis, metadata = self.stage3_refine(stage2_img, state2, config)
        dt3 = (time.perf_counter() - t2) * 1000.0

        stages = [
            StageOutput(
                name="stage_1_preprocess",
                category=self.stage_categories[0],
                image=stage1_img,
                duration_ms=dt1,
            ),
            StageOutput(
                name="stage_2_segment",
                category=self.stage_categories[1],
                image=stage2_img,
                duration_ms=dt2,
            ),
            StageOutput(
                name="stage_3_refine",
                category=self.stage_categories[2],
                image=stage3_vis,
                duration_ms=dt3,
            ),
        ]
        return PipelineResult(
            pipeline_id=self.pipeline_id,
            mask=final_mask,
            denoised=stage1_img,
            stages=stages,
            metadata=metadata,
        )
