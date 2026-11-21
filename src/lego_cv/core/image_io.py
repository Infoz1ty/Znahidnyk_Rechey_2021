"""Введення/виведення зображень та обробка крайніх випадків (критерій C2).

Зона реалізації (Етап 1): Учасник 1 (member1).
Дозволені бібліотеки: виключно ``Pillow`` для декодування/кодування PNG та ``NumPy``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class ImageLoadError(ValueError):
    """Виняток для порожніх, відсутніх, пошкоджених або непідтримуваних файлів."""


class InvalidImageError(ValueError):
    """Виняток для некоректних розмірів, типів даних або каналів масиву зображення."""


def _validate_input_file(path: Path) -> None:
    """Перевіряє існування, непорожність та розширення файлу зображення."""
    if not path.exists() or not path.is_file():
        raise ImageLoadError(f"Файл зображення не знайдено: {path}")
    if path.stat().st_size == 0:
        raise ImageLoadError(f"Файл зображення порожній (0 байт): {path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ImageLoadError(
            f"Непідтримуваний формат файлу '{path.suffix}' для {path}. "
            f"Дозволені: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )


def _open_pil_image(path: Path) -> Image.Image:
    """Безпечно відкриває й повністю декодує зображення засобами Pillow."""
    _validate_input_file(path)
    try:
        with Image.open(path) as img:
            img.load()
            return img.copy()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageLoadError(
            f"Пошкоджений або нечитабельний файл зображення {path}: {exc}"
        ) from exc


def _convert_pil_to_rgb(img: Image.Image) -> Image.Image:
    """Нормалізує довільний режим Pillow (L, RGBA, P, 16-bit) до 8-бітного RGB."""
    if img.mode == "RGB":
        return img

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        # Композитинг на білий фон усуває артефакти прозорості без втрати кольору
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")

    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        raw = np.asarray(img, dtype=np.float64)
        max_val = float(np.max(raw)) if raw.size > 0 else 0.0
        if max_val > 255.0:
            raw = raw / 256.0
        gray_u8 = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
        return Image.fromarray(gray_u8, mode="L").convert("RGB")

    return img.convert("RGB")


def load_image_rgb(
    path: Path,
    *,
    max_side_px: int | None = 1600,
    min_side_px: int = 3,
    max_allowed_side_px: int = 10000,
) -> np.ndarray:
    """Завантажує зображення у форматі ``RGB uint8`` з перевіркою крайніх випадків.

    Коректно обробляє:
    * відсутній або порожній (0 байт) файл -> ``ImageLoadError``;
    * непідтримуване розширення або битий заголовок -> ``ImageLoadError``;
    * 1-канальні монохромні (``L``) та 4-канальні (``RGBA``) зображення -> ``RGB``;
    * замалі (``< min_side_px``) або надто великі розміри -> ``InvalidImageError``.

    Args:
        path: Шлях до файлу зображення.
        max_side_px: Якщо задано — пропорційно зменшити довгу сторону до цього розміру.
        min_side_px: Мінімально допустима ширина/висота кадру.
        max_allowed_side_px: Гранично допустима сторона вхідного файлу.

    Returns:
        Контигуозний масив NumPy форми ``(H, W, 3)`` типу ``uint8``.

    Raises:
        ImageLoadError: Якщо файл відсутній, порожній, пошкоджений або непідтримуваний.
        InvalidImageError: Якщо параметри обмежень або розміри зображення некоректні.
    """
    if min_side_px < 1 or max_allowed_side_px < min_side_px:
        raise InvalidImageError(
            f"Некоректні межі розміру: min={min_side_px}, max={max_allowed_side_px}"
        )
    if max_side_px is not None and max_side_px < min_side_px:
        raise InvalidImageError(
            f"Параметр max_side_px ({max_side_px}) менший за min_side_px ({min_side_px})"
        )

    img = _open_pil_image(Path(path))
    width, height = img.size
    if width < min_side_px or height < min_side_px:
        raise InvalidImageError(
            f"Зображення замале ({width}x{height}), мінімум {min_side_px}px: {path}"
        )
    if width > max_allowed_side_px or height > max_allowed_side_px:
        raise InvalidImageError(
            f"Зображення перевищує межу ({width}x{height} > {max_allowed_side_px}px): {path}"
        )

    rgb_img = _convert_pil_to_rgb(img)
    longest_side = max(width, height)
    if max_side_px is not None and longest_side > max_side_px:
        scale = max_side_px / float(longest_side)
        new_w = max(min_side_px, int(round(width * scale)))
        new_h = max(min_side_px, int(round(height * scale)))
        rgb_img = rgb_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    arr = np.asarray(rgb_img, dtype=np.uint8)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise InvalidImageError(
            f"Очікувався 3-канальний RGB-масив, отримано форму {arr.shape}: {path}"
        )
    return np.ascontiguousarray(arr)


def load_mask_binary(
    path: Path,
    *,
    max_side_px: int | None = None,
    target_size: tuple[int, int] | None = None,
) -> np.ndarray:
    """Завантажує еталонну бінарну маску для обчислення ``IoU``.

    Підтримує одноканальні (``L``, ``1``), ``RGB`` та ``RGBA`` маски (зокрема маски
    з антиаліасингом або прозорістю по краях).

    Args:
        path: Шлях до PNG-файлу еталонної маски.
        max_side_px: Опційне пропорційне зменшення довгої сторони (``NEAREST``).
        target_size: Опційний цільовий розмір ``(width, height)`` для точного
            вирівнювання з масштабованим зображенням.

    Returns:
        Бінарний масив NumPy форми ``(H, W)`` типу ``uint8`` зі значеннями ``0`` або ``1``.

    Raises:
        ImageLoadError: Якщо файл маски відсутній, порожній або пошкоджений.
        InvalidImageError: Якщо розміри маски або ``target_size`` некоректні.
    """
    img = _open_pil_image(Path(path))
    width, height = img.size
    if width < 1 or height < 1:
        raise InvalidImageError(f"Нульові розміри маски ({width}x{height}): {path}")

    has_alpha = img.mode in ("RGBA", "LA", "PA") or (
        img.mode == "P" and "transparency" in img.info
    )
    if has_alpha:
        rgba = img.convert("RGBA")
        # Прозорі пікселі маски трактуються як фон (0)
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        gray = Image.alpha_composite(background, rgba).convert("L")
    else:
        gray = img.convert("L")

    if target_size is not None:
        target_w, target_h = target_size
        if target_w < 1 or target_h < 1:
            raise InvalidImageError(f"Некоректний target_size: {target_size}")
        if (width, height) != (target_w, target_h):
            gray = gray.resize((target_w, target_h), Image.Resampling.NEAREST)
    elif max_side_px is not None:
        if max_side_px < 1:
            raise InvalidImageError(f"Некоректний max_side_px: {max_side_px}")
        longest_side = max(width, height)
        if longest_side > max_side_px:
            scale = max_side_px / float(longest_side)
            new_w = max(1, int(round(width * scale)))
            new_h = max(1, int(round(height * scale)))
            gray = gray.resize((new_w, new_h), Image.Resampling.NEAREST)

    arr = np.asarray(gray, dtype=np.uint8)
    # Підтримка як {0, 255} (та антиаліасингу 0..255), так і масок у форматі {0, 1}
    binary = (arr > 0) if int(arr.max()) <= 1 else (arr >= 128)
    return binary.astype(np.uint8)


def save_image(image: np.ndarray, path: Path) -> None:
    """Зберігає масив NumPy (grayscale або RGB) у файл зображення, створюючи каталоги.

    Args:
        image: Зображення форми ``(H, W)`` або ``(H, W, 3)``.
        path: Цільовий шлях для збереження файлу.

    Raises:
        InvalidImageError: Якщо масив порожній або має непідтримувану форму/канали.
    """
    if not isinstance(image, np.ndarray):
        raise InvalidImageError(f"Очікувався np.ndarray, отримано {type(image).__name__}")
    if image.ndim not in (2, 3) or image.size == 0:
        raise InvalidImageError(f"Некоректна форма масиву зображення: {image.shape}")
    if image.ndim == 3 and image.shape[2] != 3:
        raise InvalidImageError(
            f"Для 3D-зображення очікується 3 канали RGB, отримано {image.shape[2]}"
        )

    if image.dtype == np.bool_:
        arr_u8 = image.astype(np.uint8) * 255
    elif image.dtype == np.uint8:
        arr_u8 = image
    else:
        arr_u8 = np.clip(np.rint(image.astype(np.float64)), 0, 255).astype(np.uint8)

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "L" if arr_u8.ndim == 2 else "RGB"
    Image.fromarray(arr_u8, mode=mode).save(target)


def discover_images(input_path: Path) -> list[Path]:
    """Повертає відсортований перелік зображень із файлу або каталогу.

    Args:
        input_path: Шлях до одного файлу або папки із вибіркою.

    Returns:
        Список шляхів до файлів із підтримуваними розширеннями (або одиничний файл).

    Raises:
        ImageLoadError: Якщо вказаний шлях не існує або не є файлом/каталогом.
    """
    target = Path(input_path)
    if not target.exists():
        raise ImageLoadError(f"Вхідний шлях не знайдено: {target}")
    if target.is_file():
        return [target]
    if target.is_dir():
        return sorted(
            item
            for item in target.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS
        )
    raise ImageLoadError(f"Непідтримуваний тип вхідного шляху: {target}")
