"""Скрипт автогенерації еталонних бінарних масок для 3D-рендерів Kaggle.

Зона реалізації (Етап 1): Учасник 2 (member2).
Усі алгоритми (адаптивна оцінка фону, ручна фільтрація зв'язних компонент,
ручний BFS flood-fill для заповнення дірок) реалізовано виключно на
стандартній бібліотеці Python + NumPy; Pillow використовується лише для I/O.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
LABELS_FIELDNAMES = [
    "filename",
    "split",
    "scenario",
    "has_mask",
    "total_count",
    "items_json",
]

# Мапінг назв папок класів Kaggle у формат міток labels.csv (<color>_<shape>_<surface>)
CLASS_FOLDER_TO_LABEL: dict[str, str] = {
    "brick_1x1": "grayscale_1x1_brick",
    "brick_1x2": "grayscale_1x2_brick",
    "brick_1x3": "grayscale_1x3_brick",
    "brick_1x4": "grayscale_1x4_brick",
    "brick_2x2": "grayscale_2x2_brick",
    "brick_2x3": "grayscale_2x3_brick",
    "brick_2x4": "grayscale_2x4_brick",
    "plate_1x1": "grayscale_1x1_brick",
    "plate_1x2": "grayscale_1x2_brick",
    "plate_1x3": "grayscale_1x3_brick",
    "plate_2x2": "grayscale_2x2_brick",
    "plate_2x3": "grayscale_2x3_brick",
    "plate_2x4": "grayscale_2x4_brick",
    "tile_1x1": "grayscale_1x1_tile",
    "tile_1x2": "grayscale_1x2_tile",
    "tile_2x2": "grayscale_2x2_tile",
    "round_1x1": "grayscale_1x1_round",
    "round_tile_2x2": "grayscale_2x2_round",
    "cheese_1x1": "grayscale_1x1_tile",
    "roof_2x2": "grayscale_2x2_brick",
    "corner_brick_2x2": "grayscale_2x2_corner_brick",
    "corner_plate_2x2": "grayscale_2x2_corner_brick",
}


def ensure_rgb_image(image: Image.Image) -> Image.Image:
    """Конвертує довільне зображення Pillow у валідний 8-бітний RGB-формат.

    Args:
        image: Відкритий об'єкт ``PIL.Image.Image``.

    Returns:
        Зображення в режимі ``RGB`` (8 біт на канал).
    """
    if image.mode == "RGB":
        return image
    if image.mode in ("RGBA", "LA"):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        return Image.alpha_composite(background, rgba).convert("RGB")
    if image.mode.startswith("I;16") or image.mode == "I":
        arr = np.asarray(image, dtype=np.float32)
        scaled = np.clip(arr / 256.0, 0.0, 255.0).astype(np.uint8)
        rgb_arr = np.stack((scaled, scaled, scaled), axis=-1)
        return Image.fromarray(rgb_arr, mode="RGB")
    return image.convert("RGB")


def estimate_border_background(
    image_rgb: np.ndarray,
    border_width: int = 3,
) -> np.ndarray:
    """Адаптивно оцінює колір фону за медіаною граничних пікселів рамки кадру.

    Args:
        image_rgb: Масив зображення форми ``(H, W, 3)`` типу ``uint8``.
        border_width: Товщина граничної смуги в пікселях.

    Returns:
        Вектор фонового кольору форми ``(3,)`` типу ``float32``.
    """
    height, width = image_rgb.shape[:2]
    bw = max(1, min(border_width, height // 2 or 1, width // 2 or 1))

    top = image_rgb[:bw, :, :].reshape(-1, 3)
    bottom = image_rgb[-bw:, :, :].reshape(-1, 3)
    left = image_rgb[:, :bw, :].reshape(-1, 3)
    right = image_rgb[:, -bw:, :].reshape(-1, 3)
    border_pixels = np.concatenate((top, bottom, left, right), axis=0)
    return np.median(border_pixels, axis=0).astype(np.float32)


def _find_root(parents: list[int], idx: int) -> int:
    """Знаходить корінь елемента у структурі Union-Find зі стисненням шляхів."""
    root = idx
    while parents[root] != root:
        root = parents[root]
    while parents[idx] != root:
        nxt = parents[idx]
        parents[idx] = root
        idx = nxt
    return root


def _union_roots(
    parents: list[int],
    sizes: list[int],
    idx_a: int,
    idx_b: int,
) -> None:
    """Об'єднує два класи еквівалентності за розміром компоненти."""
    root_a = _find_root(parents, idx_a)
    root_b = _find_root(parents, idx_b)
    if root_a == root_b:
        return
    if sizes[root_a] < sizes[root_b]:
        root_a, root_b = root_b, root_a
    parents[root_b] = root_a
    sizes[root_a] += sizes[root_b]


def _merge_adjacent_row_runs(
    prev_row_runs: list[tuple[int, int, int]],
    curr_row_runs: list[tuple[int, int, int]],
    parents: list[int],
    sizes: list[int],
) -> None:
    """Об'єднує 8-зв'язні відрізки поточного та попереднього рядків."""
    prev_idx = 0
    for curr_id, curr_start, curr_end in curr_row_runs:
        while prev_idx < len(prev_row_runs) and prev_row_runs[prev_idx][2] < curr_start:
            prev_idx += 1
        scan = prev_idx
        while scan < len(prev_row_runs) and prev_row_runs[scan][1] <= curr_end:
            _union_roots(parents, sizes, curr_id, prev_row_runs[scan][0])
            scan += 1


def remove_small_components(mask: np.ndarray, min_area: int = 32) -> np.ndarray:
    """Видаляє дрібний шум за допомогою ручної 8-зв'язної сегментації компонент.

    Використовує рядковий (scanline RLE) Union-Find без зовнішніх CV-бібліотек.

    Args:
        mask: Булева маска переднього плану форми ``(H, W)``.
        min_area: Мінімальна площа компоненти в пікселях для збереження.

    Returns:
        Очищена булева маска форми ``(H, W)``.
    """
    bool_mask = np.asarray(mask, dtype=np.bool_)
    if min_area <= 1 or not bool_mask.any():
        return bool_mask.copy()

    runs: list[tuple[int, int, int]] = []
    parents: list[int] = []
    sizes: list[int] = []
    prev_row_runs: list[tuple[int, int, int]] = []

    for row in range(bool_mask.shape[0]):
        padded_row = np.concatenate(([False], bool_mask[row], [False]))
        transitions = np.flatnonzero(padded_row[1:] != padded_row[:-1])
        curr_row_runs: list[tuple[int, int, int]] = []

        for pair_idx in range(0, len(transitions), 2):
            start = int(transitions[pair_idx])
            end = int(transitions[pair_idx + 1])
            run_id = len(runs)
            runs.append((row, start, end))
            parents.append(run_id)
            sizes.append(end - start)
            curr_row_runs.append((run_id, start, end))

        _merge_adjacent_row_runs(prev_row_runs, curr_row_runs, parents, sizes)
        prev_row_runs = curr_row_runs

    cleaned = np.zeros_like(bool_mask, dtype=np.bool_)
    for run_id, (row, start, end) in enumerate(runs):
        if sizes[_find_root(parents, run_id)] >= min_area:
            cleaned[row, start:end] = True
    return cleaned


def fill_holes_manual(mask: np.ndarray) -> np.ndarray:
    """Заповнює внутрішні дірки маски за допомогою ручного 4-зв'язного BFS flood-fill.

    Запускає пошук у ширину від зовнішньої рамки кадру по пікселях фону;
    усі фонові пікселі, недосяжні ззовні, трактуються як внутрішні дірки й заповнюються.

    Args:
        mask: Булева маска переднього плану форми ``(H, W)``.

    Returns:
        Суцільна булева маска форми ``(H, W)`` без внутрішніх отворів.
    """
    bool_mask = np.asarray(mask, dtype=np.bool_)
    ys, xs = np.nonzero(bool_mask)
    if len(xs) == 0:
        return bool_mask.copy()

    # Обмежуємо BFS охоплювальним прямокутником із 1-піксельною рамкою фону
    y_min, y_max = int(ys.min()), int(ys.max()) + 1
    x_min, x_max = int(xs.min()), int(xs.max()) + 1
    crop = bool_mask[y_min:y_max, x_min:x_max]
    padded = np.pad(crop, pad_width=1, mode="constant", constant_values=False)
    crop_h, crop_w = padded.shape

    visited = np.zeros((crop_h, crop_w), dtype=np.bool_)
    queue: deque[tuple[int, int]] = deque([(0, 0)])
    visited[0, 0] = True

    while queue:
        r, c = queue.popleft()
        if r > 0 and not visited[r - 1, c] and not padded[r - 1, c]:
            visited[r - 1, c] = True
            queue.append((r - 1, c))
        if r + 1 < crop_h and not visited[r + 1, c] and not padded[r + 1, c]:
            visited[r + 1, c] = True
            queue.append((r + 1, c))
        if c > 0 and not visited[r, c - 1] and not padded[r, c - 1]:
            visited[r, c - 1] = True
            queue.append((r, c - 1))
        if c + 1 < crop_w and not visited[r, c + 1] and not padded[r, c + 1]:
            visited[r, c + 1] = True
            queue.append((r, c + 1))

    result = bool_mask.copy()
    result[y_min:y_max, x_min:x_max] = ~visited[1:-1, 1:-1]
    return result


def build_binary_mask(
    image_rgb: np.ndarray,
    bg_tolerance: int = 12,
    min_component_area: int = 32,
) -> np.ndarray:
    """Будує строго бінарну маску (0/255) деталі LEGO на однорідному фоні.

    Args:
        image_rgb: Вхідне RGB-зображення форми ``(H, W, 3)`` типу ``uint8``.
        bg_tolerance: Допуск відхилення інтенсивності від адаптивного фону.
        min_component_area: Мінімальна площа зв'язної компоненти в пікселях.

    Returns:
        Одноканальна маска форми ``(H, W)`` типу ``uint8`` зі значеннями лише ``0`` або ``255``.
    """
    bg_color = estimate_border_background(image_rgb)
    diff = np.max(np.abs(image_rgb.astype(np.float32) - bg_color), axis=2)
    raw_mask = diff > float(bg_tolerance)
    denoised = remove_small_components(raw_mask, min_area=min_component_area)
    filled = fill_holes_manual(denoised)
    return np.where(filled, np.uint8(255), np.uint8(0))


def map_class_to_label(class_key: str) -> str:
    """Мапує назву папки класу Kaggle у стандартну мітку ``labels.csv``.

    Args:
        class_key: Нормалізована назва папки класу (наприклад, ``brick_2x4``).

    Returns:
        Мітка формату ``grayscale_<shape>_<surface>``.
    """
    normalized = class_key.strip().lower().replace(" ", "_")
    return CLASS_FOLDER_TO_LABEL.get(normalized, f"grayscale_{normalized}")


def _sanitize_stem(stem: str) -> str:
    """Замінює пробіли в імені файлу на підкреслення для безпечних CLI-шляхів."""
    return "_".join(stem.strip().split())


def _normalize_to_rgb_png(src_path: Path, dst_path: Path) -> None:
    """Зберігає зображення у валідний 8-бітний RGB PNG за шляхом ``dst_path``."""
    with Image.open(src_path) as opened:
        rgb_image = ensure_rgb_image(opened)
        needs_save = (
            src_path != dst_path
            or opened.format != "PNG"
            or opened.mode != "RGB"
        )
        if needs_save:
            rgb_image.save(dst_path, format="PNG")
    if src_path != dst_path and src_path.exists():
        src_path.unlink()


def flatten_kaggle_directory(kaggle_dir: Path) -> list[tuple[Path, str]]:
    """Приводить ``kaggle_dir`` до пласкої структури 8-бітних RGB PNG із префіксом класу.

    Якщо в каталозі є підпапки класів (наприклад, ``brick_2x4/``), переносить усі зображення
    в корінь ``kaggle_dir`` із префіксом ``<class_name>__`` (щоб уникнути колізій і зберегти
    сумісність із пласкою схемою ``dataset/README.md`` та ``runner.py``), видаляючи порожні
    підпапки та зберігаючи ``.gitkeep``.

    Args:
        kaggle_dir: Каталог ``dataset/kaggle_topdown/``.

    Returns:
        Відсортований список пар ``(шлях_до_png, class_key)``.
    """
    entries: list[tuple[Path, str]] = []

    # 1. Обробляємо підпапки класів (якщо присутні)
    for subdir in sorted(p for p in kaggle_dir.iterdir() if p.is_dir()):
        class_key = subdir.name.strip().lower().replace(" ", "_")
        for file_path in sorted(subdir.iterdir()):
            if not file_path.is_file() or file_path.name.startswith("."):
                continue
            if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            clean_stem = _sanitize_stem(file_path.stem)
            target_name = (
                f"{clean_stem}.png"
                if clean_stem.startswith(f"{class_key}__")
                else f"{class_key}__{clean_stem}.png"
            )
            target_path = kaggle_dir / target_name
            _normalize_to_rgb_png(file_path, target_path)

        # Видаляємо підпапку після перенесення, якщо вона спорожніла
        if not any(subdir.iterdir()):
            subdir.rmdir()

    # 2. Збираємо всі пласкі зображення в корені kaggle_dir
    for file_path in sorted(kaggle_dir.iterdir()):
        if not file_path.is_file() or file_path.name.startswith("."):
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        clean_stem = _sanitize_stem(file_path.stem)
        target_path = kaggle_dir / f"{clean_stem}.png"
        _normalize_to_rgb_png(file_path, target_path)

        if "__" in target_path.stem:
            class_key = target_path.stem.split("__", 1)[0].lower()
        else:
            class_key = "unknown"
        entries.append((target_path, class_key))

    entries.sort(key=lambda item: item[0].name)
    return entries


def sync_labels_csv(
    labels_csv: Path,
    kaggle_entries: list[tuple[Path, str]],
    dataset_root: Path,
) -> int:
    """Синхронізує рядки ``kaggle_topdown`` у ``dataset/labels.csv`` без зміни інших сплітів.

    Зберігає всі наявні рядки для ``real_u1``, ``real_u2`` та інших сплітів без змін,
    оновлюючи лише записи для ``split == "kaggle_topdown"``.

    Args:
        labels_csv: Шлях до файлу ``dataset/labels.csv``.
        kaggle_entries: Список пар ``(шлях_до_зображення, class_key)``.
        dataset_root: Коренева папка ``dataset/`` для формування відносних шляхів.

    Returns:
        Кількість записаних рядків для ``kaggle_topdown``.
    """
    preserved_rows: list[dict[str, str]] = []
    if labels_csv.exists():
        with labels_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                if row.get("split") != "kaggle_topdown":
                    preserved_rows.append(
                        {field: row.get(field, "") for field in LABELS_FIELDNAMES}
                    )

    kaggle_rows: list[dict[str, str]] = []
    for image_path, class_key in kaggle_entries:
        try:
            rel_path = image_path.resolve().relative_to(dataset_root.resolve()).as_posix()
        except ValueError:
            rel_path = f"kaggle_topdown/{image_path.name}"
        label = map_class_to_label(class_key)
        kaggle_rows.append(
            {
                "filename": rel_path,
                "split": "kaggle_topdown",
                "scenario": "kaggle",
                "has_mask": "true",
                "total_count": "1",
                "items_json": json.dumps([label], ensure_ascii=False),
            }
        )

    labels_csv.parent.mkdir(parents=True, exist_ok=True)
    with labels_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LABELS_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(preserved_rows)
        writer.writerows(kaggle_rows)

    return len(kaggle_rows)


def _validate_mask_file(
    mask_path: Path,
    expected_shape: tuple[int, int],
    issues: dict[str, list[str]],
) -> None:
    """Перевіряє режим, розмір, бінарність {0, 255} та заповнення збереженої маски."""
    with Image.open(mask_path) as saved_mask:
        saved_arr = np.asarray(saved_mask, dtype=np.uint8)
        if saved_mask.mode != "L" or saved_arr.shape != expected_shape:
            issues["size_mismatches"].append(mask_path.name)

    unique_vals = set(np.unique(saved_arr).tolist())
    if not unique_vals.issubset({0, 255}):
        issues["invalid_values"].append(mask_path.name)

    fg_pixels = int(np.count_nonzero(saved_arr == 255))
    if fg_pixels == 0:
        issues["empty_masks"].append(mask_path.name)
    elif fg_pixels == saved_arr.size:
        issues["full_masks"].append(mask_path.name)


def generate_kaggle_masks(
    kaggle_dir: Path,
    masks_dir: Path,
    bg_tolerance: int = 12,
    min_component_area: int = 32,
    labels_csv: Path | None = None,
) -> dict[str, object]:
    """Формує 1-канальні бінарні маски (0/255) для чистих top-down рендерів Kaggle.

    Приводить ``kaggle_dir`` до пласкої структури 8-бітних RGB PNG із префіксом класу,
    генерує маски ``<stem>_mask.png`` у ``masks_dir`` та за наявності ``labels_csv``
    синхронізує таблицю анотацій.

    Args:
        kaggle_dir: Каталог ``dataset/kaggle_topdown/``.
        masks_dir: Цільовий каталог ``dataset/kaggle_masks/``.
        bg_tolerance: Допуск відхилення інтенсивності від однорідного фону рендера.
        min_component_area: Мінімальна площа зв'язної компоненти (фільтрація шуму).
        labels_csv: Необов'язковий шлях до ``dataset/labels.csv`` для синхронізації.

    Returns:
        Словник зі статистикою генерації та валідації масок.
    """
    if not kaggle_dir.is_dir():
        raise FileNotFoundError(f"Каталог Kaggle не знайдено: {kaggle_dir}")

    masks_dir.mkdir(parents=True, exist_ok=True)
    entries = flatten_kaggle_directory(kaggle_dir)

    class_counts: Counter[str] = Counter()
    issues: dict[str, list[str]] = {
        "empty_masks": [],
        "full_masks": [],
        "invalid_values": [],
        "size_mismatches": [],
    }

    for image_path, class_key in entries:
        with Image.open(image_path) as opened:
            image_rgb = np.asarray(ensure_rgb_image(opened), dtype=np.uint8)

        mask = build_binary_mask(
            image_rgb,
            bg_tolerance=bg_tolerance,
            min_component_area=min_component_area,
        )
        mask_path = masks_dir / f"{image_path.stem}_mask.png"
        Image.fromarray(mask, mode="L").save(mask_path, format="PNG")
        _validate_mask_file(mask_path, image_rgb.shape[:2], issues)
        class_counts[class_key] += 1

    labels_written = 0
    if labels_csv is not None:
        labels_written = sync_labels_csv(
            labels_csv=labels_csv,
            kaggle_entries=entries,
            dataset_root=kaggle_dir.parent,
        )

    return {
        "total_masks": len(entries),
        "class_counts": dict(sorted(class_counts.items())),
        "labels_written": labels_written,
        **issues,
    }


def _print_report(report: dict[str, object]) -> None:
    """Виводить зведений звіт генерації та валідації масок у консоль."""
    print(f"Згенеровано масок Kaggle: {report['total_masks']}")
    class_counts = report["class_counts"]
    if isinstance(class_counts, dict):
        print("Розподіл по класах:")
        for class_name, count in class_counts.items():
            print(f"  - {class_name:<18s}: {count:3d} ({map_class_to_label(class_name)})")
    print(f"Синхронізовано рядків у labels.csv: {report['labels_written']}")
    print(f"Порожні маски (0%): {len(report['empty_masks'])}")
    print(f"Повністю заповнені маски (100%): {len(report['full_masks'])}")
    print(f"Маски з пікселями поза {{0, 255}}: {len(report['invalid_values'])}")
    print(f"Невідповідність розміру/режиму: {len(report['size_mismatches'])}")


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу CLI для генерації еталонних масок Kaggle."""
    parser = argparse.ArgumentParser(description="Генератор еталонних масок Kaggle.")
    parser.add_argument("--input", type=Path, default=Path("dataset/kaggle_topdown"))
    parser.add_argument("--output", type=Path, default=Path("dataset/kaggle_masks"))
    parser.add_argument("--labels-csv", type=Path, default=Path("dataset/labels.csv"))
    parser.add_argument("--no-labels", action="store_true", help="Не оновлювати labels.csv.")
    parser.add_argument("--tolerance", type=int, default=12)
    parser.add_argument("--min-area", type=int, default=32)
    args = parser.parse_args(argv)

    labels_target = None if args.no_labels else args.labels_csv
    report = generate_kaggle_masks(
        kaggle_dir=args.input,
        masks_dir=args.output,
        bg_tolerance=args.tolerance,
        min_component_area=args.min_area,
        labels_csv=labels_target,
    )
    _print_report(report)
    has_errors = any(
        bool(report[key])
        for key in ("empty_masks", "full_masks", "invalid_values", "size_mismatches")
    )
    return 1 if has_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
