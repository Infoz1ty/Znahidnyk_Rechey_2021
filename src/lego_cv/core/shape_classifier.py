"""Модуль А: Спільний геометричний класифікатор форми LEGO (фінал Кроку 1).

Зона реалізації (Етап 1): Учасник 1 (member1).
Усі алгоритми реалізуються вручну на NumPy без ``cv2.connectedComponents`` та ``cv2.minAreaRect``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from lego_cv.core.types import BrickDetection

# 8-сусідні зміщення за годинниковою стрілкою, починаючи з заходу (dy, dx)
_MOORE_OFFSETS: tuple[tuple[int, int], ...] = (
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
    (0, 1), (1, 1), (1, 0), (1, -1),
)
_MOORE_STEP_LEN: tuple[float, ...] = tuple(
    1.0 if idx % 2 == 0 else math.sqrt(2.0) for idx in range(8)
)
_OFFSET_TO_IDX: dict[tuple[int, int], int] = {
    offset: idx for idx, offset in enumerate(_MOORE_OFFSETS)
}


def _dsu_find(parent: list[int], node: int) -> int:
    """Знаходить корінь множини в Union-Find із компресією шляхів."""
    root = node
    while parent[root] != root:
        root = parent[root]
    while parent[node] != root:
        nxt = parent[node]
        parent[node] = root
        node = nxt
    return root


def _dsu_union(parent: list[int], first: int, second: int) -> None:
    """Об'єднує дві множини в Union-Find, прив'язуючи більший корінь до меншого."""
    root_a = _dsu_find(parent, first)
    root_b = _dsu_find(parent, second)
    if root_a == root_b:
        return
    if root_a < root_b:
        parent[root_b] = root_a
    else:
        parent[root_a] = root_b


def _link_row_runs(
    transitions: np.ndarray,
    prev_runs: list[tuple[int, int, int]],
    parent: list[int],
) -> list[tuple[int, int, int]]:
    """Зв'язує горизонтальні прогони поточного рядка з прогонами попереднього (8-зв'язність)."""
    cur_runs: list[tuple[int, int, int]] = []
    prev_ptr = 0
    prev_count = len(prev_runs)

    for pos in range(0, transitions.size, 2):
        start = int(transitions[pos])
        end = int(transitions[pos + 1])

        while prev_ptr < prev_count and prev_runs[prev_ptr][1] < start:
            prev_ptr += 1

        assigned = 0
        scan_ptr = prev_ptr
        # Умова 8-зв'язності інтервалів [s1, e1) та [s2, e2): s1 <= e2 і s2 <= e1
        while scan_ptr < prev_count and prev_runs[scan_ptr][0] <= end:
            neighbor_lbl = prev_runs[scan_ptr][2]
            if assigned == 0:
                assigned = _dsu_find(parent, neighbor_lbl)
            else:
                _dsu_union(parent, assigned, neighbor_lbl)
            scan_ptr += 1

        if assigned == 0:
            assigned = len(parent)
            parent.append(assigned)
        cur_runs.append((start, end, assigned))

    return cur_runs


def connected_components_label(binary_mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Ручне маркування 8-зв'язних компонентів (2-pass Union-Find на прогонах рядків).

    Оптимізація: замість попіксельного циклу в першому проході виділяються
    горизонтальні прогони переднього плану ``[start, end)``, які зшиваються між
    сусідніми рядками за умовою 8-зв'язності через Union-Find.

    Args:
        binary_mask: Бінарна маска форми ``(H, W)`` (``0`` — фон, ``> 0`` — об'єкт).

    Returns:
        Кортеж ``(labels, num_labels)``, де ``labels`` має форму ``(H, W)`` та тип ``int32``
        (``0`` — фон, ``1..num_labels`` — ID компонентів).

    Raises:
        ValueError: Якщо вхідна маска не є 2D-масивом.
    """
    if binary_mask.ndim != 2:
        raise ValueError(f"Очікувалась 2D-маска, отримано форму {binary_mask.shape}")

    height, width = binary_mask.shape
    labels = np.zeros((height, width), dtype=np.int32)
    fg = binary_mask > 0
    if height == 0 or width == 0 or not np.any(fg):
        return labels, 0

    parent: list[int] = [0]
    row_runs: list[list[tuple[int, int, int]]] = []
    padded_row = np.zeros(width + 2, dtype=np.bool_)
    prev_runs: list[tuple[int, int, int]] = []

    # Прохід 1: знаходження горизонтальних прогонів та об'єднання 8-сусідів
    for row_idx in range(height):
        padded_row[1:-1] = fg[row_idx]
        transitions = np.flatnonzero(padded_row[1:] != padded_row[:-1])
        cur_runs = _link_row_runs(transitions, prev_runs, parent)
        row_runs.append(cur_runs)
        prev_runs = cur_runs

    # Прохід 2: послідовна перенумерація коренів у 1..K і заповнення матриці
    root_to_final: dict[int, int] = {}
    next_id = 0
    for row_idx, runs in enumerate(row_runs):
        for start, end, raw_lbl in runs:
            root = _dsu_find(parent, raw_lbl)
            final_id = root_to_final.get(root)
            if final_id is None:
                next_id += 1
                final_id = next_id
                root_to_final[root] = final_id
            labels[row_idx, start:end] = final_id

    return labels, next_id


def _find_next_moore_step(
    submask: np.ndarray,
    curr: tuple[int, int],
    backtrack_idx: int,
) -> tuple[tuple[int, int], int, int] | None:
    """Шукає наступний граничний піксель за годинниковою стрілкою навколо ``curr``."""
    height, width = submask.shape
    prev_bg_idx = backtrack_idx
    for step in range(1, 9):
        cand_idx = (backtrack_idx + step) % 8
        dy, dx = _MOORE_OFFSETS[cand_idx]
        ny, nx = curr[0] + dy, curr[1] + dx
        if 0 <= ny < height and 0 <= nx < width and submask[ny, nx]:
            pdy, pdx = _MOORE_OFFSETS[prev_bg_idx]
            next_backtrack = _OFFSET_TO_IDX[(curr[0] + pdy - ny, curr[1] + pdx - nx)]
            return (ny, nx), cand_idx, next_backtrack
        prev_bg_idx = cand_idx
    return None


def _trace_moore_perimeter(submask: np.ndarray) -> float:
    """Обчислює зовнішній периметр компонента ручним обходом Moore-Neighbor.

    Вага ``1.0`` для ортогональних і ``sqrt(2)`` для діагональних кроків між
    центрами граничних пікселів плюс точна зовнішня поправка ``+4.0`` для
    одиничних пікселів (дає ``2*(a+b)`` для прямокутника ``a x b`` і ``pi/4`` для квадрата).
    """
    ys, xs = np.nonzero(submask)
    if ys.size == 0:
        return 0.0
    if ys.size == 1:
        return 4.0

    # Найвищий і найлівіший піксель: сусід на заході (idx=0) гарантовано фоновий
    start = (int(ys[0]), int(xs[0]))
    curr = start
    backtrack_idx = 0
    perimeter = 0.0
    first_move_idx: int | None = None

    for _ in range(4 * int(ys.size) + 16):
        step_res = _find_next_moore_step(submask, curr, backtrack_idx)
        if step_res is None:
            break
        nxt, move_idx, next_backtrack = step_res
        if curr == start and first_move_idx is not None and move_idx == first_move_idx:
            return perimeter + 4.0
        if first_move_idx is None:
            first_move_idx = move_idx
        perimeter += _MOORE_STEP_LEN[move_idx]
        curr = nxt
        backtrack_idx = next_backtrack

    return perimeter + 4.0


def _compute_pca_obb(
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[tuple[float, float], float, float, float, float]:
    """Обчислює центр мас, кут головної осі PCA та розміри OBB ``(w, h, AR)``."""
    xs_f = xs.astype(np.float64)
    ys_f = ys.astype(np.float64)
    cx = float(np.mean(xs_f))
    cy = float(np.mean(ys_f))
    dx = xs_f - cx
    dy = ys_f - cy

    cov_xx = float(np.mean(dx * dx))
    cov_yy = float(np.mean(dy * dy))
    cov_xy = float(np.mean(dx * dy))
    angle_rad = 0.5 * math.atan2(2.0 * cov_xy, cov_xx - cov_yy)

    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    proj_u = dx * cos_a + dy * sin_a
    proj_v = -dx * sin_a + dy * cos_a

    obb_width = float(np.max(proj_u) - np.min(proj_u)) + 1.0
    obb_height = float(np.max(proj_v) - np.min(proj_v)) + 1.0
    aspect_ratio = max(obb_width, obb_height) / max(min(obb_width, obb_height), 1e-9)
    return (cx, cy), obb_width, obb_height, aspect_ratio, angle_rad


def compute_component_geometry(component_mask: np.ndarray) -> dict[str, Any]:
    """Обчислює геометричні ознаки одного зв'язного компонента вручну.

    Ознаки:
    * площа ``area_px``;
    * периметр ``perimeter_px`` (ручний Moore-Neighbor обхід зовнішнього контуру);
    * коефіцієнт окружності ``circularity = 4 * pi * S / P^2``;
    * орієнтований прямокутник через PCA головних осей -> ``aspect_ratio``;
    * обмежувальний прямокутник ``bbox`` та центр мас ``center``.

    Args:
        component_mask: Булева або бінарна маска ``(H, W)`` однієї деталі.

    Returns:
        Словник геометричних ознак.

    Raises:
        ValueError: Якщо ``component_mask`` не є 2D-масивом.
    """
    if component_mask.ndim != 2:
        raise ValueError(f"Очікувалась 2D-маска, отримано форму {component_mask.shape}")

    ys, xs = np.nonzero(component_mask)
    area_px = float(ys.size)
    if area_px == 0.0:
        return {
            "area_px": 0.0,
            "perimeter_px": 0.0,
            "circularity": 0.0,
            "aspect_ratio": 1.0,
            "bbox": (0, 0, 0, 0),
            "center": (0.0, 0.0),
            "obb_width": 0.0,
            "obb_height": 0.0,
            "angle_rad": 0.0,
        }

    bbox = (int(np.min(xs)), int(np.min(ys)), int(np.max(xs)), int(np.max(ys)))
    submask = component_mask[bbox[1] : bbox[3] + 1, bbox[0] : bbox[2] + 1] > 0

    perimeter_px = _trace_moore_perimeter(submask)
    circularity = (
        (4.0 * math.pi * area_px) / (perimeter_px * perimeter_px)
        if perimeter_px > 0.0
        else 0.0
    )
    center, obb_width, obb_height, aspect_ratio, angle_rad = _compute_pca_obb(xs, ys)

    return {
        "area_px": area_px,
        "perimeter_px": perimeter_px,
        "circularity": circularity,
        "aspect_ratio": aspect_ratio,
        "bbox": bbox,
        "center": center,
        "obb_width": obb_width,
        "obb_height": obb_height,
        "angle_rad": angle_rad,
    }


def _score_shape_entry(
    entry: dict[str, Any],
    studs_area: float,
    aspect_ratio: float,
    ar_tol: float,
    area_tol: float,
) -> tuple[float, bool]:
    """Оцінює відносну похибку кандидата щодо еталонного класу та перевіряє допуски."""
    target_ar = float(entry["aspect_ratio"])
    target_area = float(entry["studs_area"])
    ar_err = abs(aspect_ratio - target_ar) / max(target_ar, 1e-9)
    area_err = abs(studs_area - target_area) / max(target_area, 1e-9)
    in_tol = (
        abs(aspect_ratio - target_ar) <= ar_tol * max(target_ar, 1.0)
        and area_err <= area_tol
    )
    return ar_err + area_err, in_tol


def _match_shape_class(
    studs_area: float,
    aspect_ratio: float,
    circularity: float,
    shape_cfg: dict[str, Any],
) -> tuple[str, bool]:
    """Зіставляє геометричні ознаки деталі з таблицею класів із конфігурації."""
    classes = shape_cfg.get("classes", [])
    if not classes:
        return "unknown", False

    round_circ_min = float(shape_cfg.get("round_circularity_min", 0.82))
    ar_tol = float(shape_cfg.get("aspect_ratio_tolerance", 0.25))
    area_tol = float(shape_cfg.get("stud_area_tolerance", 0.35))

    is_round_cand = (circularity >= round_circ_min) and (aspect_ratio <= 1.0 + ar_tol)
    pool = [item for item in classes if bool(item.get("is_round", False)) == is_round_cand]
    if not pool:
        pool = classes

    scored = [
        (_score_shape_entry(item, studs_area, aspect_ratio, ar_tol, area_tol), item)
        for item in pool
    ]
    in_tol_items = [(score, item) for (score, ok), item in scored if ok]
    if in_tol_items:
        _, chosen = min(in_tol_items, key=lambda pair: pair[0])
        return str(chosen["name"]), bool(chosen.get("is_round", False))

    (best_score, _), fallback = min(scored, key=lambda pair: pair[0][0])
    if best_score > 1.2:
        return "unknown", False
    return str(fallback["name"]), bool(fallback.get("is_round", False))


def _determine_surface_type(
    comp_id: int,
    is_round: bool,
    pipeline_metadata: dict[str, Any] | None,
) -> tuple[str, int]:
    """Визначає тип поверхні (``round`` / ``brick`` / ``tile`` / ``unknown``) та шпильки."""
    stud_count = 0
    if pipeline_metadata is not None:
        stud_counts = pipeline_metadata.get("stud_counts")
        if isinstance(stud_counts, dict) and comp_id in stud_counts:
            stud_count = int(stud_counts[comp_id])
            return ("round" if is_round else ("brick" if stud_count > 0 else "tile")), stud_count

        textures = pipeline_metadata.get("texture_features")
        if isinstance(textures, dict) and comp_id in textures:
            feat = textures[comp_id]
            if is_round:
                return "round", stud_count
            if isinstance(feat, dict):
                homogeneity = float(feat.get("homogeneity", 0.0))
                return ("tile" if homogeneity >= 0.85 else "brick"), stud_count

    return ("round" if is_round else "unknown"), stud_count


def _prepare_instance_labels(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Повертає інстанс-маску та список ID компонентів (бінарну маркує ручним CCL)."""
    unique_pos = np.unique(mask)
    unique_pos = unique_pos[unique_pos > 0]
    if unique_pos.size == 0:
        return np.zeros(mask.shape, dtype=np.int32), []

    if unique_pos.size == 1 and int(unique_pos[0]) in (1, 255):
        labels, num_labels = connected_components_label(mask > 0)
        return labels, list(range(1, num_labels + 1))

    return mask.astype(np.int32, copy=False), [int(val) for val in unique_pos]


def classify_shapes(
    mask: np.ndarray,
    config: dict[str, Any],
    pipeline_metadata: dict[str, Any] | None = None,
) -> list[BrickDetection]:
    """Класифікує геометричну форму й тип поверхні всіх деталей на масці.

    Відкидає сміття з площею ``S < min_area_ratio * stud_size_px^2`` та заповнює
    поля форми ``BrickDetection`` (поле ``color`` заповнюється на Кроці 2).

    Args:
        mask: Бінарна (0/1 або 0/255) або інстанс-маска (0, 1..K) типу ``int32``.
        config: Повний словник конфігурації (секції ``preprocessing`` та ``shape_classifier``).
        pipeline_metadata: Необов'язкові метадані пайплайну (напр., ``stud_counts`` із П4).

    Returns:
        Список попередньо заповнених ``BrickDetection``.

    Raises:
        ValueError: Якщо ``mask`` не є 2D-масивом.
    """
    if mask.ndim != 2:
        raise ValueError(f"Очікувалась 2D-маска, отримано форму {mask.shape}")

    labels, component_ids = _prepare_instance_labels(mask)
    if not component_ids:
        return []

    stud_size_px = float(config.get("preprocessing", {}).get("stud_size_px", 30.0))
    unit_area = stud_size_px * stud_size_px
    shape_cfg = config.get("shape_classifier", {})
    min_area_px = float(shape_cfg.get("min_area_ratio", 0.5)) * unit_area

    detections: list[BrickDetection] = []
    for comp_id in component_ids:
        comp_mask = labels == comp_id
        if float(np.count_nonzero(comp_mask)) < min_area_px:
            continue

        geom = compute_component_geometry(comp_mask)
        shape_class, is_round = _match_shape_class(
            studs_area=geom["area_px"] / max(unit_area, 1e-9),
            aspect_ratio=geom["aspect_ratio"],
            circularity=geom["circularity"],
            shape_cfg=shape_cfg,
        )
        surface_type, stud_count = _determine_surface_type(
            comp_id=comp_id,
            is_round=is_round,
            pipeline_metadata=pipeline_metadata,
        )
        detections.append(
            BrickDetection(
                brick_id=comp_id,
                shape_class=shape_class,
                surface_type=surface_type,
                color="Unknown",
                stud_count=stud_count,
                area_px=geom["area_px"],
                aspect_ratio=geom["aspect_ratio"],
                circularity=geom["circularity"],
                bbox=geom["bbox"],
                center=geom["center"],
            )
        )

    return detections
