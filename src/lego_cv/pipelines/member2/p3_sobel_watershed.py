"""Пайплайн 3 (Учасник 2): Градієнтний конвеєр із розділенням злиплих деталей.

Категорії стадій (3 унікальні категорії за методичкою):
1. Ретушування та покращення — ручний білатеральний фільтр + Unsharp Masking.
2. Контурний аналіз — ручний градієнтний оператор Собеля + адаптивна груба маска.
3. Сегментація зображень — ручна карта відстаней (Chamfer 3-4) + маркерний Watershed.
"""

from __future__ import annotations

from collections import deque
import heapq
import math
from typing import Any

import numpy as np

from lego_cv.core.base_pipeline import BasePipeline


def _to_grayscale_float(image: np.ndarray) -> np.ndarray:
    """Перетворює 2D або 3D RGB-зображення у 2D яскравість ``float64``."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    raise ValueError(f"Очікується 2D або 3D RGB-зображення, отримано форму {arr.shape}.")


def _accumulate_bilateral_offset(
    arr: np.ndarray,
    padded: np.ndarray,
    offset: tuple[int, int],
    radius: int,
    inv_sigmas: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Обчислює внесок одного зміщення ``(dy, dx)`` для векторизованого білатерального фільтра."""
    dy, dx = offset
    inv_two_ss2, inv_two_sr2 = inv_sigmas
    height, width = arr.shape[:2]
    is_color = arr.ndim == 3

    shifted = padded[radius + dy : radius + dy + height, radius + dx : radius + dx + width]
    diff = shifted - arr
    range_sq = np.mean(diff * diff, axis=2) if is_color else diff * diff
    spatial_w = math.exp(-float(dy * dy + dx * dx) * inv_two_ss2)
    weight = np.float32(spatial_w) * np.exp(-range_sq * np.float32(inv_two_sr2))
    contrib = shifted * weight[..., np.newaxis] if is_color else shifted * weight
    return contrib, weight


def bilateral_filter_manual(
    image: np.ndarray,
    radius: int = 3,
    sigma_spatial: float = 3.0,
    sigma_range: float = 25.0,
) -> np.ndarray:
    """Ручний білатеральний фільтр із просторовою та радіометричною гаусовими вагами.

    Векторизований за зміщеннями вікна на ``NumPy``: згладжує однорідні ділянки та
    текстуру фону, зберігаючи різкі перепади інтенсивності на бортах деталей LEGO.

    Args:
        image: Вхідне зображення ``(H, W)`` або ``(H, W, 3)`` типу ``uint8``.
        radius: Радіус квадратного вікна фільтра (``>= 0``).
        sigma_spatial: Просторове стандартне відхилення ``sigma_s > 0``.
        sigma_range: Радіометричне стандартне відхилення ``sigma_r > 0``.

    Returns:
        Згладжене зображення ``uint8`` тієї самої форми.

    Raises:
        ValueError: Якщо параметри ``radius < 0``, ``sigma_spatial <= 0`` або ``sigma_range <= 0``.
    """
    if radius < 0:
        raise ValueError(f"radius має бути невід'ємним, отримано {radius}.")
    if sigma_spatial <= 0.0 or sigma_range <= 0.0:
        raise ValueError("sigma_spatial та sigma_range мають бути додатними.")

    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0 or radius == 0:
        return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)

    is_color = arr.ndim == 3
    pad_spec = ((radius, radius), (radius, radius), (0, 0)) if is_color else (
        (radius, radius),
        (radius, radius),
    )
    padded = np.pad(arr, pad_width=pad_spec, mode="edge")
    inv_sigmas = (1.0 / (2.0 * float(sigma_spatial) ** 2), 1.0 / (2.0 * float(sigma_range) ** 2))

    weighted_sum = np.zeros_like(arr, dtype=np.float32)
    weight_total = np.zeros(arr.shape[:2], dtype=np.float32)

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            contrib, weight = _accumulate_bilateral_offset(
                arr, padded, (dy, dx), radius, inv_sigmas
            )
            weighted_sum += contrib
            weight_total += weight

    safe_den = np.where(weight_total > 0.0, weight_total, 1.0)
    filtered = weighted_sum / safe_den[..., np.newaxis] if is_color else weighted_sum / safe_den
    return np.clip(np.rint(filtered), 0.0, 255.0).astype(np.uint8)


def _gaussian_blur_separable_manual(image: np.ndarray, sigma: float) -> np.ndarray:
    """Ручне роздільне 2D-гаусове згладжування (1D по горизонталі, потім по вертикалі)."""
    arr = np.asarray(image, dtype=np.float32)
    if sigma <= 0.0 or arr.size == 0:
        return arr.copy()

    radius = max(1, int(math.ceil(3.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(offsets * offsets) / np.float32(2.0 * sigma * sigma))
    kernel /= np.sum(kernel)

    is_color = arr.ndim == 3
    height, width = arr.shape[:2]

    # Горизонтальний прохід
    pad_x = ((0, 0), (radius, radius), (0, 0)) if is_color else ((0, 0), (radius, radius))
    padded_x = np.pad(arr, pad_width=pad_x, mode="edge")
    horiz = np.zeros_like(arr, dtype=np.float32)
    for idx, coeff in enumerate(kernel):
        horiz += coeff * padded_x[:, idx : idx + width]

    # Вертикальний прохід
    pad_y = ((radius, radius), (0, 0), (0, 0)) if is_color else ((radius, radius), (0, 0))
    padded_y = np.pad(horiz, pad_width=pad_y, mode="edge")
    blurred = np.zeros_like(arr, dtype=np.float32)
    for idx, coeff in enumerate(kernel):
        blurred += coeff * padded_y[idx : idx + height, :]
    return blurred


def unsharp_mask_manual(
    image: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.2,
) -> np.ndarray:
    """Ручне нерізке маскування ``I_sharp = clip(I + amount * (I - G_sigma * I), 0, 255)``.

    Підсилює локальний контраст на межах і тонких щілинах між злиплими деталями,
    залишаючи однорідні ділянки незмінними.

    Args:
        image: Вхідне зображення ``(H, W)`` або ``(H, W, 3)`` типу ``uint8``.
        sigma: Стандартне відхилення гаусового ядра розмиття (``>= 0``).
        amount: Коефіцієнт підсилення різкості (``>= 0``).

    Returns:
        Підсилене зображення ``uint8`` тієї самої форми.

    Raises:
        ValueError: Якщо ``sigma < 0`` або ``amount < 0``.
    """
    if sigma < 0.0 or amount < 0.0:
        raise ValueError("Параметри sigma та amount мають бути невід'ємними.")

    arr = np.asarray(image, dtype=np.float32)
    if arr.size == 0 or sigma == 0.0 or amount == 0.0:
        return np.clip(np.rint(arr), 0.0, 255.0).astype(np.uint8)

    blurred = _gaussian_blur_separable_manual(arr, sigma=sigma)
    sharpened = arr + np.float32(amount) * (arr - blurred)
    return np.clip(np.rint(sharpened), 0.0, 255.0).astype(np.uint8)


def bilateral_unsharp_manual(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Стадія 1: ручний білатеральний фільтр + нерізке маскування (Unsharp Mask).

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` типу ``uint8``.
        stage_cfg: Секція ``stage1_bilateral_unsharp`` із ``configs/pipeline_p3.json``.

    Returns:
        Покращене RGB-зображення ``(H, W, 3)`` типу ``uint8``.
    """
    radius = int(stage_cfg.get("bilateral_radius", 3))
    sigma_s = float(stage_cfg.get("sigma_spatial", 3.0))
    sigma_r = float(stage_cfg.get("sigma_range", 25.0))
    unsharp_sigma = float(stage_cfg.get("unsharp_sigma", 1.0))
    unsharp_amount = float(stage_cfg.get("unsharp_amount", 0.5))

    denoised = bilateral_filter_manual(
        image_rgb,
        radius=radius,
        sigma_spatial=sigma_s,
        sigma_range=sigma_r,
    )
    return unsharp_mask_manual(
        denoised,
        sigma=unsharp_sigma,
        amount=unsharp_amount,
    )


def sobel_gradient_manual(
    image: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ручний градієнтний оператор Собеля 3x3 (``G_x``, ``G_y``, модуль градієнта).

    Ядра:
        ``S_x = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]``,
        ``S_y = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]``.

    Args:
        image: Вхідне 2D grayscale або 3D RGB зображення.

    Returns:
        Кортеж ``(gx, gy, magnitude)`` масивів ``float64`` форми ``(H, W)``.
    """
    gray = _to_grayscale_float(image)
    if gray.size == 0:
        empty = np.zeros_like(gray, dtype=np.float64)
        return empty, empty.copy(), empty.copy()

    padded = np.pad(gray, pad_width=1, mode="edge")
    top_l, top_c, top_r = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    mid_l, mid_r = padded[1:-1, :-2], padded[1:-1, 2:]
    bot_l, bot_c, bot_r = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]

    gx = -top_l + top_r - 2.0 * mid_l + 2.0 * mid_r - bot_l + bot_r
    gy = -top_l - 2.0 * top_c - top_r + bot_l + 2.0 * bot_c + bot_r
    magnitude = np.hypot(gx, gy)
    return gx, gy, magnitude


def _otsu_threshold_from_values(values_uint8: np.ndarray) -> float:
    """Ручний поріг Оцу на 1D/2D масиві ``uint8`` із захистом від однорідного кадру."""
    hist = np.bincount(values_uint8.ravel(), minlength=256).astype(np.float64)
    total = float(values_uint8.size)
    if total == 0.0 or int(np.count_nonzero(hist)) <= 1:
        return 255.0

    probs = hist / total
    bins = np.arange(256, dtype=np.float64)
    omega0 = np.cumsum(probs)
    omega1 = 1.0 - omega0
    mu_cum = np.cumsum(probs * bins)
    mu_total = mu_cum[-1]

    valid = (omega0 > 1e-12) & (omega1 > 1e-12)
    if not np.any(valid):
        return 255.0

    mu0 = np.where(valid, mu_cum / np.where(valid, omega0, 1.0), 0.0)
    mu1 = np.where(valid, (mu_total - mu_cum) / np.where(valid, omega1, 1.0), 0.0)
    between_var = np.where(valid, omega0 * omega1 * (mu0 - mu1) ** 2, -1.0)
    return float(np.argmax(between_var))


def _binary_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    """Ручна 3x3 бінарна дилатація для змикання мікророзривів маски."""
    height, width = mask.shape
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    out = np.zeros((height, width), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            out |= padded[dy : dy + height, dx : dx + width]
    return out


def _binary_erode_3x3(mask: np.ndarray) -> np.ndarray:
    """Ручна 3x3 бінарна ерозія для морфологічного закриття."""
    height, width = mask.shape
    padded = np.pad(mask, pad_width=1, mode="constant", constant_values=False)
    out = np.ones((height, width), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + height, dx : dx + width]
    return out


def _fill_holes_bfs(mask: np.ndarray) -> np.ndarray:
    """Ручний 4-зв'язний BFS flood-fill від рамки для заповнення внутрішніх дірок."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.copy()

    y_slice = slice(int(ys.min()), int(ys.max()) + 1)
    x_slice = slice(int(xs.min()), int(xs.max()) + 1)
    bg_grid = np.pad(~mask[y_slice, x_slice], pad_width=1, mode="constant", constant_values=True)
    grid_h, grid_w = bg_grid.shape

    reachable = np.zeros((grid_h, grid_w), dtype=np.bool_)
    reachable[0, 0] = True
    frontier: deque[tuple[int, int]] = deque([(0, 0)])

    while frontier:
        row, col = frontier.popleft()
        for n_row, n_col in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
            if 0 <= n_row < grid_h and 0 <= n_col < grid_w:
                if bg_grid[n_row, n_col] and not reachable[n_row, n_col]:
                    reachable[n_row, n_col] = True
                    frontier.append((n_row, n_col))

    filled = mask.copy()
    filled[y_slice, x_slice] = ~reachable[1:-1, 1:-1]
    return filled


def _compute_adaptive_bg_distance(image: np.ndarray) -> tuple[np.ndarray, float, bool]:
    """Обчислює адаптивну карту відстані до фону за рамкою кадру.

    Для кольорових фонів (напр., зелений килимок із віньєтуванням у ``real_u2``)
    використовує інваріантну до освітлення нуль-середню хроматичну відстань,
    а для ахроматичних/темних фонів (Kaggle або сірий папір) — повну RGB/яскравісну.

    Returns:
        Кортеж ``(distance_map, border_upper_quantile, is_colored_bg)``.
    """
    arr = np.asarray(image, dtype=np.float32)
    height, width = arr.shape[:2]
    bw = max(1, min(6, height // 10 or 1, width // 10 or 1))

    if arr.ndim == 2:
        border = np.concatenate(
            (arr[:bw, :].ravel(), arr[-bw:, :].ravel(), arr[:, :bw].ravel(), arr[:, -bw:].ravel())
        )
        bg_med = float(np.median(border))
        border_dists = np.abs(border - bg_med)
        return np.abs(arr - bg_med), float(np.percentile(border_dists, 98)), False

    border_rgb = np.concatenate(
        (
            arr[:bw, :, :].reshape(-1, 3),
            arr[-bw:, :, :].reshape(-1, 3),
            arr[:, :bw, :].reshape(-1, 3),
            arr[:, -bw:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    bg_rgb = np.median(border_rgb, axis=0)
    bg_chroma = bg_rgb - float(np.mean(bg_rgb))
    bg_chroma_norm = float(np.linalg.norm(bg_chroma))

    # Нуль-середня хроматичність компенсує нерівномірну яскравість лампи на кольоровому фоні
    pix_mean = np.mean(arr, axis=2, keepdims=True)
    pix_chroma = arr - pix_mean
    chroma_dist = np.linalg.norm(pix_chroma - bg_chroma, axis=2)

    border_chroma = border_rgb - np.mean(border_rgb, axis=1, keepdims=True)
    border_chroma_dists = np.linalg.norm(border_chroma - bg_chroma, axis=1)

    if bg_chroma_norm > 15.0:
        return chroma_dist, float(np.percentile(border_chroma_dists, 98)), True

    rgb_dist = np.max(np.abs(arr - bg_rgb), axis=2)
    rgb_border_dists = np.max(np.abs(border_rgb - bg_rgb), axis=1)
    return rgb_dist, float(np.percentile(rgb_border_dists, 98)), False


def segment_sobel_foreground(
    image: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Стадія 2: оператор Собеля та адаптивна груба маска переднього плану.

    Args:
        image: Покращене зображення зі Стадії 1.
        stage_cfg: Секція ``stage2_sobel_gradient`` із ``configs/pipeline_p3.json``.

    Returns:
        Кортеж ``(gradient_magnitude_uint8, rough_fg_mask_uint8)``.
    """
    _, _, mag = sobel_gradient_manual(image)
    max_mag = float(np.max(mag)) if mag.size > 0 else 0.0
    normalize_out = bool(stage_cfg.get("normalize_output", True))
    if normalize_out and max_mag > 0.0:
        grad_uint8 = np.clip(np.rint((mag / max_mag) * 255.0), 0.0, 255.0).astype(np.uint8)
    else:
        grad_uint8 = np.clip(np.rint(mag), 0.0, 255.0).astype(np.uint8)

    if mag.size == 0 or max_mag == 0.0:
        return grad_uint8, np.zeros(mag.shape, dtype=np.uint8)

    dist_map, border_p98, is_colored_bg = _compute_adaptive_bg_distance(image)
    method = str(stage_cfg.get("fg_threshold_method", "adaptive")).strip().lower()
    offset = float(stage_cfg.get("fg_threshold_offset", 0.0))
    min_bg_delta = float(stage_cfg.get("min_bg_delta", 14.0))

    # Адаптивна верхня межа шуму фону (стійка до багатомодальних кольорів деталей)
    bg_bound = max(min_bg_delta, border_p98 * 1.45 + 8.0)
    if method == "fixed":
        threshold = max(min_bg_delta, offset)
    elif method == "otsu":
        dist_u8 = np.clip(np.rint(dist_map), 0.0, 255.0).astype(np.uint8)
        otsu_t = _otsu_threshold_from_values(dist_u8)
        # Для кольорового фону обмежуємо поріг зверху, щоб яскраві деталі не відрізали темні
        threshold = min(otsu_t, bg_bound * 1.25) if is_colored_bg else max(bg_bound, otsu_t)
        threshold = max(bg_bound, threshold + offset)
    else:
        threshold = max(min_bg_delta, bg_bound + offset)

    rough_bool = dist_map > threshold
    if np.any(rough_bool) and min(rough_bool.shape) >= 5:
        # Морфологічне закриття 3x3 + заповнення дірок (усуває отвори шпильок і відблиски)
        closed = _binary_erode_3x3(_binary_dilate_3x3(rough_bool))
        rough_bool = _fill_holes_bfs(closed)

    rough_u8 = np.where(rough_bool, np.uint8(255), np.uint8(0))
    return grad_uint8, rough_u8


def distance_transform_manual(binary_mask: np.ndarray) -> np.ndarray:
    """Ручна 2-прохідна карта відстаней Chamfer 3-4 усередині бінарної маски.

    Ортогональний крок має вагу ``3``, діагональний — ``4``; результат нормується
    діленням на ``3.0`` для отримання відстані у піксельних одиницях.

    Args:
        binary_mask: Бінарна або булева маска ``(H, W)`` (``> 0`` — передній план).

    Returns:
        Карта відстаней ``float32`` форми ``(H, W)`` (``0.0`` на фоні).
    """
    fg = np.asarray(binary_mask) > 0
    height, width = fg.shape
    if not np.any(fg):
        return np.zeros((height, width), dtype=np.float32)

    inf_val = 3 * (height + width + 2)
    dist = np.zeros((height + 2, width + 2), dtype=np.int32)
    dist[1:-1, 1:-1] = np.where(fg, inf_val, 0)

    # Прямий прохід (зверху вниз, зліва направо)
    for row in range(1, height + 1):
        fg_cols = np.flatnonzero(fg[row - 1])
        if fg_cols.size == 0:
            continue
        prev = dist[row - 1]
        cols = fg_cols + 1
        top_min = np.minimum(prev[cols] + 3, np.minimum(prev[cols - 1] + 4, prev[cols + 1] + 4))
        dist[row, cols] = np.minimum(dist[row, cols], top_min)
        row_arr = dist[row]
        for col in cols:
            left_cand = int(row_arr[col - 1]) + 3
            if left_cand < int(row_arr[col]):
                row_arr[col] = left_cand

    # Зворотний прохід (знизу вгору, справа наліво)
    for row in range(height, 0, -1):
        fg_cols = np.flatnonzero(fg[row - 1])
        if fg_cols.size == 0:
            continue
        nxt = dist[row + 1]
        cols = fg_cols + 1
        bot_min = np.minimum(nxt[cols] + 3, np.minimum(nxt[cols - 1] + 4, nxt[cols + 1] + 4))
        dist[row, cols] = np.minimum(dist[row, cols], bot_min)
        row_arr = dist[row]
        for col in cols[::-1]:
            right_cand = int(row_arr[col + 1]) + 3
            if right_cand < int(row_arr[col]):
                row_arr[col] = right_cand

    return (dist[1:-1, 1:-1].astype(np.float32)) / np.float32(3.0)


class _DisjointSet:
    """Компактна структура Union-Find для ручного маркування зв'язних відрізків."""

    def __init__(self) -> None:
        self.parent: list[int] = []
        self.size: list[int] = []

    def add(self, weight: int) -> int:
        """Додає новий елемент із вагою ``weight`` і повертає його ID."""
        node_id = len(self.parent)
        self.parent.append(node_id)
        self.size.append(weight)
        return node_id

    def find(self, node_id: int) -> int:
        """Повертає корінь множини зі стисненням шляхів."""
        cur = node_id
        while self.parent[cur] != cur:
            self.parent[cur] = self.parent[self.parent[cur]]
            cur = self.parent[cur]
        return cur

    def union(self, node_a: int, node_b: int) -> None:
        """Об'єднує дві множини за розміром."""
        ra, rb = self.find(node_a), self.find(node_b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]


def _link_adjacent_runs(
    prev_runs: list[tuple[int, int, int]],
    curr_runs: list[tuple[int, int, int]],
    dsu: _DisjointSet,
) -> None:
    """Зливає 8-зв'язні відрізки сусідніх рядків у ``_DisjointSet``."""
    p_idx = 0
    for cid, c_start, c_end in curr_runs:
        while p_idx < len(prev_runs) and prev_runs[p_idx][2] < c_start:
            p_idx += 1
        scan = p_idx
        while scan < len(prev_runs) and prev_runs[scan][1] <= c_end:
            dsu.union(cid, prev_runs[scan][0])
            scan += 1


def _label_connected_components_rle(fg_mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """Швидке ручне 8-зв'язне маркування компонент (scanline RLE Union-Find)."""
    height, width = fg_mask.shape
    runs: list[tuple[int, int, int]] = []
    dsu = _DisjointSet()
    prev_runs: list[tuple[int, int, int]] = []

    for row in range(height):
        trans = np.flatnonzero(
            np.diff(np.concatenate(([0], fg_mask[row].view(np.int8), [0]))) != 0
        )
        curr_runs: list[tuple[int, int, int]] = []
        for idx in range(0, len(trans), 2):
            start, end = int(trans[idx]), int(trans[idx + 1])
            run_id = dsu.add(end - start)
            runs.append((row, start, end))
            curr_runs.append((run_id, start, end))
        _link_adjacent_runs(prev_runs, curr_runs, dsu)
        prev_runs = curr_runs

    labels = np.zeros((height, width), dtype=np.int32)
    root_to_id: dict[int, int] = {}
    comp_areas: dict[int, int] = {}
    for run_id, (row, start, end) in enumerate(runs):
        root = dsu.find(run_id)
        if root not in root_to_id:
            new_id = len(root_to_id) + 1
            root_to_id[root] = new_id
            comp_areas[new_id] = dsu.size[root]
        labels[row, start:end] = root_to_id[root]
    return labels, comp_areas


def _assign_component_markers(
    comp_mask: np.ndarray,
    dist: np.ndarray,
    markers: np.ndarray,
    params: tuple[float, float, int],
) -> int:
    """Виділяє плато-маркери всередині однієї компоненти й повертає наступний вільний ID."""
    peak_min_ratio, min_dist_sq, next_label = params
    comp_max = float(np.max(dist[comp_mask]))
    if comp_max <= 0.0:
        return next_label

    core_mask = comp_mask & (dist >= peak_min_ratio * comp_max)
    if not np.any(core_mask):
        core_mask = comp_mask & (dist >= comp_max - 1e-6)

    sub_labels, sub_areas = _label_connected_components_rle(core_mask)
    sub_peaks: list[tuple[float, float, float, int]] = []
    for sub_id in sub_areas:
        ys, xs = np.nonzero(sub_labels == sub_id)
        sub_peaks.append((float(np.max(dist[ys, xs])), float(ys.mean()), float(xs.mean()), sub_id))
    sub_peaks.sort(key=lambda item: item[0], reverse=True)

    accepted: list[tuple[float, float, int]] = []
    for _, cy, cx, sub_id in sub_peaks:
        assigned = next(
            (lbl for ay, ax, lbl in accepted if (cy - ay) ** 2 + (cx - ax) ** 2 < min_dist_sq),
            None,
        )
        if assigned is None:
            assigned = next_label
            next_label += 1
            accepted.append((cy, cx, assigned))
        markers[sub_labels == sub_id] = assigned
    return next_label


def extract_watershed_markers(
    distance_map: np.ndarray,
    binary_mask: np.ndarray,
    peak_min_ratio: float = 0.6,
    peak_min_distance_px: int = 15,
    min_marker_area_px: int = 120,
) -> tuple[np.ndarray, int]:
    """Виділяє зв'язні області-маркери ``1..K`` на плато карти відстаней із NMS-злиттям.

    Порогове виділення ``dist >= peak_min_ratio * comp_max`` у межах кожної зв'язної
    компоненти запобігає перерозділенню довгих суцільних деталей (``2x4``, ``1x4``),
    але розділяє злиплі деталі, між якими карта відстаней має звуження («перешийок»).
    Близькі за центроїдами маркери (``< peak_min_distance_px``) об'єднуються.

    Args:
        distance_map: Карта відстаней ``float32`` форми ``(H, W)``.
        binary_mask: Бінарна маска переднього плану ``(H, W)``.
        peak_min_ratio: Частка від максимуму відстані в компоненті для виділення ядра.
        peak_min_distance_px: Мінімальна відстань між центроїдами сусідніх маркерів.
        min_marker_area_px: Мінімальна площа компоненти для створення маркера.

    Returns:
        Кортеж ``(markers_int32, num_markers)``.
    """
    fg = np.asarray(binary_mask) > 0
    dist = np.asarray(distance_map, dtype=np.float32)
    markers = np.zeros(fg.shape, dtype=np.int32)
    if not np.any(fg):
        return markers, 0

    comp_labels, comp_areas = _label_connected_components_rle(fg)
    min_dist_sq = float(max(1, peak_min_distance_px)) ** 2
    next_label = 1

    for comp_id, area in comp_areas.items():
        if area < min_marker_area_px:
            continue
        next_label = _assign_component_markers(
            comp_labels == comp_id,
            dist,
            markers,
            (float(peak_min_ratio), min_dist_sq, next_label),
        )

    return markers, next_label - 1


def marker_watershed_manual(
    gradient_mag: np.ndarray,
    markers: np.ndarray,
    fg_mask: np.ndarray,
) -> np.ndarray:
    """Ручний маркерний алгоритм вододілу (Watershed) на пріоритетній черзі ``heapq``.

    Затоплює пікселі переднього плану в порядку зростання висоти градієнтного
    рельєфу ``gradient_mag`` від заданих маркерів ``1..K``, розділяючи злиплі деталі
    по градієнтних хребтах на стиках.

    Args:
        gradient_mag: Карта модуля градієнта форми ``(H, W)``.
        markers: Початкова карта маркерів ``int32`` форми ``(H, W)`` (``0`` — невідомо, ``1..K``).
        fg_mask: Маска області переднього плану ``(H, W)`` (``> 0``).

    Returns:
        Інстанс-маска ``int32`` форми ``(H, W)`` (``0`` — фон, ``1..K`` — ID деталей).
    """
    grad = np.asarray(gradient_mag, dtype=np.float32)
    fg = np.asarray(fg_mask) > 0
    labels = np.where(fg, np.asarray(markers, dtype=np.int32), np.int32(0))
    height, width = labels.shape

    seed_ys, seed_xs = np.nonzero(labels > 0)
    if len(seed_xs) == 0:
        return labels

    heap: list[tuple[float, int, int, int]] = []
    seq = 0
    for sy, sx in zip(seed_ys.tolist(), seed_xs.tolist()):
        heapq.heappush(heap, (float(grad[sy, sx]), seq, sy, sx))
        seq += 1

    while heap:
        _, _, r, c = heapq.heappop(heap)
        lbl = int(labels[r, c])
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < height and 0 <= nc < width:
                if fg[nr, nc] and labels[nr, nc] == 0:
                    labels[nr, nc] = lbl
                    heapq.heappush(heap, (float(grad[nr, nc]), seq, nr, nc))
                    seq += 1

    return labels


def _render_instance_mask_visual(instance_mask: np.ndarray) -> np.ndarray:
    """Формує наочну 3-канальну кольорову візуалізацію інстанс-маски для ``stage_3``."""
    height, width = instance_mask.shape
    vis = np.zeros((height, width, 3), dtype=np.uint8)
    palette = (
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (255, 215, 0),
        (186, 85, 211),
        (0, 206, 209),
        (255, 140, 0),
        (144, 238, 144),
    )
    for lbl in np.unique(instance_mask):
        lbl_int = int(lbl)
        if lbl_int <= 0:
            continue
        vis[instance_mask == lbl_int] = palette[(lbl_int - 1) % len(palette)]
    return vis


def distance_watershed_manual(
    gradient_mag: np.ndarray,
    rough_fg_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Стадія 3: ручна карта відстаней (Chamfer 3-4) + маркерний Watershed на ``heapq``.

    Args:
        gradient_mag: Градієнтний рельєф Собеля зі Стадії 2.
        rough_fg_mask: Попередня бінарна маска переднього плану.
        stage_cfg: Секція ``stage3_watershed`` із ``configs/pipeline_p3.json``.

    Returns:
        Кортеж ``(instance_mask_int32, watershed_visual_rgb_uint8, num_markers)``.
    """
    fg_bool = np.asarray(rough_fg_mask) > 0
    if not np.any(fg_bool):
        empty_mask = np.zeros(fg_bool.shape, dtype=np.int32)
        empty_vis = np.zeros((*fg_bool.shape, 3), dtype=np.uint8)
        return empty_mask, empty_vis, 0

    dist_map = distance_transform_manual(fg_bool)
    markers, num_markers = extract_watershed_markers(
        distance_map=dist_map,
        binary_mask=fg_bool,
        peak_min_ratio=float(stage_cfg.get("peak_min_ratio", 0.45)),
        peak_min_distance_px=int(stage_cfg.get("peak_min_distance_px", 15)),
        min_marker_area_px=int(stage_cfg.get("min_marker_area_px", 120)),
    )
    instance_mask = marker_watershed_manual(gradient_mag, markers, fg_bool)
    visual = _render_instance_mask_visual(instance_mask)
    return instance_mask, visual, num_markers


class SobelWatershedPipeline(BasePipeline):
    """Пайплайн П3 (Учасник 2): Білатеральний+Unsharp -> Собель -> Watershed."""

    pipeline_id = "p3"
    name = "Градієнтний з розділенням злиплих (Білатеральний -> Собель -> Watershed)"
    owner = "member2"
    stage_categories = (
        "Ретушування та покращення зображень",
        "Контурний аналіз",
        "Сегментація зображень",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Покращення): білатеральне згладжування та Unsharp Masking."""
        stage_cfg = config.get("stage1_bilateral_unsharp", {})
        sharpened = bilateral_unsharp_manual(image_rgb, stage_cfg)
        return sharpened, {}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Контурний аналіз): градієнтний рельєф Собеля та груба маска."""
        stage_cfg = config.get("stage2_sobel_gradient", {})
        grad_mag, rough_fg = segment_sobel_foreground(stage1_output, stage_cfg)
        next_state = dict(state)
        next_state["rough_fg_mask"] = rough_fg
        return grad_mag, next_state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Сегментація): карта відстаней і маркерний вододіл."""
        stage_cfg = config.get("stage3_watershed", {})
        instance_mask, ws_vis, num_markers = distance_watershed_manual(
            gradient_mag=stage2_output,
            rough_fg_mask=state["rough_fg_mask"],
            stage_cfg=stage_cfg,
        )
        return instance_mask, ws_vis, {
            "is_instance_mask": True,
            "num_markers": num_markers,
        }
