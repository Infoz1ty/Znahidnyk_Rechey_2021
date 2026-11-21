"""Пайплайн 4 (Учасник 2): Частотно-текстурний конвеєр аналізу поверхні та шпильок.

Категорії стадій (3 унікальні категорії за методичкою):
1. Перетворення Фур'є — ручне 2D FFT (Кулі-Тьюкі Radix-2) та фільтрація Баттерворта.
2. Аналіз текстур — ручна 8-сусідня карта LBP та ознаки матриці суміжності GLCM.
3. Перетворення Хафа — ручний Circle Hough Transform для детекції та підрахунку шпильок.
"""

from __future__ import annotations

from collections import deque
import math
from typing import Any, Sequence

import numpy as np

from lego_cv.core.base_pipeline import BasePipeline

_LBP_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
)

_GLCM_ANGLE_STEPS: dict[int, tuple[int, int]] = {
    0: (0, 1),
    45: (-1, 1),
    90: (1, 0),
    135: (1, 1),
}

_LUMA_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float64)


def _rgb_to_luma(image: np.ndarray) -> np.ndarray:
    """Перетворює 2D або 3D RGB-зображення у 2D кадр яскравості ``float64``."""
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[2] >= 3:
        return np.tensordot(arr[..., :3], _LUMA_WEIGHTS, axes=([-1], [0]))
    raise ValueError(f"Очікується 2D або 3D RGB-масив, отримано форму {arr.shape}.")


def _is_power_of_two(value: int) -> bool:
    """Перевіряє, чи є ціле додатне число степенем двійки."""
    return value > 0 and (value & (value - 1)) == 0


def _next_power_of_two(value: int) -> int:
    """Повертає найменший степінь двійки ``>= value``."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _bit_reversal_order(length: int) -> np.ndarray:
    """Генерує перестановку біт-реверсу для ітеративного Radix-2 FFT."""
    bits = int(math.log2(length))
    indices = np.arange(length, dtype=np.int64)
    reversed_idx = np.zeros(length, dtype=np.int64)
    for _ in range(bits):
        reversed_idx = (reversed_idx << 1) | (indices & 1)
        indices >>= 1
    return reversed_idx


def fft1d_manual(signal: np.ndarray, inverse: bool = False) -> np.ndarray:
    """Ручне ітеративне 1D FFT Кулі–Тьюкі (Radix-2) уздовж останньої осі.

    Працює векторизовано для 1D-векторів і 2D-батчів рядків без ``np.fft`` чи ``scipy``.

    Args:
        signal: Масив форми ``(..., N)``, де ``N`` — степінь двійки.
        inverse: Якщо ``True`` — виконує обернене перетворення IFFT (із діленням на ``N``).

    Returns:
        Комплексний масив ``complex128`` форми ``(..., N)``.

    Raises:
        ValueError: Якщо довжина останньої осі не є степенем двійки.
    """
    arr = np.asarray(signal, dtype=np.complex128)
    length = arr.shape[-1]
    if not _is_power_of_two(length):
        raise ValueError(f"Довжина сигналу для Radix-2 FFT має бути 2^k, отримано {length}.")
    if length == 1:
        return arr.copy()

    rev = _bit_reversal_order(length)
    data = arr[..., rev].copy()
    sign = 1.0 if inverse else -1.0

    stage_size = 2
    while stage_size <= length:
        half = stage_size // 2
        angles = (sign * 2.0 * math.pi / float(stage_size)) * np.arange(half, dtype=np.float64)
        twiddles = np.cos(angles) + 1j * np.sin(angles)

        reshaped = data.reshape(*data.shape[:-1], length // stage_size, stage_size)
        even = reshaped[..., :half].copy()
        odd = reshaped[..., half:] * twiddles
        reshaped[..., :half] = even + odd
        reshaped[..., half:] = even - odd
        stage_size *= 2

    if inverse:
        data /= float(length)
    return data


def fft2_manual(matrix: np.ndarray) -> np.ndarray:
    """Ручне 2D швидке перетворення Фур'є (Radix-2: по рядках, потім по стовпцях).

    Args:
        matrix: 2D-матриця форми ``(2^p, 2^q)``.

    Returns:
        2D комплексний спектр ``complex128`` тієї самої форми.

    Raises:
        ValueError: Якщо вхід не 2D або розміри не є степенями двійки.
    """
    arr = np.asarray(matrix, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"Очікується 2D-матриця для fft2_manual, отримано {arr.shape}.")
    row_spec = fft1d_manual(arr, inverse=False)
    return fft1d_manual(row_spec.T, inverse=False).T


def ifft2_manual(spectrum: np.ndarray) -> np.ndarray:
    """Ручне обернене 2D швидке перетворення Фур'є (Radix-2).

    Args:
        spectrum: 2D комплексний спектр форми ``(2^p, 2^q)``.

    Returns:
        Відновлена 2D-матриця ``complex128`` тієї самої форми.

    Raises:
        ValueError: Якщо вхід не 2D або розміри не є степенями двійки.
    """
    arr = np.asarray(spectrum, dtype=np.complex128)
    if arr.ndim != 2:
        raise ValueError(f"Очікується 2D-спектр для ifft2_manual, отримано {arr.shape}.")
    col_inv = fft1d_manual(arr.T, inverse=True).T
    return fft1d_manual(col_inv, inverse=True)


def _fftshift_2d(arr: np.ndarray) -> np.ndarray:
    """Ручне центрування нульової частоти 2D-спектра без ``np.fft.fftshift``."""
    return np.roll(np.roll(arr, arr.shape[0] // 2, axis=0), arr.shape[1] // 2, axis=1)


def _ifftshift_2d(arr: np.ndarray) -> np.ndarray:
    """Ручне обернене зміщення центрованого 2D-спектра без ``np.fft.ifftshift``."""
    return np.roll(np.roll(arr, -(arr.shape[0] // 2), axis=0), -(arr.shape[1] // 2), axis=1)


def _build_butterworth_mask(
    shape: tuple[int, int],
    low_cutoff_ratio: float,
    high_cutoff_ratio: float,
    butterworth_order: int,
    low_attenuation: float,
) -> np.ndarray:
    """Формує 2D-маску смугового фільтра Баттерворта зі збереженням DC."""
    pad_h, pad_w = shape
    cy, cx = pad_h // 2, pad_w // 2
    yy = (np.arange(pad_h, dtype=np.float64) - cy) / max(1.0, pad_h / 2.0)
    xx = (np.arange(pad_w, dtype=np.float64) - cx) / max(1.0, pad_w / 2.0)
    radius_norm = np.hypot(yy[:, np.newaxis], xx[np.newaxis, :])

    two_n = 2 * max(1, int(butterworth_order))
    filt = np.ones((pad_h, pad_w), dtype=np.float64)
    if high_cutoff_ratio > 0.0:
        filt *= 1.0 / (1.0 + (radius_norm / float(high_cutoff_ratio)) ** two_n)
    if low_cutoff_ratio > 0.0 and low_attenuation > 0.0:
        safe_r = np.maximum(radius_norm, 1e-12)
        h_hp = 1.0 / (1.0 + (float(low_cutoff_ratio) / safe_r) ** two_n)
        atten = min(1.0, max(0.0, float(low_attenuation)))
        filt *= 1.0 - atten * (1.0 - h_hp)
    # Зберігаємо DC-компоненту, щоб не втратити середню яскравість (PSNR/SSIM)
    filt[cy, cx] = 1.0
    return filt


def fft2_bandpass_filter(
    image_gray: np.ndarray,
    low_cutoff_ratio: float = 0.01,
    high_cutoff_ratio: float = 0.30,
    butterworth_order: int = 2,
    low_attenuation: float = 0.15,
) -> np.ndarray:
    """Ручна смугова фільтрація Баттерворта у частотній області зі збереженням DC.

    Доповнює кадр симетричним віддзеркаленням до найближчого степеня двійки ``(2^p, 2^q)``,
    виконує ручне 2D FFT, множить спектр на гладку характеристику Баттерворта зі збереженням
    середньої яскравості (DC-складової), виконує 2D IFFT та обрізає до вихідних розмірів.

    Args:
        image_gray: 2D-кадр яскравості ``(H, W)``.
        low_cutoff_ratio: Нормована частота зрізу ФВЧ ``D_L`` у ``[0, 1]``.
        high_cutoff_ratio: Нормована частота зрізу ФНЧ ``D_H`` у ``(0, 1.5]``.
        butterworth_order: Порядок фільтра Баттерворта ``n >= 1``.
        low_attenuation: Частка придушення наднизьких частот (``0..1``) зі збереженням DC.

    Returns:
        Відфільтрований 2D-кадр яскравості ``uint8`` форми ``(H, W)``.
    """
    gray = np.asarray(image_gray, dtype=np.float64)
    height, width = gray.shape
    if gray.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    pad_h = _next_power_of_two(height)
    pad_w = _next_power_of_two(width)
    pad_mode = "reflect" if (height > 1 and width > 1) else "edge"
    padded = np.pad(gray, ((0, pad_h - height), (0, pad_w - width)), mode=pad_mode)

    spectrum_centered = _fftshift_2d(fft2_manual(padded))
    filt = _build_butterworth_mask(
        (pad_h, pad_w),
        low_cutoff_ratio=low_cutoff_ratio,
        high_cutoff_ratio=high_cutoff_ratio,
        butterworth_order=butterworth_order,
        low_attenuation=low_attenuation,
    )
    restored = np.real(ifft2_manual(_ifftshift_2d(spectrum_centered * filt)))[:height, :width]
    return np.clip(np.rint(restored), 0.0, 255.0).astype(np.uint8)


def fft2_bandpass_manual(
    image_rgb: np.ndarray,
    stage_cfg: dict[str, Any],
) -> np.ndarray:
    """Стадія 1: ручне 2D FFT (Radix-2) та смугова фільтрація Баттерворта.

    Args:
        image_rgb: Вхідне RGB-зображення ``(H, W, 3)`` (або 2D grayscale) типу ``uint8``.
        stage_cfg: Секція ``stage1_fft_bandpass`` із ``configs/pipeline_p4.json``.

    Returns:
        Відновлений після 2D IFFT кадр яскравості ``(H, W)`` типу ``uint8``.
    """
    gray = _rgb_to_luma(image_rgb)
    return fft2_bandpass_filter(
        gray,
        low_cutoff_ratio=float(stage_cfg.get("low_cutoff_ratio", 0.01)),
        high_cutoff_ratio=float(stage_cfg.get("high_cutoff_ratio", 0.30)),
        butterworth_order=int(stage_cfg.get("butterworth_order", 2)),
        low_attenuation=float(stage_cfg.get("low_attenuation", 0.15)),
    )


def compute_lbp_map(gray_image: np.ndarray) -> np.ndarray:
    """Ручне обчислення 8-сусідньої карти локальних бінарних шаблонів (LBP).

    Сусіди обходяться за годинниковою стрілкою починаючи з верхнього лівого ``(-1, -1)``:
    біт ``k`` (``k = 0..7``) дорівнює ``1``, якщо сусід ``>= center``.

    Args:
        gray_image: 2D-зображення яскравості форми ``(H, W)``.

    Returns:
        Карта 8-бітних кодів LBP форми ``(H, W)`` типу ``uint8``.
    """
    gray = np.asarray(gray_image, dtype=np.float32)
    if gray.ndim != 2:
        raise ValueError(f"Очікується 2D-зображення для compute_lbp_map, отримано {gray.shape}.")
    if gray.size == 0:
        return np.zeros(gray.shape, dtype=np.uint8)

    height, width = gray.shape
    padded = np.pad(gray, pad_width=1, mode="edge")
    lbp = np.zeros((height, width), dtype=np.uint8)

    for bit_idx, (dy, dx) in enumerate(_LBP_OFFSETS):
        neighbor = padded[1 + dy : 1 + dy + height, 1 + dx : 1 + dx + width]
        lbp |= ((neighbor >= gray).astype(np.uint8)) << bit_idx
    return lbp


def _box_mean_integral(values: np.ndarray, window_size: int) -> np.ndarray:
    """Швидке ручне обчислення локального середнього у вікні ``w x w`` через інтеграл."""
    win = max(1, int(window_size))
    if win == 1:
        return values.astype(np.float32, copy=True)

    pad_before = win // 2
    pad_after = win - 1 - pad_before
    padded = np.pad(
        values.astype(np.float64),
        ((pad_before, pad_after), (pad_before, pad_after)),
        mode="edge",
    )
    integral = np.pad(padded, ((1, 0), (1, 0)), mode="constant", constant_values=0.0)
    integral = np.cumsum(np.cumsum(integral, axis=0), axis=1)
    height, width = values.shape
    sums = (
        integral[win : win + height, win : win + width]
        - integral[:height, win : win + width]
        - integral[win : win + height, :width]
        + integral[:height, :width]
    )
    return (sums / float(win * win)).astype(np.float32)


def _glcm_single_direction(
    quant: np.ndarray,
    valid_mask: np.ndarray,
    n_levels: int,
    delta: tuple[int, int],
    weights: tuple[np.ndarray, np.ndarray],
) -> tuple[float, float, float] | None:
    """Обчислює ``(contrast, homogeneity, energy)`` для одного напрямку ``(dr, dc)``."""
    height, width = quant.shape
    dr, dc = delta
    r_src = slice(max(0, -dr), min(height, height - dr))
    c_src = slice(max(0, -dc), min(width, width - dc))
    if r_src.start >= r_src.stop or c_src.start >= c_src.stop:
        return None

    r_dst = slice(r_src.start + dr, r_src.stop + dr)
    c_dst = slice(c_src.start + dc, c_src.stop + dc)
    pair_ok = valid_mask[r_src, c_src] & valid_mask[r_dst, c_dst]
    if not np.any(pair_ok):
        return None

    codes = quant[r_src, c_src][pair_ok] * n_levels + quant[r_dst, c_dst][pair_ok]
    glcm = np.bincount(codes, minlength=n_levels * n_levels).reshape(n_levels, n_levels).astype(
        np.float64
    )
    glcm += glcm.T
    total = float(glcm.sum())
    if total <= 0.0:
        return None

    prob = glcm / total
    return (
        float(np.sum(weights[0] * prob)),
        float(np.sum(weights[1] * prob)),
        float(np.sum(prob * prob)),
    )


def compute_glcm_features(
    gray_patch: np.ndarray,
    mask: np.ndarray | None = None,
    levels: int = 16,
    distances: Sequence[int] = (1,),
    angles_deg: Sequence[int] = (0, 45, 90, 135),
) -> dict[str, float]:
    """Ручне обчислення ознак квантованої матриці суміжності рівнів сірого (GLCM).

    Усереднює симетричні нормовані матриці GLCM за заданими відстанями та кутами
    ``0°, 45°, 90°, 135°`` і повертає ``contrast``, ``homogeneity`` та ``energy``.

    Args:
        gray_patch: 2D-фрагмент яскравості ``uint8`` форми ``(H, W)``.
        mask: Необов'язкова булева маска ``(H, W)`` пікселів, що належать деталі.
        levels: Кількість рівнів квантування (за замовчуванням 16).
        distances: Перелік відстаней зміщення ``d``.
        angles_deg: Перелік кутів у градусах (підтримуються ``0, 45, 90, 135``).

    Returns:
        Словник із ключами ``contrast``, ``homogeneity``, ``energy``.
    """
    arr = np.asarray(gray_patch, dtype=np.float64)
    if arr.ndim != 2 or arr.size == 0:
        return {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0}

    n_levels = max(2, int(levels))
    quant = np.clip((arr * float(n_levels)) // 256.0, 0, n_levels - 1).astype(np.int32)
    valid_mask = np.ones(arr.shape, dtype=np.bool_) if mask is None else (np.asarray(mask) > 0)

    idx_grid = np.arange(n_levels, dtype=np.float64)
    diff_sq = (idx_grid[:, np.newaxis] - idx_grid[np.newaxis, :]) ** 2
    weights = (diff_sq, 1.0 / (1.0 + diff_sq))

    results: list[tuple[float, float, float]] = []
    for dist in distances:
        step = max(1, int(dist))
        for angle in angles_deg:
            dr, dc = _GLCM_ANGLE_STEPS.get(int(angle) % 180, (0, 1))
            res = _glcm_single_direction(
                quant, valid_mask, n_levels, (dr * step, dc * step), weights
            )
            if res is not None:
                results.append(res)

    if not results:
        return {"contrast": 0.0, "homogeneity": 1.0, "energy": 1.0}
    inv_n = 1.0 / float(len(results))
    return {
        "contrast": sum(r[0] for r in results) * inv_n,
        "homogeneity": sum(r[1] for r in results) * inv_n,
        "energy": sum(r[2] for r in results) * inv_n,
    }


def _morph_close_3x3(mask: np.ndarray) -> np.ndarray:
    """Ручне морфологічне закриття 3x3 (дилатація -> ерозія) для очищення текстурної маски."""
    height, width = mask.shape
    pad_d = np.pad(mask, 1, mode="constant", constant_values=False)
    dilated = np.zeros((height, width), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            dilated |= pad_d[dy : dy + height, dx : dx + width]
    pad_e = np.pad(dilated, 1, mode="constant", constant_values=False)
    eroded = np.ones((height, width), dtype=np.bool_)
    for dy in range(3):
        for dx in range(3):
            eroded &= pad_e[dy : dy + height, dx : dx + width]
    return eroded


def _fill_mask_interior(mask: np.ndarray) -> np.ndarray:
    """Заповнює внутрішні отвори маски ручним BFS по фону від границі bounding-box."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return mask.copy()
    r_slice = slice(int(ys.min()), int(ys.max()) + 1)
    c_slice = slice(int(xs.min()), int(xs.max()) + 1)
    free = np.pad(~mask[r_slice, c_slice], 1, mode="constant", constant_values=True)
    gh, gw = free.shape
    seen = np.zeros((gh, gw), dtype=np.bool_)
    seen[0, 0] = True
    bfs: deque[tuple[int, int]] = deque([(0, 0)])
    while bfs:
        cr, cc = bfs.popleft()
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= nr < gh and 0 <= nc < gw and free[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                bfs.append((nr, nc))
    out = mask.copy()
    out[r_slice, c_slice] = ~seen[1:-1, 1:-1]
    return out


def _dsu_find_compressed(seg_parent: list[int], node: int) -> int:
    """Знаходить корінь у списку предків із напівстисненням шляху."""
    cur = node
    while seg_parent[cur] != cur:
        seg_parent[cur] = seg_parent[seg_parent[cur]]
        cur = seg_parent[cur]
    return cur


def _merge_segment_rows(
    prev_segs: list[tuple[int, int, int]],
    cur_segs: list[tuple[int, int, int]],
    seg_parent: list[int],
    seg_area: list[int],
) -> None:
    """Зливає 8-зв'язні рядкові відрізки сусідніх рядків."""
    p_ptr = 0
    for sid, s_col, e_col in cur_segs:
        while p_ptr < len(prev_segs) and prev_segs[p_ptr][2] < s_col:
            p_ptr += 1
        scan = p_ptr
        while scan < len(prev_segs) and prev_segs[scan][1] <= e_col:
            ra = _dsu_find_compressed(seg_parent, sid)
            rb = _dsu_find_compressed(seg_parent, prev_segs[scan][0])
            if ra != rb:
                seg_parent[rb] = ra
                seg_area[ra] += seg_area[rb]
            scan += 1


def _connected_components_bfs(binary_mask: np.ndarray, min_area: int) -> tuple[np.ndarray, int]:
    """Ручне 8-зв'язне маркування компонент із фільтрацією шуму за мінімальною площею."""
    fg = np.asarray(binary_mask) > 0
    height, width = fg.shape
    labels = np.zeros((height, width), dtype=np.int32)
    if not np.any(fg):
        return labels, 0

    seg_parent: list[int] = []
    seg_area: list[int] = []
    seg_coords: list[tuple[int, int, int]] = []
    prev_segs: list[tuple[int, int, int]] = []

    for row in range(height):
        changes = np.flatnonzero(np.diff(np.concatenate(([0], fg[row].view(np.int8), [0]))) != 0)
        cur_segs: list[tuple[int, int, int]] = []
        for idx in range(0, len(changes), 2):
            s_col, e_col = int(changes[idx]), int(changes[idx + 1])
            sid = len(seg_parent)
            seg_parent.append(sid)
            seg_area.append(e_col - s_col)
            seg_coords.append((row, s_col, e_col))
            cur_segs.append((sid, s_col, e_col))
        if prev_segs:
            _merge_segment_rows(prev_segs, cur_segs, seg_parent, seg_area)
        prev_segs = cur_segs

    root_map: dict[int, int] = {}
    for sid, (row, s_col, e_col) in enumerate(seg_coords):
        root = _dsu_find_compressed(seg_parent, sid)
        if seg_area[root] < min_area:
            continue
        if root not in root_map:
            root_map[root] = len(root_map) + 1
        labels[row, s_col:e_col] = root_map[root]
    return labels, len(root_map)


def _rgb_chroma_foreground(
    source_rgb: np.ndarray,
    border_width: int,
    min_delta: float,
) -> np.ndarray | None:
    """Повертає хроматичну маску переднього плану для кольорового фону (або ``None``)."""
    rgb_f = np.stack(
        [_box_mean_integral(source_rgb[..., ch].astype(np.float32), 3) for ch in range(3)],
        axis=-1,
    )
    border_rgb = np.concatenate(
        (
            rgb_f[:border_width, :, :].reshape(-1, 3),
            rgb_f[-border_width:, :, :].reshape(-1, 3),
            rgb_f[:, :border_width, :].reshape(-1, 3),
            rgb_f[:, -border_width:, :].reshape(-1, 3),
        ),
        axis=0,
    )
    bg_rgb = np.median(border_rgb, axis=0)
    bg_chroma = bg_rgb - float(np.mean(bg_rgb))
    if float(np.linalg.norm(bg_chroma)) <= 15.0:
        return None

    pix_chroma = rgb_f - np.mean(rgb_f, axis=2, keepdims=True)
    dist_map = np.linalg.norm(pix_chroma - bg_chroma, axis=2)
    b_chroma = border_rgb - np.mean(border_rgb, axis=1, keepdims=True)
    b_p98 = float(np.percentile(np.linalg.norm(b_chroma - bg_chroma, axis=1), 98))
    return dist_map > max(min_delta, b_p98 * 1.45 + 8.0)


def _segment_foreground_for_texture(
    gray_image: np.ndarray,
    lbp_map: np.ndarray,
    stage_cfg: dict[str, Any],
    source_rgb: np.ndarray | None,
) -> np.ndarray:
    """Формує маску кандидатів поєднанням LBP-активності та адаптивної оцінки фону."""
    height, width = gray_image.shape
    bw = max(1, min(6, height // 10 or 1, width // 10 or 1))
    min_delta = float(stage_cfg.get("min_bg_delta", 14.0))

    if source_rgb is not None and source_rgb.ndim == 3 and source_rgb.shape[:2] == (height, width):
        chroma_fg = _rgb_chroma_foreground(source_rgb, bw, min_delta)
        if chroma_fg is not None:
            return chroma_fg

    gray_f = gray_image.astype(np.float32)
    border_g = np.concatenate(
        (
            gray_f[:bw, :].ravel(),
            gray_f[-bw:, :].ravel(),
            gray_f[:, :bw].ravel(),
            gray_f[:, -bw:].ravel(),
        )
    )
    bg_med = float(np.median(border_g))
    bg_p98 = float(np.percentile(np.abs(border_g - bg_med), 98))
    intensity_fg = np.abs(gray_f - bg_med) > max(min_delta, bg_p98 * 1.45 + 8.0)

    win = int(stage_cfg.get("lbp_window_size", 9))
    lbp_f = lbp_map.astype(np.float32)
    lbp_mean = _box_mean_integral(lbp_f, win)
    lbp_var = np.maximum(0.0, _box_mean_integral(lbp_f * lbp_f, win) - lbp_mean * lbp_mean)
    if float(np.max(lbp_var)) > 0.0 and not np.any(intensity_fg):
        pct = float(stage_cfg.get("texture_threshold_percentile", 65.0))
        tex_thresh = float(np.percentile(lbp_var, pct))
        if tex_thresh > 0.0:
            return lbp_var > tex_thresh
    return intensity_fg


def _extract_component_glcm_map(
    gray_u8: np.ndarray,
    lbp_map: np.ndarray,
    labels: np.ndarray,
    num_comps: int,
    stage_cfg: dict[str, Any],
) -> dict[int, dict[str, Any]]:
    """Обчислює GLCM + LBP-ознаки для кожного виділеного компонента маски."""
    glcm_levels = int(stage_cfg.get("glcm_levels", 16))
    glcm_dists = tuple(int(d) for d in stage_cfg.get("glcm_distances", [1]))
    glcm_angles = tuple(int(a) for a in stage_cfg.get("glcm_angles_deg", [0, 45, 90, 135]))
    tile_homog = float(stage_cfg.get("tile_homogeneity_threshold", 0.78))

    texture_features: dict[int, dict[str, Any]] = {}
    for comp_id in range(1, num_comps + 1):
        comp_mask = labels == comp_id
        ys, xs = np.nonzero(comp_mask)
        if len(xs) == 0:
            continue
        r_sl = slice(int(ys.min()), int(ys.max()) + 1)
        c_sl = slice(int(xs.min()), int(xs.max()) + 1)
        feats = compute_glcm_features(
            gray_u8[r_sl, c_sl],
            mask=comp_mask[r_sl, c_sl],
            levels=glcm_levels,
            distances=glcm_dists,
            angles_deg=glcm_angles,
        )
        feats["lbp_mean"] = float(np.mean(lbp_map[comp_mask]))
        feats["surface_hint"] = "tile" if feats["homogeneity"] >= tile_homog else "brick"
        texture_features[comp_id] = feats
    return texture_features


def lbp_glcm_texture_segment(
    gray_image: np.ndarray,
    stage_cfg: dict[str, Any],
    source_rgb: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[str, Any]]]:
    """Стадія 2: ручна карта LBP, сегментація кандидатів та ознаки GLCM по кожній деталі.

    Args:
        gray_image: Відфільтроване зображення яскравості зі Стадії 1 ``(H, W)``.
        stage_cfg: Секція ``stage2_lbp_glcm`` із ``configs/pipeline_p4.json``.
        source_rgb: Необов'язкове вхідне RGB-зображення для стійкої оцінки кольорового фону.

    Returns:
        Кортеж ``(lbp_visual_uint8, candidate_labels_int32, texture_features_by_id)``.
    """
    gray_u8 = np.asarray(gray_image, dtype=np.uint8)
    lbp_map = compute_lbp_map(gray_u8)

    if gray_u8.size == 0 or int(np.max(gray_u8)) == int(np.min(gray_u8)):
        return lbp_map, np.zeros(gray_u8.shape, dtype=np.int32), {}

    fg_raw = _segment_foreground_for_texture(gray_u8, lbp_map, stage_cfg, source_rgb)
    if np.any(fg_raw) and min(fg_raw.shape) >= 5:
        fg_clean = _fill_mask_interior(_morph_close_3x3(fg_raw))
    else:
        fg_clean = fg_raw

    min_area = int(stage_cfg.get("min_component_area_px", 100))
    labels, num_comps = _connected_components_bfs(fg_clean, min_area=min_area)
    texture_features = _extract_component_glcm_map(
        gray_u8, lbp_map, labels, num_comps, stage_cfg
    )
    return lbp_map, labels, texture_features


def _sobel_xy_patch(patch_gray: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ручний 3x3 градієнт Собеля ``(gx, gy, mag)`` на 2D-фрагменті яскравості."""
    arr = np.asarray(patch_gray, dtype=np.float64)
    padded = np.pad(arr, 1, mode="edge")
    tl, tc, tr = padded[:-2, :-2], padded[:-2, 1:-1], padded[:-2, 2:]
    ml, mr = padded[1:-1, :-2], padded[1:-1, 2:]
    bl, bc, br = padded[2:, :-2], padded[2:, 1:-1], padded[2:, 2:]
    gx = -tl + tr - 2.0 * ml + 2.0 * mr - bl + br
    gy = -tl - 2.0 * tc - tr + bl + 2.0 * bc + br
    return gx, gy, np.hypot(gx, gy)


def _box_sum_3x3(plane: np.ndarray) -> np.ndarray:
    """Сума у 3x3 вікні для згладжування квантування голосів акумулятора Хафа."""
    height, width = plane.shape
    padded = np.pad(plane, 1, mode="constant", constant_values=0.0)
    out = np.zeros_like(plane, dtype=np.float32)
    for dy in range(3):
        for dx in range(3):
            out += padded[dy : dy + height, dx : dx + width]
    return out


def _vote_hough_accumulator(
    shape: tuple[int, int],
    edges: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    radii: list[int],
    mask_bool: np.ndarray | None,
) -> np.ndarray:
    """Заповнює 3D-акумулятор Хафа ``(num_r, H, W)`` градієнтним голосуванням."""
    height, width = shape
    ey, ex, cos_t, sin_t = edges
    acc = np.zeros((len(radii), height, width), dtype=np.float32)

    for r_idx, rad in enumerate(radii):
        plane = np.zeros((height, width), dtype=np.float32)
        for sign in (1.0, -1.0):
            cx = np.rint(ex + sign * float(rad) * cos_t).astype(np.int32)
            cy = np.rint(ey + sign * float(rad) * sin_t).astype(np.int32)
            in_b = (cy >= 0) & (cy < height) & (cx >= 0) & (cx < width)
            if not np.any(in_b):
                continue
            cy, cx = cy[in_b], cx[in_b]
            if mask_bool is not None:
                in_m = mask_bool[cy, cx]
                cy, cx = cy[in_m], cx[in_m]
            np.add.at(plane, (cy, cx), 1.0)
        acc[r_idx] = _box_sum_3x3(plane) / np.float32(2.0 * math.pi * float(rad))
    return acc


def _extract_hough_peaks(
    acc: np.ndarray,
    radii: list[int],
    accumulator_threshold_ratio: float,
    min_center_distance_px: int,
) -> list[tuple[int, int, int, float]]:
    """Виділяє локальні піки акумулятора Хафа з просторовим NMS."""
    max_score = float(np.max(acc))
    min_density = 0.35
    if max_score < min_density:
        return []

    thresh = max(min_density, float(accumulator_threshold_ratio) * max_score)
    best_r_idx = np.argmax(acc, axis=0)
    best_scores = np.max(acc, axis=0)
    cand_y, cand_x = np.nonzero(best_scores >= thresh)
    if len(cand_x) == 0:
        return []

    order = np.argsort(-best_scores[cand_y, cand_x])
    min_dist_sq = float(max(1, min_center_distance_px)) ** 2
    circles: list[tuple[int, int, int, float]] = []

    for idx in order:
        cy_i, cx_i = int(cand_y[idx]), int(cand_x[idx])
        if all((cx_i - ox) ** 2 + (cy_i - oy) ** 2 >= min_dist_sq for ox, oy, _, _ in circles):
            rad_i = radii[int(best_r_idx[cy_i, cx_i])]
            circles.append((cx_i, cy_i, rad_i, float(best_scores[cy_i, cx_i])))
    return circles


def hough_circles_manual(
    gray_patch: np.ndarray,
    radius_min: int = 10,
    radius_max: int = 24,
    gradient_threshold: float = 35.0,
    accumulator_threshold_ratio: float = 0.65,
    min_center_distance_px: int = 24,
    mask_patch: np.ndarray | None = None,
) -> list[tuple[int, int, int, float]]:
    """Ручне градієнтне перетворення Хафа для детекції кіл (шпильок LEGO).

    Голосує в 3D-акумуляторі ``(r, cy, cx)`` уздовж напрямку нормалі градієнта
    ``(cos theta, sin theta)`` на сильних краях, нормує за довжиною кола ``2*pi*r``
    та виконує просторове придушення немаксимумів (NMS) за ``min_center_distance_px``.

    Args:
        gray_patch: 2D-фрагмент яскравості ``(H, W)``.
        radius_min: Мінімальний радіус кола в пікселях (``>= 2``).
        radius_max: Максимальний радіус кола в пікселях (``>= radius_min``).
        gradient_threshold: Поріг модуля градієнта для вибору крайових пікселів.
        accumulator_threshold_ratio: Відносний поріг піків від максимуму акумулятора.
        min_center_distance_px: Мінімальна відстань між центрами знайдених кіл.
        mask_patch: Необов'язкова булева маска області пошуку центрів.

    Returns:
        Список кортежів ``(cx, cy, radius, score)``, відсортований за спаданням ``score``.
    """
    arr = np.asarray(gray_patch, dtype=np.float64)
    if arr.ndim != 2 or min(arr.shape) < 3:
        return []

    gx, gy, mag = _sobel_xy_patch(arr)
    edge_mask = mag >= float(gradient_threshold)
    mask_bool = (np.asarray(mask_patch) > 0) if mask_patch is not None else None
    if mask_bool is not None:
        edge_mask &= mask_bool

    ey, ex = np.nonzero(edge_mask)
    if len(ex) < 6:
        return []

    radii = list(range(max(2, int(radius_min)), max(2, int(radius_min), int(radius_max)) + 1))
    acc = _vote_hough_accumulator(
        arr.shape,
        (ey, ex, gx[ey, ex] / mag[ey, ex], gy[ey, ex] / mag[ey, ex]),
        radii,
        mask_bool,
    )
    return _extract_hough_peaks(acc, radii, accumulator_threshold_ratio, min_center_distance_px)


def _draw_circle_outline_numpy(
    canvas: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
    color_rgb: tuple[int, int, int],
) -> None:
    """Малює коло на 3D-масиві ``NumPy`` за рівнянням кола без ``cv2``."""
    height, width = canvas.shape[:2]
    num_pts = max(24, int(round(2.0 * math.pi * float(radius) * 1.5)))
    angles = np.linspace(0.0, 2.0 * math.pi, num_pts, endpoint=False)
    xs = np.rint(center_x + radius * np.cos(angles)).astype(np.int32)
    ys = np.rint(center_y + radius * np.sin(angles)).astype(np.int32)
    valid = (ys >= 0) & (ys < height) & (xs >= 0) & (xs < width)
    canvas[ys[valid], xs[valid]] = color_rgb


def _erode_mask_pixels(mask: np.ndarray, iterations: int = 2) -> np.ndarray:
    """Звужує маску на ``iterations`` пікселів (3x3), відсікаючи зовнішній борт деталі."""
    cur = np.asarray(mask, dtype=np.bool_)
    height, width = cur.shape
    for _ in range(iterations):
        pad = np.pad(cur, 1, mode="constant", constant_values=False)
        nxt = np.ones((height, width), dtype=np.bool_)
        for dy in range(3):
            for dx in range(3):
                nxt &= pad[dy : dy + height, dx : dx + width]
        if not np.any(nxt):
            return cur
        cur = nxt
    return cur


def _detect_studs_in_component(
    gray_u8: np.ndarray,
    comp_mask: np.ndarray,
    stage_cfg: dict[str, Any],
) -> list[tuple[int, int, int]]:
    """Виявляє кола-шпильки всередині bbox одного кандидата й повертає глобальні координати."""
    ys, xs = np.nonzero(comp_mask)
    if len(xs) == 0:
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    inner_mask = _erode_mask_pixels(comp_mask[y0:y1, x0:x1], iterations=3)
    found = hough_circles_manual(
        gray_u8[y0:y1, x0:x1],
        radius_min=int(stage_cfg.get("radius_min_px", 10)),
        radius_max=int(stage_cfg.get("radius_max_px", 24)),
        gradient_threshold=float(stage_cfg.get("gradient_threshold", 35.0)),
        accumulator_threshold_ratio=float(stage_cfg.get("accumulator_threshold_ratio", 0.65)),
        min_center_distance_px=int(stage_cfg.get("min_center_distance_px", 24)),
        mask_patch=inner_mask,
    )
    return [(x0 + lx, y0 + ly, rad) for lx, ly, rad, _ in found]


def hough_circles_studs_manual(
    gray_image: np.ndarray,
    candidate_labels: np.ndarray,
    stage_cfg: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Стадія 3: ручне градієнтне перетворення Хафа для підрахунку шпильок у кожній деталі.

    Запускається строго в межах bbox кожного кандидата зі Стадії 2 для збереження швидкодії.

    Args:
        gray_image: Кадр яскравості ``(H, W)`` зі Стадії 1.
        candidate_labels: Інстанс-маска кандидатів ``int32`` зі Стадії 2.
        stage_cfg: Секція ``stage3_hough_circles`` із ``configs/pipeline_p4.json``.

    Returns:
        Кортеж ``(final_mask_int32, circles_visual_rgb_uint8, stud_metadata)``.
    """
    gray_u8 = np.asarray(gray_image, dtype=np.uint8)
    labels = np.asarray(candidate_labels, dtype=np.int32)
    visual = np.stack((gray_u8, gray_u8, gray_u8), axis=-1)

    stud_counts: dict[int, int] = {}
    stud_circles: dict[int, list[tuple[int, int, int]]] = {}

    for comp_id in np.unique(labels):
        cid = int(comp_id)
        if cid <= 0:
            continue
        global_circles = _detect_studs_in_component(gray_u8, labels == cid, stage_cfg)
        for gx, gy, rad in global_circles:
            _draw_circle_outline_numpy(visual, gx, gy, rad, (0, 255, 128))
        stud_counts[cid] = len(global_circles)
        stud_circles[cid] = global_circles

    return labels, visual, {
        "is_instance_mask": True,
        "stud_counts": stud_counts,
        "stud_circles": stud_circles,
    }


class FftTextureHoughPipeline(BasePipeline):
    """Пайплайн П4 (Учасник 2): 2D FFT -> LBP/GLCM -> Кола Хафа."""

    pipeline_id = "p4"
    name = "Частотно-текстурний (2D FFT -> LBP/GLCM -> Кола Хафа)"
    owner = "member2"
    stage_categories = (
        "Перетворення Фур'є",
        "Аналіз текстур",
        "Перетворення Хафа",
    )

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 1 (Перетворення Фур'є): ручне 2D FFT та фільтрація Баттерворта."""
        stage_cfg = config.get("stage1_fft_bandpass", {})
        filtered_gray = fft2_bandpass_manual(image_rgb, stage_cfg)
        return filtered_gray, {
            "filtered_gray": filtered_gray,
            "source_rgb": np.asarray(image_rgb, dtype=np.uint8),
        }

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Стадія 2 (Аналіз текстур): ручна карта LBP та ознаки GLCM."""
        stage_cfg = dict(config.get("stage2_lbp_glcm", {}))
        stage3_cfg = config.get("stage3_hough_circles", {})
        if "tile_homogeneity_threshold" in stage3_cfg:
            stage_cfg.setdefault(
                "tile_homogeneity_threshold",
                stage3_cfg["tile_homogeneity_threshold"],
            )
        lbp_vis, candidate_labels, texture_features = lbp_glcm_texture_segment(
            stage1_output,
            stage_cfg,
            source_rgb=state.get("source_rgb"),
        )
        next_state = dict(state)
        next_state["candidate_labels"] = candidate_labels
        next_state["texture_features"] = texture_features
        return lbp_vis, next_state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict[str, Any],
        config: dict[str, Any],
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        """Стадія 3 (Перетворення Хафа): пошук кіл-шпильок усередині кандидатів."""
        del stage2_output
        stage_cfg = config.get("stage3_hough_circles", {})
        final_mask, circles_vis, stud_meta = hough_circles_studs_manual(
            gray_image=state["filtered_gray"],
            candidate_labels=state["candidate_labels"],
            stage_cfg=stage_cfg,
        )
        metadata = dict(stud_meta)
        metadata["texture_features"] = state.get("texture_features", {})
        return final_mask, circles_vis, metadata
