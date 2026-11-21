"""Оркестратор пакетного запуску пайплайнів, збереження стадій та експорту метрик.

Зона реалізації (Етап 1): Учасник 2 (member2).
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.core.color_classifier import classify_colors
from lego_cv.core.image_io import (
    SUPPORTED_EXTENSIONS,
    ImageLoadError,
    InvalidImageError,
    discover_images,
    load_image_rgb,
    load_mask_binary,
    save_image,
)
from lego_cv.core.shape_classifier import classify_shapes
from lego_cv.core.types import BrickDetection, PipelineResult, RunMetrics
from lego_cv.core.visualizer import render_all_detections, render_search_highlight
from lego_cv.metrics.noise import add_gaussian_noise, add_salt_pepper_noise
from lego_cv.metrics.quality import (
    compute_classification_metrics,
    compute_iou,
    compute_mse,
    compute_psnr,
    compute_ssim,
)

METRICS_CSV_FIELDNAMES = [
    "image_name",
    "pipeline_id",
    "psnr",
    "mse",
    "ssim",
    "iou",
    "accuracy",
    "precision",
    "recall",
    "f1_score",
    "total_ms",
    "detected_count",
]


@dataclass(slots=True)
class RunConfig:
    """Параметри одного пакетного або одиничного запуску системи.

    Attributes:
        input_path: Файл або каталог із вхідними зображеннями.
        output_dir: Коренева папка для збереження результатів.
        mode: Режим візуалізації (``all`` або ``find``).
        find_query: Запит пошуку для режиму ``find`` (наприклад, ``blue_2x4``).
        save_stages: Чи зберігати проміжні зображення ``stage_1..3``.
        eval_denoise: Чи накладати детермінований шум перед Стадією 1 для ``PSNR/MSE/SSIM``.
        masks_dir: Каталог еталонних масок для обчислення ``IoU``.
        labels_csv: Шлях до ``labels.csv`` для обчислення ``Accuracy`` і ``F1``.
    """

    input_path: Path
    output_dir: Path
    mode: str = "all"
    find_query: str | None = None
    save_stages: bool = True
    eval_denoise: bool = False
    masks_dir: Path | None = None
    labels_csv: Path | None = None


def _safe_discover_images(input_path: Path) -> list[Path]:
    """Отримує список зображень через ``image_io.discover_images`` із безпечним фолбеком."""
    try:
        return discover_images(input_path)
    except NotImplementedError:
        if not input_path.exists():
            raise ImageLoadError(f"Вхідний шлях не існує: {input_path}") from None
        if input_path.is_file():
            return [input_path]
        if input_path.is_dir():
            return sorted(
                path
                for path in input_path.iterdir()
                if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        raise ImageLoadError(f"Непідтримуваний вхідний шлях: {input_path}") from None


def _safe_load_image_rgb(path: Path, config: dict[str, Any]) -> np.ndarray:
    """Завантажує RGB-зображення через ``image_io.load_image_rgb`` із фолбеком C2."""
    prep_cfg = config.get("preprocessing", {})
    max_side = prep_cfg.get("max_side_px", 1600)
    min_side = int(prep_cfg.get("min_image_side_px", 3))
    max_allowed = int(prep_cfg.get("max_image_side_px", 8000))

    try:
        return load_image_rgb(
            path,
            max_side_px=max_side,
            min_side_px=min_side,
            max_allowed_side_px=max_allowed,
        )
    except NotImplementedError:
        if not path.is_file():
            raise ImageLoadError(f"Файл не знайдено: {path}") from None
        if path.stat().st_size == 0:
            raise ImageLoadError(f"Файл порожній (0 байт): {path}") from None
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ImageLoadError(f"Непідтримуване розширення файлу: {path}") from None
        try:
            with Image.open(path) as opened:
                opened.load()
                width, height = opened.size
                if width < min_side or height < min_side:
                    raise InvalidImageError(
                        f"Занадто мале зображення ({width}x{height}): {path}"
                    ) from None
                if width > max_allowed or height > max_allowed:
                    raise InvalidImageError(
                        f"Перевищено граничний розмір ({width}x{height}): {path}"
                    ) from None
                if opened.mode in ("RGBA", "LA"):
                    rgba = opened.convert("RGBA")
                    bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    rgb_img = Image.alpha_composite(bg, rgba).convert("RGB")
                else:
                    rgb_img = opened.convert("RGB")
                if max_side is not None and max(rgb_img.size) > int(max_side):
                    scale = float(max_side) / float(max(rgb_img.size))
                    new_size = (
                        max(1, int(round(rgb_img.width * scale))),
                        max(1, int(round(rgb_img.height * scale))),
                    )
                    rgb_img = rgb_img.resize(new_size, Image.Resampling.BILINEAR)
                return np.asarray(rgb_img, dtype=np.uint8)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if isinstance(exc, (ImageLoadError, InvalidImageError)):
                raise
            raise ImageLoadError(f"Пошкоджене зображення {path}: {exc}") from exc


def _safe_load_mask_binary(mask_path: Path) -> np.ndarray:
    """Завантажує еталонну бінарну маску через ``image_io.load_mask_binary`` із фолбеком."""
    try:
        return load_mask_binary(mask_path)
    except NotImplementedError:
        with Image.open(mask_path) as opened:
            gray = np.asarray(opened.convert("L"), dtype=np.uint8)
        return (gray > 0).astype(np.uint8)


def _safe_save_image(image: np.ndarray, path: Path) -> None:
    """Зберігає зображення (2D або 3D) у PNG через ``image_io.save_image`` із фолбеком."""
    try:
        save_image(image, path)
    except NotImplementedError:
        path.parent.mkdir(parents=True, exist_ok=True)
        arr = np.asarray(image)
        if arr.dtype == np.bool_:
            arr = np.where(arr, np.uint8(255), np.uint8(0))
        elif arr.dtype != np.uint8:
            max_val = float(np.max(arr)) if arr.size > 0 else 0.0
            if 0.0 < max_val <= 1.0:
                arr = np.clip(np.rint(arr.astype(np.float64) * 255.0), 0.0, 255.0).astype(
                    np.uint8
                )
            else:
                arr = np.clip(np.rint(arr.astype(np.float64)), 0.0, 255.0).astype(np.uint8)
        mode = "L" if arr.ndim == 2 else "RGB"
        Image.fromarray(arr, mode=mode).save(path, format="PNG")


def _safe_classify_shapes(
    mask: np.ndarray,
    config: dict[str, Any],
    metadata: dict[str, Any],
) -> list[BrickDetection]:
    """Викликає ``shape_classifier.classify_shapes`` із базовим фолбеком до злиття T1.3."""
    try:
        return classify_shapes(mask, config, pipeline_metadata=metadata)
    except NotImplementedError:
        mask_arr = np.asarray(mask)
        unique_ids = [int(val) for val in np.unique(mask_arr) if int(val) > 0]
        fallback_detections: list[BrickDetection] = []
        for brick_id in unique_ids:
            ys, xs = np.nonzero(mask_arr == brick_id)
            if len(xs) == 0:
                continue
            fallback_detections.append(
                BrickDetection(
                    brick_id=brick_id,
                    shape_class="unknown",
                    surface_type="unknown",
                    color="Unknown",
                    stud_count=0,
                    area_px=float(len(xs)),
                    aspect_ratio=1.0,
                    circularity=0.0,
                    bbox=(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                    center=(float(xs.mean()), float(ys.mean())),
                )
            )
        return fallback_detections


def _apply_eval_noise(
    image_rgb: np.ndarray,
    config: dict[str, Any],
    eval_denoise: bool,
) -> np.ndarray:
    """Накладає детермінований шум перед Стадією 1 у режимі ``--eval-denoise``."""
    noise_cfg = config.get("noise_benchmark", {})
    enabled = eval_denoise or bool(noise_cfg.get("enabled", False))
    if not enabled:
        return image_rgb

    seed = int(config.get("seed", 42))
    noise_type = str(noise_cfg.get("noise_type", "gaussian")).strip().lower()
    if noise_type == "salt_pepper":
        prob = float(noise_cfg.get("salt_pepper_prob", 0.02))
        return add_salt_pepper_noise(image_rgb, prob=prob, seed=seed)

    sigma = float(noise_cfg.get("gaussian_sigma", 15.0))
    return add_gaussian_noise(image_rgb, sigma=sigma, seed=seed)


def _compute_denoise_metrics(
    reference_rgb: np.ndarray,
    denoised: np.ndarray,
) -> tuple[float, float, float]:
    """Обчислює ``(psnr, mse, ssim)``, узгоджуючи 2D-вихід Стадії 1 з RGB-еталоном."""
    proc = np.asarray(denoised)
    if proc.ndim == 2 and reference_rgb.ndim == 3:
        ref_float = reference_rgb.astype(np.float64)
        ref_gray = (
            0.299 * ref_float[..., 0]
            + 0.587 * ref_float[..., 1]
            + 0.114 * ref_float[..., 2]
        )
        ref_target: np.ndarray = ref_gray
    else:
        ref_target = reference_rgb

    mse = compute_mse(ref_target, proc)
    psnr = compute_psnr(ref_target, proc)
    ssim = compute_ssim(ref_target, proc)
    return psnr, mse, ssim


def _load_labels_lookup(labels_csv: Path | None) -> dict[str, list[str]]:
    """Завантажує словник еталонних міток ``filename/basename -> items`` із ``labels.csv``."""
    if labels_csv is None or not labels_csv.is_file():
        return {}

    lookup: dict[str, list[str]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            raw_filename = (row.get("filename") or "").strip()
            raw_items = (row.get("items_json") or "").strip()
            if not raw_filename or not raw_items:
                continue
            try:
                parsed = json.loads(raw_items)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, list):
                continue
            labels = [str(item) for item in parsed]
            norm_rel = Path(raw_filename).as_posix()
            lookup[norm_rel] = labels
            lookup[Path(raw_filename).name] = labels
    return lookup


def _detection_to_label(det: BrickDetection) -> str:
    """Форматує знайдену деталь у рядок мітки, сумісний із ``dataset/labels.csv``."""
    color = det.color.strip().lower()
    shape = det.shape_class.strip().lower()
    surface = det.surface_type.strip().lower()
    if surface in ("", "unknown") or shape.endswith(f"_{surface}"):
        return f"{color}_{shape}"
    return f"{color}_{shape}_{surface}"


def _find_ground_truth_labels(
    image_path: Path,
    labels_lookup: dict[str, list[str]],
) -> list[str] | None:
    """Шукає еталонні мітки за відносним шляхом або іменем файлу."""
    if not labels_lookup:
        return None
    posix_str = image_path.as_posix()
    for rel_key, labels in labels_lookup.items():
        if posix_str.endswith(rel_key):
            return labels
    return labels_lookup.get(image_path.name)


def _save_image_artifacts(
    image_dir: Path,
    result: PipelineResult,
    annotated_rgb: np.ndarray,
    detections: list[BrickDetection],
    save_stages: bool,
    save_annotated: bool,
) -> None:
    """Зберігає ``stage_1..3`` PNG, ``annotated.png`` та ``detections.json``."""
    image_dir.mkdir(parents=True, exist_ok=True)
    if save_stages:
        for idx, stage in enumerate(result.stages, start=1):
            fname = (
                f"{stage.name}.png"
                if stage.name.startswith("stage_")
                else f"stage_{idx}_{stage.name}.png"
            )
            _safe_save_image(stage.image, image_dir / fname)

    if save_annotated:
        _safe_save_image(annotated_rgb, image_dir / "annotated.png")

    detections_payload = [asdict(det) for det in detections]
    (image_dir / "detections.json").write_text(
        json.dumps(detections_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _aggregate_numeric(values: list[float | None]) -> dict[str, Any]:
    """Обчислює середнє, медіану та стандартне відхилення для скінченних значень."""
    non_null = [v for v in values if v is not None]
    inf_count = sum(1 for v in non_null if math.isinf(v))
    finite = [v for v in non_null if math.isfinite(v)]
    if not finite:
        return {
            "mean": None,
            "median": None,
            "std": None,
            "count": 0,
            "inf_count": inf_count,
        }
    arr = np.asarray(finite, dtype=np.float64)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "median": round(float(np.median(arr)), 6),
        "std": round(float(np.std(arr)), 6),
        "count": len(finite),
        "inf_count": inf_count,
    }


def _format_csv_float(value: float | None) -> str:
    """Форматує числове значення метрики для запису в ``metrics.csv``."""
    if value is None:
        return ""
    if math.isinf(value):
        return "inf"
    return f"{value:.6f}"


def _write_metrics_csv(
    csv_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    """Записує пофотографійні метрики всіх оброблених кадрів у ``metrics.csv``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRICS_CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_summary_json(
    summary_path: Path,
    run_config: RunConfig,
    per_pipeline_metrics: dict[str, list[RunMetrics]],
    errors: list[dict[str, str]],
) -> None:
    """Зберігає агреговану статистику (mean/median/std) і лог помилок у ``summary.json``."""
    pipelines_summary: dict[str, Any] = {}
    for pipeline_id, metrics_list in per_pipeline_metrics.items():
        pipe_errors = [err for err in errors if err["pipeline_id"] == pipeline_id]
        pipelines_summary[pipeline_id] = {
            "processed_count": len(metrics_list),
            "error_count": len(pipe_errors),
            "metrics": {
                "psnr": _aggregate_numeric([m.psnr for m in metrics_list]),
                "mse": _aggregate_numeric([m.mse for m in metrics_list]),
                "ssim": _aggregate_numeric([m.ssim for m in metrics_list]),
                "iou": _aggregate_numeric([m.iou for m in metrics_list]),
                "accuracy": _aggregate_numeric([m.accuracy for m in metrics_list]),
                "f1_score": _aggregate_numeric([m.f1_score for m in metrics_list]),
                "total_ms": _aggregate_numeric([m.total_ms for m in metrics_list]),
            },
        }

    payload = {
        "run_config": {
            "input_path": str(run_config.input_path),
            "output_dir": str(run_config.output_dir),
            "mode": run_config.mode,
            "find_query": run_config.find_query,
            "save_stages": run_config.save_stages,
            "eval_denoise": run_config.eval_denoise,
            "masks_dir": str(run_config.masks_dir) if run_config.masks_dir else None,
            "labels_csv": str(run_config.labels_csv) if run_config.labels_csv else None,
        },
        "pipelines": pipelines_summary,
        "errors": errors,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _evaluate_ground_truth(
    image_path: Path,
    predicted_mask: np.ndarray,
    detections: list[BrickDetection],
    masks_dir: Path | None,
    labels_lookup: dict[str, list[str]],
) -> dict[str, float | None]:
    """Обчислює опційні метрики ``iou``, ``accuracy``, ``precision``, ``recall``, ``f1_score``."""
    iou: float | None = None
    if masks_dir is not None:
        mask_candidate = masks_dir / f"{image_path.stem}_mask.png"
        if mask_candidate.is_file():
            gt_mask = _safe_load_mask_binary(mask_candidate)
            if gt_mask.shape == predicted_mask.shape:
                iou = compute_iou(predicted_mask, gt_mask)

    gt_labels = _find_ground_truth_labels(image_path, labels_lookup)
    if gt_labels is None:
        return {
            "iou": iou,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1_score": None,
        }

    pred_labels = [_detection_to_label(det) for det in detections]
    cls_metrics = compute_classification_metrics(pred_labels, gt_labels)
    return {
        "iou": iou,
        "accuracy": cls_metrics["accuracy"],
        "precision": cls_metrics["precision"],
        "recall": cls_metrics["recall"],
        "f1_score": cls_metrics["f1_score"],
    }


def _process_single_image(
    image_path: Path,
    pipeline: BasePipeline,
    config: dict[str, Any],
    run_config: RunConfig,
    labels_lookup: dict[str, list[str]],
) -> tuple[RunMetrics, dict[str, Any]]:
    """Обробляє один кадр, зберігає артефакти та повертає метрики і рядок CSV."""
    image_rgb = _safe_load_image_rgb(image_path, config)
    pipeline_input = _apply_eval_noise(image_rgb, config, run_config.eval_denoise)
    result = pipeline.run(pipeline_input, config)

    psnr, mse, ssim = _compute_denoise_metrics(image_rgb, result.denoised)
    detections = _safe_classify_shapes(result.mask, config, result.metadata)
    detections = classify_colors(image_rgb, result.mask, detections, config)

    if run_config.mode == "find" and run_config.find_query:
        annotated = render_search_highlight(
            image_rgb, result.mask, detections, run_config.find_query, config
        )
    else:
        annotated = render_all_detections(image_rgb, detections, config)

    out_cfg = config.get("output", {})
    _save_image_artifacts(
        image_dir=run_config.output_dir / pipeline.pipeline_id / image_path.stem,
        result=result,
        annotated_rgb=annotated,
        detections=detections,
        save_stages=run_config.save_stages,
        save_annotated=bool(out_cfg.get("save_annotated", True)),
    )

    gt_eval = _evaluate_ground_truth(
        image_path=image_path,
        predicted_mask=result.mask,
        detections=detections,
        masks_dir=run_config.masks_dir,
        labels_lookup=labels_lookup,
    )
    total_ms = float(sum(stage.duration_ms for stage in result.stages))
    run_metric = RunMetrics(
        image_name=image_path.name,
        pipeline_id=pipeline.pipeline_id,
        psnr=psnr,
        mse=mse,
        ssim=ssim,
        iou=gt_eval["iou"],
        accuracy=gt_eval["accuracy"],
        f1_score=gt_eval["f1_score"],
        total_ms=total_ms,
        detected_count=len(detections),
    )
    csv_row = {
        "image_name": image_path.name,
        "pipeline_id": pipeline.pipeline_id,
        "psnr": _format_csv_float(psnr),
        "mse": _format_csv_float(mse),
        "ssim": _format_csv_float(ssim),
        "iou": _format_csv_float(gt_eval["iou"]),
        "accuracy": _format_csv_float(gt_eval["accuracy"]),
        "precision": _format_csv_float(gt_eval["precision"]),
        "recall": _format_csv_float(gt_eval["recall"]),
        "f1_score": _format_csv_float(gt_eval["f1_score"]),
        "total_ms": _format_csv_float(total_ms),
        "detected_count": str(len(detections)),
    }
    return run_metric, csv_row


def run_pipelines(
    pipelines_with_configs: list[tuple[BasePipeline, dict[str, Any]]],
    run_config: RunConfig,
) -> dict[str, list[RunMetrics]]:
    """Запускає один або кілька пайплайнів на вибірці й зберігає зведені CSV/JSON.

    Забезпечує стійку обробку битих, порожніх або непідтримуваних файлів (критерій C2):
    винятки ``ImageLoadError`` та ``InvalidImageError`` фіксуються у ``summary.json["errors"]``
    без зупинки пакетного прогону.

    Args:
        pipelines_with_configs: Список пар ``(pipeline, merged_config)``.
        run_config: Параметри пакетного запуску.

    Returns:
        Словник ``{pipeline_id: list[RunMetrics]}``.
    """
    run_config.output_dir.mkdir(parents=True, exist_ok=True)
    labels_lookup = _load_labels_lookup(run_config.labels_csv)

    errors: list[dict[str, str]] = []
    try:
        image_paths = _safe_discover_images(run_config.input_path)
    except ImageLoadError as exc:
        image_paths = []
        for pipeline, _ in pipelines_with_configs:
            errors.append(
                {
                    "file": str(run_config.input_path),
                    "pipeline_id": pipeline.pipeline_id,
                    "error": str(exc),
                }
            )

    per_pipeline_metrics: dict[str, list[RunMetrics]] = {}
    all_csv_rows: list[dict[str, Any]] = []
    metrics_filename = "metrics.csv"
    summary_filename = "summary.json"

    for pipeline, config in pipelines_with_configs:
        out_cfg = config.get("output", {})
        metrics_filename = str(out_cfg.get("metrics_csv", metrics_filename))
        summary_filename = str(out_cfg.get("summary_json", summary_filename))

        pipe_metrics: list[RunMetrics] = []
        for image_path in image_paths:
            try:
                metric, csv_row = _process_single_image(
                    image_path=image_path,
                    pipeline=pipeline,
                    config=config,
                    run_config=run_config,
                    labels_lookup=labels_lookup,
                )
            except (ImageLoadError, InvalidImageError) as exc:
                errors.append(
                    {
                        "file": str(image_path),
                        "pipeline_id": pipeline.pipeline_id,
                        "error": str(exc),
                    }
                )
                continue
            pipe_metrics.append(metric)
            all_csv_rows.append(csv_row)
        per_pipeline_metrics[pipeline.pipeline_id] = pipe_metrics

    _write_metrics_csv(run_config.output_dir / metrics_filename, all_csv_rows)
    _write_summary_json(
        run_config.output_dir / summary_filename,
        run_config=run_config,
        per_pipeline_metrics=per_pipeline_metrics,
        errors=errors,
    )
    return per_pipeline_metrics


def run_batch(
    pipeline: BasePipeline,
    config: dict[str, Any],
    run_config: RunConfig,
) -> list[RunMetrics]:
    """Запускає один пайплайн на всіх зображеннях, зберігає стадії та метрики у CSV/JSON.

    Забезпечує:
    * стійку обробку пошкоджених файлів (критерій C2) без аварійного падіння;
    * збереження проміжних результатів кожної стадії (``stage_1..3`` PNG);
    * запуск ``shape_classifier``, ``color_classifier`` та ``visualizer``;
    * експорт ``metrics.csv`` та ``summary.json``.

    Args:
        pipeline: Екземпляр пайплайну (``p1``..``p4``).
        config: Об'єднана JSON-конфігурація.
        run_config: Параметри запуску з CLI.

    Returns:
        Список обчислених ``RunMetrics`` для кожного успішно обробленого кадру.
    """
    results = run_pipelines([(pipeline, config)], run_config)
    return results.get(pipeline.pipeline_id, [])
