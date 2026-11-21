"""Скрипт зведення CSV-метрик усіх пайплайнів та побудови графіків для звіту (B2, C5).

Зона реалізації (Етап 3): Учасник 2 (member2) спільно з Учасником 1.
Використовує лише стандартну бібліотеку Python, ``NumPy`` та ``matplotlib`` (бекенд ``Agg``).
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

CORE_METRICS: tuple[str, ...] = (
    "psnr", "mse", "ssim", "iou", "accuracy", "f1_score", "runtime_ms",
)
STAGE_TIME_COLUMNS: tuple[str, ...] = ("stage_1_ms", "stage_2_ms", "stage_3_ms")
PIPELINE_COLORS: tuple[str, ...] = ("#2B6CB0", "#DD6B20", "#2F855A", "#805AD5", "#C53030")


def _get_pyplot() -> Any:
    """Активує non-interactive бекенд ``Agg`` і повертає ``matplotlib.pyplot``."""
    matplotlib.use("Agg")
    return importlib.import_module("matplotlib.pyplot")


def _parse_float(raw: str | None) -> float | None:
    """Безпечно перетворює рядок CSV у ``float`` (підтримує ``inf`` та порожні комірки)."""
    if raw is None:
        return None
    text = raw.strip().lower()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        return None
    if math.isnan(value):
        return None
    return value


def _load_labels_metadata(labels_csv: Path | None) -> dict[str, tuple[str, str]]:
    """Завантажує відображення ``filename/basename -> (split, scenario)`` із ``labels.csv``."""
    if labels_csv is None or not labels_csv.is_file():
        return {}

    lookup: dict[str, tuple[str, str]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            raw_fname = (row.get("filename") or "").strip()
            if not raw_fname:
                continue
            split = (row.get("split") or "unknown").strip() or "unknown"
            scenario = (row.get("scenario") or "unknown").strip() or "unknown"
            posix_name = Path(raw_fname).as_posix()
            lookup[posix_name] = (split, scenario)
            lookup[Path(raw_fname).name] = (split, scenario)
    return lookup


def load_metrics_rows(
    metrics_csv_paths: Sequence[Path],
    labels_csv: Path | None = None,
) -> list[dict[str, Any]]:
    """Зчитує та об'єднує рядки з одного або кількох файлів ``metrics.csv``.

    Args:
        metrics_csv_paths: Шляхи до файлів ``metrics.csv``, згенерованих ``runner.py``.
        labels_csv: Необов'язковий шлях до ``dataset/labels.csv`` для додавання ``split/scenario``.

    Returns:
        Список нормалізованих словників метрик по кожному кадру.

    Raises:
        FileNotFoundError: Якщо один із переданих ``metrics.csv`` не існує.
    """
    labels_meta = _load_labels_metadata(labels_csv)
    records: list[dict[str, Any]] = []

    for csv_path in metrics_csv_paths:
        if not csv_path.is_file():
            raise FileNotFoundError(f"Файл метрик не знайдено: {csv_path}")
        with csv_path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                pipeline_id = (row.get("pipeline_id") or "").strip()
                image_name = (row.get("image_name") or "").strip()
                if not pipeline_id:
                    continue
                split_val, scen_val = labels_meta.get(
                    Path(image_name).as_posix(),
                    labels_meta.get(Path(image_name).name, ("all", "all")),
                )
                split = (row.get("split") or "").strip() or split_val
                scenario = (row.get("scenario") or "").strip() or scen_val

                runtime_val = _parse_float(row.get("total_ms"))
                if runtime_val is None:
                    runtime_val = _parse_float(row.get("runtime_ms"))

                record: dict[str, Any] = {
                    "image_name": image_name,
                    "pipeline_id": pipeline_id,
                    "split": split,
                    "scenario": scenario,
                    "psnr": _parse_float(row.get("psnr")),
                    "mse": _parse_float(row.get("mse")),
                    "ssim": _parse_float(row.get("ssim")),
                    "iou": _parse_float(row.get("iou")),
                    "accuracy": _parse_float(row.get("accuracy")),
                    "precision": _parse_float(row.get("precision")),
                    "recall": _parse_float(row.get("recall")),
                    "f1_score": _parse_float(row.get("f1_score")),
                    "runtime_ms": runtime_val,
                }
                for st_col in STAGE_TIME_COLUMNS:
                    record[st_col] = _parse_float(row.get(st_col))
                records.append(record)
    return records


def aggregate_metric_values(values: Sequence[float | None]) -> dict[str, Any]:
    """Обчислює середнє, медіану та ``std`` по скінченних значеннях, окремо рахуючи ``inf``.

    Args:
        values: Послідовність значень метрики (може містити ``None`` та ``float('inf')``).

    Returns:
        Словник із ключами ``mean``, ``median``, ``std``, ``count``, ``inf_count``.
    """
    inf_count = sum(1 for val in values if val is not None and math.isinf(val))
    finite = [val for val in values if val is not None and math.isfinite(val)]
    if finite:
        vec = np.asarray(finite, dtype=np.float64)
        mean_v, med_v, std_v = float(np.mean(vec)), float(np.median(vec)), float(np.std(vec))
    else:
        mean_v, med_v, std_v = None, None, None
    return {
        "count": len(finite),
        "inf_count": inf_count,
        "mean": mean_v,
        "median": med_v,
        "std": std_v,
    }


def _aggregate_subset(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Агрегує всі основні та стадійні метрики для однієї групи рядків."""
    summary: dict[str, Any] = {"n_images": len(rows)}
    for key in CORE_METRICS + STAGE_TIME_COLUMNS:
        summary[key] = aggregate_metric_values([row.get(key) for row in rows])
    return summary


def aggregate_metrics(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Групує та агрегує метрики по пайплайнах загалом, а також за сценаріями й сплітами.

    Args:
        rows: Нормалізовані рядки з ``load_metrics_rows``.

    Returns:
        Словник із ключами ``pipelines``, ``overall``, ``by_scenario``, ``by_split``.
    """
    pipelines = sorted({str(row["pipeline_id"]) for row in rows})
    overall: dict[str, dict[str, Any]] = {}
    for pid in pipelines:
        pipe_rows = [row for row in rows if row["pipeline_id"] == pid]
        overall[pid] = _aggregate_subset(pipe_rows)

    scenarios = sorted({str(row["scenario"]) for row in rows if row.get("scenario") != "all"})
    by_scenario: dict[str, dict[str, dict[str, Any]]] = {}
    for scen in scenarios:
        by_scenario[scen] = {}
        for pid in pipelines:
            subset = [
                row for row in rows if row["pipeline_id"] == pid and row["scenario"] == scen
            ]
            if subset:
                by_scenario[scen][pid] = _aggregate_subset(subset)

    splits = sorted({str(row["split"]) for row in rows if row.get("split") != "all"})
    by_split: dict[str, dict[str, dict[str, Any]]] = {}
    for split in splits:
        by_split[split] = {}
        for pid in pipelines:
            subset = [row for row in rows if row["pipeline_id"] == pid and row["split"] == split]
            if subset:
                by_split[split][pid] = _aggregate_subset(subset)

    return {
        "pipelines": pipelines,
        "overall": overall,
        "by_scenario": by_scenario,
        "by_split": by_split,
    }


def _fmt_csv(value: float | None) -> str:
    """Форматує числове значення для ``comparison_summary.csv``."""
    return "" if value is None else f"{value:.6f}"


def _build_csv_row(
    scope: str,
    group: str,
    pipeline_id: str,
    stats: dict[str, Any],
) -> dict[str, str]:
    """Формує один плоский рядок для експорту у ``comparison_summary.csv``."""
    row: dict[str, str] = {
        "scope": scope,
        "group": group,
        "pipeline_id": pipeline_id,
        "n_images": str(stats["n_images"]),
        "psnr_inf_count": str(stats["psnr"]["inf_count"]),
    }
    for metric in CORE_METRICS:
        m_stats = stats[metric]
        row[f"{metric}_mean"] = _fmt_csv(m_stats["mean"])
        row[f"{metric}_median"] = _fmt_csv(m_stats["median"])
        row[f"{metric}_std"] = _fmt_csv(m_stats["std"])
    return row


def write_summary_csv(aggregated: dict[str, Any], csv_path: Path) -> Path:
    """Записує зведену таблицю метрик у ``comparison_summary.csv``.

    Args:
        aggregated: Результат ``aggregate_metrics``.
        csv_path: Цільовий шлях до CSV-файлу.

    Returns:
        Шлях до збереженого файлу.
    """
    fieldnames = ["scope", "group", "pipeline_id", "n_images", "psnr_inf_count"]
    for metric in CORE_METRICS:
        fieldnames.extend([f"{metric}_mean", f"{metric}_median", f"{metric}_std"])

    rows_out: list[dict[str, str]] = []
    for pid, stats in aggregated["overall"].items():
        rows_out.append(_build_csv_row("overall", "all", pid, stats))
    for scen, pipe_map in aggregated["by_scenario"].items():
        for pid, stats in pipe_map.items():
            rows_out.append(_build_csv_row("scenario", scen, pid, stats))
    for split, pipe_map in aggregated["by_split"].items():
        for pid, stats in pipe_map.items():
            rows_out.append(_build_csv_row("split", split, pid, stats))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_out)
    return csv_path


def _fmt_md_stat(stat: dict[str, Any], precision: int = 3) -> str:
    """Форматує статистику ``mean ± std (med)`` для Markdown-таблиці."""
    mean_v = stat["mean"]
    if mean_v is None:
        if stat["inf_count"] > 0:
            return f"inf ({stat['inf_count']}x)"
        return "—"
    base = (
        f"{mean_v:.{precision}f} ± {stat['std']:.{precision}f} "
        f"(med {stat['median']:.{precision}f})"
    )
    if stat["inf_count"] > 0:
        return f"{base} [+{stat['inf_count']} inf]"
    return base


def _render_md_table(pipe_map: dict[str, dict[str, Any]]) -> list[str]:
    """Генерує рядки Markdown-таблиці для одного зрізу (загалом або сценарій/спліт)."""
    lines = [
        "| Пайплайн | Кадрів | PSNR (дБ) | MSE | SSIM | IoU | Accuracy | F1 | Час (мс) |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pid, stats in pipe_map.items():
        lines.append(
            f"| `{pid}` | {stats['n_images']} "
            f"| {_fmt_md_stat(stats['psnr'], 2)} "
            f"| {_fmt_md_stat(stats['mse'], 2)} "
            f"| {_fmt_md_stat(stats['ssim'], 3)} "
            f"| {_fmt_md_stat(stats['iou'], 3)} "
            f"| {_fmt_md_stat(stats['accuracy'], 3)} "
            f"| {_fmt_md_stat(stats['f1_score'], 3)} "
            f"| {_fmt_md_stat(stats['runtime_ms'], 1)} |"
        )
    return lines


def write_summary_markdown(aggregated: dict[str, Any], md_path: Path) -> Path:
    """Формує зведену Markdown-таблицю ``comparison_summary.md`` для фінального звіту.

    Args:
        aggregated: Результат ``aggregate_metrics``.
        md_path: Цільовий шлях до Markdown-файлу.

    Returns:
        Шлях до збереженого файлу.
    """
    lines: list[str] = [
        "# Зведене порівняння пайплайнів класичної обробки зображень",
        "",
        "## 1. Загальні метрики по пайплайнах",
        "",
    ]
    if aggregated["overall"]:
        lines.extend(_render_md_table(aggregated["overall"]))
    else:
        lines.append("_Дані відсутні._")

    if aggregated["by_scenario"]:
        lines.extend(["", "## 2. Розріз за сценаріями", ""])
        for scen, pipe_map in aggregated["by_scenario"].items():
            lines.extend([f"### Сценарій `{scen}`", ""])
            lines.extend(_render_md_table(pipe_map))
            lines.append("")

    if aggregated["by_split"]:
        lines.extend(["", "## 3. Розріз за сплітами датасету", ""])
        for split, pipe_map in aggregated["by_split"].items():
            lines.extend([f"### Спліт `{split}`", ""])
            lines.extend(_render_md_table(pipe_map))
            lines.append("")

    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return md_path


def _plot_metric_bars(
    axis: Any,
    pipelines: list[str],
    overall: dict[str, dict[str, Any]],
    metric_key: str,
    title: str,
) -> None:
    """Будує стовпчикову діаграму з ``± std`` для однієї метрики на заданій осі."""
    x_pos = np.arange(len(pipelines), dtype=np.float64)
    means: list[float] = []
    stds: list[float] = []
    labels: list[str] = []

    for pid in pipelines:
        stat = overall[pid][metric_key]
        if stat["mean"] is not None:
            means.append(float(stat["mean"]))
            stds.append(float(stat["std"] or 0.0))
            suffix = "*" if stat["inf_count"] > 0 else ""
            labels.append(f"{stat['mean']:.2f}{suffix}")
        else:
            means.append(0.0)
            stds.append(0.0)
            labels.append("inf" if stat["inf_count"] > 0 else "N/A")

    colors = [PIPELINE_COLORS[i % len(PIPELINE_COLORS)] for i in range(len(pipelines))]
    bars = axis.bar(x_pos, means, yerr=stds, capsize=4, color=colors, alpha=0.88)
    axis.set_title(title)
    axis.set_xticks(x_pos)
    axis.set_xticklabels(pipelines)
    axis.grid(axis="y", linestyle="--", alpha=0.35)

    y_pad = max(means) * 0.03 if means and max(means) > 0.0 else 0.05
    for bar_rect, std_val, txt in zip(bars, stds, labels):
        top = float(bar_rect.get_height()) + std_val + y_pad
        axis.text(
            bar_rect.get_x() + bar_rect.get_width() / 2.0,
            top,
            txt,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_denoise_metrics(aggregated: dict[str, Any], out_path: Path) -> Path:
    """Будує графік ``chart_denoise_metrics.png`` (PSNR, MSE, SSIM по пайплайнах).

    Args:
        aggregated: Результат ``aggregate_metrics``.
        out_path: Шлях до вихідного PNG-файлу.

    Returns:
        Шлях до збереженого графіка.
    """
    plt = _get_pyplot()
    pipelines = aggregated["pipelines"]
    overall = aggregated["overall"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    specs = (
        ("psnr", "PSNR (дБ, вище — краще)"),
        ("mse", "MSE (нижче — краще)"),
        ("ssim", "SSIM (вище — краще)"),
    )
    if pipelines:
        for axis, (key, title) in zip(axes, specs):
            _plot_metric_bars(axis, pipelines, overall, key, title)
    else:
        for axis, (_, title) in zip(axes, specs):
            axis.set_title(title)
            axis.text(0.5, 0.5, "Немає даних", ha="center", va="center")

    fig.suptitle("Порівняння якості предобробки / шумоподавлення (Стадія 1)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _plot_grouped_seg_cls(
    axis: Any,
    pipelines: list[str],
    overall: dict[str, dict[str, Any]],
) -> None:
    """Будує згруповані стовпчики IoU / Accuracy / F1-score по пайплайнах."""
    metrics = (("iou", "IoU"), ("accuracy", "Accuracy"), ("f1_score", "F1-score"))
    x_pos = np.arange(len(pipelines), dtype=np.float64)
    bar_w = 0.25

    for idx, (key, label) in enumerate(metrics):
        vals = [
            float(overall[pid][key]["mean"]) if overall[pid][key]["mean"] is not None else 0.0
            for pid in pipelines
        ]
        offset = (idx - 1) * bar_w
        axis.bar(
            x_pos + offset,
            vals,
            width=bar_w,
            label=label,
            color=PIPELINE_COLORS[idx],
            alpha=0.88,
        )

    axis.set_title("Сегментація та класифікація (загалом)")
    axis.set_xticks(x_pos)
    axis.set_xticklabels(pipelines)
    axis.set_ylim(0.0, 1.1)
    axis.grid(axis="y", linestyle="--", alpha=0.35)
    axis.legend()


def plot_segmentation_classification(aggregated: dict[str, Any], out_path: Path) -> Path:
    """Будує графік ``chart_segmentation_classification.png`` (IoU, Accuracy, F1 + сценарії).

    Args:
        aggregated: Результат ``aggregate_metrics``.
        out_path: Шлях до вихідного PNG-файлу.

    Returns:
        Шлях до збереженого графіка.
    """
    plt = _get_pyplot()
    pipelines = aggregated["pipelines"]
    by_scenario = aggregated["by_scenario"]
    has_scenarios = len(by_scenario) > 1

    if has_scenarios and pipelines:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(13, 4.5))
        _plot_grouped_seg_cls(ax_left, pipelines, aggregated["overall"])

        scenarios = list(by_scenario.keys())
        x_pos = np.arange(len(scenarios), dtype=np.float64)
        bar_w = 0.75 / max(1, len(pipelines))
        for idx, pid in enumerate(pipelines):
            iou_vals = [
                float(by_scenario[sc][pid]["iou"]["mean"])
                if pid in by_scenario[sc] and by_scenario[sc][pid]["iou"]["mean"] is not None
                else 0.0
                for sc in scenarios
            ]
            shift = (idx - (len(pipelines) - 1) / 2.0) * bar_w
            ax_right.bar(
                x_pos + shift,
                iou_vals,
                width=bar_w,
                label=pid,
                color=PIPELINE_COLORS[idx % len(PIPELINE_COLORS)],
                alpha=0.88,
            )
        ax_right.set_title("IoU у розрізі сценаріїв")
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(scenarios)
        ax_right.set_ylim(0.0, 1.1)
        ax_right.grid(axis="y", linestyle="--", alpha=0.35)
        ax_right.legend()
    else:
        fig, axis = plt.subplots(1, 1, figsize=(7.5, 4.5))
        if pipelines:
            _plot_grouped_seg_cls(axis, pipelines, aggregated["overall"])
        else:
            axis.set_title("Сегментація та класифікація")
            axis.text(0.5, 0.5, "Немає даних", ha="center", va="center")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_runtime(aggregated: dict[str, Any], out_path: Path) -> Path:
    """Будує графік ``chart_runtime.png`` (середній час виконання загалом та по стадіях).

    Args:
        aggregated: Результат ``aggregate_metrics``.
        out_path: Шлях до вихідного PNG-файлу.

    Returns:
        Шлях до збереженого графіка.
    """
    plt = _get_pyplot()
    pipelines = aggregated["pipelines"]
    overall = aggregated["overall"]
    has_stages = any(
        overall[pid][col]["mean"] is not None
        for pid in pipelines
        for col in STAGE_TIME_COLUMNS
    )

    fig, axis = plt.subplots(1, 1, figsize=(8.0, 4.5))
    if not pipelines:
        axis.set_title("Середній час виконання (мс)")
        axis.text(0.5, 0.5, "Немає даних", ha="center", va="center")
    elif has_stages:
        x_pos = np.arange(len(pipelines), dtype=np.float64)
        bottom = np.zeros(len(pipelines), dtype=np.float64)
        for idx, col in enumerate(STAGE_TIME_COLUMNS):
            vals = np.array(
                [
                    float(overall[pid][col]["mean"])
                    if overall[pid][col]["mean"] is not None
                    else 0.0
                    for pid in pipelines
                ],
                dtype=np.float64,
            )
            axis.bar(
                x_pos,
                vals,
                bottom=bottom,
                label=f"Стадія {idx + 1}",
                color=PIPELINE_COLORS[idx],
                alpha=0.88,
            )
            bottom += vals
        axis.set_title("Середній час виконання по стадіях (мс)")
        axis.set_xticks(x_pos)
        axis.set_xticklabels(pipelines)
        axis.set_ylabel("Час (мс)")
        axis.grid(axis="y", linestyle="--", alpha=0.35)
        axis.legend()
    else:
        _plot_metric_bars(axis, pipelines, overall, "runtime_ms", "Середній час виконання (мс)")
        axis.set_ylabel("Час (мс)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def build_comparison_report(
    metrics_csv_paths: list[Path],
    output_dir: Path,
    labels_csv: Path | None = None,
) -> Path:
    """Агрегує метрики ``PSNR/MSE/SSIM/IoU/Accuracy/F1/runtime_ms`` і генерує графіки звіту.

    Створює в ``output_dir`` файли ``comparison_summary.csv``, ``comparison_summary.md``,
    ``chart_denoise_metrics.png``, ``chart_segmentation_classification.png`` та
    ``chart_runtime.png``.

    Args:
        metrics_csv_paths: Список шляхів до ``metrics.csv`` із прогонів пайплайнів.
        output_dir: Каталог збереження таблиць та графіків.
        labels_csv: Необов'язковий шлях до ``dataset/labels.csv`` для групування за сценаріями.

    Returns:
        Шлях до згенерованого файлу ``comparison_summary.md``.
    """
    rows = load_metrics_rows(metrics_csv_paths, labels_csv=labels_csv)
    aggregated = aggregate_metrics(rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(aggregated, output_dir / "comparison_summary.csv")
    md_path = write_summary_markdown(aggregated, output_dir / "comparison_summary.md")
    plot_denoise_metrics(aggregated, output_dir / "chart_denoise_metrics.png")
    plot_segmentation_classification(
        aggregated, output_dir / "chart_segmentation_classification.png"
    )
    plot_runtime(aggregated, output_dir / "chart_runtime.png")
    return md_path


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входу CLI для порівняльного аналізу результатів П1–П4."""
    parser = argparse.ArgumentParser(description="Порівняння метрик пайплайнів П1–П4.")
    parser.add_argument(
        "--metrics",
        type=Path,
        nargs="+",
        required=True,
        help="Один або кілька шляхів до metrics.csv із прогонів runner.py.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Необов'язковий шлях до dataset/labels.csv для групування за сценаріями/сплітами.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/report/generated"),
        help="Каталог для збереження таблиць і PNG-графіків звіту.",
    )
    args = parser.parse_args(argv)
    md_path = build_comparison_report(
        args.metrics,
        args.output,
        labels_csv=args.labels_csv,
    )
    print(f"Звіт порівняння збережено у: {md_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
