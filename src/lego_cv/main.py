"""Точка входу командного рядка (CLI) системи розпізнавання деталей LEGO."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence

from lego_cv.config import ConfigError, load_pipeline_config
from lego_cv.core.registry import get_pipeline, list_pipelines
from lego_cv.core.runner import RunConfig, run_pipelines


DEFAULT_CONFIG_PATH = Path("configs/default.json")


def build_parser() -> argparse.ArgumentParser:
    """Створює та налаштовує парсер аргументів командного рядка.

    Returns:
        Налаштований об'єкт ``argparse.ArgumentParser``.
    """
    parser = argparse.ArgumentParser(
        prog="lego-cv",
        description=(
            "Знахідник речей — класичне виявлення, класифікація та пошук "
            "деталей LEGO без використання ML/AI."
        ),
    )
    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="Вивести список доступних пайплайнів (p1..p4) та їхніх стадій і вийти.",
    )
    parser.add_argument(
        "--pipeline",
        choices=["p1", "p2", "p3", "p4", "all"],
        default="p1",
        help="Ідентифікатор пайплайну для запуску або 'all' для всіх 4 пайплайнів.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Шлях до одного PNG-файлу або каталогу із вибіркою зображень.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/run"),
        help="Каталог для збереження оброблених зображень, стадій і метрик.",
    )
    parser.add_argument(
        "--default-config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Шлях до базового JSON-конфігу (за замовчуванням configs/default.json).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Шлях до JSON-конфігу конкретного пайплайну (опційно).",
    )
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        help="Перевизначення параметра без зміни коду у форматі section.key=value.",
    )
    parser.add_argument(
        "--mode",
        choices=["all", "find"],
        default="all",
        help="Режим візуалізації: 'all' (усі деталі) або 'find' (підсвітка запиту).",
    )
    parser.add_argument(
        "--find",
        type=str,
        default=None,
        help="Пошуковий запит для --mode find (наприклад, 'blue_2x4' або '2x2_tile').",
    )
    parser.add_argument(
        "--save-stages",
        action="store_true",
        help="Зберігати проміжні зображення кожної стадії пайплайну.",
    )
    parser.add_argument(
        "--eval-denoise",
        action="store_true",
        help="Додавати детермінований тестовий шум (seed) для оцінки PSNR/MSE/SSIM.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Фіксований seed для відтворюваності стохастичних кроків (перекриває JSON).",
    )
    parser.add_argument(
        "--masks-dir",
        type=Path,
        default=None,
        help="Каталог з еталонними бінарними масками для обчислення IoU.",
    )
    parser.add_argument(
        "--labels-csv",
        type=Path,
        default=None,
        help="Шлях до dataset/labels.csv для обчислення Accuracy та F1-score.",
    )
    return parser


def _print_pipelines() -> None:
    """Друкує зведену інформацію про зареєстровані пайплайни."""
    print("Доступні пайплайни класичної обробки зображень:")
    for pipeline in list_pipelines():
        info = pipeline.describe()
        categories = " -> ".join(info["stage_categories"])
        print(
            f"  [{info['pipeline_id']}] {info['name']} "
            f"(відповідальний: {info['owner']}) | {categories}"
        )


def _resolve_pipeline_config_path(
    pipeline_id: str,
    explicit_config: Path | None,
) -> Path | None:
    """Повертає шлях до конфігу пайплайну (явний або стандартний за наявності)."""
    if explicit_config is not None:
        return explicit_config
    default_pipe_cfg = Path(f"configs/pipeline_{pipeline_id}.json")
    return default_pipe_cfg if default_pipe_cfg.is_file() else None


def main(argv: Sequence[str] | None = None) -> int:
    """Головна функція запуску CLI.

    Args:
        argv: Необов'язковий список аргументів (для тестування).

    Returns:
        Код завершення процесу (0 — успіх, 1/2 — помилка).
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_pipelines:
        _print_pipelines()
        return 0

    if args.input is None:
        parser.error("Потрібно вказати --input (або скористатися --list-pipelines).")

    if args.mode == "find" and not args.find:
        parser.error("У режимі --mode find обов'язково потрібно вказати --find.")

    pipeline_ids = ["p1", "p2", "p3", "p4"] if args.pipeline == "all" else [args.pipeline]
    cli_overrides = list(args.param)
    if args.seed is not None:
        cli_overrides.append(f"seed={args.seed}")

    try:
        pipelines_with_configs = []
        save_stages = bool(args.save_stages)
        for pipeline_id in pipeline_ids:
            pipeline = get_pipeline(pipeline_id)
            cfg_path = _resolve_pipeline_config_path(pipeline_id, args.config)
            config = load_pipeline_config(
                default_config_path=args.default_config,
                pipeline_config_path=cfg_path,
                cli_overrides=cli_overrides,
            )
            save_stages = save_stages or bool(config.get("output", {}).get("save_stages", True))
            pipelines_with_configs.append((pipeline, config))

        run_cfg = RunConfig(
            input_path=args.input,
            output_dir=args.output,
            mode=args.mode,
            find_query=args.find,
            save_stages=save_stages,
            eval_denoise=args.eval_denoise,
            masks_dir=args.masks_dir,
            labels_csv=args.labels_csv,
        )
        run_pipelines(pipelines_with_configs=pipelines_with_configs, run_config=run_cfg)
    except (ConfigError, NotImplementedError, ValueError, KeyError) as exc:
        print(f"[ПОМИЛКА] {exc}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
