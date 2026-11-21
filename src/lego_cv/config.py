"""Завантаження, глибоке злиття та валідація JSON-конфігурацій проєкту."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Помилка читання, формату або валідації конфігураційного файлу."""


def _is_int(value: Any) -> bool:
    """Перевіряє, що значення є цілим числом, а не булевим ``bool``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    """Перевіряє, що значення є скінченним числом (``int`` або ``float``, не ``bool``)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def load_json_file(path: Path) -> dict[str, Any]:
    """Зчитує JSON-файл і перевіряє, що кореневий елемент є словником.

    Args:
        path: Шлях до JSON-файлу.

    Returns:
        Словник із параметрами.

    Raises:
        ConfigError: Якщо файл не існує, порожній, містить невалідний JSON
            або не є JSON-об'єктом.
    """
    if not path.is_file():
        raise ConfigError(f"Конфігураційний файл не знайдено: {path}")

    raw_text = path.read_text(encoding="utf-8").strip()
    if not raw_text:
        raise ConfigError(f"Конфігураційний файл порожній: {path}")

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Помилка синтаксису JSON у файлі {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Очікувався JSON-об'єкт (dict) у файлі {path}")
    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Рекурсивно об'єднує два словники без мутації оригіналів.

    Args:
        base: Базовий словник конфігурації (наприклад, ``default.json``).
        override: Словник перевизначень (наприклад, ``pipeline_p1.json``).

    Returns:
        Новий об'єднаний словник.
    """
    merged: dict[str, Any] = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_override_value(raw_value: str) -> Any:
    """Перетворює строкове значення CLI-параметра у відповідний Python-тип."""
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def apply_cli_overrides(config: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """Застосовує точкові перевизначення вигляду ``section.key=value``.

    Дозволяє налаштовувати параметри методів із командного рядка без зміни коду.

    Args:
        config: Поточний словник конфігурації.
        overrides: Список рядків формату ``key=value`` або ``a.b.c=value``.

    Returns:
        Новий словник із застосованими змінами.

    Raises:
        ConfigError: Якщо рядок перевизначення має некоректний формат або порожній сегмент ключа.
    """
    updated = deepcopy(config)
    for item in overrides:
        if not isinstance(item, str) or "=" not in item:
            raise ConfigError(
                f"Некоректний формат параметра '{item}'. Очікується 'key=value'."
            )
        dotted_key, raw_value = item.split("=", maxsplit=1)
        raw_parts = dotted_key.split(".")
        if not raw_parts or any(not part.strip() for part in raw_parts):
            raise ConfigError(f"Некоректний ключ у параметрі '{item}'.")
        stripped_val = raw_value.strip()
        if not stripped_val:
            raise ConfigError(f"Порожнє значення у параметрі '{item}'.")

        keys = [part.strip() for part in raw_parts]
        cursor: dict[str, Any] = updated
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_override_value(stripped_val)
    return updated


def _validate_preprocessing_section(prep: Any) -> None:
    """Перевіряє коректність параметрів секції ``preprocessing``."""
    if not isinstance(prep, dict):
        raise ConfigError("Секція 'preprocessing' повинна бути JSON-об'єктом.")
    min_side = prep.get("min_image_side_px", 3)
    max_allowed = prep.get("max_image_side_px", 10000)
    max_side = prep.get("max_side_px", 1600)
    stud_size = prep.get("stud_size_px", 30)

    if not _is_int(min_side) or min_side < 1:
        raise ConfigError("Параметр 'preprocessing.min_image_side_px' має бути цілим >= 1.")
    if not _is_int(max_allowed) or max_allowed < min_side:
        raise ConfigError(
            "Параметр 'preprocessing.max_image_side_px' має бути >= min_image_side_px."
        )
    if max_side is not None and (not _is_int(max_side) or max_side < min_side):
        raise ConfigError(
            "Параметр 'preprocessing.max_side_px' має бути None або цілим >= min_image_side_px."
        )
    if not _is_number(stud_size) or float(stud_size) <= 0.0:
        raise ConfigError("Параметр 'preprocessing.stud_size_px' має бути додатним числом.")


def _validate_shape_classifier_section(shape_cfg: Any) -> None:
    """Перевіряє коректність параметрів секції ``shape_classifier``."""
    if not isinstance(shape_cfg, dict):
        raise ConfigError("Секція 'shape_classifier' повинна бути JSON-об'єктом.")

    min_area = shape_cfg.get("min_area_ratio", 0.5)
    round_circ = shape_cfg.get("round_circularity_min", 0.82)
    ar_tol = shape_cfg.get("aspect_ratio_tolerance", 0.25)
    area_tol = shape_cfg.get("stud_area_tolerance", 0.35)

    if not _is_number(min_area) or float(min_area) <= 0.0:
        raise ConfigError("Параметр 'shape_classifier.min_area_ratio' має бути > 0.")
    if not _is_number(round_circ) or not 0.0 < float(round_circ) <= 1.0:
        raise ConfigError(
            "Параметр 'shape_classifier.round_circularity_min' має бути в діапазоні (0, 1]."
        )
    if not _is_number(ar_tol) or float(ar_tol) < 0.0:
        raise ConfigError("Параметр 'shape_classifier.aspect_ratio_tolerance' має бути >= 0.")
    if not _is_number(area_tol) or float(area_tol) < 0.0:
        raise ConfigError("Параметр 'shape_classifier.stud_area_tolerance' має бути >= 0.")

    if "classes" in shape_cfg:
        classes = shape_cfg["classes"]
        if not isinstance(classes, list) or not classes:
            raise ConfigError("Параметр 'shape_classifier.classes' має бути непорожнім списком.")
        for entry in classes:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str):
                raise ConfigError("Кожен клас у 'shape_classifier.classes' має містити 'name'.")
            if not _is_number(entry.get("studs_area")) or float(entry["studs_area"]) <= 0.0:
                raise ConfigError(f"Некоректна 'studs_area' для класу '{entry['name']}'.")
            if not _is_number(entry.get("aspect_ratio")) or float(entry["aspect_ratio"]) < 1.0:
                raise ConfigError(f"Некоректний 'aspect_ratio' для класу '{entry['name']}'.")
            if not isinstance(entry.get("is_round"), bool):
                raise ConfigError(f"Поле 'is_round' для класу '{entry['name']}' має бути bool.")


def validate_config(config: dict[str, Any]) -> None:
    """Валідує ключові секції та типи об'єднаної конфігурації проєкту.

    Args:
        config: Об'єднаний словник конфігурації.

    Raises:
        ConfigError: Якщо порушено обмеження типів або діапазонів параметрів.
    """
    if not _is_int(config.get("seed")):
        raise ConfigError("Параметр 'seed' у конфігурації повинен бути цілим числом.")
    if "preprocessing" in config:
        _validate_preprocessing_section(config["preprocessing"])
    if "shape_classifier" in config:
        _validate_shape_classifier_section(config["shape_classifier"])


def load_pipeline_config(
    default_config_path: Path,
    pipeline_config_path: Path | None = None,
    cli_overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Завантажує повну конфігурацію для запуску пайплайну.

    Args:
        default_config_path: Шлях до спільного ``configs/default.json``.
        pipeline_config_path: Необов'язковий шлях до конфігу конкретного пайплайну.
        cli_overrides: Список перевизначень із прапорця ``--param``.

    Returns:
        Валідований словник конфігурації.
    """
    config = load_json_file(default_config_path)
    if pipeline_config_path is not None:
        pipeline_cfg = load_json_file(pipeline_config_path)
        config = deep_merge(config, pipeline_cfg)
    if cli_overrides:
        config = apply_cli_overrides(config, cli_overrides)
    validate_config(config)
    return config
