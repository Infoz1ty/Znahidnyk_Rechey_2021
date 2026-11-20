"""Завантаження, глибоке злиття та валідація JSON-конфігурацій проєкту."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Помилка читання, формату або валідації конфігураційного файлу."""


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
        ConfigError: Якщо рядок перевизначення має некоректний формат.
    """
    updated = deepcopy(config)
    for item in overrides:
        if "=" not in item:
            raise ConfigError(
                f"Некоректний формат параметра '{item}'. Очікується 'key=value'."
            )
        dotted_key, raw_value = item.split("=", maxsplit=1)
        keys = [part.strip() for part in dotted_key.split(".") if part.strip()]
        if not keys:
            raise ConfigError(f"Порожній ключ у параметрі '{item}'.")

        cursor: dict[str, Any] = updated
        for key in keys[:-1]:
            if key not in cursor or not isinstance(cursor[key], dict):
                cursor[key] = {}
            cursor = cursor[key]
        cursor[keys[-1]] = _parse_override_value(raw_value.strip())
    return updated


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
    if not isinstance(config.get("seed"), int):
        raise ConfigError("Параметр 'seed' у конфігурації повинен бути цілим числом.")
    return config
