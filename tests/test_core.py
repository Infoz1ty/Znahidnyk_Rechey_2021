"""Контрактні тести архітектури, конфігурації, крайніх випадків (C2) та метрик (C3)."""

from __future__ import annotations

from pathlib import Path
import pytest

from lego_cv.config import (
    ConfigError,
    apply_cli_overrides,
    deep_merge,
    load_pipeline_config,
)
from lego_cv.core.registry import get_pipeline, list_pipelines
from lego_cv.main import main as cli_main

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = REPO_ROOT / "configs"


def test_all_four_pipelines_registered_and_valid() -> None:
    """Перевіряє, що зареєстровані 4 пайплайни (по 2 на учасника) з 3+ різними категоріями."""
    pipelines = list_pipelines()
    assert [pipe.pipeline_id for pipe in pipelines] == ["p1", "p2", "p3", "p4"]

    owners = [pipe.owner for pipe in pipelines]
    assert owners.count("member1") == 2
    assert owners.count("member2") == 2

    category_sets = []
    for pipe in pipelines:
        pipe.validate_contract()
        cats = tuple(pipe.stage_categories)
        assert len(cats) >= 3
        assert len(set(cats)) == len(cats)
        category_sets.append(frozenset(cats))

    # Комбінації категорій усіх 4 пайплайнів мають бути унікальними
    assert len(set(category_sets)) == 4


def test_get_pipeline_rejects_unknown_id() -> None:
    """Перевіряє коректний виняток при запиті неіснуючого пайплайну."""
    with pytest.raises(KeyError):
        get_pipeline("p99")


def test_all_json_configs_load_and_merge_cleanly() -> None:
    """Перевіряє валідність default.json і pipeline_p1..p4.json + CLI-перевизначення."""
    default_path = CONFIGS_DIR / "default.json"
    for pid in ("p1", "p2", "p3", "p4"):
        pipe_cfg_path = CONFIGS_DIR / f"pipeline_{pid}.json"
        merged = load_pipeline_config(
            default_config_path=default_path,
            pipeline_config_path=pipe_cfg_path,
            cli_overrides=["seed=123", "preprocessing.stud_size_px=32"],
        )
        assert merged["pipeline_id"] == pid
        assert merged["seed"] == 123
        assert merged["preprocessing"]["stud_size_px"] == 32


def test_config_rejects_invalid_cli_override() -> None:
    """Перевіряє валідацію формату --param (критерій C2)."""
    with pytest.raises(ConfigError):
        apply_cli_overrides({"seed": 42}, ["broken_override_without_equals"])


def test_deep_merge_does_not_mutate_inputs() -> None:
    """Перевіряє, що злиття конфігів не мутує базовий словник."""
    base = {"a": {"x": 1, "y": 2}, "seed": 42}
    override = {"a": {"y": 99}}
    merged = deep_merge(base, override)
    assert merged == {"a": {"x": 1, "y": 99}, "seed": 42}
    assert base["a"]["y"] == 2


def test_cli_list_pipelines_succeeds(capsys: pytest.CaptureFixture[str]) -> None:
    """Перевіряє базовий запуск CLI за інструкцією з README (--list-pipelines)."""
    exit_code = cli_main(["--list-pipelines"])
    captured = capsys.readouterr()
    assert exit_code == 0
    for pid in ("[p1]", "[p2]", "[p3]", "[p4]"):
        assert pid in captured.out


@pytest.mark.skip(reason="TODO (T1.4, Учасник 1): реалізувати тести image_io для C2.")
def test_image_io_edge_cases() -> None:
    """Тести крайніх випадків: порожній/битий файл, grayscale, RGBA, розмір 3x3."""


@pytest.mark.skip(reason="TODO (T1.9, Учасник 2): реалізувати тести PSNR/MSE/SSIM/IoU.")
def test_quality_metrics_manual_values() -> None:
    """Перевірка MSE, PSNR, SSIM та IoU на малих матрицях 3x3 з ручним розрахунком."""
