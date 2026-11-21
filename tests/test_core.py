"""Контрактні тести архітектури, конфігурації, крайніх випадків (C2) та метрик (C3)."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import numpy as np
from PIL import Image
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


def _load_kaggle_mask_module():
    """Завантажує scripts/generate_kaggle_masks.py без зміни sys.path."""
    script_path = REPO_ROOT / "scripts" / "generate_kaggle_masks.py"
    spec = importlib.util.spec_from_file_location("generate_kaggle_masks", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kaggle_mask_pipeline_on_5x5_matrix_and_labels_preservation(tmp_path: Path) -> None:
    """Перевіряє ручний BFS hole-fill, фільтрацію шуму та збереження real_u1 у labels.csv."""
    mod = _load_kaggle_mask_module()

    # Синтетичний кадр 7x7: темний фон (5), квадрат 3x3 з діркою в центрі + 1 піксель шуму
    img = np.full((7, 7, 3), 5, dtype=np.uint8)
    img[2:5, 2:5] = 140
    img[3, 3] = 5  # внутрішня дірка має бути заповнена через BFS
    img[0, 6] = 180  # одиничний шумовий піксель має бути відфільтрований

    mask = mod.build_binary_mask(img, bg_tolerance=12, min_component_area=4)
    expected = np.zeros((7, 7), dtype=np.uint8)
    expected[2:5, 2:5] = 255
    assert np.array_equal(mask, expected)

    # Перевірка пласкої нормалізації, збереження маски та незмінності рядків real_u1
    dataset_dir = tmp_path / "dataset"
    kaggle_dir = dataset_dir / "kaggle_topdown"
    masks_dir = dataset_dir / "kaggle_masks"
    class_dir = kaggle_dir / "round_1x1"
    class_dir.mkdir(parents=True)
    Image.fromarray(img, mode="RGB").save(class_dir / "sample 01L.png", format="PNG")

    labels_csv = dataset_dir / "labels.csv"
    labels_csv.write_text(
        "filename,split,scenario,has_mask,total_count,items_json\n"
        'real_u1/u1_a_01_n5.png,real_u1,a,true,5,"[""red_2x4_brick""]"\n',
        encoding="utf-8",
    )

    report = mod.generate_kaggle_masks(
        kaggle_dir=kaggle_dir,
        masks_dir=masks_dir,
        bg_tolerance=12,
        min_component_area=4,
        labels_csv=labels_csv,
    )
    assert report["total_masks"] == 1
    assert report["empty_masks"] == []
    assert report["full_masks"] == []
    assert report["invalid_values"] == []
    assert (kaggle_dir / "round_1x1__sample_01L.png").is_file()
    assert (masks_dir / "round_1x1__sample_01L_mask.png").is_file()
    assert not class_dir.exists()

    with labels_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 2
    assert rows[0]["split"] == "real_u1"
    assert rows[0]["filename"] == "real_u1/u1_a_01_n5.png"
    assert rows[1]["split"] == "kaggle_topdown"
    assert rows[1]["items_json"] == '["grayscale_1x1_round"]'
