"""Контрактні тести архітектури, конфігурації, крайніх випадків (C2) та метрик (C3)."""

from __future__ import annotations

import csv
import importlib.util
import json
import math
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
from lego_cv.core.color_classifier import (
    binary_erode_manual,
    classify_colors,
    rgb_to_hsv_manual,
)
from lego_cv.core.registry import get_pipeline, list_pipelines
from lego_cv.core.types import BrickDetection
from lego_cv.main import main as cli_main
from lego_cv.metrics.noise import add_gaussian_noise, add_salt_pepper_noise
from lego_cv.metrics.quality import (
    compute_classification_metrics,
    compute_iou,
    compute_mse,
    compute_psnr,
    compute_ssim,
)

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


def _make_dummy_detection(brick_id: int) -> BrickDetection:
    """Створює тестову заготовку BrickDetection для перевірки класифікатора кольору."""
    return BrickDetection(
        brick_id=brick_id,
        shape_class="2x2",
        surface_type="brick",
        color="Unknown",
        stud_count=0,
        area_px=9.0,
        aspect_ratio=1.0,
        circularity=0.78,
        bbox=(0, 0, 4, 4),
        center=(2.0, 2.0),
    )


def test_quality_metrics_manual_values() -> None:
    """Перевірка MSE, PSNR, SSIM, IoU та F1 на малих матрицях 3x3 із ручним розрахунком."""
    ref = np.array(
        [[0, 10, 20], [0, 10, 20], [0, 10, 20]],
        dtype=np.uint8,
    )
    shifted = (ref + 10).astype(np.uint8)

    # MSE == 0 -> PSNR = inf, SSIM = 1.0
    assert compute_mse(ref, ref) == 0.0
    assert math.isinf(compute_psnr(ref, ref))
    assert compute_ssim(ref, ref) == pytest.approx(1.0, abs=1e-9)

    # Ручний розрахунок: різниця +10 у кожному пікселі -> MSE = 100.0
    assert compute_mse(ref, shifted) == pytest.approx(100.0, abs=1e-9)
    expected_psnr = 10.0 * math.log10((255.0**2) / 100.0)
    assert compute_psnr(ref, shifted) == pytest.approx(expected_psnr, abs=1e-6)

    # Ручний розрахунок SSIM: var_x == var_y == cov_xy => контрастна частка == 1.0
    c1 = (0.01 * 255.0) ** 2
    expected_ssim = (2.0 * 10.0 * 20.0 + c1) / (10.0**2 + 20.0**2 + c1)
    assert compute_ssim(ref, shifted) == pytest.approx(expected_ssim, abs=1e-9)

    # IoU на 3x3 (підтримка 0/255 та інстанс-масок 0, 1, 2): перетин = 1, об'єднання = 7
    pred_instance = np.array([[1, 2, 0], [1, 2, 0], [0, 0, 0]], dtype=np.int32)
    gt_binary = np.array([[0, 0, 0], [0, 255, 255], [0, 255, 255]], dtype=np.uint8)
    assert compute_iou(pred_instance, gt_binary) == pytest.approx(1.0 / 7.0, abs=1e-9)
    assert compute_iou(np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8)) == 1.0

    # Мультимножинні Accuracy / Precision / Recall / F1 із захистом від ділення на нуль
    cls_metrics = compute_classification_metrics(
        predicted_labels=["Red_2x4", "blue_2x2", "blue_2x2"],
        ground_truth_labels=["red_2x4", "blue_2x2", "yellow_1x1"],
    )
    assert cls_metrics["precision"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert cls_metrics["recall"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert cls_metrics["f1_score"] == pytest.approx(2.0 / 3.0, abs=1e-9)
    assert cls_metrics["accuracy"] == pytest.approx(0.5, abs=1e-9)
    assert compute_classification_metrics([], ["red_2x4"])["f1_score"] == 0.0
    assert compute_classification_metrics([], [])["f1_score"] == 1.0


def test_noise_generators_determinism_and_uint8_bounds() -> None:
    """Перевіряє відтворюваність за seed і кліпінг [0..255] для гаусового та S&P шуму."""
    base = np.full((5, 5, 3), 120, dtype=np.uint8)

    g1 = add_gaussian_noise(base, sigma=25.0, seed=42)
    g2 = add_gaussian_noise(base, sigma=25.0, seed=42)
    g3 = add_gaussian_noise(base, sigma=25.0, seed=99)
    assert g1.dtype == np.uint8 and g1.shape == base.shape
    assert np.array_equal(g1, g2)
    assert not np.array_equal(g1, g3)
    assert np.array_equal(add_gaussian_noise(base, sigma=0.0, seed=42), base)

    sp1 = add_salt_pepper_noise(base, prob=0.4, seed=42)
    sp2 = add_salt_pepper_noise(base, prob=0.4, seed=42)
    assert sp1.dtype == np.uint8 and sp1.shape == base.shape
    assert np.array_equal(sp1, sp2)
    assert np.array_equal(add_salt_pepper_noise(base, prob=0.0, seed=42), base)

    sp_full = add_salt_pepper_noise(base, prob=1.0, seed=42)
    assert set(np.unique(sp_full).tolist()).issubset({0, 255})


def test_color_classifier_erosion_hsv_cyclic_red_and_grayscale() -> None:
    """Перевіряє ручну ерозію 3x3, RGB->HSV, циклічний червоний Hue та ч/б Kaggle."""
    # Ручна ерозія 3x3 на матриці 5x5: із квадрата 3x3 лишається лише центральний піксель
    mask_5x5 = np.zeros((5, 5), dtype=np.bool_)
    mask_5x5[1:4, 1:4] = True
    eroded = binary_erode_manual(mask_5x5, kernel_size=3)
    expected_eroded = np.zeros((5, 5), dtype=np.bool_)
    expected_eroded[2, 2] = True
    assert np.array_equal(eroded, expected_eroded)

    # Ручний RGB -> HSV на еталонних кольорах
    swatch = np.array(
        [[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [0, 0, 0]]],
        dtype=np.uint8,
    )
    hsv = rgb_to_hsv_manual(swatch)
    assert hsv.dtype == np.float32
    assert hsv[0, 0].tolist() == pytest.approx([0.0, 255.0, 255.0], abs=1e-3)
    assert hsv[0, 1].tolist() == pytest.approx([120.0, 255.0, 255.0], abs=1e-3)
    assert hsv[0, 2].tolist() == pytest.approx([240.0, 255.0, 255.0], abs=1e-3)
    assert hsv[0, 3].tolist() == pytest.approx([60.0, 255.0, 255.0], abs=1e-3)
    assert hsv[0, 4].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-3)

    with (CONFIGS_DIR / "default.json").open("r", encoding="utf-8") as fh:
        default_cfg = json.load(fh)

    # Циклічний червоний Hue (355° та 5°) + тінь на межі + відблиск у центрі
    red_frame = np.full((7, 7, 3), 20, dtype=np.uint8)
    red_frame[2:5, 2:5] = [220, 10, 28]  # H ≈ 355°
    red_frame[2:5, 3] = [220, 28, 10]  # H ≈ 5°
    red_frame[3, 3] = [250, 248, 248]  # відблиск (має відфільтруватися)
    red_frame[1, 1:6] = [15, 15, 15]  # темна межа (має відсіктися ерозією)
    inst_mask = np.zeros((7, 7), dtype=np.int32)
    inst_mask[1:6, 1:6] = 1
    inst_mask[0, 0] = 2  # крихітна 1x1 синя деталь (фолбек після ерозії)
    red_frame[0, 0] = [20, 60, 220]
    inst_mask[6, 6] = 3  # чорна деталь V=20 < 40 (фолбек після фільтрації тіні)

    classified = classify_colors(
        red_frame,
        inst_mask,
        [_make_dummy_detection(1), _make_dummy_detection(2), _make_dummy_detection(3)],
        default_cfg,
    )
    assert [det.color for det in classified] == ["Red", "Blue", "Black"]

    # Чорно-білий рендер Kaggle (R == G == B) -> "Grayscale" без помилок
    gray_frame = np.full((5, 5, 3), 135, dtype=np.uint8)
    gray_mask = np.ones((5, 5), dtype=np.int32)
    gray_classified = classify_colors(
        gray_frame,
        gray_mask,
        [_make_dummy_detection(1)],
        default_cfg,
    )
    assert gray_classified[0].color == "Grayscale"


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
