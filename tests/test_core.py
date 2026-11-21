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
from lego_cv.core.base_pipeline import BasePipeline
from lego_cv.core.color_classifier import (
    binary_erode_manual,
    classify_colors,
    rgb_to_hsv_manual,
)
from lego_cv.core.registry import get_pipeline, list_pipelines
from lego_cv.core.runner import RunConfig, run_batch
from lego_cv.core.types import BrickDetection
from lego_cv.core.visualizer import (
    matches_find_query,
    render_all_detections,
    render_search_highlight,
)
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


class _DummyPipeline(BasePipeline):
    """Легкий тестовий пайплайн для перевірки оркестратора runner.py."""

    pipeline_id = "dummy"
    name = "Тестовий пайплайн"
    owner = "member2"
    stage_categories = ("Фільтрація", "Сегментація", "Морфологія")

    def stage1_preprocess(
        self,
        image_rgb: np.ndarray,
        config: dict,
    ) -> tuple[np.ndarray, dict]:
        return image_rgb.copy(), {}

    def stage2_segment(
        self,
        stage1_output: np.ndarray,
        state: dict,
        config: dict,
    ) -> tuple[np.ndarray, dict]:
        mask = (stage1_output[..., 0] > 100).astype(np.uint8) * 255
        return mask, state

    def stage3_refine(
        self,
        stage2_output: np.ndarray,
        state: dict,
        config: dict,
    ) -> tuple[np.ndarray, np.ndarray, dict]:
        inst_mask = (stage2_output > 0).astype(np.int32)
        return inst_mask, stage2_output.copy(), {}


def test_visualizer_all_and_find_modes() -> None:
    """Перевіряє розмір виходу в режимі all та затемнення фону / підсвітку в режимі find."""
    with (CONFIGS_DIR / "default.json").open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    img = np.full((60, 60, 3), 100, dtype=np.uint8)
    img[20:45, 20:45] = [20, 40, 220]  # цільова синя деталь 2x4

    mask = np.zeros((60, 60), dtype=np.int32)
    mask[20:45, 20:45] = 1
    det_blue = BrickDetection(
        brick_id=1,
        shape_class="2x4",
        surface_type="brick",
        color="Blue",
        stud_count=8,
        area_px=625.0,
        aspect_ratio=2.0,
        circularity=0.78,
        bbox=(20, 20, 44, 44),
        center=(32.0, 32.0),
    )

    assert matches_find_query(det_blue, "blue_2x4")
    assert matches_find_query(det_blue, "2x4")
    assert not matches_find_query(det_blue, "red_2x4_tile")

    all_out = render_all_detections(img, [det_blue], cfg)
    assert all_out.shape == img.shape and all_out.dtype == np.uint8
    assert np.array_equal(render_all_detections(img, [], cfg), img)

    find_out = render_search_highlight(img, mask, [det_blue], "blue_2x4", cfg)
    assert find_out.shape == img.shape and find_out.dtype == np.uint8
    # Фон у куті (0, 0) затемнений на dim_factor = 0.3 -> 100 * 0.3 = 30
    assert find_out[0, 0].tolist() == [30, 30, 30]
    # Центр цільової деталі (32, 32) зберіг оригінальну яскравість
    assert find_out[32, 32].tolist() == [20, 40, 220]

    # Порожній збіг: увесь кадр затемнюється без аварійного винятку
    no_match = render_search_highlight(img, mask, [det_blue], "red_2x2_tile", cfg)
    assert no_match[0, 0].tolist() == [30, 30, 30]
    assert no_match[32, 32].tolist() == [6, 12, 66]


def test_runner_batch_artifacts_metrics_and_corrupt_file_resilience(tmp_path: Path) -> None:
    """Перевіряє збереження stage_1..3, detections.json, metrics.csv, summary.json та C2."""
    with (CONFIGS_DIR / "default.json").open("r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    input_dir = tmp_path / "inputs"
    masks_dir = tmp_path / "masks"
    output_dir = tmp_path / "results"
    input_dir.mkdir()
    masks_dir.mkdir()

    # Валідне зображення з червоною деталлю в центрі + еталонна маска
    valid_img = np.full((20, 20, 3), 20, dtype=np.uint8)
    valid_img[5:15, 5:15] = [210, 20, 20]
    Image.fromarray(valid_img, mode="RGB").save(input_dir / "valid_01.png", format="PNG")

    gt_mask = np.zeros((20, 20), dtype=np.uint8)
    gt_mask[5:15, 5:15] = 255
    Image.fromarray(gt_mask, mode="L").save(masks_dir / "valid_01_mask.png", format="PNG")

    # Битий/порожній файл (критерій C2) не повинен зупинити прогін
    (input_dir / "broken_02.png").write_bytes(b"")

    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "filename,split,scenario,has_mask,total_count,items_json\n"
        'inputs/valid_01.png,real_u2,a,true,1,"[""red_unknown""]"\n',
        encoding="utf-8",
    )

    run_cfg = RunConfig(
        input_path=input_dir,
        output_dir=output_dir,
        mode="all",
        save_stages=True,
        eval_denoise=True,
        masks_dir=masks_dir,
        labels_csv=labels_csv,
    )
    metrics = run_batch(pipeline=_DummyPipeline(), config=cfg, run_config=run_cfg)

    assert len(metrics) == 1
    assert metrics[0].image_name == "valid_01.png"
    assert metrics[0].iou == pytest.approx(1.0, abs=1e-6)
    assert metrics[0].psnr is not None
    assert metrics[0].mse is not None
    assert metrics[0].ssim is not None

    art_dir = output_dir / "dummy" / "valid_01"
    assert (art_dir / "stage_1_preprocess.png").is_file()
    assert (art_dir / "stage_2_segment.png").is_file()
    assert (art_dir / "stage_3_refine.png").is_file()
    assert (art_dir / "annotated.png").is_file()
    assert (art_dir / "detections.json").is_file()

    metrics_csv = output_dir / "metrics.csv"
    summary_json = output_dir / "summary.json"
    assert metrics_csv.is_file() and summary_json.is_file()

    with metrics_csv.open("r", encoding="utf-8", newline="") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == 1 and csv_rows[0]["pipeline_id"] == "dummy"

    summary = json.loads(summary_json.read_text(encoding="utf-8"))
    assert summary["pipelines"]["dummy"]["processed_count"] == 1
    assert summary["pipelines"]["dummy"]["error_count"] == 1
    assert len(summary["errors"]) == 1
    assert "broken_02.png" in summary["errors"][0]["file"]


def _load_compare_pipelines_module():
    """Завантажує scripts/compare_pipelines.py без зміни sys.path."""
    script_path = REPO_ROOT / "scripts" / "compare_pipelines.py"
    spec = importlib.util.spec_from_file_location("compare_pipelines", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compare_pipelines_script_generates_tables_and_charts(tmp_path: Path) -> None:
    """Перевіряє агрегацію metrics.csv (з PSNR=inf та порожнім IoU) і генерацію 3 графіків."""
    mod = _load_compare_pipelines_module()

    metrics_csv = tmp_path / "metrics.csv"
    metrics_csv.write_text(
        "image_name,pipeline_id,psnr,mse,ssim,iou,accuracy,precision,recall,f1_score,total_ms\n"
        "img_01.png,p3,32.5,36.5,0.95,0.88,0.90,0.90,0.90,0.90,1200.0\n"
        "img_02.png,p3,inf,0.0,1.0,,,, , ,1100.0\n"
        "img_01.png,p4,30.1,63.2,0.91,0.84,0.85,0.85,0.85,0.85,1800.0\n"
        "img_02.png,p4,29.5,72.4,0.89,0.92,,,, ,900.0\n",
        encoding="utf-8",
    )
    labels_csv = tmp_path / "labels.csv"
    labels_csv.write_text(
        "filename,split,scenario,has_mask,total_count,items_json\n"
        'real_u2/img_01.png,real_u2,a,true,2,"[""red_2x4_brick""]"\n'
        'kaggle_topdown/img_02.png,kaggle_topdown,kaggle,true,1,"[""grayscale_2x4_brick""]"\n',
        encoding="utf-8",
    )

    out_dir = tmp_path / "report_out"
    md_path = mod.build_comparison_report([metrics_csv], out_dir, labels_csv=labels_csv)

    expected_files = [
        out_dir / "comparison_summary.md",
        out_dir / "comparison_summary.csv",
        out_dir / "chart_denoise_metrics.png",
        out_dir / "chart_segmentation_classification.png",
        out_dir / "chart_runtime.png",
    ]
    assert md_path == out_dir / "comparison_summary.md"
    for artifact in expected_files:
        assert artifact.is_file() and artifact.stat().st_size > 0

    with (out_dir / "comparison_summary.csv").open("r", encoding="utf-8", newline="") as fh:
        summary_rows = list(csv.DictReader(fh))
    overall_p3 = next(
        row for row in summary_rows if row["scope"] == "overall" and row["pipeline_id"] == "p3"
    )
    assert overall_p3["psnr_inf_count"] == "1"
    assert float(overall_p3["psnr_mean"]) == pytest.approx(32.5, abs=1e-6)
    assert float(overall_p3["iou_mean"]) == pytest.approx(0.88, abs=1e-6)

    md_text = md_path.read_text(encoding="utf-8")
    assert "`p3`" in md_text and "`p4`" in md_text
    assert "Сценарій `kaggle`" in md_text
