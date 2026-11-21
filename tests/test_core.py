"""Контрактні тести архітектури, конфігурації, крайніх випадків (C2) та метрик (C3)."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from lego_cv.config import (
    ConfigError,
    apply_cli_overrides,
    deep_merge,
    load_json_file,
    load_pipeline_config,
    validate_config,
)
from lego_cv.core.image_io import (
    ImageLoadError,
    InvalidImageError,
    discover_images,
    load_image_rgb,
    load_mask_binary,
    save_image,
)
from lego_cv.core.registry import get_pipeline, list_pipelines
from lego_cv.core.shape_classifier import (
    classify_shapes,
    compute_component_geometry,
    connected_components_label,
)
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


def test_image_io_edge_cases(tmp_path: Path) -> None:
    """Тести крайніх випадків (C2): відсутній/порожній/битий файл, L, RGBA, 3x3, маски."""
    # 1. Відсутній, порожній, пошкоджений та непідтримуваний файли -> ImageLoadError
    with pytest.raises(ImageLoadError):
        load_image_rgb(tmp_path / "missing.png")

    empty_file = tmp_path / "empty.png"
    empty_file.write_bytes(b"")
    with pytest.raises(ImageLoadError):
        load_image_rgb(empty_file)

    corrupt_file = tmp_path / "corrupt.png"
    corrupt_file.write_bytes(b"not-a-valid-png-header")
    with pytest.raises(ImageLoadError):
        load_image_rgb(corrupt_file)

    unsupported = tmp_path / "notes.txt"
    unsupported.write_text("hello", encoding="utf-8")
    with pytest.raises(ImageLoadError):
        load_image_rgb(unsupported)

    # 2. Монохромне (L) 3x3 та RGBA 4x4 зображення -> RGB uint8
    gray_3x3 = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]], dtype=np.uint8)
    gray_path = tmp_path / "gray_3x3.png"
    Image.fromarray(gray_3x3, mode="L").save(gray_path)
    loaded_gray = load_image_rgb(gray_path, max_side_px=None)
    assert loaded_gray.shape == (3, 3, 3)
    assert loaded_gray.dtype == np.uint8
    assert np.array_equal(loaded_gray[:, :, 0], gray_3x3)
    assert np.array_equal(loaded_gray[:, :, 1], gray_3x3)

    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[:2, :, :] = [200, 10, 20, 0]  # повна прозорість -> білий фон (255, 255, 255)
    rgba[2:, :, :] = [10, 120, 240, 255]  # непрозора ділянка
    rgba_path = tmp_path / "rgba_4x4.png"
    Image.fromarray(rgba, mode="RGBA").save(rgba_path)
    loaded_rgba = load_image_rgb(rgba_path, max_side_px=None)
    assert loaded_rgba.shape == (4, 4, 3)
    assert tuple(loaded_rgba[0, 0]) == (255, 255, 255)
    assert tuple(loaded_rgba[3, 3]) == (10, 120, 240)

    # 4. Межі розмірів та пропорційний ресайз
    tiny_path = tmp_path / "tiny_2x2.png"
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8), mode="RGB").save(tiny_path)
    with pytest.raises(InvalidImageError):
        load_image_rgb(tiny_path, min_side_px=3)

    wide_path = tmp_path / "wide_20x10.png"
    Image.fromarray(np.full((10, 20, 3), 100, dtype=np.uint8), mode="RGB").save(wide_path)
    with pytest.raises(InvalidImageError):
        load_image_rgb(wide_path, max_allowed_side_px=15)
    resized = load_image_rgb(wide_path, max_side_px=10)
    assert resized.shape == (5, 10, 3)


def test_image_io_mask_save_and_discover(tmp_path: Path) -> None:
    """Перевіряє завантаження RGBA-масок, збереження зображень та пошук у директорії."""
    mask_rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    mask_rgba[1:3, 1:3] = [220, 220, 220, 255]
    mask_rgba[0, 0] = [255, 255, 255, 0]  # прозорий піксель -> фон 0
    mask_path = tmp_path / "rgba_mask.png"
    Image.fromarray(mask_rgba, mode="RGBA").save(mask_path)
    loaded_mask = load_mask_binary(mask_path, target_size=(8, 8))
    assert loaded_mask.shape == (8, 8)
    assert set(np.unique(loaded_mask).tolist()).issubset({0, 1})
    assert loaded_mask[0, 0] == 0
    assert loaded_mask[3, 3] == 1

    # Перевірка формату маски {0, 1}
    mask_01 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    mask_01_path = tmp_path / "mask_01.png"
    Image.fromarray(mask_01, mode="L").save(mask_01_path)
    assert np.array_equal(load_mask_binary(mask_01_path), mask_01)

    rgb = np.full((4, 4, 3), 120, dtype=np.uint8)
    saved_path = tmp_path / "nested" / "saved.png"
    save_image(rgb, saved_path)
    assert saved_path.is_file()
    with pytest.raises(InvalidImageError):
        save_image(np.zeros((4, 4, 4), dtype=np.uint8), tmp_path / "bad.png")

    ignored_txt = tmp_path / "notes.txt"
    ignored_txt.write_text("ignored", encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")
    discovered = discover_images(tmp_path)
    assert all(p.suffix.lower() == ".png" for p in discovered)
    assert ignored_txt not in discovered
    with pytest.raises(ImageLoadError):
        discover_images(tmp_path / "does_not_exist")


def test_config_validation_edge_cases() -> None:
    """Перевіряє посилену валідацію config.py (bool seed, межі розмірів, порожні сегменти)."""
    with pytest.raises(ConfigError):
        validate_config({"seed": True})
    with pytest.raises(ConfigError):
        validate_config({"seed": 42, "preprocessing": {"stud_size_px": 0}})
    with pytest.raises(ConfigError):
        validate_config(
            {"seed": 42, "preprocessing": {"min_image_side_px": 10, "max_image_side_px": 5}}
        )
    with pytest.raises(ConfigError):
        apply_cli_overrides({"seed": 42}, ["a..b=1"])
    with pytest.raises(ConfigError):
        apply_cli_overrides({"seed": 42}, ["seed="])


def test_connected_components_label_manual_5x5() -> None:
    """Ручна перевірка 8-зв'язного Union-Find CCL на матриці 5x5 та порожній масці."""
    matrix = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    labels, num_labels = connected_components_label(matrix)
    assert num_labels == 2
    assert labels.dtype == np.int32
    # (0,0) і (1,1) діагонально 8-зв'язні -> один компонент 1
    assert labels[0, 0] == 1 and labels[1, 1] == 1
    # Квадрат 2x2 у правому нижньому куті -> компонент 2
    assert np.all(labels[3:5, 3:5] == 2)
    assert labels[2, 2] == 0

    empty_labels, empty_count = connected_components_label(np.zeros((4, 4), dtype=np.uint8))
    assert empty_count == 0
    assert np.all(empty_labels == 0)


def test_shape_geometry_square_rect_and_circle() -> None:
    """Перевіряє площу, периметр, окружність і PCA AR на квадраті, 1x4 (0°/45°) та колі."""
    square = np.zeros((40, 40), dtype=np.uint8)
    square[5:35, 5:35] = 1
    g_sq = compute_component_geometry(square)
    assert g_sq["area_px"] == 900.0
    assert g_sq["perimeter_px"] == pytest.approx(120.0)
    assert g_sq["circularity"] == pytest.approx(math.pi / 4.0, abs=1e-4)
    assert g_sq["aspect_ratio"] == pytest.approx(1.0, abs=1e-4)

    rect = np.zeros((60, 60), dtype=np.uint8)
    rect[10:20, 10:50] = 1
    g_rect = compute_component_geometry(rect)
    assert g_rect["area_px"] == 400.0
    assert g_rect["aspect_ratio"] == pytest.approx(4.0, abs=1e-4)

    # Повернутий на 45° прямокутник 1x4 (піврозміри 19.5 та 4.5 -> рівно 40x10 px)
    yy, xx = np.mgrid[:80, :80]
    cos45 = math.sqrt(0.5)
    u = (xx - 40.0) * cos45 + (yy - 40.0) * cos45
    v = -(xx - 40.0) * cos45 + (yy - 40.0) * cos45
    rot_rect = ((np.abs(u) <= 19.5) & (np.abs(v) <= 4.5)).astype(np.uint8)
    g_rot = compute_component_geometry(rot_rect)
    assert g_rot["aspect_ratio"] == pytest.approx(4.0, abs=0.2)

    cy, cx = np.ogrid[:40, :40]
    circle = ((cy - 20) ** 2 + (cx - 20) ** 2 <= 15**2).astype(np.uint8)
    g_circ = compute_component_geometry(circle)
    assert g_circ["circularity"] >= 0.82
    assert g_circ["aspect_ratio"] == pytest.approx(1.0, abs=0.05)

    g_empty = compute_component_geometry(np.zeros((5, 5), dtype=np.uint8))
    assert g_empty["area_px"] == 0.0
    assert g_empty["circularity"] == 0.0


def test_classify_shapes_filters_noise_and_supports_instance_masks() -> None:
    """Перевіряє фільтрацію сміття, класифікацію 2x2 / 1x4 / 1x1_round та інстанс-маски."""
    cfg = load_json_file(CONFIGS_DIR / "default.json")  # stud_size_px = 30, unit_area = 900

    canvas = np.zeros((160, 240), dtype=np.uint8)
    canvas[5:15, 5:15] = 1  # сміття 100 px (< 450 px) — має бути відфільтровано
    canvas[20:80, 20:80] = 1  # квадрат 60x60 -> 2x2
    canvas[20:50, 100:220] = 1  # прямокутник 30x120 -> 1x4
    yy, xx = np.ogrid[:160, :240]
    canvas[(yy - 125) ** 2 + (xx - 50) ** 2 <= 15**2] = 1  # коло r=15 -> 1x1_round

    dets = classify_shapes(canvas, cfg)
    classes = {det.shape_class for det in dets}
    assert len(dets) == 3
    assert classes == {"2x2", "1x4", "1x1_round"}
    round_det = next(det for det in dets if det.shape_class == "1x1_round")
    assert round_det.surface_type == "round"

    # Інстанс-маска (2 суміжні деталі з різними ID, як після П3 Watershed) + метадані П4
    inst_mask = np.zeros((100, 160), dtype=np.int32)
    inst_mask[20:80, 15:75] = 1
    inst_mask[20:80, 75:135] = 2
    inst_dets = classify_shapes(
        inst_mask,
        cfg,
        pipeline_metadata={"stud_counts": {1: 4, 2: 0}},
    )
    assert len(inst_dets) == 2
    by_id = {det.brick_id: det for det in inst_dets}
    assert by_id[1].shape_class == "2x2" and by_id[1].surface_type == "brick"
    assert by_id[1].stud_count == 4
    assert by_id[2].shape_class == "2x2" and by_id[2].surface_type == "tile"
    assert by_id[2].stud_count == 0


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
