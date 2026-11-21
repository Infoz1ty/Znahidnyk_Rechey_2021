# Пост-рев'ю: T1.4 — Автотести ядра Учасника 1 (`tests/test_core.py`)

## Завдання та змінені файли
* **Таск із `docs/tasks.md`:** `T1.4` (`tests/test_core.py` — критерії C2 і C3).
* **Змінені файли:**
  * `tests/test_core.py`
  * `docs/tasks.md` (позначено `T1.4` виконаним)
  * `docs/reviews/task_T1.4_core_tests_review.md`

## Що зроблено та як працює логіка
* Знято `@pytest.mark.skip` із `test_image_io_edge_cases` і додано 5 цільових тест-кейсів для всієї зони Учасника 1 в ядрі (без зміни тесту Учасника 2 `T1.9`):
  * `test_image_io_edge_cases` та `test_image_io_mask_save_and_discover`: перевіряють відсутні, порожні (0 байт), пошкоджені й непідтримувані файли (`ImageLoadError`), монохромні `L 3×3`, альфа-композитинг `RGBA 4×4`, межі розміру (`InvalidImageError`), пропорційний ресайз, `RGBA` та `{0, 1}` маски, `save_image` і `discover_images`.
  * `test_config_validation_edge_cases`: перевіряє відхилення `seed=True` (`bool`), нульового `stud_size_px`, інвертованих меж розміру й некоректних CLI-перевизначень (`a..b=1`, `seed=`).
  * `test_connected_components_label_manual_5x5`: перевіряє 8-зв'язний Union-Find CCL на ручній матриці `5×5` (включно з діагональною зв'язністю та порожньою маскою).
  * `test_shape_geometry_square_rect_and_circle`: перевіряє точні аналітичні значення (`area`, `perimeter`, `circularity = pi/4`, `aspect_ratio`) на квадраті `30×30`, прямокутнику `1×4` (при `0°` і повороті `45°`), колі `r=15` та порожній масці.
  * `test_classify_shapes_filters_noise_and_supports_instance_masks`: перевіряє відсів сміття, класифікацію `2x2` / `1x4` / `1x1_round` і збереження роздільних ID інстанс-маски разом із метаданими `stud_counts`.

## Перевірка вимог проєкту (Checklist)
- [x] Алгоритми обробки зображень реалізовані вручну (тести перевіряють виключно ручні реалізації Учасника 1; `Pillow` використовується лише для створення тестових PNG).
- [x] Параметри методів винесені в конфіг або аргументи CLI (тест `classify_shapes` читає `configs/default.json`).
- [x] Додано обробку крайніх випадків (усі межові сценарії критерію C2 покрито окремими assert/raises).
- [x] Публічні функції та нетривіальні кроки мають докстрінги/коментарі, `pylint tests/test_core.py` — `10.00/10`, `pytest` — 100% pass по всіх активних тестах.

## Потенційні вузькі місця та рекомендації
1. Для малих дискретних кіл (`r < 10 px`) 8-зв'язний ланцюговий периметр Moore-Neighbor має помітну дискретну похибку, тому тест `classify_shapes` перевіряє класифікацію `1x1_round` на проєктному масштабі з `configs/default.json` (`stud_size_px = 30`, `r = 15 px`), де `circularity = 0.8413 >= 0.82`.
2. Після реалізації `color_classifier.py` і `runner.py` Учасником 2 варто додати інтеграційний тест повного ланцюжка `load_image_rgb -> pipeline -> classify_shapes -> classify_colors`.

## Нотатка для `META-USAGE`
* **Суть запиту:** Реалізація автотестів ядра для `image_io`, `config` і `shape_classifier` у `tests/test_core.py` (таск `T1.4`, критерії C2–C3).
* **Згенеровані фрагменти:** 6 тестових функцій у `tests/test_core.py` (`test_image_io_edge_cases`, `test_image_io_mask_save_and_discover`, `test_config_validation_edge_cases`, `test_connected_components_label_manual_5x5`, `test_shape_geometry_square_rect_and_circle`, `test_classify_shapes_filters_noise_and_supports_instance_masks`).
