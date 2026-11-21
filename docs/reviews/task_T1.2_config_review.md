# Пост-рев'ю: T1.2 — Валідація JSON-конфігурації та CLI-перевизначень (`config.py`)

## Завдання та змінені файли
* **Таск із `docs/tasks.md`:** `T1.2` (`src/lego_cv/config.py`).
* **Змінені файли:**
  * `src/lego_cv/config.py`
  * `configs/default.json` (`max_image_side_px`: `8000` → `10000`)
  * `docs/tasks.md` (позначено `T1.2` виконаним)
  * `docs/reviews/task_T1.2_config_review.md`

## Що зроблено та як працює логіка
* Посилено `apply_cli_overrides`: перевіряється наявність `=`, відсутність порожніх сегментів у точковому шляху (`a..b=1`, `.a=1`, `a.=1`) та непорожність значення після `=`, що виключає приховані помилки при передачі `--param`.
* Додано `validate_config` (викликається автоматично в `load_pipeline_config`) із допоміжними перевірками `_is_int`, `_is_number`, `_validate_preprocessing_section`, `_validate_shape_classifier_section`:
  * виправлено класичний крайній випадок Python, де `isinstance(True, int) == True` (`seed=true` тепер відхиляється як `ConfigError`);
  * перевіряються межі розмірів зображень (`1 <= min_image_side_px <= max_image_side_px`, `stud_size_px > 0`) і допуски/класи секції `shape_classifier`.
* У `configs/default.json` підвищено `max_image_side_px` до `10000`, щоб реальні фото роздільністю `8160×4592` із `dataset/real_u2/` проходили валідацію без помилок.

## Перевірка вимог проєкту (Checklist)
- [x] Алгоритми обробки зображень реалізовані вручну (модуль використовує лише стандартну бібліотеку Python: `copy`, `json`, `pathlib`).
- [x] Параметри методів винесені в конфіг або аргументи CLI (без хардкоду в логіці — підтримуються `default.json`, `pipeline_p*.json` і `--param`).
- [x] Додано обробку крайніх випадків (`bool` замість `int`, порожні сегменти ключів, некоректні межі розмірів і класів форми).
- [x] Публічні функції та нетривіальні кроки мають Google-style докстрінги/коментарі, `pylint src/lego_cv/config.py` — `10.00/10` (0 зауважень).

## Потенційні вузькі місця та рекомендації
1. `_parse_override_value` трактує значення на кшталт `nan`/`Infinity` як валідний JSON у Python (`json.loads`); за потреби варто додатково перевіряти `math.isfinite` для числових параметрів.
2. Покрити нові гілки валідації (`seed=true`, `a..b=1`, невалідний `stud_size_px`, `max_side_px < min_image_side_px`) автотестами в `T1.4`.

## Нотатка для `META-USAGE`
* **Суть запиту:** Доповнення валідації JSON-конфігів і `--param` у `config.py` та актуалізація `max_image_side_px` у `configs/default.json` (таск `T1.2`).
* **Згенеровані фрагменти:** `validate_config`, `_validate_preprocessing_section`, `_validate_shape_classifier_section`, оновлений `apply_cli_overrides` у `src/lego_cv/config.py`.
