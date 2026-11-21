# Пост-рев'ю: T2.8 — Пайплайн П2, Стадія 3: Ручний детектор Кенні та злиття границь (`canny_edge_merge_manual`)

## Завдання та змінені файли
* **Таск із `docs/tasks.md`:** `T2.8` (`src/lego_cv/pipelines/member1/p2_contrast_canny.py`).
* **Змінені файли:**
  * `src/lego_cv/pipelines/member1/p2_contrast_canny.py`
  * `docs/tasks.md` (позначено `T2.8` виконаним)
  * `docs/reviews/task_T2.8_p2_canny_merge_review.md`

## Що зроблено та як працює логіка
* **Ручні градієнти Собеля (`_sobel_gradients_2d`):** 2D-згортка з класичними ненормованими ядрами $S_x, S_y$ через зрізи `NumPy` на reflect-доповненому каналі яскравості, обчислення модуля $M = \sqrt{G_x^2 + G_y^2}$ (`np.hypot`) та напрямку $\theta = \operatorname{atan2}(G_y, G_x)$.
* **Векторизоване придушення немаксимумів (`_non_maximum_suppression`):** квантування кута градієнта на 4 напрямки (`0°`, `45°`, `90°`, `135°` з урахуванням системи координат зображення `+y` вниз) та суворе одностороннє порівняння (`>= prev` і `> next`) для формування рівно 1-піксельних тонких контурів на ступінчастих краях.
* **Подвійний поріг і гістерезис (`_hysteresis_bfs`, `canny_edge_detector_manual`):** 8-зв'язний BFS на `collections.deque` від сильних ребер (`>= high_threshold`) по слабких (`[low_threshold, high_threshold)`), повертає бінарну візуалізацію границь `uint8 (0/255)`.
* **Злиття з маскою Оцу (`canny_edge_merge_manual`):** замикання дрібних розривів контуру (`edge_close_kernel_size`), заповнення замкнених контурів через ручний `fill_holes_manual`, відсіювання тонких незамкнених відростків відкриттям та об'єднання з пороговою маскою Оцу (`merge_with_otsu`) у фінальну `int32` маску `{0, 1}`.

## Перевірка вимог проєкту (Checklist)
- [x] Алгоритми обробки зображень реалізовані вручну (тільки `collections.deque` та `NumPy` + повторне використання ручних функцій Учасника 1 із `p1_color_morph.py`; без `cv2.Canny`/`Sobel`).
- [x] Параметри методів винесені в конфіг або аргументи CLI (`low_threshold`, `high_threshold`, `merge_with_otsu`, `edge_close_kernel_size`, `fill_closed_contours` із `configs/pipeline_p2.json`).
- [x] Додано обробку крайніх випадків (однорідні зображення, невалідні пороги `low < 0` або `high < low`, малі матриці `5×5` без руйнування контурів відкриттям).
- [x] Публічні функції та нетривіальні кроки мають Google-style докстрінги/коментарі, `pylint src/lego_cv/pipelines/member1/p2_contrast_canny.py` — `10.00/10` (0 зауважень).

## Потенційні вузькі місця та рекомендації
1. Пороги Кенні (`low_threshold`, `high_threshold`) і `clip_limit` CLAHE взаємопов'язані: вищий контраст CLAHE посилює градієнти фактури фону, тому їх потрібно калібрувати спільно в `T2.9`.
2. Якщо у `stage3_canny` контур деталі має широкий розрив (> `edge_close_kernel_size`), заповнення контуру може не спрацювати, і тоді об'єкт підхоплюється маскою Оцу (`merge_with_otsu: true`).
3. Покрити у `T2.10` тест ступінчастого вертикального краю на матриці `5×5` і перевірку помилки при `high_threshold < low_threshold`.

## Нотатка для `META-USAGE`
* **Суть запиту:** Ручна реалізація детектора границь Кенні (Собель, NMS, подвійний поріг, BFS-гістерезис) та злиття контурів із маскою Оцу для Стадії 3 П2 (таск `T2.8`).
* **Згенеровані фрагменти:** `_sobel_gradients_2d`, `_non_maximum_suppression`, `_hysteresis_bfs`, `canny_edge_detector_manual`, `canny_edge_merge_manual` у `src/lego_cv/pipelines/member1/p2_contrast_canny.py`.
