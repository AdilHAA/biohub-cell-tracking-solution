# Biohub — Cell Tracking During Development

Решение для [соревнования Biohub на Kaggle](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development). **MISIS cell** — 172-е место из 3 947 команд, серебряная медаль. Участвовал один. Итоговый score на private leaderboard — **0.92433**. [Сертификат Kaggle](https://www.kaggle.com/certification/competitions/adilkhabibullin/biohub-cell-tracking-during-development).

## Задача

По 3D time-lapse microscopy эмбрионов данио нужно найти центры клеток, связать их между соседними кадрами и восстановить деления. Результат — lineage graph. Метрика сочетает adjusted edge Jaccard и division Jaccard.

## Решение

1. **Detection.** Temporal 3D U-Net находит центры клеток. Spatial TTA и второй detector помогают находить клетки, которые пропускает один проход.
2. **Coordinate refinement.** V1284 head получает 224 frozen U-Net features: 32 признака в центре детекции и шесть 32-мерных разностей с соседями по осям. MLP `224 → 32 → 3` предсказывает `(Δz, Δy, Δx)` со сдвигом до 2 µm. В финале усредняются поправки 13 heads; признаки для linking берутся в уточнённых координатах через trilinear sampling.
3. **Linking.** Node transformer оценивает связи между кадрами. ILP выбирает рёбра с учётом ограничений lineage graph: у клетки не больше одного родителя и двух потомков.
4. **Post-processing.** Motion relinking, gap closing и safe division rules исправляют разрывы и ветвления. DeepCenter проверяет добавляемые точки. Короткие компоненты фильтруются, координаты треков сглаживаются.

Оба выбранных сабмита используют один refinement ensemble. **v92** возвращает уверенные детекции возле открытых концов треков (`readmit`) и использует low-score peaks для заполнения пропусков. **v96** использует geometric fusion и фиксированный post-processing с `velocity_weight = 0.25`.

| Финальный сабмит | Public | Private |
| --- | ---: | ---: |
| v92 — readmit + mean13 | 0.957 | **0.92433** |
| v96 — geometric fusion + mean13 | 0.954 | 0.921 |

## Мой вклад

- Собрал 123 651 пару «детекция — размеченный центр» из 195 клипов и обучил refinement head предсказывать смещение в микрометрах. Настроил веса примеров по эмбрионам.
- Проверял head через grouped CV по клипам, leave-one-embryo-out и ablation с нулевой поправкой и случайным jitter. На 18 held-out клипах ранняя версия снизила среднее расстояние до размеченного центра с 1.582 до 1.307 µm.
- Собрал ensemble из 13 heads с разными seeds и двумя вариантами балансировки. Встроил его в оба финальных pipeline до linking.
- Для v96 зафиксировал параметры post-processing и убрал повторный sweep из inference. На четырёх видимых клипах CSV совпал побайтово с v93, а время прогона сократилось с 3.7 до 0.81 часа.

Detector и linker взяты из Biohub Tracking Support Pack (pilkwang). Варианты post-processing основаны на Frontier Readmit (thtennant) и Geometric Fusion (amanatar); архитектура refinement head — на Biohub x138 (anvithpothula).

## Код

- `infer.py` — inference для v92 и v96, сборка `submission.csv`.
- `settings.py`, `postprocess.py` — параметры и post-processing lineage graph.
- `train_refinement.py` — обучение 13 heads с двумя вариантами балансировки.
- `upstream/` — detector, linker и coordinate refinement.
