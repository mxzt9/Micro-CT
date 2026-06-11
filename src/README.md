# Micro-CT Segmentation Pipeline

Цель проекта — **качественная сегментация порового пространства** на micro-CT
разных пород и размеров кубов + честная оценка топологии маски.

Контур проницаемости (PoreSpy/OpenPNM/GNN/PNM-решатель) из проекта удалён:
без независимого ground truth (LBM/эксперимент) обучать предсказание k
некорректно — таргет был циркулярным (считался из тех же HP-проводимостей).

## Getting Started

Весь пайплайн — в одном ноутбуке:

```
src/full_pipeline.ipynb
```

Скопируй в Colab, поставь зависимости (`pip install -r src/requirements.txt`)
и выполняй секции по порядку.

## Structure

```
src/
  full_pipeline.ipynb      ← единый ноутбук (весь пайплайн)
  requirements.txt
  README.md

  utils/                   ← основные модули (не трогать — отовсюду импорты)
    __init__.py
    common.py              ← базовые 3D-слои (ConvGNAct3D, DoubleConv3D, UNet3D)
    adaptive_routing.py    ← AdaptiveRoutedUNet3D, TopologyAdaptiveRoutedUNet3D
    data.py                ← BereaPatchDataset, CubeSizeBatchSampler, write_patch_indices
    losses.py              ← BCEDiceLoss, SoftClDiceLoss (связность скелета), aux/topo лоссы
    seg_metrics.py         ← clDice, числа Бетти, связная пористость, перколяция
    topology.py            ← PH-топология (cubical_persistence_summary, TOPOLOGY_FEATURE_DIM)
    network.py             ← извлечение сети (PoreSpy/OpenPNM) — только для visualize.py
    training.py            ← EarlyStopping, MetricTracker
    dependencies.py        ← check_required_dependencies

  scripts/                 ← CLI-скрипты
    train.py                              ← обучение сегментации (есть --cldice-weight)
    compare_models.py                     ← сравнение архитектур
    visualize.py                          ← 3D-визуализация
    precompute_topology_cache.py          ← быстрый прекомпьют PH-кэша

  tests/                   ← pytest-тесты
    test_data_multirock.py
    test_segmentation_quality.py          ← clDice, Betti, перколяция
```

## Data Layout

```
data/
  Berea/
    grayscale.raw
    grayscale_filtered.raw
    binary.raw
  <other_rock>/
    grayscale_filtered.raw
    binary.raw
datasets/
  Berea/
    index_64.csv
    index_128.csv
    index_192.csv
  <other_rock>/
    ...
models/
outputs/
```

## Pipeline Sections (in order)

| Section | Description |
|---------|-------------|
| **0** Setup & Environment Check | Paths, torch, CUDA, gudhi |
| **1** Prepare Data | Scan rocks, write index CSVs (one-time) |
| **2** Train Segmentation Model | BCE+Dice + **clDice** + aux + topo-head |
| **3** Кривые обучения | loss/Dice/clDice по эпохам из history |
| **4** Validate Segmentation | Dice + clDice + Betti + связная пористость + перколяция, графики, срезы |

## Лосс и метрики

| Член лосса | Что ловит | Вес по умолчанию |
|---|---|---|
| BCE + Dice | повоксельная точность | 1.0 |
| SoftClDiceLoss | разрывы / ложные перемычки скелета | 0.3 (`CLDICE_WEIGHT`) |
| auxiliary_physics_loss | пористость, перколяция | 0.05 |
| topology_prediction_loss | PH-саммари (регуляризация фич) | 0.01 |

Валидация считает не только Dice: воксельный Dice почти не чувствует разрыв
горла в 3 вокселя, а связная пористость и числа Бетти от этого меняются в разы.
Смотри scatter «Dice vs clDice» — точки сильно ниже диагонали значат
«воксели хорошие, топология разрушена».

## Tests

```
python -m pytest src/tests -q
```
