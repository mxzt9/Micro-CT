# Micro-CT Digital Core Pipeline

## Getting Started

The entire pipeline is in a single notebook:

```
src/full_pipeline.ipynb
```

Copy it to Colab, install dependencies (`pip install -r src/requirements.txt`),
and run the sections you need. Each section is self-contained.

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
    losses.py              ← BCEDiceLoss, auxiliary_physics_loss, topology_prediction_loss
    topology.py            ← PH-топология (cubical_persistence_summary, TOPOLOGY_FEATURE_DIM)
    network.py             ← PoreSpy/OpenPNM экстракция, PoreNetworkData, PH-фичи
    pipeline.py            ← DigitalCorePipeline (сегментация → экстракция → GNN)
    pnm_gnn.py             ← PoreNetworkPermeabilityModel, DifferentiablePNMSolver
    training.py            ← EarlyStopping, MetricTracker
    dependencies.py        ← check_required_dependencies

  scripts/                 ← CLI-скрипты
    train.py                              ← обучение сегментации
    compare_models.py                     ← сравнение архитектур
    visualize.py                          ← 3D-визуализация
    precompute_topology_cache.py          ← быстрый прекомпьют PH-кэша
    check_graph_orientation.py            ← проверка ориентации графов

  tests/                   ← pytest-тесты
    test_digital_core_pipeline.py
    test_data_multirock.py
    test_network_extraction_smoke.py
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
  networks/
```

## Pipeline Sections (in order)

| Section | Description |
|---------|-------------|
| **0** Setup & Environment Check | Paths, torch, CUDA, gudhi |
| **1** Prepare Data | Scan rocks, write index CSVs (one-time) |
| **2** Train Segmentation Model | Train TopologyAdaptiveRoutedUNet3D |
| **3** Extract Pore Networks | PoreSpy/OpenPNM → .pt files |
| **4** Train GNN Permeability Model | GNN on extracted networks |
| **5** Run Full Pipeline | Load both models, run one cube |
| **6** Validate Segmentation | Dice/BCE/error rate by rock + size |
| **7** Compare Variants (optional) | Topology vs Adaptive on 64³ |

## Colab Notes

- Install: `!pip install -r src/requirements.txt`
- Mount Drive if data is there: `from google.colab import drive; drive.mount('/content/drive')`
- Run only needed sections (e.g. skip Section 1 if datasets/ ready)

## Import Convention

Import from `utils`, not from individual modules:

```python
from utils import (
    BereaPatchDataset,
    DigitalCorePipeline,
    TopologyAdaptiveRoutedUNet3D,
    PoreNetworkPermeabilityModel,
    check_required_dependencies,
)
```

## Fast Training Modes

Section 2 has `TRAIN_MODE`:

- **quick**: real data with caps on samples/batches per epoch
- **full**: full epoch over all data

CLI equivalent:

```bash
python src/scripts/train.py --mode quick --num-workers 2 --model topology
```

Architecture comparison (Section 7 equivalent):

```bash
python src/scripts/compare_models.py --cube-size 64 --epochs 1 --samples-per-group 4
```

## Notes

- PoreSpy/OpenPNM extraction is **not differentiable** — training is staged by design.
- `BereaPatchDataset(..., cube_size=[64, 128, 192], balance=True)` balances by rock + cube_size.
- `TopologyAdaptiveRoutedUNet3D` consumes PH features `[B, 6]` computed from grayscale only;
  binary-derived topology is used only as an auxiliary loss target (no label leakage).
- Use `sliding_window_inference_3d(model, x, window_size=128, overlap=0.5)` for large volumes at inference.
- `PoreNetworkData` stores graph tensors: coords, edge_index, node_attr, edge_attr, log_g_hp.