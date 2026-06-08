# Digital Core Training Package

Portable package for staged Digital Core training:

`raw/segmented cube -> FiLM UNet segmentation -> PoreSpy/OpenPNM network -> features + PH -> GNN log(g) -> differentiable PNM -> kx, ky, kz`

## Install

```bash
pip install -r src/requirements.txt
```

`gudhi` is required. The environment notebook checks this explicitly.

## Expected Data Layout

Place or mount your data in the project root:

```text
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
    index_64.csv
    index_128.csv
    index_192.csv
models/
outputs/
```

Each index CSV must contain: `z,y,x,split`. `cube_size` and `rock` columns are optional and are filled by the dataset loader when absent.

The old single-rock layout is still supported:

```text
data/
  Berea_2d25um_grayscale.raw
  Berea_2d25um_grayscale_filtered.raw
  Berea_2d25um_binary.raw
dataset_128/
  index_128.csv
```

## Notebooks

Run in this order:

1. `src/notebooks/00_prepare_data.ipynb`  
   Discovers rock folders and writes `index_64.csv`, `index_128.csv`, `index_192.csv`.

2. `src/notebooks/00_environment_check.ipynb`  
   Verifies imports, CUDA, `gudhi`, and a tiny model forward.

3. `src/notebooks/01_train_segmentation.ipynb`  
   Trains `FiLMRoutedUNet3D` on `raw -> binary pore mask` using cube sizes `64, 128, 192`.

4. `src/notebooks/02_extract_networks_openpnm.ipynb`  
   Uses PoreSpy/OpenPNM to convert segmented cubes into pore-network `.pt` files.

5. `src/notebooks/03_train_gnn_pnm.ipynb`  
   Trains `PoreNetworkPermeabilityModel` on extracted network tensors and target `k`.

6. `src/notebooks/04_run_full_pipeline.ipynb`  
   Runs the orchestrated pipeline on one cube and saves a summary CSV.

## Fast Training Modes

`src/notebooks/01_train_segmentation.ipynb` has `TRAIN_MODE`:

- `quick`: real data with `SAMPLES_PER_GROUP`, `MAX_TRAIN_BATCHES`, and `MAX_VAL_BATCHES` caps. Use it for iteration before long runs.
- `full`: full real-data epoch.

For 11 rocks, prefer fixed-budget epochs first: cap samples per `rock + cube_size`, train `64/128` more often, and schedule `192` less frequently until the architecture and losses are stable.

For faster real runs outside Jupyter:

```bash
python src/tools/train_segmentation.py --mode quick --num-workers 2
```

`00_prepare_data.ipynb` can write `porosity` and `percolates_z/y/x` into index CSVs (`COMPUTE_AUX_TARGETS = True`). Training then reuses those labels instead of recomputing connected components for every sampled cube.

## Modules

Import from `utils`, not from old research notebooks:

```python
from utils import (
    BereaPatchDataset,
    MultiScaleNoiseConsistencyDataset,
    MultiRockPatchDataset,
    FiLMRoutedUNet3D,
    DigitalCorePipeline,
    PoreNetworkPermeabilityModel,
)
```

## Notes

- The pipeline is orchestrated end-to-end, but PoreSpy/OpenPNM extraction is not differentiable.
- Training is staged by design: segmentation first, network extraction second, GNN/PNM third.
- `BereaPatchDataset(..., cube_size=[64, 128, 192], balance=True)` dynamically discovers rocks and balances train epochs by `rock + cube_size`.
- `MultiScaleNoiseConsistencyDataset(..., view_cube_sizes=[64, 128, 192])` returns centered noisy views of the same patch for rock-embedding consistency training.
- `FiLMRoutedUNet3D(...)(x, return_dict=True)` returns `logits`, `rock_embedding`, `decoder_embedding`, `router_alpha`, `porosity_logit`, and `percolation_logits`.
- Use `sliding_window_inference_3d(model, x, window_size=128, overlap=0.5)` for large volumes at inference time.
- `PoreNetworkData` stores tensors needed by GNN/PNM: `coords`, `edge_index`, `node_attr`, `edge_attr`, `log_g_hp`, `domain_size`, `metadata`.
