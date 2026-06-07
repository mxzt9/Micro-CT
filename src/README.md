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
  Berea_2d25um_grayscale.raw
  Berea_2d25um_grayscale_filtered.raw
  Berea_2d25um_binary.raw
dataset_128/
  index_128.csv
models/
outputs/
```

`index_128.csv` must contain: `z,y,x,split`.

## Notebooks

Run in this order:

1. `src/notebooks/00_environment_check.ipynb`  
   Verifies imports, CUDA, `gudhi`, and a tiny model forward.

2. `src/notebooks/01_train_segmentation.ipynb`  
   Trains `FiLMRoutedUNet3D` on `raw -> binary pore mask`.

3. `src/notebooks/02_extract_networks_openpnm.ipynb`  
   Uses PoreSpy/OpenPNM to convert segmented cubes into pore-network `.pt` files.

4. `src/notebooks/03_train_gnn_pnm.ipynb`  
   Trains `PoreNetworkPermeabilityModel` on extracted network tensors and target `k`.

5. `src/notebooks/04_run_full_pipeline.ipynb`  
   Runs the orchestrated pipeline on one cube and saves a summary CSV.

## Modules

Import from `utils`, not from old research notebooks:

```python
from utils import (
    BereaPatchDataset,
    FiLMRoutedUNet3D,
    DigitalCorePipeline,
    PoreNetworkPermeabilityModel,
)
```

## Notes

- The pipeline is orchestrated end-to-end, but PoreSpy/OpenPNM extraction is not differentiable.
- Training is staged by design: segmentation first, network extraction second, GNN/PNM third.
- `PoreNetworkData` stores tensors needed by GNN/PNM: `coords`, `edge_index`, `node_attr`, `edge_attr`, `log_g_hp`, `domain_size`, `metadata`.
