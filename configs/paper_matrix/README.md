# Paper Matrix (Direct-Run)

This folder provides a unified, directly runnable experiment matrix for the
forest 5-band dataset.

## Unified Settings

- Dataset: `configs/_base_/datasets/forest.py`
- Classes: 2 (`background`, `AF`)
- Input: 5 bands
- Backbone wrapper: `SevenToThreeResNetV1c` (`5ch -> 3ch` adapter)
- Backbone core: `ResNetV1c-50` (ImageNet pretrained in inner backbone)
- Crop size: `512x512`
- Schedule: `configs/_base_/schedules/schedule_20k.py` (epoch-based, 100 epochs)
- Norm: `SyncBN` (for multi-GPU training)
- Dual-4090 overrides in each matrix config:
  - `train_dataloader.batch_size=2`
  - `train_dataloader.num_workers=4`
  - `optim_wrapper.accumulative_counts=4`

## Matrix Configs (Main: CE + Dice)

- `fcn_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `deeplabv3_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `gcnet_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `nonlocal_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `pam_only_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `cam_only_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `dnl_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `danet_r50-d8_4xb4-20k_forest-5band-matrix.py`
- `ann_r50-d8_4xb4-20k_forest-5band-matrix.py`

All main matrix configs use:

- Decode head: `CrossEntropyLoss + DiceLoss`
- Auxiliary head: `CrossEntropyLoss + DiceLoss` (scaled to 0.4)

## CE-Only Control

The original CE-only matrix is kept under:

- `configs/paper_matrix/ce_only/`

## Run One Model

```powershell
python train.py configs/paper_matrix/danet_r50-d8_4xb4-20k_forest-5band-matrix.py
```

## Run Whole Matrix

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_paper_matrix.ps1
```

## Run Whole Matrix On 2x4090 (Recommended)

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_paper_matrix_2gpu_amp.ps1
```

## Run CE-Only Control Matrix

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_paper_matrix_ce_only.ps1
```

## Run CE-Only Control On 2x4090

```powershell
powershell -ExecutionPolicy Bypass -File tools/run_paper_matrix_ce_only_2gpu_amp.ps1
```
