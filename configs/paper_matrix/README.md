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
- Norm: `BN` (for single-GPU training)
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
$configs = @(
  "configs/paper_matrix/fcn_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/deeplabv3_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/gcnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/nonlocal_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/pam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/cam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/dnl_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/danet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ann_r50-d8_4xb4-20k_forest-5band-matrix.py"
)
foreach ($cfg in $configs) {
  python train.py $cfg
}
```

## Run Whole Matrix On 2x4090 (Recommended)

```powershell
$configs = @(
  "configs/paper_matrix/fcn_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/deeplabv3_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/gcnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/nonlocal_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/pam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/cam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/dnl_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/danet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ann_r50-d8_4xb4-20k_forest-5band-matrix.py"
)
foreach ($cfg in $configs) {
  python train.py $cfg --amp
}
```

## Run CE-Only Control Matrix

```powershell
$configs = @(
  "configs/paper_matrix/ce_only/fcn_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/deeplabv3_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/gcnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/nonlocal_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/pam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/cam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/dnl_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/danet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/ann_r50-d8_4xb4-20k_forest-5band-matrix.py"
)
foreach ($cfg in $configs) {
  python train.py $cfg
}
```

## Run CE-Only Control On 2x4090

```powershell
$configs = @(
  "configs/paper_matrix/ce_only/fcn_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/deeplabv3_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/gcnet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/nonlocal_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/pam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/cam_only_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/dnl_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/danet_r50-d8_4xb4-20k_forest-5band-matrix.py",
  "configs/paper_matrix/ce_only/ann_r50-d8_4xb4-20k_forest-5band-matrix.py"
)
foreach ($cfg in $configs) {
  python train.py $cfg --amp
}
```
