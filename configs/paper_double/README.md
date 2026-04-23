# Paper Double-Attention Experiments

Configs in this folder are designed for dual-attention ablation on the forest
5-band setting.

## Unified Training Policy

All configs inherit `configs/_base_/schedules/schedule_20k.py`, so they follow
the same training policy used for cross-family comparison:

- Optimizer: `AdamW`
- Base LR: `1e-4`
- LR schedule: `LinearLR warmup + PolyLR`
- Training length: `100 epochs`
- Gradient accumulation: `accumulative_counts=4`
- Adapter LR multiplier: `backbone.adapter lr_mult=5`
- Validation interval: `50` epochs
- Checkpoint interval: `3` epochs

## Configs

- `d0_danet_parallel_r50_5band.py` (DANet baseline, PAM||CAM)
- `d1_cam_to_pam_r50_5band.py` (serial CAM -> PAM)
- `d2_pam_to_cam_r50_5band.py` (serial PAM -> CAM)
- `d3_parallel_concat_r50_5band.py` (parallel concat + 1x1 fusion)
- `d4_parallel_residual_r50_5band.py` (parallel concat + 1x1 + residual)
- `d5_gated_fusion_r50_5band.py` (parallel gated fusion)

## Example

```bash
python train.py configs/paper_double/d1_cam_to_pam_r50_5band.py
```
