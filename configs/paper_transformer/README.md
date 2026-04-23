# Paper Transformer (5-Band)

This folder provides 5-band transformer baselines for the forest dataset.

Models:

- `swin_tiny_upernet_5band_forest.py`
- `mask2former_swin_t_5band_forest.py`

Both configs use a 5->3 adapter wrapper (`SevenToThreeResNetV1c`) so ImageNet
pretrained 3-channel Swin weights remain usable.

Run:

```bash
python train.py configs/paper_transformer/swin_tiny_upernet_5band_forest.py
python train.py configs/paper_transformer/mask2former_swin_t_5band_forest.py
```
