# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser

import cv2
import numpy as np
import tifffile
import torch
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def read_multiband_tif_for_vis(img_path: str, max_size: int = 1000000):
    """读取多波段 tif，只取 3 个波段用于可视化，并做必要的缩放与 uint8 归一化。
    返回: vis_img_uint8 (H,W,3), ori_h, ori_w, resized_h, resized_w
    """
    print("=" * 60)
    print(f"Reading multi-band image: {img_path}")

    multi_band_img = tifffile.imread(img_path)
    print(f"  Original shape: {multi_band_img.shape}")
    print(f"  Original dtype: {multi_band_img.dtype}")

    # 选择3个波段用于可视化
    if multi_band_img.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W) or (H,W,C), got: {multi_band_img.shape}")

    # 常见两种布局：(C,H,W) 或 (H,W,C)
    if multi_band_img.shape[0] >= 3 and multi_band_img.shape[0] < 100:
        # 更像 (C,H,W)
        print("  Detected format: (Channels, Height, Width)")
        vis_img = multi_band_img[[0, 1, 2], :, :].transpose(1, 2, 0)  # -> (H,W,3)
    elif multi_band_img.shape[2] >= 3 and multi_band_img.shape[2] < 100:
        # 更像 (H,W,C)
        print("  Detected format: (Height, Width, Channels)")
        vis_img = multi_band_img[:, :, :3]
    else:
        raise ValueError(f"Cannot infer channel layout from shape: {multi_band_img.shape}")

    print(f"  Selected 3 bands, new shape: {vis_img.shape}")

    ori_h, ori_w = vis_img.shape[:2]

    # 取消自动缩放
    need_resize = False
    new_h, new_w = ori_h, ori_w  # 不进行缩放

    # 归一化到 0-255
    print("  Normalizing to uint8...")
    img_min = float(vis_img.min())
    img_max = float(vis_img.max())
    print(f"    Value range: [{img_min:.2f}, {img_max:.2f}]")

    if img_max > img_min:
        vis_img_uint8 = ((vis_img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        print("    Warning: min == max, creating zero image")
        vis_img_uint8 = np.zeros_like(vis_img, dtype=np.uint8)

    print(f"  Final visualization image shape: {vis_img_uint8.shape}")
    print(f"  Final dtype: {vis_img_uint8.dtype}")
    print(f"  Final range: [{vis_img_uint8.min()}, {vis_img_uint8.max()}]")

    return vis_img_uint8, ori_h, ori_w, new_h, new_w, need_resize

def main():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--out-file", default=None, help="Path to output file")
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.5,
        help="Opacity of painted segmentation map. In (0, 1] range.",
    )
    parser.add_argument(
        "--with-labels",
        action="store_true",
        default=False,
        help="Whether to display the class labels.",
    )
    parser.add_argument("--title", default="result", help="The image identifier.")
    parser.add_argument(
        "--vis-max-size",
        type=int,
        default=2048,
        help="Max side length for visualization image (only for display/save).",
    )
    args = parser.parse_args()

    # Build model
    print("=" * 60)
    print("Loading model...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)
    print("Model loaded successfully!")

    # 1) 准备可视化图像（只用于 overlay，不影响推理）
    vis_img_uint8, ori_h, ori_w, vis_h, vis_w, vis_resized = read_multiband_tif_for_vis(
        args.img, max_size=args.vis_max_size
    )

    # 2) 推理（使用原始 tif 路径，让 mmseg 的 pipeline 自己读、自己处理）
    print("=" * 60)
    print("Running inference on original image...")
    result = inference_model(model, args.img)
    print("  Inference completed successfully!")
    print(f"  Prediction shape: {tuple(result.pred_sem_seg.data.shape)}")

    # 3) 关键修复：不要去改 result.pred_sem_seg.data（PixelData 会检查 shape）
    #    而是把可视化图像 resize 到预测的 HxW，保证 show_result_pyplot 叠加一致
    pred_h, pred_w = result.pred_sem_seg.data.shape[-2:]
    if (vis_img_uint8.shape[0] != pred_h) or (vis_img_uint8.shape[1] != pred_w):
        print("=" * 60)
        print("Resizing visualization image to match prediction...")
        print(f"  Visualization image: {vis_img_uint8.shape[0]}×{vis_img_uint8.shape[1]}")
        print(f"  Prediction size:     {pred_h}×{pred_w}")
        vis_img_uint8 = cv2.resize(vis_img_uint8, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)
        print(f"  New visualization image shape: {vis_img_uint8.shape}")

    # 4) 可视化 + 保存
    print("=" * 60)
    print("Generating visualization...")

    show_result_pyplot(
        model,
        vis_img_uint8,  # 传 numpy 数组
        result,         # 原始推理结果，不修改 PixelData
        title=args.title,
        opacity=args.opacity,
        with_labels=args.with_labels,
        draw_gt=False,
        show=False,      # 不弹窗
        out_file=args.out_file,
    )

    if args.out_file:
        print(f"✓ Success! Result saved to: {args.out_file}")
    else:
        print("✓ Success! (No out_file specified, not saved.)")

    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()
