# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmengine.structures import PixelData
from mmseg.structures import SegDataSample
import cv2
import numpy as np
import tifffile
import torch
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def read_multiband_tif(img_path: str):
    """读取多波段 tif 文件
    返回：
        vis_img_uint8: 用于可视化的3波段RGB图像 (H,W,3) uint8
        full_bands: 完整的多波段数据，用于推理
    """
    print("=" * 60)
    print(f"Reading multi-band image: {img_path}")

    multi_band_img = tifffile.imread(img_path)
    print(f"  Original shape: {multi_band_img.shape}")
    print(f"  Original dtype: {multi_band_img.dtype}")

    if multi_band_img.ndim != 3:
        raise ValueError(f"Expected 3D array (C,H,W) or (H,W,C), got: {multi_band_img.shape}")

    # 判断数据格式
    if multi_band_img.shape[0] >= 3 and multi_band_img.shape[0] < 100:
        # (C,H,W) 格式
        print("  Detected format: (Channels, Height, Width)")
        num_bands = multi_band_img.shape[0]
        print(f"  Total bands: {num_bands}")
        
        # 保存完整波段用于推理
        full_bands = multi_band_img
        
        # 选择3个波段用于可视化
        vis_bands = multi_band_img[[0, 1, 2], :, :].transpose(1, 2, 0)  # -> (H,W,3)
        
    elif multi_band_img.shape[2] >= 3 and multi_band_img.shape[2] < 100:
        # (H,W,C) 格式
        print("  Detected format: (Height, Width, Channels)")
        num_bands = multi_band_img.shape[2]
        print(f"  Total bands: {num_bands}")
        
        # 保存完整波段用于推理，转换为 (C,H,W)
        full_bands = multi_band_img.transpose(2, 0, 1)
        
        # 选择3个波段用于可视化
        vis_bands = multi_band_img[:, :, :3]
        
    else:
        raise ValueError(f"Cannot infer channel layout from shape: {multi_band_img.shape}")

    print(f"  Visualization bands shape: {vis_bands.shape}")
    print(f"  Full bands for inference shape: {full_bands.shape}")

    # 归一化可视化图像到 0-255
    print("  Normalizing visualization image to uint8...")
    img_min = float(vis_bands.min())
    img_max = float(vis_bands.max())
    print(f"    Value range: [{img_min:.2f}, {img_max:.2f}]")

    if img_max > img_min:
        vis_img_uint8 = ((vis_bands - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        print("    Warning: min == max, creating zero image")
        vis_img_uint8 = np.zeros_like(vis_bands, dtype=np.uint8)

    print(f"  Final visualization image - shape: {vis_img_uint8.shape}, dtype: {vis_img_uint8.dtype}")
    print(f"  Value range: [{vis_img_uint8.min()}, {vis_img_uint8.max()}]")

    return vis_img_uint8, full_bands


def slide_inference(model, img_path, crop_size, stride, device):
    """使用滑窗方式对大图进行推理（直接处理tensor，避免文件IO）
    
    Args:
        model: 分割模型
        img_path: 图像路径
        crop_size: 裁剪尺寸 [h, w]
        stride: 滑动步长 [h, w]
        device: 设备
        
    Returns:
        pred_mask: 预测掩码 (H, W)
    """
    # 读取完整图像
    full_img = tifffile.imread(img_path)
    
    # 判断格式并转换为 (C, H, W)
    if full_img.shape[0] >= 3 and full_img.shape[0] < 100:
        # 已经是 (C, H, W)
        pass
    elif full_img.shape[2] >= 3 and full_img.shape[2] < 100:
        # (H, W, C) -> (C, H, W)
        full_img = full_img.transpose(2, 0, 1)
    
    C, H, W = full_img.shape
    crop_h, crop_w = crop_size
    stride_h, stride_w = stride
    
    print(f"\n  Image size: {H} x {W}")
    print(f"  Number of bands: {C}")
    
    # 计算需要的窗口数量
    num_h = (H - crop_h) // stride_h + 1
    num_w = (W - crop_w) // stride_w + 1
    
    # 如果图像不能被完整覆盖，增加一行/列
    if (num_h - 1) * stride_h + crop_h < H:
        num_h += 1
    if (num_w - 1) * stride_w + crop_w < W:
        num_w += 1
    
    total_windows = num_h * num_w
    print(f"  Total windows: {num_h} x {num_w} = {total_windows}")
    
    # 初始化结果和计数矩阵
    num_classes = model.decode_head.num_classes
    pred_sum = np.zeros((num_classes, H, W), dtype=np.float32)
    count_mat = np.zeros((H, W), dtype=np.float32)
    
    # 将图像转为 tensor
    full_img_tensor = torch.from_numpy(full_img.astype(np.float32))
    
    # 获取数据预处理器的均值和标准差
    if hasattr(model, 'data_preprocessor'):
        mean = model.data_preprocessor.mean.view(-1, 1, 1).cpu()
        std = model.data_preprocessor.std.view(-1, 1, 1).cpu()
    else:
        mean = torch.zeros(C, 1, 1)
        std = torch.ones(C, 1, 1)
    
    model.eval()
    
    window_idx = 0
    with torch.no_grad():
        for i in range(num_h):
            for j in range(num_w):
                window_idx += 1
                
                # 计算窗口位置
                start_h = min(i * stride_h, H - crop_h)
                start_w = min(j * stride_w, W - crop_w)
                end_h = start_h + crop_h
                end_w = start_w + crop_w
                
                # 裁剪窗口
                window = full_img_tensor[:, start_h:end_h, start_w:end_w].clone()
                
                # 数据预处理（归一化）
                window = (window - mean) / std
                
                # 添加 batch 维度
                window = window.unsqueeze(0).to(device)
                
                try:
                    # 推理
                    result = model.test_step({'inputs': window, 'data_samples': None})
                    
                    if isinstance(result, list):
                        result = result[0]
                    
                    # 获取预测结果
                    if hasattr(result, 'seg_logits'):
                        pred_logits = result.seg_logits.data.cpu().numpy()
                        if pred_logits.ndim == 4:  # (1, C, H, W)
                            pred_logits = pred_logits.squeeze(0)
                    elif hasattr(result, 'pred_sem_seg'):
                        # 从类别索引转换为one-hot
                        pred_cls = result.pred_sem_seg.data.cpu().numpy().squeeze()
                        pred_logits = np.zeros((num_classes, crop_h, crop_w), dtype=np.float32)
                        for c in range(num_classes):
                            pred_logits[c] = (pred_cls == c).astype(np.float32)
                    else:
                        print(f"\n  Warning: Unexpected result format at window [{i},{j}]")
                        continue
                    
                    # 累加到结果中
                    pred_sum[:, start_h:end_h, start_w:end_w] += pred_logits
                    count_mat[start_h:end_h, start_w:end_w] += 1
                    
                except Exception as e:
                    print(f"\n  Warning: Failed to process window [{i},{j}]: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    continue
                
                # 进度显示
                if window_idx % max(1, total_windows // 20) == 0 or window_idx == total_windows:
                    progress = window_idx / total_windows * 100
                    print(f"  Progress: {window_idx}/{total_windows} ({progress:.1f}%)", end='\r')
    
    print()  # 换行
    
    # 平均并取最大类别
    pred_sum = pred_sum / (count_mat[None, :, :] + 1e-8)
    pred_mask = np.argmax(pred_sum, axis=0).astype(np.uint8)
    
    print("  Sliding window inference completed!")
    
    return pred_mask


def main():
    parser = ArgumentParser()
    parser.add_argument("img", help="Image file")
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument("--out-file", default=None, help="Path to output PNG visualization file")
    parser.add_argument("--out-tif", default=None, help="Path to output TIF prediction mask (class indices)")
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
        "--slide-inference",
        action="store_true",
        default=False,
        help="Use sliding window inference for large images"
    )
    parser.add_argument(
        "--crop-size",
        type=int,
        nargs=2,
        default=[512, 512],
        help="Crop size for sliding window inference (height, width)"
    )
    parser.add_argument(
        "--stride",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Stride for sliding window inference (height, width)"
    )
    args = parser.parse_args()

    # Build model
    print("=" * 60)
    print("Loading model...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == "cpu":
        model = revert_sync_batchnorm(model)
    
    # 修改 test_pipeline 以保持原始分辨率
    print("Modifying test_pipeline to keep original resolution...")
    cfg = model.cfg
    new_test_pipeline = []
    for transform in cfg.test_pipeline:
        # 跳过 Resize 操作
        if transform['type'] != 'Resize':
            new_test_pipeline.append(transform)
        else:
            print(f"  Removed Resize transform: {transform}")
    cfg.test_pipeline = new_test_pipeline
    print(f"  Updated test_pipeline: {[t['type'] for t in cfg.test_pipeline]}")
    
    print("Model loaded successfully!")
    print(f"  Model expects {model.backbone.in_channels if hasattr(model.backbone, 'in_channels') else 'unknown'} input channels")
    print(f"  Model predicts {model.decode_head.num_classes} classes")

    # 读取图像：vis_img_uint8用于可视化，full_bands用于推理
    vis_img_uint8, full_bands = read_multiband_tif(args.img)

# 使用完整波段进行推理
    print("=" * 60)
    print("Running inference with full bands...")
    if args.slide_inference:
        # 滑窗推理模式
        print(f"Using sliding window inference:")
        print(f"  Crop size: {args.crop_size}")
        print(f"  Stride: {args.stride}")
        
        pred_mask = slide_inference(model, args.img, args.crop_size, args.stride, args.device)
        
        # --- 修改后的部分 ---
        # 1. 创建 SegDataSample 实例
        result = SegDataSample()
        
        # 2. 将预测掩码封装进 PixelData
        # 注意：PixelData 期望的数据形状通常是 (C, H, W)，对于语义分割 C=1
        pred_mask_tensor = torch.from_numpy(pred_mask).unsqueeze(0) 
        result.pred_sem_seg = PixelData(data=pred_mask_tensor)
        # -------------------
        
    else:
        # 标准推理模式
        try:
            result = inference_model(model, args.img)
            pred_mask = result.pred_sem_seg.data.cpu().numpy().squeeze()
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower():
                print("\n" + "="*60)
                print("❌ CUDA out of memory!")
                print("Your image is too large for available GPU memory.")
                print("\nSuggested solutions:")
                print("  1. Use sliding window inference (RECOMMENDED):")
                print(f"     Add these arguments: --slide-inference --crop-size 512 512 --stride 256 256")
                print("  2. Use CPU (slower):")
                print(f"     Add this argument: --device cpu")
                print("  3. Use a GPU with more memory")
                print("="*60 + "\n")
                import sys
                sys.exit(1)
            else:
                raise
    
    print("  Inference completed successfully!")
    print(f"  Prediction shape: {pred_mask.shape}")
    
    # 检查预测结果的类别分布
    unique, counts = np.unique(pred_mask, return_counts=True)
    print("\n  Prediction class distribution:")
    total_pixels = pred_mask.size
    for cls, count in zip(unique, counts):
        percentage = count / total_pixels * 100
        print(f"    Class {cls}: {count:,} pixels ({percentage:.2f}%)")

    # 保存类别索引的 TIF 文件
    if args.out_tif:
        print("=" * 60)
        print(f"Saving prediction mask to TIF: {args.out_tif}")
        print(f"  Mask shape: {pred_mask.shape}")
        print(f"  Mask dtype: {pred_mask.dtype}")
        print(f"  Unique values: {unique.tolist()}")
        
        # 保存为 TIF 文件
        tifffile.imwrite(args.out_tif, pred_mask.astype(np.uint8))
        print(f"✓ TIF mask saved successfully!")

    # 生成彩色可视化图像（可选）
    if args.out_file:
        # 调整可视化图像尺寸以匹配预测结果
        pred_h, pred_w = pred_mask.shape
        if (vis_img_uint8.shape[0] != pred_h) or (vis_img_uint8.shape[1] != pred_w):
            print("=" * 60)
            print("Resizing visualization image to match prediction...")
            print(f"  Visualization image: {vis_img_uint8.shape[0]}×{vis_img_uint8.shape[1]}")
            print(f"  Prediction size:     {pred_h}×{pred_w}")
            vis_img_uint8 = cv2.resize(vis_img_uint8, (pred_w, pred_h), interpolation=cv2.INTER_LINEAR)
            print(f"  New visualization image shape: {vis_img_uint8.shape}")

        print("=" * 60)
        print("Generating color visualization...")

        show_result_pyplot(
            model,
            vis_img_uint8,  # 传入3波段可视化图像
            result,         # 使用7波段推理的结果
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=False,
            show=False,
            out_file=args.out_file,
        )

        print(f"✓ Color visualization saved to: {args.out_file}")

    print("=" * 60)
    print("All done!")


if __name__ == "__main__":
    main()