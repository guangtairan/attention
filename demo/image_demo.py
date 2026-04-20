# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from mmengine.model import revert_sync_batchnorm
from mmseg.apis import inference_model, init_model, show_result_pyplot
import tifffile
import numpy as np
import cv2
import torch
from mmengine.structures import PixelData

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument('--out-file', default=None, help='Path to output file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--with-labels',
        action='store_true',
        default=False,
        help='Whether to display the class labels.')
    parser.add_argument(
        '--title', default='result', help='The image identifier.')
    args = parser.parse_args()


    # Build model
    print("=" * 60)
    print("Loading model...")
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)
    print("Model loaded successfully!")
    
    # ===== 检查test_cfg =====
    print("\nChecking test_cfg:")
    test_cfg = model.cfg.model.get('test_cfg', None)
    if test_cfg:
        print(f"  mode: {test_cfg.get('mode', 'NOT SET')}")
        print(f"  crop_size: {test_cfg.get('crop_size', 'NOT SET')}")
        print(f"  stride: {test_cfg.get('stride', 'NOT SET')}")
    else:
        print("  WARNING: test_cfg not set! Sliding window may not work!")

    # ============ 准备可视化图像 ============
    print("=" * 60)
    print(f"Reading multi-band image: {args.img}")
    
    try:
        multi_band_img = tifffile.imread(args.img)
        print(f"  Original shape: {multi_band_img.shape}")
        print(f"  Original dtype: {multi_band_img.dtype}")
    except Exception as e:
        print(f"Error reading image with tifffile: {e}")
        raise
    
    # 选择3个波段用于可视化
    if multi_band_img.ndim == 3:
        if multi_band_img.shape[0] == 7:  # (C, H, W) 格式
            print("  Detected format: (Channels, Height, Width)")
            vis_img = multi_band_img[[0, 1, 2], :, :].transpose(1, 2, 0)
        elif multi_band_img.shape[2] == 7:  # (H, W, C) 格式
            print("  Detected format: (Height, Width, Channels)")
            vis_img = multi_band_img[:, :, :3]
        else:
            raise ValueError(f"Expected 7 channels, got shape: {multi_band_img.shape}")
    else:
        raise ValueError(f"Expected 3D array, got shape: {multi_band_img.shape}")
    
    print(f"  Selected 3 bands, new shape: {vis_img.shape}")
    
    # 记录原始尺寸
    ori_h, ori_w = vis_img.shape[:2]
    
    # 检查是否需要resize(避免内存问题)
    max_size = 2048
    need_resize = False
    if max(ori_h, ori_w) > max_size:
        scale = max_size / max(ori_h, ori_w)
        new_h, new_w = int(ori_h * scale), int(ori_w * scale)
        print(f"  Image too large, resizing for visualization: {ori_h}×{ori_w} → {new_h}×{new_w}")
        vis_img = cv2.resize(vis_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        need_resize = True
    else:
        print(f"  Image size OK: {ori_h}×{ori_w}")
        new_h, new_w = ori_h, ori_w
    
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
    
    # ============ 推理 ============
    print("=" * 60)
    print("Running inference on original image...")
    try:
        result = inference_model(model, args.img)
        print("  Inference completed successfully!")
        print(f"  Prediction shape: {result.pred_sem_seg.data.shape}")
    except Exception as e:
        print(f"  Error during inference: {e}")
        raise
    
    # ============ 调整预测结果尺寸 ============
    # 获取预测结果
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    pred_h, pred_w = pred_mask.shape
    print(f"\nPrediction size: {pred_h}×{pred_w}")
    print(f"Original image size: {ori_h}×{ori_w}")
    print(f"Visualization size: {new_h}×{new_w}")
    
    # 如果预测尺寸和可视化尺寸不匹配,需要resize
    if pred_h != new_h or pred_w != new_w:
        print("\n⚠️ Prediction size != Visualization size, resizing prediction...")
        
        pred_mask_resized = cv2.resize(
            pred_mask.astype(np.uint8),
            (new_w, new_h),
            interpolation=cv2.INTER_NEAREST
        )
        print(f"  Resized prediction shape: {pred_mask_resized.shape}")
        
        # ===== 关键修改:创建新的PixelData对象 =====
        from mmseg.structures import SegDataSample
        
        # 创建新的result对象
        new_result = SegDataSample()
        new_result.pred_sem_seg = PixelData()
        
        # 设置新的预测数据
        if torch.cuda.is_available() and args.device != 'cpu':
            new_result.pred_sem_seg.data = torch.from_numpy(pred_mask_resized).unsqueeze(0).cuda()
        else:
            new_result.pred_sem_seg.data = torch.from_numpy(pred_mask_resized).unsqueeze(0)
        
        # 复制其他元数据
        if hasattr(result, 'img_path'):
            new_result.img_path = result.img_path
        
        result = new_result
        print(f"  Updated result shape: {result.pred_sem_seg.data.shape}")
    
    # ============ 可视化 ============
    print("=" * 60)
    print("Generating visualization...")
    
    try:
        show_result_pyplot(
            model,
            vis_img_uint8,
            result,
            title=args.title,
            opacity=args.opacity,
            with_labels=args.with_labels,
            draw_gt=False,
            show=False,
            out_file=args.out_file
        )
        print(f"✓ Success! Result saved to: {args.out_file}")
    except Exception as e:
        print(f"  Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print("=" * 60)
    print("All done!")


if __name__ == '__main__':
    main()
