from osgeo import gdal
import numpy as np
import os
import torch
import cv2
from mmseg.apis import init_model, inference_model
from tqdm import tqdm


# =========================================================
# 1. 基础配置
# =========================================================

# 模型配置与权重
# 注意：config_file 和 checkpoint_file 必须严格对应同一个模型
config_file = r'E:\Desktop\mmsegmentation-five\work_dirs\mask2former_r50_8xb2-90k_forest-512x1024\20260322_230639\vis_data\config.py'
checkpoint_file = r'E:\Desktop\mmsegmentation-five\work_dirs\mask2former_r50_8xb2-90k_forest-512x1024\epoch_100.pth'

# 输入输出目录（都写目录，不要写单个文件）
input_folder = r'E:\Desktop\7.14peizhun\ronghe'
output_folder = r'E:\Desktop\7.14peizhun\yuce\mask2former\256rgb'

# patch 设置
crop_size = (512, 512)
overlap = 256

# 推理设备
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# 输出设置
NODATA_VALUE = 255
BINARY_OUTPUT = True          # True: 输出 0/1/255；False: 保留原类别
PRINT_PIXEL_STATS = True

# padding 设置
PAD_MODE = 'reflect'          # 推荐 reflect；可选 'constant'

# =========================================================

# 2. 波段配置（5波段场景最关键）
# =========================================================

# TIFF 中读取哪些波段（0-based）
INPUT_BAND_INDEXES = [0, 1, 2, 3, 4]

# 实际送给模型哪些波段
# 如果你的模型训练就是 5 通道，这里就保持 5 个
MODEL_BAND_INDEXES = [0, 1, 2, 3, 4]

# 生成有效区 mask 时，用哪些波段判断
# 通常建议只用 RGB
MASK_BAND_INDEXES = [0, 1, 2]

# 用于可视化显示为 RGB 的波段
# 如果你的 5 波段顺序是 [R,G,B,NIR,DEM]，这里写 [0,1,2]
# 如果顺序是 [NIR,R,G,B,DEM]，这里应写 [1,2,3]
VIS_BAND_INDEXES = [0, 1, 2]

# 判定有效区阈值
VALID_PIXEL_THRESHOLD = 0

# =========================================================
# 3. 效果图可视化配置
# =========================================================

SAVE_VISUALIZATION = True

# 预测为 1 的区域覆盖颜色（浅灰）
OVERLAY_COLOR = [220, 220, 220]

# 透明度
# 1.0 = 完全覆盖
# 0.8 = 轻微透底图
OVERLAY_ALPHA = 1.0

# 是否做百分位拉伸，提升底图观感
USE_PERCENTILE_STRETCH = True
STRETCH_LOW = 2
STRETCH_HIGH = 98


# =========================================================
# 4. 配置检查
# =========================================================

def validate_settings():
    if not os.path.isfile(config_file):
        raise FileNotFoundError(f'config_file 不存在: {config_file}')

    if not os.path.isfile(checkpoint_file):
        raise FileNotFoundError(f'checkpoint_file 不存在: {checkpoint_file}')

    if not os.path.isdir(input_folder):
        raise FileNotFoundError(f'input_folder 不存在: {input_folder}')

    if crop_size[0] <= 0 or crop_size[1] <= 0:
        raise ValueError(f'crop_size 必须为正数，当前: {crop_size}')

    if overlap < 0:
        raise ValueError(f'overlap 不能小于 0，当前: {overlap}')

    if overlap >= crop_size[0] or overlap >= crop_size[1]:
        raise ValueError(
            f'overlap 必须小于 crop_size，当前 overlap={overlap}, crop_size={crop_size}'
        )

    if len(INPUT_BAND_INDEXES) == 0:
        raise ValueError('INPUT_BAND_INDEXES 不能为空')

    if len(MODEL_BAND_INDEXES) == 0:
        raise ValueError('MODEL_BAND_INDEXES 不能为空')

    if len(MASK_BAND_INDEXES) == 0:
        raise ValueError('MASK_BAND_INDEXES 不能为空')

    if len(VIS_BAND_INDEXES) != 3:
        raise ValueError('VIS_BAND_INDEXES 必须正好包含 3 个波段，用于 RGB 可视化')

    if max(MODEL_BAND_INDEXES) >= len(INPUT_BAND_INDEXES):
        raise ValueError('MODEL_BAND_INDEXES 超出了 INPUT_BAND_INDEXES 读取后的通道范围')

    if max(MASK_BAND_INDEXES) >= len(INPUT_BAND_INDEXES):
        raise ValueError('MASK_BAND_INDEXES 超出了 INPUT_BAND_INDEXES 读取后的通道范围')

    if max(VIS_BAND_INDEXES) >= len(INPUT_BAND_INDEXES):
        raise ValueError('VIS_BAND_INDEXES 超出了 INPUT_BAND_INDEXES 读取后的通道范围')

    config_name = os.path.basename(config_file).lower()
    ckpt_name = os.path.basename(checkpoint_file).lower()

    if ('danet' in config_name and 'deeplab' in ckpt_name) or \
       ('deeplab' in config_name and 'danet' in ckpt_name):
        print('[警告] config_file 和 checkpoint_file 看起来不像同一个模型，请确认严格匹配。')


# =========================================================
# 5. 读取 TIFF + 生成 mask
# =========================================================

def read_tiff_with_gdal(file_path):
    """
    读取多波段 TIFF，并生成有效区域 mask
    - 按 INPUT_BAND_INDEXES 读取指定波段
    - 按 MASK_BAND_INDEXES 判断有效区
    """
    dataset = gdal.Open(file_path)
    if dataset is None:
        return None, None, None, None

    count = dataset.RasterCount
    if count <= 0:
        return None, None, None, None

    required_max_band = max(INPUT_BAND_INDEXES) + 1
    if count < required_max_band:
        raise ValueError(
            f'影像波段数不足，需要至少 {required_max_band} 个波段，实际只有 {count} 个。文件: {file_path}'
        )

    bands = []
    for idx in INPUT_BAND_INDEXES:
        band = dataset.GetRasterBand(idx + 1).ReadAsArray()
        if band is None:
            raise ValueError(f'读取波段失败: band={idx + 1}, file={file_path}')
        bands.append(band)

    img = np.dstack(bands).astype(np.float32)

    # 只用指定波段判断有效区
    mask_part = img[:, :, MASK_BAND_INDEXES]

    if VALID_PIXEL_THRESHOLD > 0:
        valid_mask = np.sum(mask_part, axis=2) > VALID_PIXEL_THRESHOLD
    else:
        valid_mask = np.any(mask_part != 0, axis=2)

    mask = np.where(valid_mask, 255, 0).astype(np.uint8)

    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()

    return img, mask, geo_transform, projection


# =========================================================
# 6. padding
# =========================================================

def pad_image(image, crop_size, pad_mode='reflect'):
    """
    将图像 pad 到 crop_size 的整数倍
    """
    h, w = image.shape[:2]
    pad_h = (crop_size[0] - h % crop_size[0]) % crop_size[0]
    pad_w = (crop_size[1] - w % crop_size[1]) % crop_size[1]

    if image.ndim == 3:
        pad_width = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        pad_width = ((0, pad_h), (0, pad_w))

    if pad_h == 0 and pad_w == 0:
        return image

    if pad_mode == 'reflect':
        return np.pad(image, pad_width, mode='reflect')
    elif pad_mode == 'constant':
        return np.pad(image, pad_width, mode='constant', constant_values=0)
    else:
        raise ValueError(f'不支持的 PAD_MODE: {pad_mode}')


# =========================================================
# 7. 滑窗切块（带 overlap）
# =========================================================

def crop_image(image, crop_size, overlap):
    """
    按滑窗方式裁成多个 patch
    step = crop_size - overlap
    """
    height, width = image.shape[:2]
    step_h = crop_size[0] - overlap
    step_w = crop_size[1] - overlap

    patches = []
    coords = []

    for i in range(0, height - crop_size[0] + 1, step_h):
        for j in range(0, width - crop_size[1] + 1, step_w):
            patch = image[i:i + crop_size[0], j:j + crop_size[1]]
            patches.append(patch)
            coords.append((i, j, i + crop_size[0], j + crop_size[1]))

    return patches, coords


# =========================================================
# 8. patch 推理
# =========================================================

def predict_patches_mmseg(model, patches):
    """
    对每个 patch 推理
    """
    pred_patches = []

    for patch in patches:
        patch_for_model = patch[:, :, MODEL_BAND_INDEXES]

        result = inference_model(model, patch_for_model)

        if hasattr(result, 'pred_sem_seg'):
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        else:
            pred_mask = result[0]

        pred_patches.append(pred_mask.astype(np.uint8))

    return pred_patches


# =========================================================
# 9. stitch 拼接
# =========================================================

def stitch_images(patches, coords, image_shape, overlap):
    """
    将 patch 拼回整图
    只取 patch 中心区域，减少边缘效应
    """
    stitched_image = np.full(image_shape, NODATA_VALUE, dtype=np.uint8)

    for patch, (top, left, bottom, right) in zip(patches, coords):
        off = overlap // 2
        h_patch, w_patch = patch.shape

        # 大图目标区域
        t_top = top + off
        t_bottom = bottom - off
        t_left = left + off
        t_right = right - off

        # patch 来源区域
        s_top = off
        s_bottom = h_patch - off
        s_left = off
        s_right = w_patch - off

        # 边界 patch 靠外侧不裁
        if top == 0:
            t_top, s_top = top, 0
        if left == 0:
            t_left, s_left = left, 0
        if bottom >= image_shape[0]:
            t_bottom, s_bottom = image_shape[0], h_patch
        if right >= image_shape[1]:
            t_right, s_right = image_shape[1], w_patch

        # 防御性裁剪
        t_h = t_bottom - t_top
        t_w = t_right - t_left
        s_h = s_bottom - s_top
        s_w = s_right - s_left

        valid_h = min(t_h, s_h)
        valid_w = min(t_w, s_w)

        if valid_h > 0 and valid_w > 0:
            stitched_image[
                t_top:t_top + valid_h,
                t_left:t_left + valid_w
            ] = patch[
                s_top:s_top + valid_h,
                s_left:s_left + valid_w
            ]

    return stitched_image


# =========================================================
# 10. 后处理
# =========================================================

def postprocess_prediction(final_pred, mask):
    """
    后处理
    - 二分类时压成 0/1
    - 背景区域设为 255
    """
    if BINARY_OUTPUT:
        final_pred = np.where(final_pred > 0, 1, 0).astype(np.uint8)
    else:
        final_pred = final_pred.astype(np.uint8)

    if mask is not None:
        final_pred[mask == 0] = NODATA_VALUE

    return final_pred


# =========================================================
# 11. 保存 GeoTIFF
# =========================================================

def save_tiff_with_geo(output_path, image_data, geo_transform, projection, nodata_val):
    driver = gdal.GetDriverByName('GTiff')
    h, w = image_data.shape

    out_raster = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    if out_raster is None:
        raise RuntimeError(f'无法创建输出文件: {output_path}')

    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)

    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(image_data)
    out_band.SetNoDataValue(nodata_val)
    out_band.FlushCache()

    out_raster.FlushCache()
    out_band = None
    out_raster = None


# =========================================================
# 12. 可视化辅助函数
# =========================================================

def stretch_to_uint8(rgb, low=2, high=98):
    """
    将 RGB 拉伸到 0~255，便于可视化
    rgb: H x W x 3
    """
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.uint8)

    for c in range(3):
        band = rgb[:, :, c]
        p_low = np.percentile(band, low)
        p_high = np.percentile(band, high)

        if p_high <= p_low:
            out[:, :, c] = np.clip(band, 0, 255).astype(np.uint8)
        else:
            band = (band - p_low) / (p_high - p_low) * 255.0
            band = np.clip(band, 0, 255)
            out[:, :, c] = band.astype(np.uint8)

    return out


def create_visualization(image_multiband, final_pred,
                         vis_band_indexes=(0, 1, 2),
                         overlay_color=(220, 220, 220),
                         alpha=1.0,
                         use_percentile_stretch=True,
                         stretch_low=2,
                         stretch_high=98):
    """
    生成效果图
    - 底图：原始影像 RGB 可视化
    - 叠加：预测为 1 的区域覆盖浅灰色
    """
    rgb = image_multiband[:, :, vis_band_indexes].copy()

    if use_percentile_stretch:
        rgb_vis = stretch_to_uint8(rgb, low=stretch_low, high=stretch_high)
    else:
        rgb_vis = np.clip(rgb, 0, 255).astype(np.uint8)

    vis = rgb_vis.copy()
    target_mask = (final_pred == 1)

    overlay_color = np.array(overlay_color, dtype=np.float32)

    if alpha >= 1.0:
        vis[target_mask] = overlay_color.astype(np.uint8)
    else:
        vis[target_mask] = (
            vis[target_mask].astype(np.float32) * (1.0 - alpha) +
            overlay_color * alpha
        ).astype(np.uint8)

    return vis


def save_visualization_png(output_png_path, vis_img):
    """
    保存 PNG 效果图
    OpenCV 保存时需要 BGR
    """
    vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(output_png_path, vis_bgr)
    if not ok:
        raise RuntimeError(f'保存 PNG 失败: {output_png_path}')


# =========================================================
# 13. 打印统计信息
# =========================================================

def print_stats(final_pred, file_name):
    if not PRINT_PIXEL_STATS:
        return

    unique, counts = np.unique(final_pred, return_counts=True)
    stats = dict(zip(unique.tolist(), counts.tolist()))
    print(f'[{file_name}] Pixel distribution: {stats}')

    if NODATA_VALUE in stats:
        total = final_pred.size
        nodata_ratio = stats[NODATA_VALUE] / total
        print(f'[{file_name}] NoData ratio: {nodata_ratio:.2%}')


# =========================================================
# 14. 单张处理流程
# =========================================================

def process_single_tiff(model, input_path, output_tif_path):
    img, mask, geo, proj = read_tiff_with_gdal(input_path)
    if img is None:
        print(f'[跳过] 读取失败: {input_path}')
        return

    # pad
    padded_img = pad_image(img, crop_size, pad_mode=PAD_MODE)

    # crop with overlap
    patches, coords = crop_image(padded_img, crop_size, overlap)
    if len(patches) == 0:
        print(f'[跳过] 没有生成 patch: {input_path}')
        return

    # inference
    pred_patches = predict_patches_mmseg(model, patches)

    # stitch
    stitched = stitch_images(pred_patches, coords, padded_img.shape[:2], overlap)

    # 裁回原图大小
    final_pred = stitched[:img.shape[0], :img.shape[1]]

    # postprocess
    final_pred = postprocess_prediction(final_pred, mask)

    # 保存分类结果 GeoTIFF
    save_tiff_with_geo(output_tif_path, final_pred, geo, proj, NODATA_VALUE)
    print(f'[完成] Prediction TIFF saved to: {output_tif_path}')

    # 保存可视化 PNG
    if SAVE_VISUALIZATION:
        base_name = os.path.splitext(os.path.basename(input_path))[0]
        vis_output_path = os.path.join(output_folder, f'vis_{base_name}.png')

        vis_img = create_visualization(
            image_multiband=img,
            final_pred=final_pred,
            vis_band_indexes=VIS_BAND_INDEXES,
            overlay_color=OVERLAY_COLOR,
            alpha=OVERLAY_ALPHA,
            use_percentile_stretch=USE_PERCENTILE_STRETCH,
            stretch_low=STRETCH_LOW,
            stretch_high=STRETCH_HIGH
        )

        save_visualization_png(vis_output_path, vis_img)
        print(f'[完成] Visualization PNG saved to: {vis_output_path}')

    print_stats(final_pred, os.path.basename(input_path))


# =========================================================
# 15. 主函数
# =========================================================

def main():
    validate_settings()

    print('================= 推理配置 =================')
    print(f'config_file         : {config_file}')
    print(f'checkpoint_file     : {checkpoint_file}')
    print(f'input_folder        : {input_folder}')
    print(f'output_folder       : {output_folder}')
    print(f'device              : {device}')
    print(f'crop_size           : {crop_size}')
    print(f'overlap             : {overlap}')
    print(f'PAD_MODE            : {PAD_MODE}')
    print(f'BINARY_OUTPUT       : {BINARY_OUTPUT}')
    print(f'INPUT_BAND_INDEXES  : {INPUT_BAND_INDEXES}')
    print(f'MODEL_BAND_INDEXES  : {MODEL_BAND_INDEXES}')
    print(f'MASK_BAND_INDEXES   : {MASK_BAND_INDEXES}')
    print(f'VIS_BAND_INDEXES    : {VIS_BAND_INDEXES}')
    print(f'SAVE_VISUALIZATION  : {SAVE_VISUALIZATION}')
    print('===========================================')

    model = init_model(config_file, checkpoint_file, device=device)

    os.makedirs(output_folder, exist_ok=True)

    tiff_files = [
        f for f in os.listdir(input_folder)
        if f.lower().endswith(('.tif', '.tiff'))
    ]

    if not tiff_files:
        print(f'[提示] 输入目录中未找到 TIFF 文件: {input_folder}')
        return

    for tiff_file in tqdm(tiff_files, desc='Processing TIFF files'):
        input_path = os.path.join(input_folder, tiff_file)
        base_name = os.path.splitext(tiff_file)[0]
        output_tif_path = os.path.join(output_folder, f'pred_{base_name}.tif')

        try:
            process_single_tiff(model, input_path, output_tif_path)
        except Exception as e:
            print(f'[错误] 处理失败: {input_path}')
            print(f'       原因: {e}')


if __name__ == '__main__':
    main()