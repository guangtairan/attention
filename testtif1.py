from osgeo import gdal
import numpy as np
import os
import torch
from mmseg.apis import init_model, inference_model
import mmcv
from tqdm import tqdm

# ================= 配置区域 =================
config_file = r'E:\Desktop\mmsegmentation-seven\configs\deeplabv3\deeplabv3_r50-d8_4xb4-20k_forest-512x512.py' 
checkpoint_file = r'E:\Desktop\mmsegmentation-seven\work_dirs\danet_r50-d8_4xb4-20k_forest-512x512\best_mIoU_epoch_100.pth'
input_folder = r'E:\Desktop\7.14peizhun\ronghe'
output_folder = r'E:\Desktop\7.14peizhun\yuce'


crop_size = (512, 512)
overlap = 128
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NODATA_VALUE = 255  # 背景无效值（用255标记背景区域）
# ===========================================

def read_tiff_with_gdal(file_path):
    """读取TIFF文件，生成背景掩膜"""
    dataset = gdal.Open(file_path)
    if dataset is None:
        return None, None, None, None

    count = dataset.RasterCount
    bands = []
    for i in range(1, count + 1):
        bands.append(dataset.GetRasterBand(i).ReadAsArray())

    img = np.dstack(bands)
    
    # 生成背景掩膜（只用前3个波段判断）
    if img.shape[2] >= 3:
        rgb_part = img[:, :, :3]
        valid_mask = np.any(rgb_part != 0, axis=2)
    else:
        valid_mask = np.any(img != 0, axis=2)
    
    mask = np.where(valid_mask, 255, 0).astype(np.uint8)
    
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    return img, mask, geo_transform, projection

def pad_image(image, crop_size):
    """使用零填充"""
    h, w = image.shape[:2]
    pad_h = (crop_size[0] - h % crop_size[0]) % crop_size[0]
    pad_w = (crop_size[1] - w % crop_size[1]) % crop_size[1]
    
    if len(image.shape) == 3:
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), 
                     mode='constant', constant_values=0)
    else:
        return np.pad(image, ((0, pad_h), (0, pad_w)), 
                     mode='constant', constant_values=0)

def crop_image(image, crop_size, overlap):
    """裁剪图像为多个patch"""
    height, width = image.shape[:2]
    step_size = (crop_size[0] - overlap, crop_size[1] - overlap)
    patches, coords = [], []
    for i in range(0, height - crop_size[0] + 1, step_size[0]):
        for j in range(0, width - crop_size[1] + 1, step_size[1]):
            patches.append(image[i:i+crop_size[0], j:j+crop_size[1]])
            coords.append((i, j, i+crop_size[0], j+crop_size[1]))
    return patches, coords

def predict_patches_mmseg(model, patches):
    """使用MMSeg模型预测patches"""
    pred_patches = []
    for patch in patches:
        result = inference_model(model, patch)
        if hasattr(result, 'pred_sem_seg'):
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        else:
            pred_mask = result[0]
        pred_patches.append(pred_mask.astype(np.uint8))
    return pred_patches

def stitch_images(patches, coords, image_shape, overlap):
    """改进的拼接函数，修复边界问题"""
    stitched_image = np.full(image_shape, NODATA_VALUE, dtype=np.uint8)
    
    for patch, (top, left, bottom, right) in zip(patches, coords):
        off = overlap // 2
        h_patch, w_patch = patch.shape
        
        # 目标位置（在拼接图上）
        t_top = top + off
        t_bottom = bottom - off
        t_left = left + off
        t_right = right - off
        
        # 源位置（在patch上）
        s_top = off
        s_bottom = h_patch - off
        s_left = off
        s_right = w_patch - off
        
        # 边界处理：第一块和最后一块不裁剪边缘
        if top == 0:
            t_top, s_top = top, 0
        if left == 0:
            t_left, s_left = left, 0
        if bottom >= image_shape[0]:
            t_bottom, s_bottom = image_shape[0], h_patch
        if right >= image_shape[1]:
            t_right, s_right = image_shape[1], w_patch
        
        # 安全检查：确保尺寸匹配
        t_h, t_w = t_bottom - t_top, t_right - t_left
        s_h, s_w = s_bottom - s_top, s_right - s_left
        valid_h = min(t_h, s_h)
        valid_w = min(t_w, s_w)
        
        # 拼接
        stitched_image[t_top:t_top+valid_h, t_left:t_left+valid_w] = \
            patch[s_top:s_top+valid_h, s_left:s_left+valid_w]
            
    return stitched_image

def save_tiff_with_geo(output_path, image_data, geo_transform, projection, nodata_val):
    """保存单通道TIFF，设置NoData值"""
    driver = gdal.GetDriverByName('GTiff')
    h, w = image_data.shape
    out_raster = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)
    
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(image_data)
    out_band.SetNoDataValue(nodata_val)
    out_raster.FlushCache()

def main():
    print(f"Loading model from: {config_file}")
    model = init_model(config_file, checkpoint_file, device=device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    for tiff_file in tqdm(tiff_files, desc="Processing TIFF files"):
        input_path = os.path.join(input_folder, tiff_file)
        output_path = os.path.join(output_folder, f"pred_{tiff_file}")
        
        # 1. 读取数据（生成mask）
        img, mask, geo, proj = read_tiff_with_gdal(input_path)
        if img is None: 
            print(f"Failed to read: {input_path}")
            continue
        
        # 2. 预测流程
        padded_img = pad_image(img, crop_size)
        patches, coords = crop_image(padded_img, crop_size, overlap)
        pred_patches = predict_patches_mmseg(model, patches)
        stitched = stitch_images(pred_patches, coords, padded_img.shape[:2], overlap)
        
        # 3. 裁剪回原图大小
        final_pred = stitched[:img.shape[0], :img.shape[1]]
        
        # 4. 确保只有0和1值（二值化处理）
        # 如果模型输出已经是0/1，这步可以跳过
        # 如果模型输出可能有其他值，取阈值二值化
        final_pred = np.where(final_pred > 0, 1, 0).astype(np.uint8)
        
        # 5. 应用背景掩膜（将背景区域标记为255）
        if mask is not None:
            final_pred[mask == 0] = NODATA_VALUE
        
        # 6. 保存单通道TIFF（只有0, 1, 255三种值）
        save_tiff_with_geo(output_path, final_pred, geo, proj, NODATA_VALUE)
        print(f"Prediction saved to: {output_path}")
        
        # 打印统计信息
        unique, counts = np.unique(final_pred, return_counts=True)
        print(f"  Pixel distribution: {dict(zip(unique, counts))}")

if __name__ == '__main__':
    main()