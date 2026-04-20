from osgeo import gdal
import numpy as np
import os
import torch
from mmseg.apis import init_model, inference_model
import mmcv
from tqdm import tqdm

# ================= 配置区域 =================
config_file = r'E:\Desktop\mmsegmentation-seven\configs\danet\danet_r50-d8_4xb4-20k_forest-512x512.py' 
checkpoint_file = r'work_dirs/danet_r50-d8_4xb4-20k_forest-512x512/best_mIoU_epoch_93.pth'
input_folder = r'E:\Desktop\5.20\融合'
output_folder = r'E:\Desktop\5.20\预测1'

crop_size = (512, 512)
overlap = 128  
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NODATA_VALUE = 255  # 预测结果中的透明背景值
# ===========================================

def read_tiff_with_gdal(file_path):
    dataset = gdal.Open(file_path)
    if dataset is None:
        return None, None, None, None

    count = dataset.RasterCount
    bands = []
    for i in range(1, count + 1):
        bands.append(dataset.GetRasterBand(i).ReadAsArray())

    img = np.dstack(bands)
    
    # === 【关键修改】更严格的 Mask 生成逻辑 ===
    # 为了防止 DEM 或 IR 波段在背景处有杂值，我们只使用 RGB (前3个波段) 来定义形状。
    # 假设前3个通道是 RGB。如果你的顺序不一样，请调整切片 [:3]
    if img.shape[2] >= 3:
        # 只检查前3个波段。只要 RGB 均不为0，才视为有效区域。
        # 使用 np.any 检查这三个波段中是否有任意一个非0 (避免纯黑像素被误删)
        # 或者更严格：检查 sum > 10 (避免压缩噪声)
        rgb_part = img[:, :, :3]
        valid_mask = np.any(rgb_part != 0, axis=2)
    else:
        # 如果波段少于3个，检查所有波段
        valid_mask = np.any(img != 0, axis=2)
    
    # 转换为 0/255 的掩膜
    mask = np.where(valid_mask, 255, 0).astype(np.uint8)
    
    # 【可选】调试打印：看看 Mask 到底覆盖了多少区域
    # print(f"  Valid Area Ratio: {np.mean(mask == 255):.2%}")
    
    geo_transform = dataset.GetGeoTransform()
    projection = dataset.GetProjection()
    
    return img, mask, geo_transform, projection

def pad_image(image, crop_size):
    h, w = image.shape[:2]
    pad_h = (crop_size[0] - h % crop_size[0]) % crop_size[0]
    pad_w = (crop_size[1] - w % crop_size[1]) % crop_size[1]
    # 使用 reflect 填充，减少边缘硬截断对模型的影响
    if len(image.shape) == 3:
        return np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        return np.pad(image, ((0, pad_h), (0, pad_w)), mode='reflect')

def crop_image(image, crop_size, overlap):
    height, width = image.shape[:2]
    step_size = (crop_size[0] - overlap, crop_size[1] - overlap)
    patches, coords = [], []
    for i in range(0, height - crop_size[0] + 1, step_size[0]):
        for j in range(0, width - crop_size[1] + 1, step_size[1]):
            patches.append(image[i:i+crop_size[0], j:j+crop_size[1]])
            coords.append((i, j, i+crop_size[0], j+crop_size[1]))
    return patches, coords

def predict_patches_mmseg(model, patches):
    pred_patches = []
    # 这里的 batch_size 可以设大一点以加速，但为了简单保持循环
    for patch in patches:
        result = inference_model(model, patch)
        if hasattr(result, 'pred_sem_seg'):
            pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
        else:
            pred_mask = result[0]
        pred_patches.append(pred_mask.astype(np.uint8))
    return pred_patches

def stitch_images(patches, coords, image_shape, overlap):
    # 初始化整个画布为 NODATA
    stitched_image = np.full(image_shape, NODATA_VALUE, dtype=np.uint8)
    
    for patch, (top, left, bottom, right) in zip(patches, coords):
        off = overlap // 2
        
        # 简单的中心裁剪逻辑
        h_patch, w_patch = patch.shape
        
        # 目标位置
        t_top = top + off
        t_bottom = bottom - off
        t_left = left + off
        t_right = right - off
        
        # 源位置
        s_top = off
        s_bottom = h_patch - off
        s_left = off
        s_right = w_patch - off
        
        # 边界处理：防止第一块和最后一块切多了
        if top == 0: t_top, s_top = top, 0
        if left == 0: t_left, s_left = left, 0
        if bottom == image_shape[0]: t_bottom, s_bottom = bottom, h_patch
        if right == image_shape[1]: t_right, s_right = right, w_patch
            
        # 安全检查：防止维度不匹配
        t_h, t_w = t_bottom - t_top, t_right - t_left
        s_h, s_w = s_bottom - s_top, s_right - s_left
        
        # 如果尺寸有微小偏差，取最小的一致尺寸
        valid_h = min(t_h, s_h)
        valid_w = min(t_w, s_w)
        
        stitched_image[t_top:t_top+valid_h, t_left:t_left+valid_w] = \
            patch[s_top:s_top+valid_h, s_left:s_left+valid_w]
            
    return stitched_image

def save_tiff_with_geo(output_path, image_data, geo_transform, projection, nodata_val):
    driver = gdal.GetDriverByName('GTiff')
    h, w = image_data.shape
    out_raster = driver.Create(output_path, w, h, 1, gdal.GDT_Byte)
    out_raster.SetGeoTransform(geo_transform)
    out_raster.SetProjection(projection)
    
    out_band = out_raster.GetRasterBand(1)
    out_band.WriteArray(image_data)
    out_band.SetNoDataValue(nodata_val) # 设置透明值
    out_raster.FlushCache()

def main():
    print(f"Loading model from: {config_file}")
    model = init_model(config_file, checkpoint_file, device=device)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    for tiff_file in tqdm(tiff_files):
        input_path = os.path.join(input_folder, tiff_file)
        output_path = os.path.join(output_folder, f"pred_{tiff_file}")
        
        # 1. 读取数据
        img, mask, geo, proj = read_tiff_with_gdal(input_path)
        if img is None: continue
        
        # 2. 预测流程
        padded_img = pad_image(img, crop_size)
        patches, coords = crop_image(padded_img, crop_size, overlap)
        pred_patches = predict_patches_mmseg(model, patches)
        stitched = stitch_images(pred_patches, coords, padded_img.shape[:2], overlap)
        
        # 3. 裁剪回原图大小
        final_pred = stitched[:img.shape[0], :img.shape[1]]
        
        # 4. 【核心修复】应用掩膜进行“雕刻”
        # 确保只有 Mask 范围内有值，其他地方强制设为 NODATA
        if mask is not None:
            final_pred[mask == 0] = NODATA_VALUE
            
        save_tiff_with_geo(output_path, final_pred, geo, proj, NODATA_VALUE)

if __name__ == '__main__':
    main()