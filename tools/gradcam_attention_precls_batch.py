#!/usr/bin/env python
"""Batch Grad-CAM generator for segmentation models.

Goal:
- Use exactly 5 validation images.
- Load epoch-100 checkpoints.
- Hook at "before final classification conv, after attention-enhanced feature".
- Use panoramic target: foreground logit spatial mean.

Notes on hook location:
- Standard decode heads: hook `model.decode_head.conv_seg` *input*
  via forward-pre-hook.
- Mask2FormerHead: hook `model.decode_head.pixel_decoder` output
  `mask_features` via forward-hook.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


def bootstrap_repo_path() -> None:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


# =========================
# User-editable experiment list
# =========================
# Fill these paths before full run.
MODEL_RUNS: List[Dict[str, str]] = [
    dict(
        name='ccnet',
        config='configs/paper_matrix/ccnet_r50-d8_4xb4-20k_forest-5band-matrix.py',
        checkpoint='work_dirs/ccnet_r50-d8_4xb4-20k_forest-5band-matrix/epoch_100.pth'),
    dict(
        name='danet',
        config='configs/paper_matrix/danet_r50-d8_4xb4-20k_forest-5band-matrix.py',
        checkpoint='work_dirs/danet_r50-d8_4xb4-20k_forest-5band-matrix/epoch_100.pth'),
    dict(
        name='fcn',
        config='configs/paper_matrix/fcn_r50-d8_4xb4-20k_forest-5band-matrix.py',
        checkpoint='work_dirs/fcn_r50-d8_4xb4-20k_forest-5band-matrix/epoch_100.pth'),
]

# Exactly 5 validation images are expected for official runs.
VAL_IMAGE_PATHS: List[str] = [
    r'data/dataset5pix/forest/images/validation/cropped_2_17.tif',
    r'data/dataset5pix/forest/images/validation/cropped_4_31.tif',
    r'data/dataset5pix/forest/images/validation/cropped_6_23.tif',
    r'data/dataset5pix/forest/images/validation/cropped_9_17.tif',
    r'data/dataset5pix/forest/images/validation/cropped_11_15.tif',
]


# Heads with explicit attention in decode head; used for reporting/check.
ATTN_HEADS = {
    'CCHead': ['cca'],
    'DAHead': ['pam', 'cam'],
    'PAMHead': ['pam'],
    'CAMHead': ['cam'],
    'NLHead': ['nl_block'],
    'DNLHead': ['dnl_block'],
    'GCHead': ['gc_block'],
    'ANNHead': ['fusion', 'context'],
    'DualAttentionExperimentHead': ['pam', 'cam'],
    'MHSA2DExperimentHead': ['q_proj', 'k_proj', 'v_proj'],
    'MHSATokenExperimentHead': ['mhsa'],
    'Mask2FormerHead': ['pixel_decoder', 'transformer_decoder'],
}


@dataclass
class CamRecord:
    model_name: str
    config: str
    checkpoint: str
    image_path: str
    hook_path: str
    decode_head_type: str
    target_desc: str
    out_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Grad-CAM batch runner')
    parser.add_argument('--device', default='cuda:0', help='cuda:0 or cpu')
    parser.add_argument(
        '--out-root',
        default='Grad-CAM',
        help='Output root folder')
    parser.add_argument(
        '--fg-index',
        type=int,
        default=-1,
        help='Foreground class index. -1 means auto infer + sanity check.')
    parser.add_argument(
        '--fg-prob-switch-thr',
        type=float,
        default=1e-6,
        help='If selected fg prob mean < thr and alt class is high, auto switch.')
    parser.add_argument(
        '--target-mode',
        choices=['region_weighted', 'panoramic'],
        default='panoramic',
        help='Grad-CAM backward target mode.')
    parser.add_argument(
        '--auto-select-foreground',
        action='store_true',
        help='Auto-select 5 images with highest predicted foreground ratio.')
    parser.add_argument(
        '--val-dir',
        default='data/dataset5pix/forest/images/validation',
        help='Validation image directory used by auto selection.')
    parser.add_argument(
        '--auto-select-k',
        type=int,
        default=5,
        help='Number of images to auto select.')
    parser.add_argument(
        '--auto-select-model-index',
        type=int,
        default=0,
        help='Use which MODEL_RUNS index as selector model.')
    parser.add_argument(
        '--auto-select-source',
        choices=['gt', 'pred'],
        default='gt',
        help='Use GT masks or model prediction for auto selection.')
    parser.add_argument(
        '--self-check',
        action='store_true',
        help='Run synthetic forward/backward check only')
    parser.add_argument(
        '--self-check-config',
        action='append',
        default=[],
        help='Config(s) for self-check. Repeatable.')
    parser.add_argument(
        '--self-check-size',
        type=int,
        default=64,
        help='Synthetic H=W used in self-check')
    return parser.parse_args()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def normalize01(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    xmin = float(x.min())
    xmax = float(x.max())
    return (x - xmin) / (xmax - xmin + eps)


def normalize_cam_for_vis(cam: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Robust normalization for visualization.

    Uses percentile clipping to avoid near-black maps caused by outliers or
    tiny dynamic ranges.
    """
    cam = cam.astype(np.float32)
    p1 = float(np.percentile(cam, 1))
    p99 = float(np.percentile(cam, 99))
    if p99 <= p1 + eps:
        return normalize01(cam, eps=eps)
    cam = np.clip((cam - p1) / (p99 - p1 + eps), 0, 1)
    return cam


def to_uint8(img01: np.ndarray) -> np.ndarray:
    return np.clip(img01 * 255.0, 0, 255).astype(np.uint8)


def read_multiband_image(path: str) -> np.ndarray:
    # 5-band TIFF cannot be reliably read by OpenCV backend in mmcv.
    # Try tifffile first, then GDAL, and fallback to mmcv for <=4 channels.
    try:
        import tifffile
        img = tifffile.imread(path)
        if img is not None:
            img = np.asarray(img)
            if img.ndim == 2:
                img = img[..., None]
            elif img.ndim == 3 and img.shape[0] <= 16 and img.shape[-1] > 16:
                # CHW -> HWC heuristic for remote-sensing rasters.
                img = np.transpose(img, (1, 2, 0))
            return img
    except Exception:
        pass

    try:
        from osgeo import gdal
        ds = gdal.Open(path)
        if ds is not None:
            arr = ds.ReadAsArray()  # C,H,W or H,W
            img = np.asarray(arr)
            if img.ndim == 2:
                img = img[..., None]
            else:
                img = np.transpose(img, (1, 2, 0))
            return img
    except Exception:
        pass

    import mmcv
    img = mmcv.imread(path, flag='unchanged')
    if img is None:
        raise RuntimeError(f'Failed to read image with tifffile/gdal/mmcv: {path}')
    if img.ndim == 2:
        img = img[..., None]
    return img


def visual_rgb_from_multiband(img: np.ndarray) -> np.ndarray:
    """Create display RGB from multiband image using first 3 channels."""
    if img.shape[2] >= 3:
        rgb = img[..., :3].astype(np.float32)
    elif img.shape[2] == 2:
        c0 = img[..., 0:1].astype(np.float32)
        c1 = img[..., 1:2].astype(np.float32)
        rgb = np.concatenate([c0, c1, c0], axis=2)
    else:
        c0 = img[..., 0:1].astype(np.float32)
        rgb = np.concatenate([c0, c0, c0], axis=2)

    # Robust percentile normalization for visualization.
    out = np.zeros_like(rgb, dtype=np.float32)
    for i in range(3):
        ch = rgb[..., i]
        lo = np.percentile(ch, 2)
        hi = np.percentile(ch, 98)
        if hi <= lo:
            out[..., i] = normalize01(ch)
        else:
            out[..., i] = np.clip((ch - lo) / (hi - lo), 0, 1)
    return out


def colorize_cam(cam01: np.ndarray) -> np.ndarray:
    import cv2

    cam_u8 = to_uint8(cam01)
    cam_color = cv2.applyColorMap(cam_u8, cv2.COLORMAP_JET)
    cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)
    return cam_color.astype(np.float32) / 255.0


def make_standalone_colorbar(height: int = 512,
                             vmin: float = 0.0,
                             vmax: float = 1.0) -> np.ndarray:
    """Create a standalone vertical JET colorbar image with tick labels."""
    import cv2

    h = int(height)
    bar_w = max(18, h // 24)
    pad = max(8, h // 64)
    label_w = 58

    canvas = np.ones((h, pad + bar_w + label_w, 3), dtype=np.uint8) * 255

    # Top is high value (red), bottom is low value (blue).
    grad = np.linspace(1.0, 0.0, h, dtype=np.float32).reshape(h, 1)
    grad = np.repeat(grad, bar_w, axis=1)
    bar_rgb = to_uint8(colorize_cam(grad))

    bx0 = pad
    bx1 = bx0 + bar_w
    canvas[:, bx0:bx1, :] = bar_rgb
    cv2.rectangle(canvas, (bx0, 0), (bx1 - 1, h - 1), (0, 0, 0), 1)

    tx = bx1 + 6
    cv2.putText(canvas, f'{vmax:.1f}', (tx, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'{0.5 * (vmin + vmax):.1f}', (tx, h // 2 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'{vmin:.1f}', (tx, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    return canvas.astype(np.float32) / 255.0


def overlay(rgb01: np.ndarray, heat01: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    heat_rgb = colorize_cam(heat01)
    return np.clip((1 - alpha) * rgb01 + alpha * heat_rgb, 0, 1)


def save_png(path: str, img01: np.ndarray) -> None:
    import cv2

    img_u8 = to_uint8(img01)
    if img_u8.ndim == 3 and img_u8.shape[2] == 3:
        # cv2/mmcv expect BGR when writing 3-channel images.
        img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img_u8)


def get_num_input_channels(cfg) -> int:
    mean = cfg.get('data_preprocessor', {}).get('mean', None)
    if isinstance(mean, (list, tuple)) and len(mean) > 0:
        return int(len(mean))
    return 3


def resolve_hook_target_and_report(model) -> Tuple[torch.nn.Module, str, str, str, str]:
    """Resolve hook module and mode.

    Returns:
        hook_module, hook_path, decode_head_type, attn_hint, hook_mode
    """
    decode_head = model.decode_head
    decode_head_type = decode_head.__class__.__name__
    if decode_head_type == 'Mask2FormerHead':
        if not hasattr(decode_head, 'pixel_decoder'):
            raise RuntimeError('Mask2FormerHead has no pixel_decoder.')
        hook_mod = decode_head.pixel_decoder
        hook_path = 'decode_head.pixel_decoder -> mask_features (forward_hook output[0])'
        hook_mode = 'mask2former_mask_features'
    else:
        if not hasattr(decode_head, 'conv_seg'):
            raise RuntimeError(
                f'{decode_head_type} has no `conv_seg`; this script expects '
                'BaseDecodeHead-style classifier conv.')
        hook_mod = decode_head.conv_seg
        hook_path = 'decode_head.conv_seg (forward_pre_hook: input tensor)'
        hook_mode = 'conv_seg_input'

    if decode_head_type in ATTN_HEADS:
        attn_hint = 'attention modules in head: ' + ', '.join(ATTN_HEADS[decode_head_type])
    else:
        attn_hint = ('no explicit decode-head attention module mapping; '
                     'using last pre-classifier feature')
    return hook_mod, hook_path, decode_head_type, attn_hint, hook_mode


def prepare_data_for_grad(model, img_path: str):
    from mmseg.apis.inference import _preprare_data

    data, _ = _preprare_data(img_path, model)
    # Keep preprocessing path consistent with official inference.
    data = model.data_preprocessor(data, False)
    inputs = data['inputs']
    data_samples = data['data_samples']
    return inputs, data_samples


def unpack_main_logits(seg_logits):
    # DAHead in tensor mode returns tuple(pam_cam_out, pam_out, cam_out)
    if isinstance(seg_logits, (tuple, list)):
        return seg_logits[0]
    return seg_logits


def forward_main_seg_logits(model, inputs: torch.Tensor, data_samples):
    """Forward and return primary segmentation logits for Grad-CAM target."""
    decode_head_type = model.decode_head.__class__.__name__
    if decode_head_type == 'Mask2FormerHead':
        # Explicit path because EncoderDecoder(mode='tensor') does not pass
        # batch_data_samples into Mask2FormerHead.forward.
        x = model.extract_feat(inputs)
        all_cls_scores, all_mask_preds = model.decode_head(x, data_samples)
        mask_cls_results = all_cls_scores[-1]  # [B, Q, C+1]
        mask_pred_results = all_mask_preds[-1]  # [B, Q, H, W]
        cls_score = torch.softmax(mask_cls_results, dim=-1)[..., :-1]
        mask_pred = mask_pred_results.sigmoid()
        seg_logits = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        return seg_logits

    # Use the same path as official inference/evaluation to avoid behavior
    # mismatch observed with `mode='tensor'` on some decode heads.
    batch_img_metas = [sample.metainfo for sample in data_samples]
    seg_logits = model.encode_decode(inputs, batch_img_metas)
    return seg_logits


def foreground_panoramic_target(seg_logits: torch.Tensor, fg_index: int) -> torch.Tensor:
    if seg_logits.dim() != 4:
        raise RuntimeError(f'Expected 4D seg logits, got shape {tuple(seg_logits.shape)}')
    channels = int(seg_logits.shape[1])
    if channels == 1:
        fg_map = seg_logits[:, 0, :, :]
    else:
        if fg_index < 0 or fg_index >= channels:
            raise ValueError(f'fg_index={fg_index} out of range for channels={channels}')
        fg_map = seg_logits[:, fg_index, :, :]
    return fg_map.mean()


def foreground_region_weighted_target(seg_logits: torch.Tensor, fg_index: int) -> torch.Tensor:
    """Region-weighted foreground target.

    target = sum(fg_logit * weight_mask) / (sum(weight_mask) + eps)
    where weight_mask is predicted foreground soft mask (detached).
    """
    if seg_logits.dim() != 4:
        raise RuntimeError(f'Expected 4D seg logits, got shape {tuple(seg_logits.shape)}')

    b, c, _, _ = seg_logits.shape
    if c == 1:
        fg_logit = seg_logits[:, 0, :, :]
        weight = torch.sigmoid(seg_logits[:, 0, :, :]).detach()
    else:
        if fg_index < 0 or fg_index >= c:
            raise ValueError(f'fg_index={fg_index} out of range for channels={c}')
        fg_logit = seg_logits[:, fg_index, :, :]
        weight = torch.softmax(seg_logits, dim=1)[:, fg_index, :, :].detach()

    eps = 1e-6
    num = (fg_logit * weight).sum(dim=(1, 2))
    den = weight.sum(dim=(1, 2)) + eps
    target = (num / den).mean()

    # Safety fallback for pathological all-zero weights.
    if not torch.isfinite(target):
        target = fg_logit.mean()
    return target


def build_target(seg_logits: torch.Tensor, fg_index: int, target_mode: str) -> torch.Tensor:
    if target_mode == 'region_weighted':
        return foreground_region_weighted_target(seg_logits, fg_index)
    if target_mode == 'panoramic':
        return foreground_panoramic_target(seg_logits, fg_index)
    raise ValueError(f'Unsupported target_mode: {target_mode}')


def prediction_from_logits(seg_logits: torch.Tensor, fg_index: int) -> np.ndarray:
    with torch.no_grad():
        if seg_logits.shape[1] == 1:
            prob = torch.sigmoid(seg_logits[:, 0:1, :, :])
            pred = (prob > 0.5).long()
        else:
            pred = seg_logits.argmax(dim=1, keepdim=True)
    return pred[0, 0].detach().cpu().numpy().astype(np.int32)


def get_palette_from_model(model) -> Optional[List[List[int]]]:
    try:
        palette = model.dataset_meta.get('palette', None)
    except Exception:
        palette = None
    if not isinstance(palette, (list, tuple)) or len(palette) == 0:
        return None
    out: List[List[int]] = []
    for c in palette:
        if not isinstance(c, (list, tuple)) or len(c) < 3:
            return None
        out.append([int(c[0]), int(c[1]), int(c[2])])
    return out


def default_palette(num_classes: int, fg_index: int) -> List[List[int]]:
    # Background dark blue; foreground yellow for quick visual contrast.
    pal = [[30, 60, 160] for _ in range(max(2, num_classes))]
    if 0 <= fg_index < len(pal):
        pal[fg_index] = [255, 210, 0]
    return pal


def colorize_pred(pred: np.ndarray,
                  num_classes: int,
                  palette: Optional[List[List[int]]],
                  fg_index: int) -> np.ndarray:
    h, w = pred.shape
    if palette is None or len(palette) < num_classes:
        palette = default_palette(num_classes, fg_index)
    palette_np = np.asarray(palette, dtype=np.uint8)
    idx = np.clip(pred.astype(np.int64), 0, len(palette_np) - 1)
    out = palette_np[idx.reshape(-1)].reshape(h, w, 3)
    return out.astype(np.float32) / 255.0


def class_ratio_map(pred: np.ndarray) -> Dict[str, float]:
    total = float(pred.size)
    unique, counts = np.unique(pred.astype(np.int64), return_counts=True)
    return {str(int(u)): float(c / total) for u, c in zip(unique, counts)}


def compute_cam_from_precls(feat: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
    # feat/grad: [B, C, h, w]
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feat).sum(dim=1, keepdim=True))
    return cam


def infer_fg_index_from_meta(model, channels: int) -> int:
    if channels <= 1:
        return 0
    classes = None
    try:
        classes = model.dataset_meta.get('classes', None)
    except Exception:
        classes = None
    if isinstance(classes, (list, tuple)) and len(classes) == channels:
        bg_candidates = {'background', 'bg', 'backgroud'}
        bg_idx = None
        for i, n in enumerate(classes):
            if str(n).strip().lower() in bg_candidates:
                bg_idx = i
                break
        if bg_idx is not None:
            for i in range(channels):
                if i != bg_idx:
                    return i
    # Fallback for multi-class when metadata is unavailable.
    return 1 if channels > 1 else 0


def auto_fix_fg_index(model,
                      image_paths: List[str],
                      device: str,
                      fg_index: int,
                      switch_thr: float = 1e-6) -> int:
    """Auto infer/correct foreground index to reduce all-blue/all-red pitfalls."""
    if not image_paths:
        return fg_index

    # Probe first image once.
    inputs, data_samples = prepare_data_for_grad(model, image_paths[0])
    inputs = inputs.to(device)
    seg_logits = forward_main_seg_logits(model, inputs, data_samples)
    channels = int(seg_logits.shape[1])
    if channels <= 1:
        return 0

    if fg_index < 0:
        fg_index = infer_fg_index_from_meta(model, channels)
        print(f'==> [fg-index] auto inferred from metadata: {fg_index}')

    if channels != 2:
        return fg_index

    alt = 1 - fg_index
    with torch.no_grad():
        sm = torch.softmax(seg_logits, dim=1)
        p_fg = float(sm[:, fg_index, :, :].mean().item())
        p_alt = float(sm[:, alt, :, :].mean().item())

    if p_fg < switch_thr and p_alt > 0.1:
        print(
            f'==> [fg-index] auto-switch {fg_index} -> {alt} '
            f'(mean prob fg={p_fg:.3e}, alt={p_alt:.3e})')
        return alt
    print(
        f'==> [fg-index] keep {fg_index} '
        f'(mean prob fg={p_fg:.3e}, alt={p_alt:.3e})')
    return fg_index


def list_val_images(val_dir: str) -> List[str]:
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f'Validation dir not found: {val_dir}')
    exts = {'.tif', '.tiff', '.png', '.jpg', '.jpeg', '.bmp'}
    files = []
    for n in sorted(os.listdir(val_dir)):
        p = os.path.join(val_dir, n)
        if os.path.isfile(p) and os.path.splitext(n.lower())[1] in exts:
            files.append(p)
    if not files:
        raise RuntimeError(f'No image files found under: {val_dir}')
    return files


def fg_ratio_from_logits(seg_logits: torch.Tensor, fg_index: int) -> float:
    with torch.no_grad():
        if seg_logits.shape[1] == 1:
            prob = torch.sigmoid(seg_logits[:, 0:1, :, :])
            pred = (prob > 0.5).float()
            return float(pred.mean().item())
        pred = seg_logits.argmax(dim=1)
        return float((pred == fg_index).float().mean().item())


def auto_select_foreground_images(model_runs: List[Dict[str, str]],
                                  device: str,
                                  val_dir: str,
                                  fg_index: int,
                                  k: int,
                                  model_index: int,
                                  out_root: str,
                                  source: str = 'gt') -> List[str]:
    from mmseg.apis import init_model

    all_imgs = list_val_images(val_dir)
    scored = []

    if source == 'gt':
        # val_dir expected like: .../images/validation
        val_dir_norm = os.path.normpath(val_dir)
        parent = os.path.dirname(val_dir_norm)          # .../images
        root = os.path.dirname(parent)                  # ...
        ann_dir = os.path.join(root, 'annotations', 'validation')
        if not os.path.isdir(ann_dir):
            raise FileNotFoundError(
                f'GT annotation dir not found for auto-select-source=gt: {ann_dir}')
        import mmcv
        for p in all_imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            ann_path = os.path.join(ann_dir, f'{stem}.png')
            if not os.path.exists(ann_path):
                continue
            gt = mmcv.imread(ann_path, flag='unchanged')
            if gt is None:
                continue
            if gt.ndim == 3:
                gt = gt[..., 0]
            ratio = float((gt == fg_index).mean())
            scored.append((ratio, p))
    else:
        if not model_runs:
            raise ValueError('MODEL_RUNS is empty, cannot auto select images.')
        if model_index < 0 or model_index >= len(model_runs):
            raise ValueError(f'auto_select_model_index out of range: {model_index}')

        spec = model_runs[model_index]
        cfg = spec['config']
        ckpt = spec['checkpoint']
        if not os.path.exists(cfg):
            raise FileNotFoundError(f'Config not found: {cfg}')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

        print(f'==> [auto-select] selector model: {spec["name"]}')
        model = init_model(cfg, ckpt, device=device)
        model.eval()
        try:
            for p in all_imgs:
                inputs, data_samples = prepare_data_for_grad(model, p)
                inputs = inputs.to(device)
                seg_logits = forward_main_seg_logits(model, inputs, data_samples)
                ratio = fg_ratio_from_logits(seg_logits, fg_index)
                scored.append((ratio, p))
        finally:
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not scored:
        raise RuntimeError('Auto selection found no valid candidates.')

    scored.sort(key=lambda x: x[0], reverse=True)
    selected = [p for _, p in scored[:k]]

    ensure_dir(out_root)
    save_path = os.path.join(out_root, 'auto_selected_images.json')
    payload = {
        'source': source,
        'val_dir': val_dir,
        'k': k,
        'selected': [{'path': p, 'fg_ratio': r} for r, p in scored[:k]],
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f'==> [auto-select] saved: {save_path}')
    return selected


def run_one_image(model,
                  img_path: str,
                  fg_index: int,
                  device: str,
                  out_dir: str,
                  target_mode: str) -> Dict[str, str]:
    ensure_dir(out_dir)

    hook_mod, hook_path, decode_head_type, attn_hint, hook_mode = resolve_hook_target_and_report(model)
    capture: Dict[str, Optional[torch.Tensor]] = {'feat': None}

    if hook_mode == 'mask2former_mask_features':
        def out_hook(_module, _inputs, output):
            feat = output[0] if isinstance(output, (tuple, list)) else output
            feat.retain_grad()
            capture['feat'] = feat

        handle = hook_mod.register_forward_hook(out_hook)
    else:
        def pre_hook(_module, inputs):
            feat = inputs[0]
            feat.retain_grad()
            capture['feat'] = feat

        handle = hook_mod.register_forward_pre_hook(pre_hook)

    try:
        model.zero_grad(set_to_none=True)
        img_raw = read_multiband_image(img_path)
        rgb01 = visual_rgb_from_multiband(img_raw)
        out_h, out_w = int(rgb01.shape[0]), int(rgb01.shape[1])

        inputs, data_samples = prepare_data_for_grad(model, img_path)
        inputs = inputs.to(device)

        seg_logits = forward_main_seg_logits(model, inputs, data_samples)
        target = build_target(seg_logits, fg_index, target_mode)
        target.backward()

        feat = capture['feat']
        if feat is None or feat.grad is None:
            raise RuntimeError('Failed to capture pre-classifier feature/gradient.')
        cam = compute_cam_from_precls(feat, feat.grad)
        cam = torch.nn.functional.interpolate(
            cam, size=(out_h, out_w), mode='bilinear', align_corners=False)
        cam01 = normalize_cam_for_vis(cam[0, 0].detach().cpu().numpy())

        seg_logits_up = torch.nn.functional.interpolate(
            seg_logits, size=(out_h, out_w), mode='bilinear', align_corners=False)
        pred = prediction_from_logits(seg_logits_up, fg_index)
        num_classes = int(max(2, seg_logits_up.shape[1]))
        palette = get_palette_from_model(model)
        pred_color01 = colorize_pred(pred, num_classes, palette, fg_index)
        if seg_logits_up.shape[1] == 1:
            pred_fg01 = (pred > 0).astype(np.float32)
        else:
            pred_fg01 = (pred == int(fg_index)).astype(np.float32)
        ov01 = overlay(rgb01, cam01, alpha=0.45)

        stem = os.path.splitext(os.path.basename(img_path))[0]
        p_orig = os.path.join(out_dir, f'{stem}_orig.png')
        p_pred = os.path.join(out_dir, f'{stem}_pred.png')
        p_pred_idx = os.path.join(out_dir, f'{stem}_pred_idx.png')
        p_pred_fg = os.path.join(out_dir, f'{stem}_pred_fgmask.png')
        p_cam = os.path.join(out_dir, f'{stem}_cam.png')
        p_overlay = os.path.join(out_dir, f'{stem}_overlay.png')

        save_png(p_orig, rgb01)
        save_png(p_pred, pred_color01)
        # Keep raw class-index map for strict reproducibility/debugging.
        save_png(p_pred_idx, np.repeat(pred.astype(np.float32)[..., None], 3, axis=2))
        save_png(p_pred_fg, np.repeat(pred_fg01[..., None], 3, axis=2))
        # Save plain CAM/overlay; a standalone shared colorbar is saved once per run.
        save_png(p_cam, colorize_cam(cam01))
        save_png(p_overlay, ov01)

        meta = {
            'image_path': img_path,
            'decode_head_type': decode_head_type,
            'hook_path': hook_path,
            'attention_hint': attn_hint,
            'target': target_mode,
            'saved': {
                'orig': p_orig,
                'pred': p_pred,
                'pred_idx': p_pred_idx,
                'pred_fgmask': p_pred_fg,
                'cam': p_cam,
                'overlay': p_overlay,
            },
            'class_ratio': class_ratio_map(pred),
        }
        return meta
    finally:
        handle.remove()


def run_batch(model_runs: List[Dict[str, str]],
              image_paths: List[str],
              device: str,
              out_root: str,
              fg_index: int,
              fg_prob_switch_thr: float = 1e-6,
              target_mode: str = 'panoramic') -> None:
    from mmseg.apis import init_model

    if len(image_paths) != 5:
        raise ValueError(f'Expected exactly 5 validation images, got {len(image_paths)}')
    for p in image_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f'Image not found: {p}')

    ensure_dir(out_root)
    shared_colorbar = os.path.join(out_root, 'cam_colorbar.png')
    save_png(shared_colorbar, make_standalone_colorbar(height=512))
    summary = []

    for spec in model_runs:
        name = spec['name']
        cfg = spec['config']
        ckpt = spec['checkpoint']
        model_out = os.path.join(out_root, name)
        ensure_dir(model_out)

        if not os.path.exists(cfg):
            raise FileNotFoundError(f'Config not found: {cfg}')
        if not os.path.exists(ckpt):
            raise FileNotFoundError(f'Checkpoint not found: {ckpt}')

        print(f'==> [{name}] load model')
        model = init_model(cfg, ckpt, device=device)
        model.eval()
        model_fg_index = auto_fix_fg_index(
            model=model,
            image_paths=image_paths,
            device=device,
            fg_index=fg_index,
            switch_thr=fg_prob_switch_thr)

        model_meta = {
            'name': name,
            'config': cfg,
            'checkpoint': ckpt,
            'fg_index': model_fg_index,
            'target_mode': target_mode,
            'shared_colorbar': shared_colorbar,
            'images': [],
        }
        for img_path in image_paths:
            print(f'    - Grad-CAM: {img_path}')
            one = run_one_image(model, img_path, model_fg_index, device, model_out, target_mode)
            model_meta['images'].append(one)

        meta_path = os.path.join(model_out, 'gradcam_meta.json')
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(model_meta, f, ensure_ascii=False, indent=2)
        summary.append(model_meta)

        # release GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary_path = os.path.join(out_root, 'gradcam_summary.json')
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f'All done. Summary saved to: {summary_path}')


def self_check(configs: List[str], device: str, size: int, fg_index: int) -> None:
    """Minimal verification:
    - Build model from config
    - Resolve pre-classifier hook
    - Run synthetic forward/backward once
    """
    from mmseg.apis import init_model

    if not configs:
        raise ValueError('Please pass at least one --self-check-config')

    for cfg in configs:
        if not os.path.exists(cfg):
            raise FileNotFoundError(f'Config not found: {cfg}')
        print(f'[self-check] {cfg}')
        model = init_model(cfg, checkpoint=None, device=device)
        model.eval()
        hook_mod, hook_path, decode_head_type, attn_hint, hook_mode = resolve_hook_target_and_report(model)
        print(f'  decode_head={decode_head_type}')
        print(f'  hook={hook_path}')
        print(f'  note={attn_hint}')

        cap = {'feat': None}

        if hook_mode == 'mask2former_mask_features':
            def out_hook(_m, _inputs, output):
                feat = output[0] if isinstance(output, (tuple, list)) else output
                feat.retain_grad()
                cap['feat'] = feat

            handle = hook_mod.register_forward_hook(out_hook)
        else:
            def pre_hook(_m, inputs):
                feat = inputs[0]
                feat.retain_grad()
                cap['feat'] = feat

            handle = hook_mod.register_forward_pre_hook(pre_hook)
        try:
            c = get_num_input_channels(model.cfg)
            x = torch.randn(1, c, size, size, device=device)
            model.zero_grad(set_to_none=True)
            # For Mask2Former, minimal metainfo sample is required in forward.
            if decode_head_type == 'Mask2FormerHead':
                from mmseg.structures.seg_data_sample import SegDataSample
                sample = SegDataSample()
                sample.set_metainfo({'img_shape': (size, size), 'ori_shape': (size, size)})
                data_samples = [sample]
            else:
                data_samples = None

            seg_logits = forward_main_seg_logits(model, x, data_samples)
            target = foreground_region_weighted_target(seg_logits, fg_index)
            target.backward()
            if cap['feat'] is None or cap['feat'].grad is None:
                raise RuntimeError('Hook or gradient capture failed.')
            _ = compute_cam_from_precls(cap['feat'], cap['feat'].grad)
            print('  self-check: PASS')
        finally:
            handle.remove()
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main() -> None:
    bootstrap_repo_path()
    args = parse_args()
    if args.self_check:
        self_check(args.self_check_config, args.device, args.self_check_size, args.fg_index)
        return

    if not MODEL_RUNS:
        raise ValueError('MODEL_RUNS is empty. Fill model config/checkpoint list first.')
    if args.auto_select_foreground:
        image_paths = auto_select_foreground_images(
            model_runs=MODEL_RUNS,
            device=args.device,
            val_dir=args.val_dir,
            fg_index=args.fg_index,
            k=args.auto_select_k,
            model_index=args.auto_select_model_index,
            out_root=args.out_root,
            source=args.auto_select_source)
    else:
        image_paths = VAL_IMAGE_PATHS

    if not image_paths:
        raise ValueError('No image paths available. Set VAL_IMAGE_PATHS or use --auto-select-foreground')

    run_batch(
        MODEL_RUNS,
        image_paths,
        args.device,
        args.out_root,
        args.fg_index,
        fg_prob_switch_thr=args.fg_prob_switch_thr,
        target_mode=args.target_mode)


if __name__ == '__main__':
    main()
