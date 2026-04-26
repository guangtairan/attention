#!/usr/bin/env python
"""Profile Params/FLOPs/latency/FPS for MMSeg models.

Example:
python scripts/profile_model_cost.py \
  configs/paper_double/d5_gated_fusion_r50_5band.py \
  --input-shape 5 512 512 --device cuda --warmup 50 --iters 200
"""
# #你给它一个 config（可选 checkpoint），它会：

# 构建 mmseg 模型
# 用你指定输入尺寸（默认 5x512x512）测试
# 先 warmup 再正式计时
# 打印 JSON，或保存到 --out 文件
import argparse
import json
import time
from typing import Optional, Tuple

import torch
from mmengine.config import Config
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.utils import import_modules_from_strings

from mmseg.registry import MODELS
from mmseg.utils import register_all_modules


def parse_args():
    parser = argparse.ArgumentParser(description='Profile model cost.')
    parser.add_argument('config', help='Config path')
    parser.add_argument('--checkpoint', default=None, help='Checkpoint path')
    parser.add_argument(
        '--input-shape',
        type=int,
        nargs=3,
        metavar=('C', 'H', 'W'),
        default=[5, 512, 512],
        help='Input tensor shape as C H W')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument(
        '--amp', action='store_true', help='Use autocast for latency test')
    parser.add_argument(
        '--out',
        default=None,
        help='Optional json output path. Example: work_dirs/d5_cost.json')
    return parser.parse_args()


def _build_model(cfg: Config, checkpoint: Optional[str], device: str):
    if 'custom_imports' in cfg:
        import_modules_from_strings(**cfg.custom_imports)

    model_cfg = cfg.model.copy()
    model_cfg['pretrained'] = None
    model = MODELS.build(model_cfg)
    if checkpoint:
        load_checkpoint(model, checkpoint, map_location='cpu')
    model.to(device)
    model.eval()
    return model


def _count_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _flops_str(model, input_shape: Tuple[int, int, int]) -> str:
    try:
        from mmengine.analysis import get_model_complexity_info
    except Exception:
        return 'N/A (mmengine.analysis unavailable)'

    class TensorForwardWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            try:
                return self.inner(inputs=x, mode='tensor')
            except TypeError:
                # Some decode heads (e.g. Mask2Former) require samples even in
                # internal forward. Fallback to predict mode for profiling.
                return self.inner(inputs=x, mode='predict')

    wrapper = TensorForwardWrapper(model).eval()
    try:
        result = get_model_complexity_info(
            wrapper, input_shape, show_table=False, show_arch=False)
        if isinstance(result, dict):
            return str(result.get('flops_str', result.get('flops', 'N/A')))
        return str(result)
    except Exception as e:
        return f'N/A ({type(e).__name__}: {e})'


@torch.no_grad()
def _latency_ms(model,
                input_shape: Tuple[int, int, int],
                batch_size: int,
                device: str,
                warmup: int,
                iters: int,
                use_amp: bool) -> float:
    c, h, w = input_shape
    x = torch.randn(batch_size, c, h, w, device=device)

    def _run_once():
        if use_amp and device == 'cuda':
            with torch.cuda.amp.autocast():
                try:
                    _ = model(inputs=x, mode='tensor')
                except TypeError:
                    _ = model(inputs=x, mode='predict')
        else:
            try:
                _ = model(inputs=x, mode='tensor')
            except TypeError:
                _ = model(inputs=x, mode='predict')

    for _ in range(warmup):
        _run_once()
    if device == 'cuda':
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _run_once()
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000.0 / iters / batch_size


def main():
    args = parse_args()
    register_all_modules(init_default_scope=True)
    cfg = Config.fromfile(args.config)

    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but not available.')

    model = _build_model(cfg, args.checkpoint, args.device)
    input_shape = tuple(args.input_shape)

    params = _count_params(model)
    flops = _flops_str(model, input_shape)
    latency = _latency_ms(
        model=model,
        input_shape=input_shape,
        batch_size=args.batch_size,
        device=args.device,
        warmup=args.warmup,
        iters=args.iters,
        use_amp=args.amp)
    fps = 1000.0 / latency if latency > 0 else float('inf')

    result = {
        'config': args.config,
        'checkpoint': args.checkpoint,
        'device': args.device,
        'input_shape_chw': list(input_shape),
        'batch_size': args.batch_size,
        'amp': args.amp,
        'params': params,
        'params_m': params / 1e6,
        'flops': flops,
        'latency_ms_per_image': latency,
        'fps': fps,
        'warmup': args.warmup,
        'iters': args.iters,
    }

    print(json.dumps(result, indent=2))

    if args.out:
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
