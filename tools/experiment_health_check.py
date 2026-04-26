#!/usr/bin/env python
"""Experiment health checker for mmsegmentation configs.

Checks each target config in three stages:
1) parse: Config.fromfile
2) build: MODELS.build(cfg.model)
3) smoke (optional): one synthetic forward+backward step

Example:
  python tools/experiment_health_check.py --group matrix --group ce_only --smoke
  python tools/experiment_health_check.py --group all --smoke --device cuda --batch-size 2
  python tools/experiment_health_check.py --config-glob "configs/paper_matrix/*.py"
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional


GROUP_PATTERNS: Dict[str, List[str]] = {
    "matrix": ["configs/paper_matrix/*.py"],
    "ce_only": ["configs/paper_matrix/ce_only/*.py"],
    "double": ["configs/paper_double/d*.py"],
    "transformer": ["configs/paper_transformer/*.py"],
}
GROUP_PATTERNS["all"] = (
    GROUP_PATTERNS["matrix"]
    + GROUP_PATTERNS["ce_only"]
    + GROUP_PATTERNS["double"]
    + GROUP_PATTERNS["transformer"]
)


def bootstrap_repo_path() -> None:
    """Ensure repository root is on sys.path for custom_imports."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


@dataclass
class CheckResult:
    config: str
    parse_ok: bool = False
    build_ok: bool = False
    smoke_ok: Optional[bool] = None
    stage: str = ""
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "config": self.config,
            "parse_ok": self.parse_ok,
            "build_ok": self.build_ok,
            "smoke_ok": self.smoke_ok,
            "stage": self.stage,
            "error": self.error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Health check mmseg experiment configs")
    parser.add_argument(
        "--group",
        action="append",
        choices=sorted(GROUP_PATTERNS.keys()),
        help="Preset config group to include (repeatable).",
    )
    parser.add_argument(
        "--config-glob",
        action="append",
        default=[],
        help="Additional glob pattern(s) for config files.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one synthetic forward+backward step per config.",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for smoke test. auto prefers CUDA when available.",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Synthetic batch size for smoke test.")
    parser.add_argument("--height", type=int, default=64, help="Synthetic input height.")
    parser.add_argument("--width", type=int, default=64, help="Synthetic input width.")
    parser.add_argument(
        "--report",
        default="",
        help="JSON report path. Default: work_dirs/health_check/health_check_<timestamp>.json",
    )
    parser.add_argument("--fail-fast", action="store_true", help="Stop at first failed config.")
    return parser.parse_args()


def collect_configs(groups: Optional[List[str]], extra_globs: List[str]) -> List[str]:
    patterns: List[str] = []
    if groups:
        for g in groups:
            patterns.extend(GROUP_PATTERNS[g])
    patterns.extend(extra_globs)

    if not patterns:
        patterns = GROUP_PATTERNS["all"]

    configs = []
    for pattern in patterns:
        configs.extend(glob.glob(pattern))

    # de-dup + deterministic order
    uniq = sorted({os.path.normpath(p) for p in configs})
    return uniq


def detect_device(device_arg: str):
    import torch

    if device_arg == "cpu":
        return "cpu"
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available")
        return "cuda"
    return "cuda" if torch.cuda.is_available() else "cpu"


def infer_input_channels(cfg) -> int:
    mean = cfg.get("data_preprocessor", {}).get("mean", None)
    if isinstance(mean, (list, tuple)) and len(mean) > 0:
        return int(len(mean))

    backbone = cfg.get("model", {}).get("backbone", {})
    if isinstance(backbone, dict):
        if "in_channels" in backbone and isinstance(backbone["in_channels"], int):
            return int(backbone["in_channels"])
        nested = backbone.get("backbone_cfg", {})
        if isinstance(nested, dict) and isinstance(nested.get("in_channels", None), int):
            return int(nested["in_channels"])

    return 3


def infer_num_classes(cfg) -> int:
    decode_head = cfg.get("model", {}).get("decode_head", {})
    if isinstance(decode_head, dict):
        num_classes = decode_head.get("num_classes", None)
        if isinstance(num_classes, int) and num_classes > 1:
            return num_classes
    return 2


def run_smoke(model, cfg, device: str, batch_size: int, height: int, width: int) -> None:
    import torch
    from mmengine.structures import PixelData
    from mmseg.structures import SegDataSample

    channels = infer_input_channels(cfg)
    num_classes = infer_num_classes(cfg)

    inputs = torch.randn(batch_size, channels, height, width, device=device)
    samples = []
    for _ in range(batch_size):
        s = SegDataSample()
        gt = torch.randint(0, num_classes, (1, height, width), dtype=torch.long, device=device)
        s.gt_sem_seg = PixelData(data=gt)
        samples.append(s)

    losses = model.forward(inputs=inputs, data_samples=samples, mode="loss")
    total = None
    for v in losses.values():
        if isinstance(v, torch.Tensor):
            total = v if total is None else total + v
        elif isinstance(v, (list, tuple)):
            for t in v:
                if isinstance(t, torch.Tensor):
                    total = t if total is None else total + t

    if total is None:
        raise RuntimeError("No tensor loss found in model losses")

    total.backward()


def make_default_report_path() -> str:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join("help", "health_check")
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, f"health_check_{stamp}.json")


def main() -> int:
    args = parse_args()
    bootstrap_repo_path()

    configs = collect_configs(args.group, args.config_glob)
    if not configs:
        print("No configs found from given groups/globs.")
        return 2

    import torch
    from mmengine.config import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules

    register_all_modules(init_default_scope=True)

    device = detect_device(args.device)
    print(f"Device: {device}")
    print(f"Configs: {len(configs)}")

    results: List[CheckResult] = []

    for idx, config_path in enumerate(configs, start=1):
        print(f"[{idx}/{len(configs)}] {config_path}")
        result = CheckResult(config=config_path)
        model = None
        try:
            cfg = Config.fromfile(config_path)
            result.parse_ok = True
            result.stage = "parse"

            model = MODELS.build(cfg.model)
            result.build_ok = True
            result.stage = "build"

            if args.smoke:
                model = model.to(device)
                model.train()
                run_smoke(
                    model=model,
                    cfg=cfg,
                    device=device,
                    batch_size=args.batch_size,
                    height=args.height,
                    width=args.width,
                )
                result.smoke_ok = True
                result.stage = "smoke"
            else:
                result.smoke_ok = None
        except Exception:
            result.error = traceback.format_exc()
            if args.smoke and result.smoke_ok is None:
                result.smoke_ok = False
        finally:
            results.append(result)
            try:
                del model
            except Exception:
                pass
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if args.fail_fast and result.error:
            break

    parse_ok = sum(r.parse_ok for r in results)
    build_ok = sum(r.build_ok for r in results)
    smoke_total = sum(1 for r in results if r.smoke_ok is not None)
    smoke_ok = sum(1 for r in results if r.smoke_ok is True)
    failed = [r for r in results if r.error]

    print("=== Summary ===")
    print(f"Parse: {parse_ok}/{len(results)}")
    print(f"Build: {build_ok}/{len(results)}")
    if args.smoke:
        print(f"Smoke: {smoke_ok}/{smoke_total}")
    print(f"Failed: {len(failed)}")

    for r in failed:
        last = r.error.strip().splitlines()[-1] if r.error else ""
        print(f"- {r.config}: {last}")

    report_path = args.report or make_default_report_path()
    report_dir = os.path.dirname(report_path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    payload = {
        "timestamp": dt.datetime.now().isoformat(),
        "args": vars(args),
        "device": device,
        "summary": {
            "total": len(results),
            "parse_ok": parse_ok,
            "build_ok": build_ok,
            "smoke_total": smoke_total,
            "smoke_ok": smoke_ok,
            "failed": len(failed),
        },
        "results": [r.to_dict() for r in results],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Report written: {report_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
