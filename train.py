# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp
import os
import torch.nn as nn
from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner
import torch

from mmseg.registry import RUNNERS

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='resume from the latest checkpoint in the work_dir automatically')
    parser.add_argument(
        '--amp',
        action='store_true',
        default=False,
        help='enable automatic-mixed-precision training')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args                                                                    


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    # enable automatic-mixed-precision training
    if args.amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # resume training
    cfg.resume = args.resume

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)

    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
    has_adapter = hasattr(runner.model.backbone, 'adapter')
    if has_adapter:
        w0 = runner.model.backbone.adapter.weight.detach().cpu().clone()
        print('[DEBUG] adapter weight snapshot saved')

        m = runner.model
        w = m.backbone.adapter.weight.detach().cpu()
        print('adapter nonzero:', int((w != 0).sum()))
        print('adapter diag:',
              float(w[0, 0, 0, 0]),
              float(w[1, 1, 0, 0]),
              float(w[2, 2, 0, 0]))
        print('BACKBONE TYPE:', type(m.backbone).__name__)
        print('ADAPTER WEIGHT:', m.backbone.adapter.weight.shape)

        if hasattr(runner.model.backbone, 'backbone'):
            inner = runner.model.backbone.backbone
            print('INNER BACKBONE:', type(inner).__name__)
            print('INNER TOP MODULES:', list(inner._modules.keys()))
            for name, mod in inner.named_modules():
                if isinstance(mod, nn.Conv2d):
                    print('FIRST CONV NAME:', name)
                    print('FIRST CONV WEIGHT:', mod.weight.shape)
                    break
    else:
        print('[DEBUG] backbone has no adapter, skip adapter debug.')

    # start training
    runner.train()

    if has_adapter:
        w1 = runner.model.backbone.adapter.weight.detach().cpu()
        print('[DEBUG] adapter weight change:',
              float((w1 - w0).abs().mean()))

if __name__ == '__main__':
    main()

