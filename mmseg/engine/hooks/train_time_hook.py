# Copyright (c) OpenMMLab. All rights reserved.
import json
import os.path as osp
import time
from datetime import datetime
from typing import List

from mmengine.hooks import Hook

from mmseg.registry import HOOKS


@HOOKS.register_module()
class TrainTimeHook(Hook):
    """Record epoch-level and total wall-clock training time.

    The hook writes a summary json to:
    ``{work_dir}/train_time_summary.json``.
    """

    priority = 'LOWEST'

    def __init__(self, summary_filename: str = 'train_time_summary.json') -> None:
        self.summary_filename = summary_filename
        self._train_start: float = 0.0
        self._epoch_start: float = 0.0
        self._epoch_seconds: List[float] = []

    def before_train(self, runner) -> None:
        self._epoch_seconds = []
        self._train_start = time.perf_counter()
        runner.logger.info('[TrainTimeHook] Start wall-clock timer.')

    def before_train_epoch(self, runner) -> None:
        self._epoch_start = time.perf_counter()

    def after_train_epoch(self, runner) -> None:
        elapsed = time.perf_counter() - self._epoch_start
        self._epoch_seconds.append(elapsed)
        runner.logger.info(
            f'[TrainTimeHook] Epoch {runner.epoch + 1} wall time: '
            f'{elapsed:.2f}s ({elapsed / 60.0:.2f} min)')

    def after_train(self, runner) -> None:
        total_seconds = time.perf_counter() - self._train_start
        avg_epoch_seconds = (
            sum(self._epoch_seconds) / len(self._epoch_seconds)
            if self._epoch_seconds else 0.0)

        summary = {
            'generated_at': datetime.now().isoformat(timespec='seconds'),
            'work_dir': runner.work_dir,
            'max_epochs': getattr(runner.train_loop, 'max_epochs', None),
            'epochs_recorded': len(self._epoch_seconds),
            'total_seconds': total_seconds,
            'total_minutes': total_seconds / 60.0,
            'total_hours': total_seconds / 3600.0,
            'avg_epoch_seconds': avg_epoch_seconds,
            'avg_epoch_minutes': avg_epoch_seconds / 60.0,
            'epoch_seconds': self._epoch_seconds,
        }

        out_path = osp.join(runner.work_dir, self.summary_filename)
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        runner.logger.info(
            '[TrainTimeHook] Total wall time: '
            f'{total_seconds:.2f}s ({total_seconds / 3600.0:.3f} h).')
        runner.logger.info(f'[TrainTimeHook] Summary saved to: {out_path}')
