# Copyright (c) OpenMMLab. All rights reserved.
import time

from mmengine.hooks import Hook
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS


@HOOKS.register_module()
class EpochTimerHook(Hook):
    """Timer hook for epoch-based training.

    It logs epoch wall-clock time and writes it to visualizer backends
    (e.g. TensorBoard) as `train/epoch_time_sec`.
    """

    priority = 'NORMAL'

    def __init__(self, log_key: str = 'train/epoch_time_sec') -> None:
        self.log_key = log_key
        self._epoch_start = 0.0
        self._visualizer = Visualizer.get_current_instance()

    def before_train_epoch(self, runner) -> None:
        self._epoch_start = time.perf_counter()

    def after_train_epoch(self, runner) -> None:
        epoch_time = time.perf_counter() - self._epoch_start
        runner.logger.info(
            f'[EpochTimerHook] Epoch {runner.epoch + 1} time: '
            f'{epoch_time:.2f}s ({epoch_time / 60.0:.2f} min)')
        self._visualizer.add_scalars(
            {self.log_key: epoch_time}, step=runner.epoch + 1)
