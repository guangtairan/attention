# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict

import torch
from mmengine.hooks import Hook
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS


@HOOKS.register_module()
class EpochLossHook(Hook):
    """Aggregate training loss by epoch and write to visualization backends.

    This hook keeps default iter-level loss logging unchanged and adds an
    epoch-level curve (e.g. TensorBoard tag ``train/loss_epoch``).
    """

    priority = 'NORMAL'

    def __init__(self, log_key: str = 'train/loss_epoch') -> None:
        self.log_key = log_key
        self._loss_sum = 0.0
        self._loss_count = 0
        self._visualizer = None

    def before_train_epoch(self, runner) -> None:
        self._loss_sum = 0.0
        self._loss_count = 0

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: Any = None,
                         outputs: Dict[str, Any] = None) -> None:
        loss_value = self._extract_loss_value(outputs)
        if loss_value is None:
            return
        self._loss_sum += loss_value
        self._loss_count += 1

    def after_train_epoch(self, runner) -> None:
        if self._loss_count == 0:
            return
        if self._visualizer is None:
            self._visualizer = getattr(runner, 'visualizer', None)
            if self._visualizer is None:
                try:
                    self._visualizer = Visualizer.get_current_instance()
                except RuntimeError:
                    runner.logger.warning(
                        '[EpochLossHook] No active visualizer, skip writing '
                        'epoch loss to visual backends.')
                    return
        epoch_loss = self._loss_sum / self._loss_count
        self._visualizer.add_scalars({self.log_key: epoch_loss},
                                     step=runner.epoch + 1)
        runner.logger.info(
            f'[EpochLossHook] Epoch {runner.epoch + 1} mean loss: {epoch_loss:.6f}'
        )

    @staticmethod
    def _extract_loss_value(outputs: Dict[str, Any]) -> float:
        if outputs is None:
            return None

        # Most common mmengine path.
        log_vars = outputs.get('log_vars')
        if isinstance(log_vars, dict) and 'loss' in log_vars:
            return float(log_vars['loss'])

        # Fallback: tensor/scalar loss in outputs.
        raw_loss = outputs.get('loss')
        if raw_loss is None:
            return None
        if isinstance(raw_loss, torch.Tensor):
            return float(raw_loss.detach().mean().cpu().item())
        return float(raw_loss)
