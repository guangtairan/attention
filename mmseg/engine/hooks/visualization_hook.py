# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import os.path as osp
import warnings
from typing import Optional, Sequence
import torch
import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample


@HOOKS.register_module()
class SegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of visualization. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.interval = interval
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for visualization will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False or mode == 'train':
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            for output in outputs:
                img_path = output.img_path
                img_bytes = fileio.get(
                    img_path, backend_args=self.backend_args)
                # img = mmcv.imfrombytes(img_bytes, channel_order='rgb')
                # 用 tifffile 解码 7 通道（不要用 channel_order='rgb'，否则会走 cv2.cvtColor）
                img = mmcv.imfrombytes(img_bytes, flag='unchanged', backend='tifffile')

                # 兼容 CHW(7,H,W) -> HWC(H,W,7)
                if img.ndim == 3 and img.shape[0] == 5 and img.shape[-1] != 5:
                     img = img.transpose(1, 2, 0)
                 # 只取前三个波段做“真彩 RGB”
                img = img[..., :3]   # 先按你说的“前三个波段就是可见光”
                # 如果你发现颜色不对（常见是数据存的顺序是 B,G,R），就改成：
                # img = img[..., [2, 1, 0]]

                # 为了能正常写图/可视化：拉伸到 0~255 并转 uint8（对 uint16/float 很关键）
                if img.dtype != np.uint8:
                    img_f = img.astype(np.float32)
                    out = np.empty_like(img_f, dtype=np.uint8)
                    for c in range(img_f.shape[2]):
                        lo, hi = np.percentile(img_f[..., c], (2, 98))
                        if hi <= lo:
                             out[..., c] = 0
                        else:
                            x = (img_f[..., c] - lo) / (hi - lo)
                            out[..., c] = (np.clip(x, 0, 1) * 255).astype(np.uint8)
                    img = out
                if not hasattr(self, '_printed_img_shape'):
                    runner.logger.info(f'[SegVis] decoded img shape={img.shape}, dtype={img.dtype}')
                    self._printed_img_shape = True
                    
                
                window_name = f'{mode}_{osp.basename(img_path)}'
                if not hasattr(self, "_printed_pred"):
                    pred = output.pred_sem_seg.data  # (1,H,W)
                    u, c = torch.unique(pred, return_counts=True)
                    runner.logger.info(f"[PRED] unique={u.tolist()}, counts={c.tolist()}")
                    # 顺便看 GT
                    gt = output.gt_sem_seg.data
                    ug, cg = torch.unique(gt, return_counts=True)
                    runner.logger.info(f"[GT] unique={ug.tolist()}, counts={cg.tolist()}")
                    self._printed_pred = True
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    data_sample=output,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)
