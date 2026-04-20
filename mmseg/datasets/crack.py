# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from abc import ABC
import mmengine.fileio as fileio
 
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
 
 
@DATASETS.register_module()
class CRACKDataset(BaseSegDataset):
    """My Crack dataset.
    Args:
        split (str): Split txt file
    """
 
    METAINFO = dict(
        classes=('background', 'feldspar', 'mica'),
        palette=[[0, 0, 0], [255, 0, 0], [0, 255, 0]]  # feldspar 为红色，mica 为绿色
    )
 
    def __init__(self,
                 img_suffix='.jpg',
                 seg_map_suffix='.png',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        assert fileio.exists(
            self.data_prefix['img_path'], backend_args=self.backend_args)


