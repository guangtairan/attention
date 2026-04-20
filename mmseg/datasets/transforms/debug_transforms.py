import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugImgOnce(BaseTransform):
    def transform(self, results):
        if not results.get('_dbg', False):
            img = results['img']
            mins = img.reshape(-1, img.shape[-1]).min(axis=0)
            maxs = img.reshape(-1, img.shape[-1]).max(axis=0)
            print('[DBG] shape/dtype:', img.shape, img.dtype)
            print('[DBG] per-channel min:', mins)
            print('[DBG] per-channel max:', maxs)
            results['_dbg'] = True
        return results
@TRANSFORMS.register_module()
class ScaleBand7To255(BaseTransform):
    def transform(self, results):
        img = results['img'].astype(np.float32)
        b7 = img[..., 6]
        # 用当前样本的 min/max 做一个快速缩放（先验证问题，之后再换成全局统计）
        mn, mx = float(b7.min()), float(b7.max())
        if mx > mn:
            img[..., 6] = (b7 - mn) / (mx - mn) * 255.0
        results['img'] = img
        return results   
import numpy as np
from mmcv.transforms import BaseTransform
from mmseg.registry import TRANSFORMS

@TRANSFORMS.register_module()
class DebugGTOnce(BaseTransform):
    def transform(self, results):
        if results.get('_dbg_gt', False):
            return results
        gt = results.get('gt_seg_map', None)
        if gt is not None:
            u = np.unique(gt)
            fg = int((gt == 1).sum())
            total = int(gt.size)
            print('[DBG-GT] unique=', u, 'fg=', fg, 'ratio=', fg/total)
            print('[DBG-GT] img=', results.get('img_path'), 'ann=', results.get('seg_map_path'))
        results['_dbg_gt'] = True
        return results
