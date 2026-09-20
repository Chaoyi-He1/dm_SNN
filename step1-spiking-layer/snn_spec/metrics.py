"""验收指标的定义(软件侧实现,行为级仿真器与 SPICE 比对时共用)。

- 发放事件 precision/recall 按正负分:三值神经元对 +1 和 -1 事件分别统计;全零输出不会被"通过"。
- 符号错误率:软硬件都发放但符号相反的事件,占两边都发放的事件的比例。
- 状态误差:||S_ckt - S_sw||_F 分别除以命题上界 B(稳定性检查)和校准集实际状态尺度(真实偏差)。
- 一致率:逐元素相等的比例,只作辅助观察量(它会被永不发放的电路"通过")。
"""
import torch


def _prf(pred_mask, ref_mask):
    tp = (pred_mask & ref_mask).sum().item()
    n_pred = pred_mask.sum().item()
    n_ref = ref_mask.sum().item()
    precision = tp / n_pred if n_pred > 0 else (1.0 if n_ref == 0 else 0.0)
    recall = tp / n_ref if n_ref > 0 else 1.0
    return precision, recall, tp, n_pred, n_ref


def spike_prf_by_sign(ckt, ref):
    """ckt, ref: 同形状的脉冲张量,取值 {-1,0,+1}(二值时只有 {0,1})。"""
    pp, pr, ptp, pn, prn = _prf(ckt == 1, ref == 1)
    np_, nr, ntp, nn_, nrn = _prf(ckt == -1, ref == -1)
    return dict(pos_precision=pp, pos_recall=pr, neg_precision=np_, neg_recall=nr,
                pos_tp=ptp, pos_pred=pn, pos_ref=prn, neg_tp=ntp, neg_pred=nn_, neg_ref=nrn)


def sign_error_rate(ckt, ref):
    both = (ckt != 0) & (ref != 0)
    n_both = both.sum().item()
    if n_both == 0:
        return 0.0
    return ((torch.sign(ckt) != torch.sign(ref)) & both).sum().item() / n_both


def state_error(S_ckt, S_sw, bound, ref_scale=None):
    """返回 by_bound = ||ΔS||_F / bound 与 by_scale = ||ΔS||_F / ref_scale(未给 ref_scale 时用 ||S_sw||_F)。"""
    diff = (S_ckt - S_sw).norm().item()
    scale = ref_scale if ref_scale is not None else S_sw.norm().item()
    by_scale = diff / scale if scale > 0 else float("nan")
    return dict(abs=diff, by_bound=diff / bound, by_scale=by_scale)


def spike_agreement(ckt, ref):
    return (ckt == ref).float().mean().item()
