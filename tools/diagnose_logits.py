#!/usr/bin/env python3
"""Chẩn đoán AUC = NaN: logit tràn ở FP16 hay mô hình mất ổn định thật ở FP32?

    python tools/diagnose_logits.py --config config_loku_iu_kaggle.yaml --seed 42 \
        --ckpt <.../checkpoints/latest.pt> --override "..."

Ba giả thuyết cần tách bạch:
  A. logit KHÔNG hữu hạn ngay ở FP32   -> mô hình sụp đổ thật về mặt số học.
  B. logit hữu hạn ở cả hai, AUC vẫn NaN -> lỗi hàm metric.
  C. logit hữu hạn ở FP32 nhưng tràn dưới autocast FP16 lúc eval -> artefact precision,
     KHÔNG phải mô hình sụp đổ. Kết luận trong báo cáo phải viết khác hẳn A.

Checkpoint của P3 CHỈ chứa trọng số LoRA; phép trừ bù FILA (W* = W − B*A*) không được
ghi ra đĩa nên phải dựng lại bằng setup_experiment (tất định với cùng seed/dữ liệu).
"""
import argparse
import os
import sys

import numpy as np
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import training.adv_common as C


def collect_logits(model, dataset, device, cfg, autocast, batch_size=16):
    """Trả (logit_ảnh, logit_văn_bản, nhãn) — chạy dưới autocast hoặc FP32 thuần."""
    from torch.utils.data import DataLoader
    C.set_eval_autocast(autocast)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    li, lt, ys = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
            inputs, label_raw, _ = C.get_model_inputs(cfg, batch, device)
            with C._eval_autocast():
                out = C.safe_forward(model, inputs)
            li.append(out[1].float().cpu()); lt.append(out[3].float().cpu())
            ys.append(label_raw.long().view(-1).cpu())
    return torch.cat(li), torch.cat(lt), torch.cat(ys)


def report(tag, li, lt, ys):
    print(f'\n--- {tag} ---')
    for name, x in (('logit_ảnh', li), ('logit_văn_bản', lt)):
        fin = torch.isfinite(x)
        print(f'  {name:14} shape {tuple(x.shape)}  hữu hạn {int(fin.sum())}/{x.numel()}'
              f'  NaN {int(torch.isnan(x).sum())}  Inf {int(torch.isinf(x).sum())}')
        if int(fin.sum()):
            v = x[fin]
            print(f'  {"":14} min {v.min():.4g}  max {v.max():.4g}  |max| {v.abs().max():.4g}'
                  f'   (ngưỡng tràn FP16 = 65504)')
    u, c = torch.unique(ys, return_counts=True)
    print(f'  nhãn: {dict(zip(u.tolist(), c.tolist()))}')
    return bool(torch.isfinite(li).all() and torch.isfinite(lt).all())


def try_auc(li, lt, ys):
    """Tính AUC bằng ĐÚNG hàm của pipeline để tái hiện NaN."""
    from joint_img_txt.metrics import compute_auc
    from scipy.special import softmax
    n = int(ys.max().item()) + 1
    oh = np.eye(max(n, 2))[ys.numpy()]
    for name, x in (('ảnh', li), ('văn bản', lt)):
        try:
            p = softmax(x.numpy(), axis=1)
            print(f'  AUC ({name}): {compute_auc(oh, p)}'
                  f'   | softmax có NaN: {bool(np.isnan(p).any())}')
        except Exception as e:
            print(f'  AUC ({name}): LỖI {type(e).__name__}: {e}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='latest.pt / cand_S2.pt (chỉ chứa LoRA)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--override', default=None)
    ap.add_argument('--scheme', default='uni_nokd')
    a = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(a.config), 'p3')
    cfg_d['random_seed'] = int(a.seed)
    cfg_d = C.apply_overrides(cfg_d, a.override)
    cfg = C.Cfg(cfg_d)
    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('🔧 Dựng lại pipeline P3 (Fisher + FILA) để tái tạo W* — checkpoint chỉ có LoRA...')
    ctx = C.setup_experiment(cfg, device)

    print(f'📦 Nạp {a.ckpt}')
    payload = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    state = payload.get('trainable_state', payload.get('lora_state', payload))
    print(f'   epoch trong checkpoint: {payload.get("epoch")}  |  {len(state)} tensor')

    from joint_img_txt.model import ImageTextModel
    from peft import get_peft_model
    base = ImageTextModel.from_pretrained(ctx['base_p']).to(device)
    peft = get_peft_model(base, ctx['peft_cfg'])
    C.apply_fila_subtraction(peft, ctx.get('fila_subtraction', {}))
    peft.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)
    model = peft.merge_and_unload()

    ds = ctx['datasets']
    for split in ('forget', 'test_final'):
        print(f'\n{"=" * 66}\nTẬP: {split}  (n = {len(ds[split])})\n{"=" * 66}')
        li16, lt16, ys = collect_logits(model, ds[split], device, cfg, autocast=True)
        ok16 = report('FP16 autocast (đúng như lúc chạy thật)', li16, lt16, ys)
        try_auc(li16, lt16, ys)

        li32, lt32, _ = collect_logits(model, ds[split], device, cfg, autocast=False)
        ok32 = report('FP32 (tắt autocast)', li32, lt32, ys)
        try_auc(li32, lt32, ys)

        print(f'\n>>> KẾT LUẬN cho {split}:')
        if not ok32:
            print('    TRƯỜNG HỢP A — logit KHÔNG hữu hạn ngay ở FP32.')
            print('    Mô hình mất ổn định số thật sự. Báo cáo: dùng checkpoint S2 làm kết quả chính.')
        elif ok32 and not ok16:
            print('    TRƯỜNG HỢP C — FP32 hữu hạn, FP16 tràn.')
            print('    NaN là ARTEFACT của autocast lúc eval, KHÔNG phải mô hình sụp đổ.')
            print('    Phải sửa câu giải thích trong báo cáo và dùng số FP32.')
        else:
            print('    FP32 và FP16 đều hữu hạn — nếu AUC vẫn NaN thì là TRƯỜNG HỢP B (lỗi metric).')
    C.set_eval_autocast(True)


if __name__ == '__main__':
    main()
