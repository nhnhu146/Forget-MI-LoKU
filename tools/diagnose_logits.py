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
    """Trả (logit_ảnh, logit_văn_bản, nhãn, onehot).

    Lấy one-hot ĐÚNG như perf_metrics làm (batch[5]) — không tự dựng bằng np.eye theo
    số lớp có mặt. Mô hình có đầu ra 4 lớp trong khi IU chỉ dùng lớp 0/1, nên dựng
    one-hot 2 cột sẽ lệch chiều với logit 4 cột và gây IndexError giả."""
    from torch.utils.data import DataLoader
    C.set_eval_autocast(autocast)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    li, lt, ys, ohs = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device) if torch.is_tensor(t) else t for t in batch)
            inputs, label_raw, _ = C.get_model_inputs(cfg, batch, device)
            with C._eval_autocast():
                out = C.safe_forward(model, inputs)
            li.append(out[1].float().cpu()); lt.append(out[3].float().cpu())
            ys.append(label_raw.long().view(-1).cpu())
            oh = batch[5].cpu().numpy()
            if oh.ndim == 1 or oh.shape[-1] == 1:
                oh = np.eye(4)[np.clip(oh.flatten().astype(int), 0, 3)]
            ohs.append(oh.astype(int))
    return torch.cat(li), torch.cat(lt), torch.cat(ys), np.concatenate(ohs, axis=0)


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


def try_auc(li, lt, oh, cfg):
    """Tái hiện ĐÚNG đường tính của perf_metrics: dùng logit ẢNH (outputs[1]),
    one-hot của dataset, và cùng output_channel_encoding."""
    from joint_img_txt.metrics import compute_auc
    from scipy.special import softmax
    enc = str(getattr(cfg, 'output_channel_encoding', 'multiclass'))
    for name, x in (('ảnh  <- perf_metrics dùng cái này', li), ('văn bản', lt)):
        try:
            p = softmax(x.numpy(), axis=1)
            auc, pairwise = compute_auc(oh.tolist(), p.tolist(), output_channel_encoding=enc)
            valid = [a for a in auc if not (isinstance(a, float) and np.isnan(a))]
            mean = float(np.mean(valid)) if valid else float('nan')
            print(f'  AUC ({name}): tung kenh {auc}')
            print(f'  {"":8} -> mean cua kenh hop le = {mean}'
                  f'   | softmax co NaN: {bool(np.isnan(p).any())}')
        except Exception as e:
            print(f'  AUC ({name}): LOI {type(e).__name__}: {e}')


def run_perf(model, dataset, device, cfg, autocast, tag):
    """Goi THANG perf_metrics — dung ham that su sinh ra so trong bang ket qua."""
    C.set_eval_autocast(autocast)
    try:
        m = C.perf_metrics(model, dataset, device, cfg,
                           batch_size=int(getattr(cfg, 'eval_batch_size', 16)))
        print(f'  perf_metrics [{tag}]: AUC={m.get("AUC")}  Macro_F1={m.get("Macro_F1")}'
              f'  Pairwise_AUC={m.get("Pairwise_AUC")}')
    except Exception as e:
        print(f'  perf_metrics [{tag}]: LOI {type(e).__name__}: {e}')


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
    inc = peft.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)

    # KIEM TRA BAT BUOC: strict=False nuot im lang moi khoa khong khop. Neu cau hinh LoRA
    # dung lai khac luc train (vi du quen override 'more' -> 8 target thay vi 16) thi model
    # chi nhan duoc mot phan trong so, va moi so do sau do deu vo nghia.
    unexpected = list(getattr(inc, 'unexpected_keys', []) or [])
    loaded = len(state) - len(unexpected)
    print(f'   nap {loaded}/{len(state)} tensor tu checkpoint')
    if unexpected:
        print(f'   ❌ {len(unexpected)} khoa trong checkpoint KHONG co cho trong model, vi du:')
        for k in unexpected[:6]:
            print(f'      {k}')
        raise SystemExit(
            'Cau hinh LoRA dung lai KHAC luc train. Truyen dung override cua run do — '
            'voi P3-NoKD-More can:\n'
            '  lora_extra_target_modules=attention.output.dense|intermediate.dense|output.dense\n'
            '  lora_image_last_k_blocks=2')
    n_tr = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    print(f'   tham so LoRA cua model dung lai: {n_tr:,}')
    model = peft.merge_and_unload()

    ds = ctx['datasets']
    for split in ('forget', 'test_final'):
        print(f'\n{"=" * 66}\nTẬP: {split}  (n = {len(ds[split])})\n{"=" * 66}')
        li16, lt16, ys, oh = collect_logits(model, ds[split], device, cfg, autocast=True)
        ok16 = report('FP16 autocast (đúng như lúc chạy thật)', li16, lt16, ys)
        try_auc(li16, lt16, oh, cfg)
        run_perf(model, ds[split], device, cfg, True, 'FP16')

        li32, lt32, _, _ = collect_logits(model, ds[split], device, cfg, autocast=False)
        ok32 = report('FP32 (tắt autocast)', li32, lt32, ys)
        try_auc(li32, lt32, oh, cfg)
        run_perf(model, ds[split], device, cfg, False, 'FP32')

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
