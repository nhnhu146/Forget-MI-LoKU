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
import re
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
    """Goi THANG perf_metrics — dung ham that su sinh ra so trong bang ket qua.
    Tra AUC (float, co the la nan) hoac None neu nem loi."""
    C.set_eval_autocast(autocast)
    try:
        m = C.perf_metrics(model, dataset, device, cfg,
                           batch_size=int(getattr(cfg, 'eval_batch_size', 16)))
        print(f'  perf_metrics [{tag}]: AUC={m.get("AUC")}  Macro_F1={m.get("Macro_F1")}'
              f'  Pairwise_AUC={m.get("Pairwise_AUC")}')
        return m.get('AUC')
    except Exception as e:
        print(f'  perf_metrics [{tag}]: LOI {type(e).__name__}: {e}')
        return None


def infer_lora_cfg(keys):
    """Suy cấu hình LoRA TỪ CHÍNH checkpoint, thay vì trông chờ người gọi truyền đúng
    override. Sai cấu hình là hỏng âm thầm (strict=False bỏ qua khóa lệch) nên tự suy
    an toàn hơn. Trả (danh sách target văn bản bổ sung, số khối ảnh cuối).

    Khóa có dạng:
      base_model.model.text_model.bert.encoder.layer.0.intermediate.dense.lora_A...
      base_model.model.img_model.layer7.0.conv1.lora_A...
    """
    txt = []
    # 'output.dense' và 'attention.output.dense' đều khớp hậu tố '.output.dense' →
    # tách bằng cách xét ký tự đứng trước.
    if any(re.search(r'\.attention\.output\.dense\.lora_', k) for k in keys):
        txt.append('attention.output.dense')
    if any(re.search(r'\.intermediate\.dense\.lora_', k) for k in keys):
        txt.append('intermediate.dense')
    if any(re.search(r'(?<!attention)\.output\.dense\.lora_', k)
           and not re.search(r'\.attention\.output\.dense\.lora_', k) for k in keys):
        txt.append('output.dense')

    # CHỈ đếm khối thật sự có LoRA. trainable_state còn chứa BUFFER (BN running stats…)
    # của mọi img_model.layerN, đếm cả buffer sẽ ra 7 khối thay vì 2.
    lora_keys = [k for k in keys if '.lora_' in k]
    layers = sorted({int(m.group(1)) for k in lora_keys
                     for m in [re.search(r'img_model\.layer(\d+)\.', k)] if m})
    # resolve_image_targets lấy k khối CUỐI → số khối riêng biệt chính là k
    return txt, (len(layers) if layers else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--ckpt', required=True, help='latest.pt / cand_S2.pt (chỉ chứa LoRA)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--override', default=None)
    ap.add_argument('--scheme', default='uni_nokd')
    ap.add_argument('--no-infer', action='store_true',
                    help='Tắt suy cấu hình LoRA từ checkpoint (dùng nguyên config + override)')
    a = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(a.config), 'p3')
    cfg_d['random_seed'] = int(a.seed)
    cfg_d = C.apply_overrides(cfg_d, a.override)

    # ---- Nạp checkpoint TRƯỚC setup_experiment ----
    # Vừa để suy ra cấu hình LoRA, vừa để hỏng thì hỏng ngay chứ không sau 8 phút Fisher.
    print(f'📦 Nạp {a.ckpt}')
    payload = torch.load(a.ckpt, map_location='cpu', weights_only=False)
    state = payload.get('trainable_state', payload.get('lora_state', payload))
    print(f'   epoch trong checkpoint: {payload.get("epoch")}  |  {len(state)} tensor')

    if not a.no_infer:
        txt_extra, img_k = infer_lora_cfg(list(state.keys()))
        if txt_extra:
            cfg_d['lora_extra_target_modules'] = '|'.join(txt_extra)
        if img_k:
            cfg_d['lora_image_last_k_blocks'] = img_k
        print(f'🔎 Suy từ checkpoint: lora_extra_target_modules={cfg_d.get("lora_extra_target_modules", "(không có)")}'
              f'  lora_image_last_k_blocks={cfg_d.get("lora_image_last_k_blocks")}')

    cfg = C.Cfg(cfg_d)
    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('🔧 Dựng lại pipeline P3 (Fisher + FILA) để tái tạo W* — checkpoint chỉ có LoRA...')
    ctx = C.setup_experiment(cfg, device)

    from joint_img_txt.model import ImageTextModel
    from peft import get_peft_model
    # Giải phóng model_unlearn của ctx: từ đây chỉ cần base_p / peft_cfg / fila_subtraction
    # / datasets. Bớt một model FP32 đầy đủ trên GPU cho chắc.
    ctx.pop('model_unlearn', None)
    torch.cuda.empty_cache()
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
    # Chiều ngược lại: model có NHIỀU chỗ LoRA hơn checkpoint. Khi đó khóa thiếu rơi vào
    # missing_keys — vốn luôn rất lớn (mọi trọng số nền không nằm trong trainable_state)
    # nên không dùng missing_keys để phát hiện được. Đối chiếu SỐ TENSOR LoRA hai bên.
    n_ck = sum(1 for k in state if '.lora_' in k)
    n_md = sum(1 for n_, _ in peft.named_parameters() if '.lora_' in n_)
    print(f'   tensor LoRA: checkpoint {n_ck} · model dựng lại {n_md}')
    if n_ck != n_md:
        raise SystemExit(
            f'Số tensor LoRA lệch ({n_ck} vs {n_md}): model dựng lại có nhiều/ít vị trí LoRA '
            f'hơn checkpoint. Cấu hình suy ra chưa khớp — kiểm tra lora_r và danh sách target.')
    n_tr = sum(p.numel() for p in peft.parameters() if p.requires_grad)
    print(f'   tham so LoRA cua model dung lai: {n_tr:,}')
    model = peft.merge_and_unload()

    ds = ctx['datasets']
    for split in ('forget', 'test_final'):
        print(f'\n{"=" * 66}\nTẬP: {split}  (n = {len(ds[split])})\n{"=" * 66}')
        li16, lt16, ys, oh = collect_logits(model, ds[split], device, cfg, autocast=True)
        report('FP16 autocast (đúng như lúc chạy thật)', li16, lt16, ys)
        try_auc(li16, lt16, oh, cfg)
        auc16 = run_perf(model, ds[split], device, cfg, True, 'FP16')

        li32, lt32, _, _ = collect_logits(model, ds[split], device, cfg, autocast=False)
        ok32_img = bool(torch.isfinite(li32).all())
        report('FP32 (tắt autocast)', li32, lt32, ys)
        try_auc(li32, lt32, oh, cfg)
        auc32 = run_perf(model, ds[split], device, cfg, False, 'FP32')

        # Kết luận phải dựa vào ĐÚNG đại lượng đã sinh ra NaN trong bảng, tức AUC của
        # perf_metrics (nhánh ẢNH). Bản trước kết luận theo tính hữu hạn của logit VĂN BẢN
        # nên in "TRƯỜNG HỢP C" trong khi perf_metrics vẫn ra AUC hợp lệ — sai.
        nan16 = auc16 is None or (isinstance(auc16, float) and np.isnan(auc16))
        nan32 = auc32 is None or (isinstance(auc32, float) and np.isnan(auc32))
        print(f'\n>>> KẾT LUẬN cho {split}:  (AUC của perf_metrics: FP16={auc16}  FP32={auc32})')
        if nan16 and not nan32:
            print('    TRƯỜNG HỢP C — FP16 cho NaN, FP32 cho số hợp lệ.')
            print('    NaN trong bảng là ARTEFACT của autocast lúc eval, KHÔNG phải mô hình sụp đổ.')
            print('    Sửa câu giải thích Bảng 5 và dùng số FP32.')
        elif nan16 and nan32 and not ok32_img:
            print('    TRƯỜNG HỢP A — cả hai đều NaN và logit ảnh KHÔNG hữu hạn ở FP32.')
            print('    Mô hình mất ổn định số thật sự. Dùng checkpoint S2 làm kết quả chính.')
        elif nan16 and nan32 and ok32_img:
            print('    TRƯỜNG HỢP B — logit ảnh hữu hạn mà AUC vẫn NaN → lỗi hàm metric.')
            print('    Phải sửa hàm đánh giá rồi tính lại AUC cho CẢ OG/GOLD/Forget-MI/P3 trên IU.')
        else:
            print('    KHÔNG TÁI HIỆN ĐƯỢC NaN — cả FP16 lẫn FP32 đều cho AUC hợp lệ.')
            print('    Nghĩa là mô hình dựng lại KHÁC mô hình lúc chạy thật, hoặc giá trị NaN')
            print('    trong bảng đến từ nguyên nhân khác. KHÔNG được dùng số ở đây để thay')
            print('    vào Bảng 5 khi chưa giải thích được chênh lệch.')
    C.set_eval_autocast(True)


if __name__ == '__main__':
    main()
