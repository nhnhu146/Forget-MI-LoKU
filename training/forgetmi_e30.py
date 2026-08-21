#!/usr/bin/env python3
"""
forgetmi_e30.py — Driver E30 cho Đề xuất 1 (P3-cand) / P3 / P6
===============================================================
Gọi THẲNG các hàm của ``adv_e30``. KHÔNG hoán tên hàm ở đâu cả — bản trước dùng
monkeypatch và dính hai lỗi runtime chỉ lộ khi chạy thật trên Kaggle.

Trọng số loss, scheme, ablation, phân rank: import NGUYÊN từ ``forgetmi_p3_cand`` /
``forgetmi_p3`` / ``forgetmi_p6``, không chép lại — phương pháp không đổi, chỉ dữ liệu
và giao thức chọn checkpoint đổi.

Khác driver cũ đúng ba điểm:
  1. ``adv_e30.setup_experiment_e30``  → retain = TOÀN BỘ D_r, test_final = TOÀN BỘ D_t
  2. ``adv_e30.run_training_e30``      → không S_val, không ce_selector, LR hằng số
  3. ``adv_e30.finalize_and_eval_e30`` → CSV chỉ có hàng 'last' (E30)

Chạy — Đề xuất 1 (P3-NoKD-More):
  python training/forgetmi_e30.py --method p3cand --scheme uni_nokd \\
      --config config_advanced_kaggle.yaml --seed 42 \\
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=p3_m3_e30_s42"

Các biến thể khác:
  --method p3                      (trọng số cố định của forgetmi_p3)
  --method p6                      (phân rank theo Fisher)
  --method p3cand --ablate ihl     (ablation, y như driver cũ)

Baseline Forget-MI / NegGrad+ / CF-k KHÔNG chạy qua đây (chúng không dùng adv_common)
— dùng ``training/forgetmi_partial_e30.py``.

LƯU Ý: số ra từ đây KHÔNG so trực tiếp được với run cũ (retain và tập test đã khác
kích thước, LR khác chế độ). Bảng đối chứng phải cùng chạy bằng nhánh _e30.
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch

import training.adv_common as C
import training.adv_e30 as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', type=str, default='p3cand',
                    choices=['p3cand', 'p3', 'p6'])
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--scheme', type=str, default='uni_nokd',
                    help='chỉ dùng với --method p3cand: eq | uni | multi | ret | uni_nokd')
    ap.add_argument('--ablate', type=str, default='none',
                    choices=['none', 'fisher_fila', 'ihl', 'mu_mr'],
                    help='chỉ dùng với --method p3cand')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    ap.add_argument('--fresh', action='store_true')
    ap.add_argument('--allow-core-drift', action='store_true',
                    help='chạy tiếp dù 5 hàm nguồn bên adv_common đã đổi (KHÔNG khuyến khích)')
    cli = ap.parse_args()

    # Chốt chặn lệch âm thầm: adv_e30 chép 5 hàm của adv_common; nếu bản gốc đã đổi
    # mà bản chép chưa đồng bộ thì mọi con số sau đây đều sai giao thức.
    E.assert_core_unchanged(strict=not cli.allow_core_drift)

    extra = {}
    rank_alloc = None
    extra_row = {}

    if cli.method == 'p3cand':
        import training.forgetmi_p3_cand as P
        method_tag = f'p3cand_{cli.scheme}'
        base_scheme, w4, extra = P.resolve_scheme(cli.scheme)
        # ---- Ablation: chạm ĐÚNG thành phần bị bỏ (sao y forgetmi_p3_cand.main) ----
        if cli.ablate == 'fisher_fila':
            extra['loku_random_init'] = True      # bỏ cả Fisher lẫn FILA/W*
        elif cli.ablate == 'ihl':
            extra['lambda_ihl'] = 0.0
        elif cli.ablate == 'mu_mr':
            w_uu, w_ur, _, _ = w4
            w4 = (w_uu, w_ur, 0.0, 0.0)           # KHÔNG chuẩn hóa lại
    elif cli.method == 'p3':
        import training.forgetmi_p3 as P
        method_tag = 'p3'
    else:
        import training.forgetmi_p6 as P
        method_tag = 'p6'

    cfg_d = C.flatten_method_config(C.load_config(cli.config), 'p3')
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d.update(extra)                           # vd uni_nokd → lambda_kd=0
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    cfg_d['ce_selector'] = 0                      # adv_e30 không hỗ trợ (cần tập 'sel')
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")

    # ---- weight_fn: lấy nguyên của driver gốc ----
    if cli.method == 'p3cand':
        weight_fn, w = P.make_weight_fn(cfg, w4, ablated=(cli.ablate != 'none'))
        print(f"🟢 Device: {device}  |  E30  |  scheme={cli.scheme} (base={base_scheme})")
        if cli.ablate != 'none':
            print(f"🔬 ABLATION = {cli.ablate}  (bỏ đúng thành phần này)")
        print(f"   Forget-MI block (tổng=1): w_uu={w[1]:.4f}  w_ur={w[0]:.4f}  "
              f"w_mu={w[2]:.4f}  w_mr={w[3]:.4f}")
        print(f"   Ngoài block: λ_CE={w[4]:.3f}  λ_KD={w[5]:.3f}  λ_IHL={w[6]:.3f}")
        extra_row = {'scheme': cli.scheme, 'ablate': cli.ablate,
                     'w_uu': round(w[1], 4), 'w_ur': round(w[0], 4),
                     'w_mu': round(w[2], 4), 'w_mr': round(w[3], 4),
                     'lambda_kd': w[5], 'lambda_ihl': w[6]}
    elif cli.method == 'p3':
        weight_fn = P.make_weight_fn(cfg)
        print(f"🟢 Device: {device}  |  E30  |  METHOD=p3")
    else:
        desired_ranks = {}
        rank_alloc = P.make_rank_alloc(cfg, desired_ranks)
        weight_fn = P.make_weight_fn(cfg)
        print(f"🟢 Device: {device}  |  E30  |  METHOD=p6  |  gate=frozen")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = str(getattr(cfg, 'id', f'{method_tag}_e30'))
    C.save_resolved_config(cfg_d, out_dir, f'{method_tag}_e30', run_id)
    if cli.fresh:
        import shutil
        cp = os.path.join(out_dir, 'checkpoints')
        if os.path.exists(cp):
            shutil.rmtree(cp)

    ctx = E.setup_experiment_e30(cfg, device, rank_alloc_fn=rank_alloc)
    ctx['out_dir'] = out_dir

    if cli.method == 'p6':
        C.verify_lora_ranks(ctx['model_unlearn'], desired_ranks)
        extra_row = {'gate_mode': 'frozen', 'p6_n_adapters': len(desired_ranks),
                     'p6_rank_high': int(getattr(cfg, 'p6_rank_high', 16))}

    print(f"🚀 {method_tag} E30 ({cfg.unlearn_epochs} epoch, không selector)...")
    timing, total_steps = E.run_training_e30(cfg, ctx, device, method_tag, weight_fn)

    csv_path = str(getattr(cfg, 'results_csv_path',
                           os.path.join(out_dir, 'results_e30.csv')))
    E.finalize_and_eval_e30(cfg, ctx, device, method_tag, run_id, timing, total_steps,
                            csv_path, extra_row=extra_row)
    print(f"✅ {method_tag} E30 done.")


if __name__ == '__main__':
    main()
