#!/usr/bin/env python3
"""
forgetmi_p3.py — PHƯƠNG ÁN 3: Quên đồng thời ở đầu ra + không gian biểu diễn
============================================================================
  L = λ_IHL·L_IHL                      # quên ĐẦU RA (LoKU inverted-hinge, bounded [0,2])
    + λ_UU·L_UU_b + λ_MU·L_MU_b        # quên BIỂU DIỄN đơn/đa phương thức — hinge ĐẨY-XA chặn TRẦN
    + λ_UR·L_UR   + λ_MR·L_MR          # GIỮ biểu diễn retain gần model gốc (hinge Forget-MI)
    + λ_CE·L_CE   + λ_KD·L_KD          # GIỮ utility: CE(D_r) + KD(ul‖og)(D_r)

Hinge đẩy-xa chặn TRẦN (use_noise=true, tham chiếu og_rnd là bản nhiễu σ=0.1):
  L_UU_b = ReLU(m_u − d_u),  L_MU_b = ReLU(m_m − d_m);  m = quantile-0.90 của {d_i} tại epoch 0.

30 epoch (ngân sách Forget-MI). Trọng số CỐ ĐỊNH suốt quá trình (khác P5). Chọn checkpoint
bằng S_val (validation, không dùng test/retrained), báo cáo cả last(E29). Eval trên D_t_final.

Chạy:
  python training/forgetmi_p3.py --config config_advanced_kaggle.yaml --seed 42 \
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=p3_3per"
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C

METHOD = 'p3'


def p3_weights(cfg):
    """Trọng số cố định (nền 2 paper: UR=1, UU=1, MU=2, MR=2; IHL/CE/KD=1)."""
    return (float(getattr(cfg, 'lambda_ur', 1.0)), float(getattr(cfg, 'lambda_uu', 1.0)),
            float(getattr(cfg, 'lambda_mu', 2.0)), float(getattr(cfg, 'lambda_mr', 2.0)),
            float(getattr(cfg, 'lambda_ce', 1.0)), float(getattr(cfg, 'lambda_kd', 1.0)),
            float(getattr(cfg, 'lambda_ihl', 1.0)))


def make_weight_fn(cfg):
    w = p3_weights(cfg)
    def weight_fn(cfg, ctx, epoch, global_frac):
        return w, True, 0.0        # forget luôn bật, không guard
    return weight_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    ap.add_argument('--fresh', action='store_true')
    cli = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(cli.config), METHOD)
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")
    print(f"🟢 Device: {device}  |  METHOD={METHOD}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = str(getattr(cfg, 'id', METHOD))
    C.save_resolved_config(cfg_d, out_dir, METHOD, run_id)
    if cli.fresh:
        import shutil
        cp = os.path.join(out_dir, 'checkpoints')
        if os.path.exists(cp):
            shutil.rmtree(cp)

    ctx = C.setup_experiment(cfg, device)
    ctx['out_dir'] = out_dir

    print(f"🚀 P3 unlearning ({cfg.unlearn_epochs} epoch)...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, make_weight_fn(cfg))

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path)
    print("✅ P3 done.")


if __name__ == '__main__':
    main()
