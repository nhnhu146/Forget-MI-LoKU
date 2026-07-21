#!/usr/bin/env python3
"""
forgetmi_p4.py — PHƯƠNG ÁN 4: Huấn luyện HAI GIAI ĐOẠN
======================================================
  Giai đoạn 1 (epoch < k): quên–giữ ĐỒNG THỜI (như P3)
      L1 = λ_IHL·IHL + λ_UU·UU_b + λ_MU·MU_b + λ_UR·UR + λ_MR·MR + λ_CE·CE + λ_KD·KD
  Giai đoạn 2 (epoch ≥ k): CHỈ phục hồi utility trên D_r
      L2 = λ_UR·UR + λ_MR·MR + λ_CE·CE + λ_KD·KD  (+ guard·IHL nếu p4_stage2_guard>0)

Rủi ro: GĐ2 kéo về gần og trên D_r có thể GIÁN TIẾP khôi phục hành vi trên D_f (chia sẻ
tham số). Hai chốt:
  - guard nhỏ (mặc định 0.05·IHL) giữ áp lực quên rất nhẹ; đặt p4_stage2_guard=0 = bản thuần.
  - selector S_val VẪN đánh giá IHL/UU/MU trên D_f (no-grad) mỗi epoch → phát hiện 'nhớ lại'
    và KHÔNG chọn checkpoint đã rò rỉ ngược.

k (p4_stage1_epochs) nên SWEEP {5,10,15,20} qua --override. 30 epoch, chọn bằng S_val.

Chạy (ví dụ sweep k=15):
  python training/forgetmi_p4.py --config config_advanced_kaggle.yaml --seed 42 \
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=p4_k15_3per,p4_stage1_epochs=15"
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C

METHOD = 'p4'


def make_weight_fn(cfg):
    w = (float(getattr(cfg, 'lambda_ur', 1.0)), float(getattr(cfg, 'lambda_uu', 1.0)),
         float(getattr(cfg, 'lambda_mu', 2.0)), float(getattr(cfg, 'lambda_mr', 2.0)),
         float(getattr(cfg, 'lambda_ce', 1.0)), float(getattr(cfg, 'lambda_kd', 1.0)),
         float(getattr(cfg, 'lambda_ihl', 1.0)))
    k = int(getattr(cfg, 'p4_stage1_epochs', 15))
    guard = float(getattr(cfg, 'p4_stage2_guard', 0.05))

    def weight_fn(cfg, ctx, epoch, global_frac):
        if epoch < k:
            return w, True, 0.0                    # GĐ1: quên–giữ đồng thời
        return w, False, guard                     # GĐ2: chỉ D_r (+ guard·IHL)
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
    print(f"🟢 Device: {device}  |  METHOD={METHOD}  |  k={getattr(cfg, 'p4_stage1_epochs', 15)} "
          f"guard={getattr(cfg, 'p4_stage2_guard', 0.05)}")
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

    print(f"🚀 P4 two-stage unlearning ({cfg.unlearn_epochs} epoch)...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, make_weight_fn(cfg))

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path,
                        extra_row={'p4_stage1_epochs': int(getattr(cfg, 'p4_stage1_epochs', 15)),
                                   'p4_stage2_guard': float(getattr(cfg, 'p4_stage2_guard', 0.05))})
    print("✅ P4 done.")


if __name__ == '__main__':
    main()
