#!/usr/bin/env python3
"""
forgetmi_main.py — PHƯƠNG PHÁP CHÍNH = P3 + P5 + P6
===================================================
  P3: quên đa tầng (IHL của LoKU + UU/MU = −Dist của Forget-MI) + UR/MR/CE/KD.
  P5: cân bằng quên↔giữ theo tiến trình (g_f=e^{−αq}, tổng trọng số gần hằng) + preset theo tỉ lệ.
  P6: phân bổ rank LoRA theo Fisher chuẩn hóa riêng từng nhánh (α/r=2). Fusion gate luôn cố định.

Chọn checkpoint bằng S_val trên VALIDATION (KHÔNG dùng test/retrained). Retrained chỉ để
đánh giá HẬU NGHIỆM (1−Sim). Báo cáo cả last(E29).

ABLATION SELECTOR (chứng minh S_val có lợi): chạy 2 lần trên main
   selector='sval'  (mặc định, chọn bằng S_val)
   selector='valce' (chọn bằng val_ce — chỉ quan tâm utility, dễ chọn gần og nhưng chưa quên đủ)
qua --override "selector=valce,id=main_valce_3per".

Chạy:
  python training/forgetmi_main.py --config config_advanced_kaggle.yaml --seed 42 \
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=main_3per"
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C
from training.forgetmi_p6 import make_rank_alloc      # P6: phân rank theo Fisher
from training.forgetmi_p5 import make_weight_fn        # P5: lịch trọng số động

METHOD = 'main'


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

    selector = str(getattr(cfg, 'selector', 'sval')).lower()
    select_by = 'val_ce' if selector in ('valce', 'val_ce') else 'S_val'
    print(f"🟢 Device: {device}  |  METHOD={METHOD}  |  selector={select_by}  "
          f"gate=frozen (cố định, không train)")
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

    # ----- P6: setup có phân rank theo Fisher (gate luôn cố định) -----
    desired_ranks = {}
    rank_alloc = make_rank_alloc(cfg, desired_ranks)
    ctx = C.setup_experiment(cfg, device, rank_alloc_fn=rank_alloc)
    ctx['out_dir'] = out_dir
    C.verify_lora_ranks(ctx['model_unlearn'], desired_ranks)

    # ----- P5: lịch trọng số động -----
    weight_fn = make_weight_fn(cfg)

    print(f"🚀 MAIN (P3+P5+P6) unlearning ({cfg.unlearn_epochs} epoch, selector={select_by})...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, weight_fn, select_by=select_by)

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path,
                        extra_row={'selector': select_by, 'gate_mode': 'frozen',
                                   'p6_n_adapters': len(desired_ranks)})
    print("✅ MAIN done.")


if __name__ == '__main__':
    main()
