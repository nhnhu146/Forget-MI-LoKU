#!/usr/bin/env python3
"""
forgetmi_p6.py — PHƯƠNG ÁN 6: Phân bổ rank LoRA theo Fisher & theo modality
===========================================================================
Điểm lớp:  s_ℓ = mean(F_ℓ^rel),  F^rel = Fisher_forget / (Fisher_retain + ε).
CHUẨN HÓA RIÊNG từng nhánh (ảnh vs văn bản — gradient khác thang đo), cắt PERCENTILE
trong mỗi nhánh:
   top 30%     → r = 16
   30–70%      → r = 8
   bottom 30%  → r = 0 (KHÔNG gắn adapter)
GIỮ α/r = 2 (r=8→α=16, r=16→α=32) để ablation rank không lẫn cường độ LoRA
(alpha_pattern tự suy trong build_peft). Adapter riêng ảnh/text là tự nhiên vì tên module
tách biệt. Fusion gate LUÔN cố định (không train, không có ablation gate_mode nữa):
LoKU chỉ tối ưu θ_FILA và Forget-MI không tối ưu gate.

⚠️ PEFT rank_pattern khớp bằng regex `.*\\.{key}$` — hợp cho Linear (SciBERT) nhưng có thể
lệch với Conv2d (ResNet). Sau khi wrap, gọi verify_lora_ranks() để KIỂM CHỨNG rank thực tế
và CẢNH BÁO nếu lệch (không chạy sai trong im lặng).

Loss = như P3 (trọng số cố định); khác biệt của P6 nằm ở CẤU TRÚC adapter, không phải loss.
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C

METHOD = 'p6'


def make_rank_alloc(cfg, desired_out):
    """Trả rank_alloc_fn(f_imp, r_imp, target_modules, cfg) → (target_modules_full,
    rank_pattern, alpha_pattern=None). Ghi rank mong muốn vào desired_out để verify sau."""
    r_hi = int(getattr(cfg, 'p6_rank_high', 16))
    r_mid = int(getattr(cfg, 'p6_rank_mid', 8))
    top_pct = float(getattr(cfg, 'p6_top_pct', 0.30))
    bot_pct = float(getattr(cfg, 'p6_bottom_pct', 0.30))
    base_r = int(cfg.lora_r)

    def alloc(f_imp, r_imp, target_modules, cfg):
        scores = {}
        for k in f_imp:
            if not k.endswith('.weight'):
                continue
            module = k[:-len('.weight')]
            rel = (f_imp[k] / (r_imp.get(k, torch.zeros_like(f_imp[k])) + 1e-6)).clamp(0, 1e3)
            scores[module] = float(rel.mean().item())
        img = {m: s for m, s in scores.items() if 'img_model' in m}
        txt = {m: s for m, s in scores.items() if 'img_model' not in m}
        new_targets, rank_pattern, desired = [], {}, {}
        for bname, branch in (('image', img), ('text', txt)):
            if not branch:
                continue
            items = sorted(branch.items(), key=lambda x: -x[1])   # cao→thấp
            n = len(items)
            n_top = max(1, int(round(top_pct * n)))
            n_bot = int(round(bot_pct * n))
            for i, (m, s) in enumerate(items):
                if i < n_top:
                    r = r_hi
                elif i >= n - n_bot and n_bot > 0:
                    r = 0                                          # drop adapter
                else:
                    r = r_mid
                if r > 0:
                    desired[m] = r
                    new_targets.append(m)
                    if r != base_r:                                # r_mid=base → dùng default
                        key = m.split('.', 1)[1] if '.' in m else m
                        rank_pattern[key] = r
            kept = sum(1 for m in branch if m in desired)
            print(f"   [{bname}] {len(branch)} module → giữ {kept} "
                  f"(top{n_top}=r{r_hi}, drop{n_bot})")
        desired_out.clear(); desired_out.update(desired)
        return new_targets, (rank_pattern or None), None
    return alloc


def make_weight_fn(cfg):
    w = (float(getattr(cfg, 'lambda_ur', 1.0)), float(getattr(cfg, 'lambda_uu', 1.0)),
         float(getattr(cfg, 'lambda_mu', 2.0)), float(getattr(cfg, 'lambda_mr', 2.0)),
         float(getattr(cfg, 'lambda_ce', 1.0)), float(getattr(cfg, 'lambda_kd', 1.0)),
         float(getattr(cfg, 'lambda_ihl', 1.0)))
    def weight_fn(cfg, ctx, epoch, global_frac):
        return w, True, 0.0
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
    print(f"🟢 Device: {device}  |  METHOD={METHOD}  |  gate=frozen (cố định, không train)")
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

    desired_ranks = {}
    rank_alloc = make_rank_alloc(cfg, desired_ranks)
    ctx = C.setup_experiment(cfg, device, rank_alloc_fn=rank_alloc)
    ctx['out_dir'] = out_dir

    # ----- KIỂM CHỨNG rank thực tế đã áp đúng chưa -----
    C.verify_lora_ranks(ctx['model_unlearn'], desired_ranks)

    print(f"🚀 P6 rank-adaptive unlearning ({cfg.unlearn_epochs} epoch)...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, make_weight_fn(cfg))

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path,
                        extra_row={'gate_mode': 'frozen',
                                   'p6_n_adapters': len(desired_ranks),
                                   'p6_rank_high': int(getattr(cfg, 'p6_rank_high', 16))})
    print("✅ P6 done.")


if __name__ == '__main__':
    main()
