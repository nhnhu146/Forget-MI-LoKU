#!/usr/bin/env python3
"""
forgetmi_p5.py — PHƯƠNG ÁN 5: Trọng số ĐỘNG theo quy mô & tiến trình quên
=========================================================================
Lịch theo GLOBAL STEP chuẩn hóa q = s/(S−1) ∈ [0,1]:
  g_f(q) = exp(−α·q)           (α = schedule_decay_rate)
  Trọng số QUÊN giảm dần:  λ_UU(q)=g_f·λ_UU⁰, λ_MU(q)=g_f·λ_MU⁰, λ_IHL(q)=g_f·λ_IHL⁰
  Trọng số GIỮ tăng dần, GIỮ TỔNG GẦN HẰNG (chống trôi độ lớn loss):
      C = F⁰ + R⁰ ,  forget_total(q) = g_f·F⁰ ,  retain_scale = (C − forget_total)/R⁰
      λ_UR/MR/CE/KD(q) = retain_scale · λ⁰   ⇒  λ_forget(q) + λ_retain(q) = C  (∀q)

Preset ban đầu theo FORGET_PCT (LỊCH THỰC NGHIỆM lấy cảm hứng từ kết quả Forget-MI,
KHÔNG phải quy luật tổng quát đã chứng minh):
   3%: ưu tiên ĐƠN phương thức   → λ_UU⁰ > λ_MU⁰
   6%: ưu tiên ĐA phương thức    → λ_MU⁰ ≥ λ_UU⁰
  10%: ưu tiên GIỮ LẠI mạnh      → λ_UR⁰+λ_MR⁰+λ_KD⁰ > λ_UU⁰+λ_MU⁰

30 epoch, per-batch step, chọn checkpoint bằng S_val, báo cáo cả last(E29).
"""
import os
import sys
import argparse
import math

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C

METHOD = 'p5'

# preset base (UR, UU, MU, MR, CE, KD, IHL) — có thể override qua config p5_preset_*
P5_PRESETS = {
    '3per':  (1.0, 1.5, 1.0, 1.0, 1.0, 1.0, 1.0),   # UU>MU (đơn phương thức)
    '6per':  (1.0, 1.0, 1.5, 1.0, 1.0, 1.0, 1.0),   # MU≥UU (đa phương thức)
    '10per': (2.0, 1.0, 1.0, 2.0, 1.0, 1.5, 1.0),   # retain-heavy: UR+MR+KD(5.5) > UU+MU(2)
    'default': (1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0),
}


def detect_pct(forget_set_path):
    base = os.path.basename(str(forget_set_path))
    for p in ('3per', '6per', '10per'):
        if p in base:
            return p
    return 'default'


def _preset(cfg):
    pct = detect_pct(cfg.forget_set_path)
    # config có thể override: p5_preset_3per = "1.0,1.5,1.0,1.0,1.0,1.0,1.0"
    override = getattr(cfg, f'p5_preset_{pct}', None)
    if isinstance(override, str) and override.strip():
        vals = tuple(float(x) for x in override.split(','))
        if len(vals) == 7:
            return pct, vals
    return pct, P5_PRESETS.get(pct, P5_PRESETS['default'])


def make_weight_fn(cfg):
    pct, base = _preset(cfg)
    ur0, uu0, mu0, mr0, ce0, kd0, ihl0 = base
    decay = float(getattr(cfg, 'schedule_decay_rate', 3.0))
    F0 = uu0 + mu0 + ihl0                 # tổng khối quên
    R0 = ur0 + mr0 + ce0 + kd0            # tổng khối giữ
    C_tot = F0 + R0
    print(f"📅 P5 preset[{pct}] base UR/UU/MU/MR/CE/KD/IHL={base}  decay={decay}  "
          f"F0={F0:.2f} R0={R0:.2f} C={C_tot:.2f}")

    def weight_fn(cfg, ctx, epoch, global_frac):
        q = min(max(global_frac, 0.0), 1.0)
        g_f = math.exp(-decay * q)
        forget_total = g_f * F0
        retain_scale = (C_tot - forget_total) / max(R0, 1e-6)
        w = (ur0 * retain_scale, uu0 * g_f, mu0 * g_f, mr0 * retain_scale,
             ce0 * retain_scale, kd0 * retain_scale, ihl0 * g_f)
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

    print(f"🚀 P5 dynamic-weight unlearning ({cfg.unlearn_epochs} epoch)...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, make_weight_fn(cfg))

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path,
                        extra_row={'p5_pct': detect_pct(cfg.forget_set_path),
                                   'schedule_decay_rate': float(getattr(cfg, 'schedule_decay_rate', 3.0))})
    print("✅ P5 done.")


if __name__ == '__main__':
    main()
