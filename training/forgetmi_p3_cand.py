#!/usr/bin/env python3
"""forgetmi_p3_cand.py — P3 với các WEIGHTING SCHEME của Forget-MI (candidate search).

CƠ SỞ: Forget-MI Eq.(5) định nghĩa
    L = w_uu·L_UU + w_ur·L_UR + w_mu·L_MU + w_mr·L_MR ,  với  w_uu+w_ur+w_mu+w_mr = 1
và Mục 3 của paper liệt kê các thiết lập trọng số được khảo sát:
    (2) "Trọng số bằng nhau cho tất cả các mất mát"            → Equal
    (3) "trọng số cao hơn cho các mất mát đa phương thức w_mu và w_mr" → Multimodal
    (4) "trọng số cao hơn cho các mất mát đơn phương thức w_uu và w_ur" → Unimodal
    (5) "trọng số cao hơn cho các mất mát giữ lại w_ur và w_mr"        → Retention

⚠️ NGUỒN CỦA TỪNG BỘ SỐ (đọc kỹ trước khi trích dẫn trong báo cáo):
  * eq     — SUY TRỰC TIẾP từ paper ("bằng nhau" + tổng = 1) ⇒ 0.25 mỗi thành phần. CHẮC.
  * multi  — LẤY TỪ config.yaml GỐC của tác giả (alpha=1 UR, beta=1 UU, theta=2 MU,
             gamma=2 MR) rồi chuẩn hóa cho tổng = 1 ⇒ (1,1,2,2)/6. CHẮC.
  * uni    — paper CHỈ nói "w_uu, w_ur cao hơn", KHÔNG cho số. Ở đây lấy ĐỐI XỨNG của
             multi (đảo vai trò uni↔multi, giữ nguyên tỉ lệ 2:1) ⇒ (2,2,1,1)/6. SUY RA.
  * ret    — paper CHỈ nói "w_ur, w_mr cao hơn", KHÔNG cho số. Lấy cùng tỉ lệ 2:1
             ⇒ (1,2,1,2)/6 theo thứ tự (uu,ur,mu,mr). SUY RA.
  Khi viết báo cáo phải ghi rõ uni/ret là suy ra theo đối xứng, không phải số công bố.

MỌI THỨ CÒN LẠI GIỮ NGUYÊN so với P3: empirical Fisher per-sample, relative Fisher,
FILA + W*, LoRA-only, IHL, noisy reference, UU/MU = −Dist, UR/MR = +Dist (không cap),
Gate theo code gốc (tạo mới mỗi batch, riêng vai trò), 30 epoch, 1 optimizer update
cuối mỗi epoch, cùng seed/split/selector. Chỉ thay ĐÚNG bộ trọng số đang khảo sát.

Chạy:
  python training/forgetmi_p3_cand.py --config config_advanced_kaggle.yaml --seed 42 \
      --scheme eq --override "forget_set_path=./data_splits/forget_set_3per.csv,id=p3eq_3per"
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C

METHOD = 'p3cand'

# (w_uu, w_ur, w_mu, w_mr) — TỔNG PHẢI = 1 (Forget-MI Eq.(5))
SCHEMES = {
    'eq':    (0.25,    0.25,    0.25,    0.25),      # Equal        — từ paper
    'multi': (1 / 6,   1 / 6,   1 / 3,   1 / 3),     # Multimodal   — từ config.yaml gốc (1,1,2,2)
    'uni':   (1 / 3,   1 / 3,   1 / 6,   1 / 6),     # Unimodal     — suy đối xứng từ multi
    'ret':   (1 / 6,   1 / 3,   1 / 6,   1 / 3),     # Retention    — suy theo tỉ lệ 2:1
}
# Candidate phái sinh: dùng scheme khác + tắt KD (KD không thuộc Forget-MI lẫn LoKU).
DERIVED = {
    'uni_nokd': ('uni', {'lambda_kd': 0.0}),
}


def resolve_scheme(name):
    """Trả (tên_scheme_gốc, weights4, override_bổ_sung)."""
    name = str(name).lower()
    if name in DERIVED:
        base, extra = DERIVED[name]
        return base, SCHEMES[base], dict(extra)
    if name not in SCHEMES:
        raise SystemExit(f"--scheme phải thuộc {sorted(list(SCHEMES) + list(DERIVED))}, nhận '{name}'")
    return name, SCHEMES[name], {}


def make_weight_fn(cfg, w4):
    """w4 = (w_uu, w_ur, w_mu, w_mr). combined_batch_loss nhận thứ tự
    (λ_UR, λ_UU, λ_MU, λ_MR, λ_CE, λ_KD, λ_IHL)."""
    w_uu, w_ur, w_mu, w_mr = w4
    total = w_uu + w_ur + w_mu + w_mr
    assert abs(total - 1.0) < 1e-9, f"Tổng trọng số block Forget-MI phải = 1, đang là {total}"
    w = (float(w_ur), float(w_uu), float(w_mu), float(w_mr),
         float(getattr(cfg, 'lambda_ce', 1.0)),
         float(getattr(cfg, 'lambda_kd', 1.0)),
         float(getattr(cfg, 'lambda_ihl', 1.0)))

    def weight_fn(cfg, ctx, epoch, global_frac):
        return w, True, 0.0        # forget luôn bật, không guard — giống P3
    return weight_fn, w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--scheme', type=str, required=True,
                    help='eq | uni | multi | ret | uni_nokd')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    ap.add_argument('--fresh', action='store_true')
    cli = ap.parse_args()

    base_scheme, w4, extra = resolve_scheme(cli.scheme)

    cfg_d = C.flatten_method_config(C.load_config(cli.config), 'p3')
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d.update(extra)                      # ví dụ uni_nokd → lambda_kd=0
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")

    weight_fn, w = make_weight_fn(cfg, w4)
    print(f"🟢 Device: {device}  |  METHOD={METHOD}  |  scheme={cli.scheme} (base={base_scheme})")
    print(f"   Forget-MI block (tổng=1): w_uu={w[1]:.4f}  w_ur={w[0]:.4f}  "
          f"w_mu={w[2]:.4f}  w_mr={w[3]:.4f}")
    print(f"   Ngoài block: λ_CE={w[4]:.3f}  λ_KD={w[5]:.3f}  λ_IHL={w[6]:.3f}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = str(getattr(cfg, 'id', f'{METHOD}_{cli.scheme}'))
    C.save_resolved_config(cfg_d, out_dir, f'{METHOD}_{cli.scheme}', run_id)
    if cli.fresh:
        import shutil
        cp = os.path.join(out_dir, 'checkpoints')
        if os.path.exists(cp):
            shutil.rmtree(cp)

    ctx = C.setup_experiment(cfg, device)
    ctx['out_dir'] = out_dir

    print(f"🚀 P3-{cli.scheme.upper()} unlearning ({cfg.unlearn_epochs} epoch)...")
    timing, best, total_steps = C.run_training(cfg, ctx, device, METHOD, weight_fn)

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, total_steps, csv_path,
                        extra_row={'scheme': cli.scheme, 'w_uu': round(w[1], 4),
                                   'w_ur': round(w[0], 4), 'w_mu': round(w[2], 4),
                                   'w_mr': round(w[3], 4), 'lambda_kd': w[5]})
    print(f"✅ P3-{cli.scheme.upper()} done.")


if __name__ == '__main__':
    main()
