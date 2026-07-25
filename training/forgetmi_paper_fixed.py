#!/usr/bin/env python3
"""
forgetmi_paper_fixed.py — Forget-MI SỬA THEO ĐÚNG BÀI BÁO
==========================================================
Bản `forgetmi_partial.py` TRUNG THÀNH code gốc (giữ cả lỗi). File này sửa các lỗi đó
**theo đúng công thức trong bài báo** (Hardan et al., MICCAI 2025), KHÔNG thêm ý tưởng mới.

Công thức bài báo (Mục 2):
  L_UU = −Dist( [F_ul(I_f), F_ul(T_f)] , [F_og(Ĩ_f), F_og(T̃_f)] )        (1)
  L_MU = −Dist( F_ul((I_f,T_f))        , F_og((Ĩ_f,T̃_f))       )        (2)
  L_UR = +Dist( [F_ul(I_r), F_ul(T_r)] , [F_og(I_r), F_og(T_r)] )        (3)
  L_MR = +Dist( F_ul((I_r,T_r))        , F_og((I_r,T_r))        )        (4)
  L    = w_uu·L_UU + w_ur·L_UR + w_mu·L_MU + w_mr·L_MR ,  Σw = 1          (5)
  Dist = Euclid · Ĩ,T̃ = forget có nhiễu (ảnh Gaussian μ=0,σ=0.1; text char/word)

BẢNG SỬA (lỗi code gốc → bài báo nói gì → sửa thế nào)
┌───┬──────────────────────────────────┬─────────────────────────────┬──────────────────────────┐
│ 1 │ Gate tạo MỚI ngẫu nhiên MỖI batch│ "joint embeddings using a   │ MỘT gate duy nhất, tạo 1 │
│   │ và F_ul / F_og dùng 2 gate KHÁC  │ multimodal adaptation gate" │ lần, DÙNG CHUNG cho cả   │
│   │ nhau ⇒ L_MU/L_MR là nhiễu        │ (1 khối kiến trúc, cố định) │ F_ul lẫn F_og, đóng băng │
│ 2 │ og_frgt_joint dùng nhầm gate +   │ Eq(2) chỉ cần F_og trên     │ bỏ (code chết)           │
│   │ truyền img 2 lần                 │ bản NHIỄU                   │                          │
│ 3 │ optimizer.step() 1 lần/epoch,    │ không nêu; chuẩn huấn luyện │ step MỖI batch, train từ │
│   │ epoch 0 không update ⇒ 29 update │ là cập nhật theo batch      │ epoch 0 (30×13≈390 step) │
│ 4 │ hinge min(L_UR, margin) trên     │ Eq(3),(4) là khoảng cách    │ bỏ hinge                 │
│   │ UR/MR (không có trong paper)     │ THUẦN, không có ngưỡng      │                          │
│ 5 │ model_og.train() (BN batch-stats)│ Fig.2: F_og **Frozen**      │ .eval() + requires_grad=F │
│ 6 │ use_noise=True lật dấu → cực     │ Eq(1),(2) LUÔN là −Dist     │ luôn −Dist               │
│   │ TIỂU hóa khoảng cách             │ (đẩy RA XA bản nhiễu)       │                          │
│ 7 │ trọng số không theo preset paper │ Eq(5) Σw=1; best@3% =       │ preset equal/unimodal/   │
│   │                                  │ **Unimodal**                │ multimodal/retention     │
└───┴──────────────────────────────────┴─────────────────────────────┴──────────────────────────┘

Đánh giá trên `D_t_final` (75% test) bằng CÙNG pipeline với P3–P6/main ⇒ số SO SÁNH ĐƯỢC.
F_ul được tinh chỉnh TOÀN BỘ (paper không dùng LoRA).
"""
import os
import sys
import time
import argparse

# Chống phân mảnh VRAM (train FULL 113M fp32 trên ảnh 2048² rất sát trần T4 14.5GB).
# PHẢI đặt TRƯỚC khi import torch (trước lần cấp phát CUDA đầu tiên).
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

import training.adv_common as C

METHOD = 'forgetmi_paper'

# Preset trọng số theo Bảng 2 bài báo — thứ tự (w_uu, w_ur, w_mu, w_mr), Σ = 1 (Eq 5).
# Paper: best@3% = Unimodal · best@6% = Multimodal · best@10% = Retention.
PRESETS = {
    'equal':      (0.25, 0.25, 0.25, 0.25),
    'unimodal':   (0.40, 0.40, 0.10, 0.10),   # ưu tiên w_uu, w_ur
    'multimodal': (0.10, 0.10, 0.40, 0.40),   # ưu tiên w_mu, w_mr
    'retention':  (0.10, 0.40, 0.10, 0.40),   # ưu tiên w_ur, w_mr
}
BEST_BY_PCT = {'3per': 'unimodal', '6per': 'multimodal', '10per': 'retention'}


def resolve_weights(cfg):
    """Trả (w_uu, w_ur, w_mu, w_mr) đã chuẩn hóa Σ=1 (Eq 5)."""
    name = str(getattr(cfg, 'weight_preset', 'auto')).lower()
    if name == 'auto':
        base = os.path.basename(str(cfg.forget_set_path))
        pct = next((p for p in ('3per', '6per', '10per') if p in base), '3per')
        name = BEST_BY_PCT[pct]
        print(f"   weight_preset=auto → '{name}' (paper: best cho {pct})")
    if name not in PRESETS:
        raise ValueError(f"weight_preset '{name}' không hợp lệ. Chọn: {list(PRESETS)} hoặc 'auto'.")
    w = PRESETS[name]
    s = float(sum(w))
    w = tuple(x / s for x in w)                      # Eq(5): Σw = 1
    print(f"📐 Trọng số [{name}] w_uu={w[0]:.2f} w_ur={w[1]:.2f} w_mu={w[2]:.2f} w_mr={w[3]:.2f} (Σ={sum(w):.2f})")
    return w, name


def build_shared_gate(device, seed):
    """SỬA #1: MỘT gate duy nhất, tạo MỘT lần, ĐÓNG BĂNG, dùng chung cho F_ul lẫn F_og.
    Bản gốc tạo Gate ngẫu nhiên MỚI mỗi batch và còn dùng 2 gate KHÁC nhau cho 2 model
    ⇒ L_MU/L_MR so hai phép chiếu ngẫu nhiên độc lập = nhiễu thuần (mà chúng chiếm tới
    w_mu+w_mr của tổng loss). Paper coi gate là 1 khối kiến trúc cố định."""
    from training.joint_embedding import Gate
    g = torch.Generator().manual_seed(int(seed))
    torch.manual_seed(int(seed))
    gate = Gate(768, 768).to(device)
    for p in gate.parameters():
        p.requires_grad = False
    gate.eval()
    print(f"🔗 Gate DÙNG CHUNG (đóng băng, seed={seed}) — sửa lỗi gate-ngẫu-nhiên-mỗi-batch")
    return gate


def paper_batch_loss(cfg, model_ul, model_og, gate, fb, rb, retb, device, w):
    """Loss đúng Eq(1)–(5). F_og đóng băng/eval (SỬA #5); luôn −Dist cho UU/MU (SỬA #6);
    KHÔNG hinge (SỬA #4); joint dùng gate CHUNG (SỬA #1)."""
    w_uu, w_ur, w_mu, w_mr = w
    f_in, _, _ = C.get_model_inputs(cfg, fb, device)      # (I_f, T_f) sạch
    n_in, _, _ = C.get_model_inputs(cfg, rb, device)      # (Ĩ_f, T̃_f) nhiễu
    r_in, _, _ = C.get_model_inputs(cfg, retb, device)    # (I_r, T_r)

    with torch.no_grad():                                  # F_og FROZEN
        og_n_i, _, og_n_t, _ = model_og(**n_in)[:4]        # F_og trên NHIỄU
        og_r_i, _, og_r_t, _ = model_og(**r_in)[:4]        # F_og trên retain
    ul_f_i, _, ul_f_t, _ = model_ul(**f_in)[:4]            # F_ul trên forget
    ul_r_i, _, ul_r_t, _ = model_ul(**r_in)[:4]            # F_ul trên retain

    cat = lambda a, b: torch.cat((a, b), dim=-1)
    # (1) UU: đẩy F_ul(D_f) RA XA F_og(D̃_f) — luôn dấu ÂM
    L_uu = -C.euclidean_distance(cat(ul_f_i, ul_f_t), cat(og_n_i, og_n_t)).mean()
    # (2) MU: như (1) nhưng trên joint embedding (gate CHUNG)
    L_mu = -C.euclidean_distance(gate(ul_f_i, ul_f_t), gate(og_n_i, og_n_t)).mean()
    # (3) UR: giữ biểu diễn đơn phương thức trên D_r — khoảng cách THUẦN, không hinge
    L_ur = C.euclidean_distance(cat(ul_r_i, ul_r_t), cat(og_r_i, og_r_t)).mean()
    # (4) MR: giữ joint trên D_r
    L_mr = C.euclidean_distance(gate(ul_r_i, ul_r_t), gate(og_r_i, og_r_t)).mean()

    loss = w_uu * L_uu + w_ur * L_ur + w_mu * L_mu + w_mr * L_mr   # (5)
    comp = {'UU': L_uu.item(), 'UR': L_ur.item(), 'MU': L_mu.item(),
            'MR': L_mr.item(), 'Total': loss.item()}
    return loss, comp


def train(cfg, ctx, gate, w, device):
    """SỬA #3: optimizer.step() MỖI batch và train NGAY từ epoch 0."""
    model_ul, model_og = ctx['model_unlearn'], ctx['model_og']
    optimizer = ctx['optimizer']
    datasets = ctx['datasets']
    grad_clip = float(getattr(cfg, 'max_grad_norm', 1.0))
    history = getattr(cfg, 'history_csv_path', None)
    t_train = 0.0
    total_steps = 0
    wall0 = time.time()

    for epoch in range(int(cfg.unlearn_epochs)):
        _t = time.time()
        model_ul.train()
        model_og.eval()                                    # SỬA #5: F_og luôn eval
        # paper: mỗi epoch lấy tập retain CÙNG CỠ với forget (zip tự cắt theo forget)
        retain_dl = DataLoader(datasets['retain'], sampler=RandomSampler(datasets['retain']),
                               batch_size=cfg.unlearn_batch_size, num_workers=ctx['nw'])
        agg, steps = {}, 0
        for fb, rb, retb in zip(ctx['forget_dl'], ctx['rand_dl'], retain_dl):
            loss, comp = paper_batch_loss(cfg, model_ul, model_og, gate, fb, rb, retb, device, w)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model_ul.parameters() if p.requires_grad], grad_clip)
            optimizer.step()
            total_steps += 1; steps += 1
            for k, v in comp.items():
                agg[k] = agg.get(k, 0.0) + v
        avg = {k: v / max(steps, 1) for k, v in agg.items()}
        t_train += time.time() - _t

        print(f"[{METHOD} E{epoch:02d}] L={avg['Total']:+.3f}  UU={avg['UU']:+.3f}  "
              f"UR={avg['UR']:+.3f}  MU={avg['MU']:+.3f}  MR={avg['MR']:+.3f}  (steps={total_steps})")
        if not np.isfinite(avg['Total']):
            raise RuntimeError(f"Loss phân kỳ (nan/inf) ở epoch {epoch} — UU/MU là −Dist KHÔNG chặn "
                               f"(đúng bài báo). Giảm learning_rate hoặc đổi weight_preset.")
        if history:
            C.append_history_row(history, {
                'method': METHOD, 'id': str(getattr(cfg, 'id', '')), 'seed': int(cfg.random_seed),
                'epoch': epoch + 1, 'total_loss': avg['Total'], 'uu': avg['UU'], 'ur': avg['UR'],
                'mu': avg['MU'], 'mr': avg['MR'], 'cum_optimizer_steps': total_steps,
            })

    timing = {'train_h': t_train / 3600, 'wall_h': (time.time() - wall0) / 3600,
              'fisher_h': 0.0, 'load_h': ctx.get('load_h', 0.0),
              'last_epoch': int(cfg.unlearn_epochs) - 1}
    return timing, total_steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    cli = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(cli.config), METHOD)
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d.setdefault('unlearn_epochs', 30)          # paper: 30 epoch
    cfg_d.setdefault('learning_rate', 1e-5)         # paper: 1e-4 hoặc 1e-5
    cfg_d.setdefault('use_noise', True)             # paper chính: nhiễu μ=0, σ=0.1
    cfg_d.setdefault('noise_mean', 0.0)
    cfg_d.setdefault('noise_std', 0.1)
    cfg_d.setdefault('weight_preset', 'auto')
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    # CAP BATCH (tự bảo vệ, không phụ thuộc notebook): train FULL 113M fp32 trên ảnh 2048²
    # rất nặng → batch 16 OOM trên T4 14.5GB. Không cho vượt paper_max_batch (mặc định 8).
    # Nếu vẫn OOM: chạy với --override paper_max_batch=4.
    _cap = int(cfg_d.get('paper_max_batch', 8))
    for _k in ('unlearn_batch_size', 'eval_batch_size'):
        cfg_d[_k] = min(int(cfg_d.get(_k, _cap)), _cap)
    print(f"   🧯 batch cap: unlearn={cfg_d['unlearn_batch_size']} eval={cfg_d['eval_batch_size']} "
          f"(paper_max_batch={_cap})")
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")
    print(f"🟢 Device: {device}  |  METHOD={METHOD} (Forget-MI sửa theo bài báo)")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    run_id = str(getattr(cfg, 'id', METHOD))
    C.save_resolved_config(cfg_d, out_dir, METHOD, run_id)

    # ---- models + data (dùng chung pipeline với P3–P6 ⇒ eval trên D_t_final) ----
    ctx = C.build_base_models(cfg, device)
    print("📚 Building dataset (4-tập holdout)...")
    _t = time.time()
    datasets, _ = C.build_dataset(cfg, ctx['tokenizer'])
    ctx['datasets'] = datasets
    ctx['load_h'] = (time.time() - _t) / 3600

    # SỬA #5: F_og đóng băng hoàn toàn
    for p in ctx['model_og'].parameters():
        p.requires_grad = False
    ctx['model_og'].eval()
    # TIẾT KIỆM VRAM: model_re chỉ dùng ở eval cuối (CosSim) → đẩy sang CPU lúc train,
    # đưa lại GPU trước final_evaluation. (Train full-model fp32 đã rất sát trần T4.)
    ctx['model_re'] = ctx['model_re'].cpu()
    torch.cuda.empty_cache()
    # paper: tinh chỉnh TOÀN BỘ F_ul (không LoRA)
    for p in ctx['model_unlearn'].parameters():
        p.requires_grad = True
    ctx['trainable'], ctx['total'] = C.count_params(ctx['model_unlearn'])
    print(f"📊 Trainable: {ctx['trainable']:,}/{ctx['total']:,} "
          f"({100*ctx['trainable']/ctx['total']:.1f}%) — toàn bộ model (đúng paper)")

    gate = build_shared_gate(device, int(cfg.random_seed))
    w, preset_name = resolve_weights(cfg)

    ctx['optimizer'] = AdamW([p for p in ctx['model_unlearn'].parameters() if p.requires_grad],
                             lr=float(cfg.learning_rate),
                             weight_decay=float(getattr(cfg, 'weight_decay', 0.01)))
    nw = min(int(getattr(cfg, 'num_cpu_workers', 2)), 2)
    ctx['nw'] = nw
    ctx['forget_dl'] = DataLoader(datasets['forget'],
        sampler=C.AlignedSampler(len(datasets['forget']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw)
    ctx['rand_dl'] = DataLoader(datasets['random'],
        sampler=C.AlignedSampler(len(datasets['random']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw)

    print(f"🚀 Forget-MI (paper-fixed) — {cfg.unlearn_epochs} epoch, lr={cfg.learning_rate}, "
          f"preset={preset_name}, use_noise={cfg.use_noise} (σ={cfg.noise_std})")
    timing, total_steps = train(cfg, ctx, gate, w, device)
    timing['method_gpu_peak_gb'] = (torch.cuda.max_memory_allocated() / 1e9
                                    if torch.cuda.is_available() else 0.0)

    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    ctx['model_unlearn'].eval()
    torch.cuda.empty_cache()
    ctx['model_re'] = ctx['model_re'].to(device)      # đưa lại GPU cho CosSim
    C.final_evaluation(ctx['model_unlearn'], ctx['model_re'], datasets, cfg, device,
                       METHOD, run_id, checkpoint_kind='last',
                       selected_epoch=timing['last_epoch'] + 1, timing=timing,
                       trainable=ctx['trainable'], total=ctx['total'],
                       total_optimizer_steps=total_steps, csv_path=csv_path,
                       gold_available=ctx['gold_available'],
                       extra_row={'weight_preset': preset_name, 'lr': float(cfg.learning_rate),
                                  'use_noise': bool(cfg.use_noise),
                                  'noise_std': float(cfg.noise_std)})
    print("✅ Forget-MI (paper-fixed) done.")


if __name__ == '__main__':
    main()
