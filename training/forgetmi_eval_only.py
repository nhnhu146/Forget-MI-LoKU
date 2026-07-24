#!/usr/bin/env python3
"""
forgetmi_eval_only.py — Đánh giá HẬU NGHIỆM 1 model bất kỳ trên CÙNG split mới
==============================================================================
Sau khi đổi test split (25% D_nm_val / 75% D_t_final), PHẢI đánh giá lại Original,
Retrained và MỌI baseline (Forget-MI, NegGrad, CF-k, EU-k, LoKU) trên chính D_t_final
để so sánh công bằng — KHÔNG ghép với số cũ tính trên toàn bộ test.

Dùng ĐÚNG hàm final_evaluation + build_dataset của adv_common → cùng holdout/seed/metric.
Chỉ FORWARD (không train), nên nhanh.

Ví dụ:
  # Original
  python training/forgetmi_eval_only.py --config config_advanced_kaggle.yaml \
      --label og --model_type pretrained \
      --model_path /kaggle/input/.../base_model/training_original_model
  # Retrained (gold)
  python training/forgetmi_eval_only.py --config config_advanced_kaggle.yaml \
      --label re --model_type pretrained \
      --model_path /kaggle/input/.../retrained_model/model_retrained_3per
  # Forget-MI baseline (state_dict .pth của epoch chọn)
  python training/forgetmi_eval_only.py --config config_advanced_kaggle.yaml \
      --label forgetmi --model_type state_dict --model_path /path/epoch_29/model_state_dict.pth
"""
import os
import sys
import argparse

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import training.adv_common as C


def load_eval_model(model_path, model_type, base_p, device):
    from joint_img_txt.model import ImageTextModel
    if model_type == 'pretrained':
        path = C.ensure_model_path(model_path, "eval-model")
        model = ImageTextModel.from_pretrained(path).to(device)
    else:  # state_dict (.pth)
        model = ImageTextModel.from_pretrained(base_p).to(device)
        sd = torch.load(model_path, map_location='cpu', weights_only=False)
        # bóc wrapper: baseline dual-checkpoint lưu {'model_state':...}, bản per-epoch lưu raw
        if isinstance(sd, dict):
            for k in ('model_state', 'state_dict', 'trainable_state'):
                if k in sd and isinstance(sd[k], dict):
                    print(f"   (bóc checkpoint key '{k}', epoch={sd.get('epoch', '?')})")
                    sd = sd[k]
                    break
        missing, unexpected = model.load_state_dict(sd, strict=False)
        n_total = len(model.state_dict())
        print(f"   load_state_dict: nạp {n_total - len(missing)}/{n_total} khóa "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")
        # CHẶN nạp hỏng: nếu thiếu quá nửa → model gần như còn nguyên BASE ⇒ số báo cáo sẽ SAI
        if len(missing) > 0.5 * n_total:
            raise RuntimeError(
                f"Nạp state_dict HỎNG: thiếu {len(missing)}/{n_total} khóa → model gần như là "
                f"BASE (chưa unlearn), số sẽ sai. Kiểm tra định dạng {model_path}.")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--label', type=str, required=True,
                    help="Nhãn checkpoint_kind trong CSV (og/re/forgetmi/cfk/euk/neggrad/loku...)")
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--model_type', type=str, default='pretrained',
                    choices=['pretrained', 'state_dict'])
    ap.add_argument('--method', type=str, default='reference',
                    help="Cột 'method' trong CSV (mặc định 'reference')")
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    cli = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(cli.config), 'p3')  # dùng common là đủ
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")
    print(f"🟢 Device: {device}  |  EVAL-ONLY label={cli.label}")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    from transformers import BertTokenizer
    from joint_img_txt.model import ImageTextModel
    base_p = C.ensure_model_path(cfg.base_model_path, "base")
    re_p = C.ensure_model_path(cfg.retrained_model_path, "retrained")
    tok = BertTokenizer.from_pretrained(
        base_p if os.path.exists(os.path.join(base_p, "vocab.txt")) else cfg.bert_pretrained_dir)

    print("📚 Building dataset (CÙNG 4-tập holdout)...")
    datasets, _ = C.build_dataset(cfg, tok)

    model_re = ImageTextModel.from_pretrained(re_p).to(device).half()
    gold_available = os.path.abspath(re_p) != os.path.abspath(base_p)
    model = load_eval_model(cli.model_path, cli.model_type, base_p, device)
    trainable, total = C.count_params(model)

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_advanced.csv')))
    run_id = f"{cli.label}_{os.path.basename(str(cfg.forget_set_path))}"

    C.final_evaluation(model, model_re, datasets, cfg, device, cli.method, run_id,
        checkpoint_kind=cli.label, selected_epoch=-1,
        timing={'train_h': 0.0, 'fisher_h': 0.0, 'wall_h': 0.0},
        trainable=trainable, total=total, total_optimizer_steps=0,
        csv_path=csv_path, gold_available=gold_available,
        extra_row={'eval_only': True, 'source_model_path': cli.model_path})
    print(f"✅ Eval-only [{cli.label}] done → {csv_path}")


if __name__ == '__main__':
    main()
