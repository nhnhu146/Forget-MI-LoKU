#!/usr/bin/env python3
"""
eval_only_e30.py — Đánh giá một checkpoint có sẵn trên phân hoạch E30
=====================================================================
Dùng cho OG và GOLD (retrained): chúng là checkpoint tải về, KHÔNG cần huấn luyện lại
— chỉ cần đo lại trên tập test mới (531 thay vì 398).

Khác ``forgetmi_eval_only.py`` đúng MỘT dòng: gọi ``adv_e30.build_dataset_e30`` thay
cho ``adv_common.build_dataset``. Hàm nạp model (``load_eval_model``) và hàm chấm
(``final_evaluation``) dùng NGUYÊN của bản gốc, không chép.

Chạy:
  python training/eval_only_e30.py --config config_advanced_kaggle.yaml --seed 42 \\
      --label og_mimic_3per_e30 --model_type pretrained --model_path <thư mục OG> \\
      --method reference --override "..."

  # với checkpoint .pth (vd baseline last.pt):
  ... --model_type state_dict --model_path <đường/dẫn/last.pt>
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
from training.forgetmi_eval_only import load_eval_model     # dùng nguyên, không chép


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--label', type=str, required=True)
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--model_type', type=str, default='pretrained',
                    choices=['pretrained', 'state_dict'])
    ap.add_argument('--method', type=str, default='reference')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    ap.add_argument('--allow-core-drift', action='store_true')
    cli = ap.parse_args()

    E.assert_core_unchanged(strict=not cli.allow_core_drift)

    cfg_d = C.flatten_method_config(C.load_config(cli.config), 'p3')
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d = C.apply_overrides(cfg_d, cli.override)
    cfg_d['ce_selector'] = 0
    cfg = C.Cfg(cfg_d)

    C.set_seed(int(cfg.random_seed))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")
    print(f"🟢 Device: {device}  |  EVAL-ONLY E30  label={cli.label}")

    from transformers import BertTokenizer
    from joint_img_txt.model import ImageTextModel
    base_p = C.ensure_model_path(cfg.base_model_path, "base")
    re_p = C.ensure_model_path(cfg.retrained_model_path, "retrained")
    tok = BertTokenizer.from_pretrained(
        base_p if os.path.exists(os.path.join(base_p, "vocab.txt")) else cfg.bert_pretrained_dir)

    print("📚 Building dataset (4 tập code gốc — E30)...")
    datasets, _ = E.build_dataset_e30(cfg, tok)     # <-- KHÁC BẢN GỐC ĐÚNG DÒNG NÀY

    model_re = ImageTextModel.from_pretrained(re_p).to(device).half()
    gold_available = os.path.abspath(re_p) != os.path.abspath(base_p)
    model = load_eval_model(cli.model_path, cli.model_type, base_p, device)
    trainable, total = C.count_params(model)

    out_dir = cfg.output_dir
    os.makedirs(out_dir, exist_ok=True)
    csv_path = str(getattr(cfg, 'results_csv_path', os.path.join(out_dir, 'results_e30.csv')))
    run_id = f"{cli.label}_{os.path.basename(str(cfg.forget_set_path))}"

    C.final_evaluation(model, model_re, datasets, cfg, device, cli.method, run_id,
        checkpoint_kind=cli.label, selected_epoch=-1,
        timing={'train_h': 0.0, 'fisher_h': 0.0, 'wall_h': 0.0},
        trainable=trainable, total=total, total_optimizer_steps=0,
        csv_path=csv_path, gold_available=gold_available,
        extra_row={'eval_only': True, 'source_model_path': cli.model_path,
                   'split_protocol': 'e30_original'})
    print(f"✅ Eval-only E30 [{cli.label}] done → {csv_path}")


if __name__ == '__main__':
    main()
