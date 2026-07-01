#!/usr/bin/env python
"""Simple unlearning BASELINES on the SAME pipeline as Forget-MI (controlled comparison).

Methods (chọn qua --method), theo đúng các baseline trong Forget-MI paper (Table 1):
  - neggrad : NegGrad+  — fine-tune trên retain Dr + ĐẢO gradient (−CE) trên forget Df.
  - cfk     : CF-k      — freeze k=2 layer đầu của MỖI encoder, fine-tune phần còn lại trên Dr.
  - euk     : EU-k      — freeze k=2 layer đầu, REINIT phần còn lại, train lại trên Dr.

Tái dùng nguyên: build_dataset (data giống hệt baseline/LoKU), get_model_inputs,
và _final_evaluation (MIA/AUC/CosSim + ghi CSV cùng format — chỉ khác 'method').
→ So sánh CÓ KIỂM SOÁT: mọi method cùng data/eval/seed, chỉ khác thuật toán unlearning.

Eval CHỈ 1 LẦN sau epoch cuối (không per-epoch) cho nhanh.

Loss phân loại = CE(image logits) + CE(text logits) = CE(outputs[1]) + CE(outputs[3]),
khớp cách train model gốc (train_iu_model.py) và head của kiến trúc late-fusion.

Setup khớp paper: retain được lấy mẫu bằng kích thước forget mỗi epoch (matched sampling),
lr/epoch/batch lấy từ config baseline → công bằng với Forget-MI.

Ví dụ:
    WANDB_MODE=disabled PYTHONPATH=. python scripts/unlearn_baselines.py \
        --config config_baseline_kaggle.yaml --method neggrad --seed 42 \
        --override "base_model_path=...,retrained_model_path=...,forget_set_path=...,\
text_data_dir=...,img_data_dir=...,data_split_path=...,id=neggrad_3per,\
output_dir=/kaggle/working/baseline_output"
"""
import argparse
import copy
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import BertTokenizer

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from joint_img_txt.model import ImageTextModel
from training.forgetmi_partial import (
    _load_config, _init_wandb, build_dataset, get_model_inputs, _final_evaluation,
)

# "First k=2 layers" của mỗi encoder (thích ứng CF-k/EU-k cho kiến trúc multimodal):
#   ảnh: conv1 + bn1 (layer đầu) + layer1 (layer thứ 2);  text: embeddings + encoder.layer.0.
FREEZE_PREFIXES = (
    'img_model.conv1', 'img_model.bn1', 'img_model.layer1',
    'text_model.bert.embeddings', 'text_model.bert.encoder.layer.0.',
)


def _set_seed(s):
    import random
    import numpy as np
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def _ce(outputs, labels):
    """CE trên image logits (outputs[1]) + text logits (outputs[3])."""
    return F.cross_entropy(outputs[1], labels) + F.cross_entropy(outputs[3], labels)


def _freeze_first_k(model):
    frozen = trainable = 0
    for n, p in model.named_parameters():
        if any(n.startswith(pre) for pre in FREEZE_PREFIXES):
            p.requires_grad = False
            frozen += p.numel()
        else:
            p.requires_grad = True
            trainable += p.numel()
    return frozen, trainable


def _reinit_unfrozen(model):
    n_reset = 0
    for name, module in model.named_modules():
        if not name or any(name.startswith(pre) for pre in FREEZE_PREFIXES):
            continue
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            n_reset += 1
    print(f"   ♻️  reset_parameters on {n_reset} unfrozen modules")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--method', required=True, choices=['neggrad', 'cfk', 'euk'])
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--override', default='')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--lr', type=float, default=None)
    ap.add_argument('--neggrad_lambda', type=float, default=1.0,
                    help='Trọng số cho số hạng −CE(forget) của NegGrad+')
    cli = ap.parse_args()

    _set_seed(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('🔴 WARNING: no GPU — sẽ rất chậm.')

    # Config: tái dùng loader của forgetmi_partial (xử lý --seed / --override).
    from types import SimpleNamespace
    cfg = _load_config(SimpleNamespace(config=cli.config, seed=cli.seed, override=cli.override))
    config = _init_wandb(cfg)

    lr = cli.lr if cli.lr is not None else float(getattr(config, 'learning_rate', 1e-5))
    wd = float(getattr(config, 'weight_decay', 0.01))
    bs = int(getattr(config, 'unlearn_batch_size', 16))
    workers = int(getattr(config, 'num_cpu_workers', 2))
    max_grad_norm = float(getattr(config, 'max_grad_norm', 1.0))

    base_out = str(getattr(config, 'output_dir', '/kaggle/working/baseline_output'))
    output_dir = os.path.join(base_out, f"{cli.method}_{getattr(config, 'id', 'run')}")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrained_dir)
    print(f"📦 Load F_og: {config.base_model_path}")
    model_og = ImageTextModel.from_pretrained(config.base_model_path).to(device)
    model_ul = copy.deepcopy(model_og).to(device)
    try:
        model_re = ImageTextModel.from_pretrained(config.retrained_model_path).to(device)
    except Exception as e:
        print(f"⚠️  Load retrained fail ({e}) → dùng model_og làm dummy (dist_vs_re INVALID)")
        model_re = copy.deepcopy(model_og).to(device)
    for p in model_re.parameters():
        p.requires_grad = False
    del model_og
    torch.cuda.empty_cache()

    print("📦 build_dataset (reuse forgetmi_partial)...")
    dataset, num_labels = build_dataset(config, tokenizer)
    retain_set, forget_set = dataset['retain'], dataset['forget']
    n_forget = len(forget_set)
    print(f"🎯 retain={len(retain_set)} forget={n_forget} test={len(dataset['test'])} num_labels={num_labels}")
    if n_forget == 0 or len(retain_set) == 0:
        raise SystemExit("retain/forget rỗng — kiểm tra forget_set / data_split / cache.")

    total_params = sum(p.numel() for p in model_ul.parameters())
    if cli.method == 'euk':
        print("♻️  EU-k: reinit các layer KHÔNG bị freeze")
        _reinit_unfrozen(model_ul)
    if cli.method in ('cfk', 'euk'):
        frozen, trainable = _freeze_first_k(model_ul)
        print(f"🧊 {cli.method.upper()}: frozen={frozen:,} | trainable={trainable:,} "
              f"({100.0 * trainable / total_params:.1f}%)")
    else:
        trainable = total_params  # NegGrad+ = full fine-tune

    params = [p for p in model_ul.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    # retain lấy mẫu = kích thước forget MỖI epoch (khớp setup paper) → nhanh + công bằng.
    retain_loader = DataLoader(
        retain_set,
        sampler=RandomSampler(retain_set, replacement=True, num_samples=n_forget),
        batch_size=bs, num_workers=workers, pin_memory=True)
    forget_loader = DataLoader(
        forget_set, sampler=RandomSampler(forget_set),
        batch_size=bs, num_workers=workers, pin_memory=True)

    model_ul.train()
    t0 = time.time()
    print(f"\n{'=' * 70}\n🚂 {cli.method.upper()} — epochs={cli.epochs} lr={lr} bs={bs}\n{'=' * 70}")
    for epoch in range(cli.epochs):
        run_loss, n = 0.0, 0
        if cli.method == 'neggrad':
            steps = min(len(forget_loader), len(retain_loader))
            pbar = tqdm(zip(forget_loader, retain_loader), total=steps, desc=f"E{epoch:02d}/{cli.epochs}")
            for f_batch, r_batch in pbar:
                f_in, f_lab, _ = get_model_inputs(config, f_batch, device)
                r_in, r_lab, _ = get_model_inputs(config, r_batch, device)
                f_lab = f_lab.long().view(-1)
                r_lab = r_lab.long().view(-1)
                optimizer.zero_grad()
                r_out = model_ul(**r_in)
                f_out = model_ul(**f_in)
                loss = _ce(r_out, r_lab) - cli.neggrad_lambda * _ce(f_out, f_lab)
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                run_loss += loss.item()
                n += 1
                pbar.set_postfix(loss=f"{run_loss / n:.4f}")
        else:  # cfk / euk: fine-tune trên retain (Dr) bằng CE
            pbar = tqdm(retain_loader, desc=f"E{epoch:02d}/{cli.epochs}")
            for r_batch in pbar:
                r_in, r_lab, _ = get_model_inputs(config, r_batch, device)
                r_lab = r_lab.long().view(-1)
                optimizer.zero_grad()
                r_out = model_ul(**r_in)
                loss = _ce(r_out, r_lab)
                loss.backward()
                nn.utils.clip_grad_norm_(params, max_grad_norm)
                optimizer.step()
                run_loss += loss.item()
                n += 1
                pbar.set_postfix(loss=f"{run_loss / n:.4f}")
        print(f"[E{epoch:02d}/{cli.epochs}] loss={run_loss / max(1, n):+.4f} "
              f"elapsed={(time.time() - t0) / 60:.1f}m")

    elapsed_h = (time.time() - t0) / 3600
    print(f"⏱  {cli.method.upper()} done in {elapsed_h * 60:.1f} min ({elapsed_h:.2f}h)")

    # Eval 1 LẦN (không per-epoch) → tái dùng eval baseline + ghi CSV method-labeled.
    _final_evaluation(config, output_dir, device, model_ul, model_re, dataset,
                      elapsed_h, trainable, total_params, tracker=None, method=cli.method)


if __name__ == '__main__':
    main()
