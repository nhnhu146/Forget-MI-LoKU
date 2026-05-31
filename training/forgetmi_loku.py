#!/usr/bin/env python3
"""
Forget-MI-LoKU: Forget-MI + LoRA + Fisher-guided init
------------------------------------------------------
Improvements over baseline Forget-MI (forgetmi_partial.py):
  1. Trainable params ~1-3% of full model (LoRA adapters on BERT attention only)
  2. Fisher Information (cross-entropy gradients) used to initialize LoRA in the
     "forget-relevant" subspace via importance-weighted SVD (soft init: base weights
     untouched, LoRA injected at small scale)
  3. Bounded losses (relu-hinge) so L_uu/L_md cannot diverge to -inf
  4. Optional anchor term toward model_re (gold-standard retrained) to help convergence
  5. Full evaluation pipeline (MIA + CosSim + AUC + F1) + efficiency metrics
     (wall-clock time, GPU peak memory, trainable params) logged at end
"""
import os
import sys
import time
import random
import csv
import yaml
import argparse
import shutil
from datetime import datetime
from functools import reduce

# Make project root importable
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from tqdm import tqdm
import wandb
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler, RandomSampler, Subset
from transformers import BertTokenizer
from peft import LoraConfig, get_peft_model

# Forget-MI shared modules
from training.joint_embedding import Gate
from joint_img_txt.model_utils import (
    CXRImageTextDataset, EdemaClassificationProcessor, EdemaMultiLabelClassificationProcessor,
    RandomTranslateCrop, CenterCrop,
)
from joint_img_txt.model import ImageTextModel
from joint_img_txt.convert_examples_to_features import (
    convert_examples_to_features, convert_examples_to_features_multilabel,
)


# ============================================================================
# Utilities
# ============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def euclidean_distance(a, b):
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1) + 1e-8)


class AlignedSampler(Sampler):
    def __init__(self, n, shuffle=False, seed=None):
        self.n = n; self.shuffle = shuffle; self.seed = seed

    def __iter__(self):
        idx = list(range(self.n))
        if self.shuffle:
            if self.seed is not None:
                torch.manual_seed(self.seed)
            idx = torch.randperm(self.n).tolist()
        return iter(idx)

    def __len__(self):
        return self.n


def get_module(model, path):
    return reduce(getattr, path.split('.'), model)


def count_params(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def _model_dtype(model):
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    return torch.float32


def safe_forward(model, inputs):
    """
    Forward pass that auto-casts floating inputs to match model dtype.
    Fixes 'Input type FloatTensor and weight type HalfTensor' errors when
    calling fp16 frozen models outside autocast.
    """
    dtype = _model_dtype(model)
    casted = {
        k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != dtype else v)
        for k, v in inputs.items()
    }
    return model(**casted)


# ============================================================================
# Dataset
# ============================================================================

def data_split(split_list_path, forget_ids_path, validation_ratio=0.1):
    random.seed(0)
    train_l, train_ids = {}, {}
    test_l, test_ids = {}, {}
    rand_l, rand_ids = {}, {}
    forget_l, forget_ids = {}, {}

    with open(split_list_path, 'r') as f:
        forget_set = pd.read_csv(forget_ids_path).astype(str).subject_id.values
        reader = csv.reader(f); next(reader)
        for row in reader:
            if row[3] == 'edeme_severity':
                continue
            try:
                sev = float(row[3])
            except ValueError:
                continue
            if row[0] in forget_set:
                forget_l[row[2]] = [sev]; forget_ids[row[2]] = row[1]
                rand_l[row[2]] = [sev];   rand_ids[row[2]]   = row[1]
            else:
                if row[-1] == 'TEST':
                    test_l[row[2]] = [sev]; test_ids[row[2]] = row[1]
                else:
                    train_l[row[2]] = [sev]; train_ids[row[2]] = row[1]

    from sklearn.model_selection import train_test_split
    tr_ids = list(train_ids.keys()); tr_vals = list(train_ids.values())
    tr_labs = list(train_l.values())
    tr_ids, val_ids, tr_vals, val_vals, tr_labs, val_labs = train_test_split(
        tr_ids, tr_vals, tr_labs, test_size=validation_ratio, random_state=42, stratify=tr_labs
    )
    val_l = dict(zip(val_ids, val_labs)); val_id_map = dict(zip(val_ids, val_vals))
    retain_l = dict(zip(tr_ids, tr_labs)); retain_id_map = dict(zip(tr_ids, tr_vals))
    return (retain_l, retain_id_map, val_l, val_id_map,
            test_l, test_ids, rand_l, rand_ids, forget_l, forget_ids)


def build_dataset(args, tokenizer):
    processor = (EdemaMultiLabelClassificationProcessor()
                 if args.output_channel_encoding == 'multilabel'
                 else EdemaClassificationProcessor())
    num_labels = len(processor.get_labels())
    get_features = (convert_examples_to_features_multilabel
                    if args.output_channel_encoding == 'multilabel'
                    else convert_examples_to_features)

    cached = os.path.join(args.text_data_dir,
                          f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")
    cached_noisy = os.path.join(args.text_data_dir,
                                f"cachednoisyfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")

    if os.path.exists(cached) and os.path.exists(cached_noisy) and not args.reprocess_input_data:
        # weights_only=False needed for PyTorch 2.6+ (cached files contain custom InputFeatures class)
        features = torch.load(cached, weights_only=False)
        noisy_features = torch.load(cached_noisy, weights_only=False)
    else:
        synonyms = pd.read_csv(args.synonyms_dir)
        examples = processor.get_all_examples(args.text_data_dir)
        noisy_examples = processor.get_noisy_examples(args.text_data_dir, synonyms, args.text_noise_level)
        features = get_features(examples, processor.get_labels(), args.max_seq_length, tokenizer)
        noisy_features = get_features(noisy_examples, processor.get_labels(), args.max_seq_length, tokenizer)
        torch.save(features, cached); torch.save(noisy_features, cached_noisy)

    all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in features}
    noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in noisy_features}

    (retain_l, retain_ids, val_l, val_ids,
     test_l, test_ids, rand_l, rand_ids, forget_l, forget_ids) = \
        data_split(args.data_split_path, args.forget_set_path, args.validation_ratio)

    def extract(mapping, ids):
        t, m, s, l = {}, {}, {}, {}
        for rid in ids:
            t[rid], m[rid], s[rid], l[rid] = mapping[rid]
        return t, m, s, l

    train_trans = RandomTranslateCrop(2048)
    eval_trans = CenterCrop(2048)

    datasets = {}
    for name, ids, labels, trans, txt_map in [
        ('retain', retain_ids, retain_l, train_trans, all_txt),
        ('validation', val_ids, val_l, eval_trans, all_txt),
        ('test', test_ids, test_l, eval_trans, all_txt),
        ('forget', forget_ids, forget_l, train_trans, all_txt),
        ('random', rand_ids, rand_l, train_trans, noisy_txt),
    ]:
        valid = [d for d, r in ids.items() if r in txt_map]
        if len(valid) < len(ids):
            print(f"⚠️  {len(ids) - len(valid)} items in '{name}' set skipped (text not found)")
        f_ids = {d: ids[d] for d in valid}
        f_labels = {d: labels[d] for d in valid}
        tk, mk, sg, lb = extract(txt_map, f_ids.values())
        datasets[name] = CXRImageTextDataset(
            args.id, tk, mk, sg, lb, f_ids, args.img_data_dir, f_labels,
            args.data_split_path, transform=trans,
            output_channel_encoding=args.output_channel_encoding,
        )
        print(f"  [{name}] {len(datasets[name])} samples")
    return datasets, num_labels


def get_model_inputs(args, batch, device):
    image, label_raw, txt_ids, txt_mask, txt_seg, label_onehot, report_id = batch
    inputs = {
        'input_img': image.to(device),
        'input_ids': txt_ids.to(device),
        'attention_mask': txt_mask.to(device),
        'token_type_ids': txt_seg.to(device),
        'bert_pool_last_hidden': getattr(args, 'bert_pool_last_hidden', False),
        'bert_pool_use_img': getattr(args, 'bert_pool_use_img', False),
        'bert_pool_img_lowerlevel': getattr(args, 'bert_pool_img_lowerlevel', False),
    }
    return inputs, label_raw.to(device), report_id.to(device)


# ============================================================================
# Fisher Information & LoKU SVD initialization
# ============================================================================

def compute_fisher_importance(model, dataloader, device, target_modules, args,
                              max_samples=512):
    """
    Approximate Fisher Information: E[(∂ log p(y|x) / ∂θ)^2]
    Uses cross-entropy at the TRUE labels (img + txt logits).
    """
    model.eval()
    importance = {
        name: torch.zeros_like(p.data)
        for name, p in model.named_parameters()
        if any(tm in name for tm in target_modules) and 'weight' in name
    }
    n = 0
    for batch in tqdm(dataloader, desc="Fisher"):
        if n >= max_samples:
            break
        inputs, labels, _ = get_model_inputs(args, batch, device)
        labels = labels.long().view(-1)
        model.zero_grad()
        outputs = model(**inputs)
        # img_logits = outputs[1], txt_logits = outputs[3]
        loss = F.cross_entropy(outputs[1], labels) + F.cross_entropy(outputs[3], labels)
        loss.backward()
        for name, p in model.named_parameters():
            if name in importance and p.grad is not None:
                importance[name] += p.grad.data.pow(2)
        n += batch[0].size(0)
    if n == 0:
        print("⚠️  No samples for Fisher. Returning zeros.")
        return importance
    return {k: v / n for k, v in importance.items()}


def apply_loku_soft_init(model, forget_imp, retain_imp, target_modules,
                         r, init_scale=0.05):
    """
    Soft LoKU init: base weights UNTOUCHED. LoRA initialized via
    importance-weighted truncated SVD, scaled small so initial forward
    behaves like the original model.

    For each target weight W:
      imp = forget_imp / (retain_imp + eps)   # relative forget-importance
      Wp  = sqrt(imp) * W                      # scale rows by importance
      U,S,V = svd_lowrank(Wp, r)
      lora_A = (V * sqrt(S)).T * scale
      lora_B = (U * sqrt(S))   * scale
    """
    matched, skipped = 0, 0
    for name, f_imp in forget_imp.items():
        r_imp = retain_imp.get(name, torch.zeros_like(f_imp))
        # Relative importance, capped to avoid blowups
        rel = (f_imp / (r_imp + 1e-6)).clamp(0, 1e3)
        path = name.replace('.weight', '')
        try:
            mod = get_module(model, path)
            base = mod.base_layer
            lora_A = mod.lora_A['default']
            lora_B = mod.lora_B['default']
        except (AttributeError, KeyError):
            skipped += 1
            continue

        W = base.weight.data.float()
        row_imp = rel.mean(dim=1).sqrt() + 1e-6
        Wp = row_imp[:, None] * W
        try:
            U, S, V = torch.svd_lowrank(Wp, q=r)
        except Exception:
            skipped += 1
            continue

        S_sqrt = torch.sqrt(S + 1e-8)
        new_A = (V * S_sqrt).t() * init_scale
        new_B = (U * S_sqrt) * init_scale
        lora_A.weight.data = new_A.to(lora_A.weight.dtype)
        lora_B.weight.data = new_B.to(lora_B.weight.dtype)
        matched += 1
    print(f"✅ LoKU soft-init applied to {matched} layers ({skipped} skipped)")


# ============================================================================
# Checkpoint
# ============================================================================

def save_ckpt(out_dir, epoch, model, gates, optimizer, metrics):
    cp = os.path.join(out_dir, "checkpoints")
    os.makedirs(cp, exist_ok=True)
    lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k.lower()}
    payload = {
        'epoch': epoch,
        'lora_state': lora_state,
        'gates': {n: g.state_dict() for n, g in gates.items()},
        'optimizer': optimizer.state_dict(),
        'metrics': metrics,
    }
    torch.save(payload, os.path.join(cp, "latest.pt"))


def load_ckpt(out_dir, model, gates, optimizer):
    p = os.path.join(out_dir, "checkpoints", "latest.pt")
    if not os.path.exists(p):
        return -1
    try:
        cp = torch.load(p, map_location='cpu', weights_only=False)
        model.load_state_dict(cp['lora_state'], strict=False)
        for n, g in gates.items():
            if n in cp['gates']:
                g.load_state_dict(cp['gates'][n])
        optimizer.load_state_dict(cp['optimizer'])
        print(f"📂 Resumed from epoch {cp['epoch']}")
        return cp['epoch']
    except Exception as e:
        print(f"⚠️  Checkpoint incompatible ({e}); starting fresh.")
        return -1


# ============================================================================
# Evaluation (MIA + CosSim + AUC + F1)
# ============================================================================

@torch.no_grad()
def per_sample_ce(model, dataset, device, args, batch_size=32):
    """Per-sample cross-entropy on img_logits (for MIA)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, labels, _ = get_model_inputs(args, batch, device)
        outputs = safe_forward(model, inputs)
        logits = outputs[1].float()
        losses = F.cross_entropy(logits, labels.long().view(-1), reduction='none')
        out.append(losses.cpu().numpy())
    return np.concatenate(out)


def run_mia(model, retain_ds, test_ds, forget_ds, device, args, batch_size=32, seed=42):
    """SVM-based MIA. Returns fraction of forget predicted as 'member' (label=1)."""
    from sklearn.svm import SVC
    print("  computing per-sample losses...")
    rl = per_sample_ce(model, retain_ds, device, args, batch_size).reshape(-1, 1)
    tl = per_sample_ce(model, test_ds, device, args, batch_size).reshape(-1, 1)
    fl = per_sample_ce(model, forget_ds, device, args, batch_size).reshape(-1, 1)
    n = min(len(rl), len(tl))
    rng = np.random.default_rng(seed)
    r_sub = rl[rng.choice(len(rl), n, replace=False)]
    t_sub = tl[rng.choice(len(tl), n, replace=False)]
    X = np.vstack([r_sub, t_sub])
    y = np.concatenate([np.ones(n), np.zeros(n)])
    clf = SVC(C=3, kernel='rbf', gamma='auto', random_state=seed)
    clf.fit(X, y)
    return float(clf.predict(fl).mean())


@torch.no_grad()
def cosine_sim_models(model_a, model_b, dataset, device, args, batch_size=32):
    """Mean cosine similarity of img_logits between two models on a dataset.
    Handles fp16/fp32 dtype mismatch via safe_forward."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sims = []
    model_a.eval(); model_b.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, _, _ = get_model_inputs(args, batch, device)
        la = safe_forward(model_a, inputs)[1].float()
        lb = safe_forward(model_b, inputs)[1].float()
        sims.append(F.cosine_similarity(la, lb, dim=1).cpu().numpy())
    return float(np.concatenate(sims).mean())


@torch.no_grad()
def perf_metrics(model, dataset, device, args, batch_size=32, num_classes=4):
    """Compute AUC + Macro-F1 + Accuracy on a dataset.
    Uses label_onehot from batch (compute_auc/get_acc_f1 expect [N, num_classes])."""
    from joint_img_txt.metrics import compute_auc, get_acc_f1
    from scipy.special import softmax
    from scipy.stats import logistic
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits, all_labels_oh = [], []
    model.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, label_raw, _ = get_model_inputs(args, batch, device)
        # batch structure: image[0], label_raw[1], txt_ids[2], txt_mask[3], txt_seg[4], label_onehot[5], report_id[6]
        label_oh = batch[5]
        outputs = safe_forward(model, inputs)
        all_logits.append(outputs[1].float().cpu().numpy())
        # Ensure one-hot [N, num_classes]
        oh_np = label_oh.cpu().numpy()
        if oh_np.ndim == 1 or oh_np.shape[-1] == 1:
            # label_raw fallback: convert raw class index to one-hot
            raw = oh_np.flatten().astype(int)
            oh_np = np.eye(num_classes)[np.clip(raw, 0, num_classes - 1)]
        all_labels_oh.append(oh_np.astype(int))
    logits = np.concatenate(all_logits, axis=0)
    labels_oh = np.concatenate(all_labels_oh, axis=0)
    if args.output_channel_encoding == 'multilabel':
        probs = logistic.cdf(logits)
    else:
        probs = softmax(logits, axis=1)
    try:
        auc, _ = compute_auc(labels_oh.tolist(), probs.tolist(),
                             output_channel_encoding=args.output_channel_encoding)
        # filter out NaN per-class AUCs
        valid = [a for a in auc if not (isinstance(a, float) and np.isnan(a))]
        auc_mean = float(np.mean(valid)) if valid else float('nan')
    except Exception as e:
        print(f"   [auc warn] {e}"); auc_mean = float('nan')
    try:
        f1m, _, _ = get_acc_f1(labels_oh, probs, args.output_channel_encoding)
        macro_f1 = float(f1m['macro_f1']); f1 = float(f1m['f1'])
        accuracy = float(f1m['accuracy'])
    except Exception as e:
        print(f"   [f1 warn] {e}")
        macro_f1, f1, accuracy = float('nan'), float('nan'), float('nan')
    return {'AUC': auc_mean, 'Macro_F1': macro_f1, 'F1': f1, 'Accuracy': accuracy}


# ============================================================================
# Training loop
# ============================================================================

def unlearn(args, out_dir, device, model_og, model_ul, model_re, gates, optimizer,
            datasets, weights, tracker=None):
    alpha, beta, theta, gamma, eta = weights  # last is re-anchor weight
    start_epoch = load_ckpt(out_dir, model_ul, gates, optimizer) + 1

    model_og.eval(); model_re.eval()
    for p in model_og.parameters(): p.requires_grad = False
    for p in model_re.parameters(): p.requires_grad = False

    nw = min(int(getattr(args, 'num_cpu_workers', 2)), 2)
    forget_dl = DataLoader(datasets['forget'],
                           sampler=AlignedSampler(len(datasets['forget']), shuffle=True, seed=42),
                           batch_size=args.unlearn_batch_size, num_workers=nw, pin_memory=False)
    rand_dl = DataLoader(datasets['random'],
                         sampler=AlignedSampler(len(datasets['random']), shuffle=True, seed=42),
                         batch_size=args.unlearn_batch_size, num_workers=nw, pin_memory=False)

    forget_margin = float(getattr(args, 'forget_margin', 8.0))
    use_re_anchor = eta > 0
    grad_clip = float(getattr(args, 'grad_clip', 1.0))
    # Classification loss weights
    kappa_ret = float(getattr(args, 'kappa_cls_retain', 1.0))   # keep retain classification correct
    kappa_frg = float(getattr(args, 'kappa_cls_forget', 0.5))   # NegGrad weight (legacy)
    cls_clamp = float(getattr(args, 'cls_forget_clamp', 5.0))   # cap forget CE to avoid divergence
    # Uniform-prior unlearning (NEW): push forget predictions toward uniform distribution
    # Better than NegGrad: Forget F1 ↓ (random predictions) + MIA ↓ (forget loss ≈ test loss)
    uniform_weight = float(getattr(args, 'uniform_prior_weight', 0.0))   # 0 = disabled (legacy NegGrad)
    eval_subset = Subset(datasets['retain'],
                         list(range(min(200, len(datasets['retain'])))))

    best_cossim = -1.0; patience = 0
    margin_ukr = margin_mkr = None
    train_start = time.time()

    for epoch in range(start_epoch, args.unlearn_epochs):
        model_ul.train()
        for g in gates.values(): g.train()

        retain_dl = DataLoader(datasets['retain'],
                               sampler=RandomSampler(datasets['retain']),
                               batch_size=args.unlearn_batch_size,
                               num_workers=nw)
        epoch_iter = tqdm(zip(forget_dl, rand_dl, retain_dl), desc=f"Epoch {epoch}")
        agg = {'UU': 0, 'MD': 0, 'UKR': 0, 'MKR': 0, 'RE_anchor': 0,
               'CLS_ret': 0, 'CLS_frg': 0, 'Total': 0}
        steps = 0

        for fb, rb, retb in epoch_iter:
            ret_in, ret_lbl, _ = get_model_inputs(args, retb, device)
            f_in, f_lbl, _ = get_model_inputs(args, fb, device)
            rnd_in, _, _ = get_model_inputs(args, rb, device)

            # Frozen forward passes (fp16 via autocast)
            with torch.no_grad(), torch.autocast(device_type='cuda'):
                og_ret_i, _, og_ret_t, _ = model_og(**ret_in)[:4]
                og_rnd_i, _, og_rnd_t, _ = model_og(**rnd_in)[:4]
                if use_re_anchor:
                    re_ret_i, _, re_ret_t, _ = model_re(**ret_in)[:4]
            og_ret_i = og_ret_i.float(); og_ret_t = og_ret_t.float()
            og_rnd_i = og_rnd_i.float(); og_rnd_t = og_rnd_t.float()
            if use_re_anchor:
                re_ret_i = re_ret_i.float(); re_ret_t = re_ret_t.float()

            # Unlearning model forward (trainable) — also capture logits for classification loss
            ul_ret_i, ul_ret_il, ul_ret_t, ul_ret_tl = model_ul(**ret_in)[:4]
            ul_frg_i, ul_frg_il, ul_frg_t, ul_frg_tl = model_ul(**f_in)[:4]

            # Joint embeddings
            ul_ret_j = gates['ul_ret'](ul_ret_i, ul_ret_t)
            ul_frg_j = gates['ul_frg'](ul_frg_i, ul_frg_t)
            og_rnd_j = gates['og_rnd'](og_rnd_i, og_rnd_t)
            og_ret_j = gates['og_ret'](og_ret_i, og_ret_t)

            # ----- Bounded forget losses (relu-hinge: push only until margin) -----
            ul_frg_c = torch.cat((ul_frg_i, ul_frg_t), dim=-1)
            og_rnd_c = torch.cat((og_rnd_i, og_rnd_t), dim=-1)
            d_uu = euclidean_distance(ul_frg_c, og_rnd_c).mean()
            d_md = euclidean_distance(ul_frg_j, og_rnd_j).mean()
            if args.use_noise:
                # original "minimize distance to noisy version" — keep direction
                L_uu = d_uu; L_md = d_md
            else:
                # push apart, but only up to forget_margin (avoids -inf divergence)
                L_uu = F.relu(forget_margin - d_uu)
                L_md = F.relu(forget_margin - d_md)

            # ----- Retain losses with correct-direction hinge -----
            ul_ret_c = torch.cat((ul_ret_i, ul_ret_t), dim=-1)
            og_ret_c = torch.cat((og_ret_i, og_ret_t), dim=-1)
            d_ukr = euclidean_distance(ul_ret_c, og_ret_c).mean()
            d_mkr = euclidean_distance(ul_ret_j, og_ret_j).mean()
            if margin_ukr is None:
                margin_ukr = (d_ukr + 1.0).detach()
                margin_mkr = (d_mkr + 1.0).detach()
            # penalize ONLY when retain drift exceeds margin (correct hinge direction)
            L_ukr = F.relu(d_ukr - margin_ukr) + 0.1 * d_ukr  # small constant pull
            L_mkr = F.relu(d_mkr - margin_mkr) + 0.1 * d_mkr

            # ----- Optional anchor: pull retain repr toward model_re's retain repr -----
            L_re = torch.tensor(0.0, device=device)
            if use_re_anchor:
                re_ret_c = torch.cat((re_ret_i, re_ret_t), dim=-1)
                L_re = euclidean_distance(ul_ret_c, re_ret_c).mean()

            # ----- Classification losses (provides gradient signal to img/txt classifier heads) -----
            # Retain: standard CE (keep classification correct)
            ret_y = ret_lbl.long().view(-1)
            f_y = f_lbl.long().view(-1)
            L_cls_ret = 0.5 * (F.cross_entropy(ul_ret_il, ret_y) + F.cross_entropy(ul_ret_tl, ret_y))

            # Forget: 2 options based on uniform_weight
            #   uniform_weight = 0 → NegGrad (legacy): push CE up, can over-train
            #   uniform_weight > 0 → Uniform Prior: push softmax toward uniform (better for MIA)
            if uniform_weight > 0:
                # Uniform prior: KL(softmax(logits) || uniform)
                n_cls = ul_frg_il.size(-1)
                log_p_img = F.log_softmax(ul_frg_il, dim=-1)
                log_p_txt = F.log_softmax(ul_frg_tl, dim=-1)
                # KL(p || uniform) = sum p * log(p / (1/n)) = sum p*log(p) + log(n)
                # Equivalent: minimize -entropy + log(n). Just minimize -entropy.
                L_cls_frg_raw = -0.5 * (
                    -(log_p_img.exp() * log_p_img).sum(dim=-1).mean()
                    -(log_p_txt.exp() * log_p_txt).sum(dim=-1).mean()
                )  # negative entropy (want to MAXIMIZE entropy → min negative entropy)
                L_cls_frg = uniform_weight * L_cls_frg_raw
            else:
                # NegGrad: maximize CE on forget (capped)
                L_cls_frg_raw = 0.5 * (F.cross_entropy(ul_frg_il, f_y) + F.cross_entropy(ul_frg_tl, f_y))
                L_cls_frg = -torch.clamp(L_cls_frg_raw, max=cls_clamp) * kappa_frg

            loss = (alpha * L_ukr + beta * L_uu
                    + theta * L_md + gamma * L_mkr
                    + eta * L_re
                    + kappa_ret * L_cls_ret + L_cls_frg)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model_ul.parameters() if p.requires_grad] +
                [p for g in gates.values() for p in g.parameters()],
                max_norm=grad_clip,
            )
            optimizer.step()

            agg['UU'] += L_uu.item(); agg['MD'] += L_md.item()
            agg['UKR'] += L_ukr.item(); agg['MKR'] += L_mkr.item()
            agg['RE_anchor'] += L_re.item()
            agg['CLS_ret'] += L_cls_ret.item()
            agg['CLS_frg'] += L_cls_frg.item()
            agg['Total'] += loss.item()
            steps += 1

        avg = {k: v / max(steps, 1) for k, v in agg.items()}

        # Eval: CosSim model_ul vs model_re on retain subset
        cossim = cosine_sim_models(model_ul, model_re, eval_subset, device, args,
                                   batch_size=args.eval_batch_size)
        log = {'epoch': epoch, 'cossim_vs_re': cossim,
               **{f'loss/{k}': v for k, v in avg.items()}}
        wandb.log(log)
        epoch_line = (f"[E{epoch:02d}] loss={avg['Total']:+.3f}  UU={avg['UU']:+.3f}  "
                      f"MD={avg['MD']:+.3f}  UKR={avg['UKR']:+.3f}  MKR={avg['MKR']:+.3f}  "
                      f"RE={avg['RE_anchor']:+.3f}  CLS_ret={avg['CLS_ret']:+.3f}  "
                      f"CLS_frg={avg['CLS_frg']:+.3f}  | CosSim(ul,re)={cossim:.4f}")
        print(epoch_line)
        if tracker is not None:
            tracker.log_epoch_line(epoch_line)
        save_ckpt(out_dir, epoch, model_ul, gates, optimizer, avg)

        # Early stopping on CosSim (higher = better)
        if cossim > best_cossim + 1e-4:
            best_cossim = cossim; patience = 0
        else:
            patience += 1
            if patience >= int(getattr(args, 'early_stop_patience', 4)):
                print(f"⏹  Early stop at epoch {epoch} (best CosSim={best_cossim:.4f})")
                break

    elapsed_h = (time.time() - train_start) / 3600.0
    return elapsed_h


# ============================================================================
# Model path discovery (handles nested-zip extraction)
# ============================================================================

def ensure_model_path(path, desc="model"):
    if os.path.exists(os.path.join(path, "pytorch_model.bin")):
        return path
    import glob
    nested = glob.glob(os.path.join(path, "**", "pytorch_model.bin"), recursive=True)
    if nested:
        return os.path.dirname(nested[0])
    base = os.path.basename(path.rstrip('/'))
    for m in glob.glob("/content/**/pytorch_model.bin", recursive=True):
        if base in m and "checkpoint" not in m:
            return os.path.dirname(m)
    print(f"❌ Cannot locate {desc} at {path}")
    return path


# ============================================================================
# Main
# ============================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    ap.add_argument("--fresh", action="store_true",
                    help="Delete prior checkpoints before training")
    ap.add_argument("--exp", type=str, default=None,
                    help="Experiment short name → auto-track to experiments/exp_NNN_<name>.md")
    ap.add_argument("--hypothesis", type=str, default=None,
                    help="One-line hypothesis for the experiment (auto-fill section 1)")
    cli = ap.parse_args()

    # ----- Optional experiment tracker -----
    tracker = None
    if cli.exp:
        try:
            from scripts.exp_tracker import ExpTracker
            tracker = ExpTracker(name=cli.exp, hypothesis=cli.hypothesis)
        except Exception as e:
            print(f"⚠️  ExpTracker init failed ({e}); continuing without tracking")
            tracker = None

    with open(cli.config, 'r') as f:
        raw = yaml.safe_load(f)
    if 'parameters' in raw:
        cfg = {k: (v['value'] if isinstance(v, dict) and 'value' in v
                   else (v['values'][0] if isinstance(v, dict) and 'values' in v
                         else v))
               for k, v in raw['parameters'].items()}
    else:
        cfg = raw

    wandb.init(project=cfg.get('project', 'forget_exp'),
               entity=cfg.get('entity', None),
               config=cfg, mode=os.environ.get('WANDB_MODE', 'online'))
    config = wandb.config
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = config.output_dir
    os.makedirs(out_dir, exist_ok=True)
    if cli.fresh:
        cp_dir = os.path.join(out_dir, "checkpoints")
        if os.path.exists(cp_dir):
            shutil.rmtree(cp_dir)
            print("🧹 Removed old checkpoints")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # ----- Load models -----
    base_p = ensure_model_path(config.base_model_path, "base")
    re_p = ensure_model_path(config.retrained_model_path, "retrained")
    model_og = ImageTextModel.from_pretrained(base_p).to(device)         # fp32 for Fisher
    model_unlearn = ImageTextModel.from_pretrained(base_p).to(device)    # fp32 for LoRA training
    model_re = ImageTextModel.from_pretrained(re_p).to(device).half()    # fp16 frozen
    tok = BertTokenizer.from_pretrained(
        base_p if os.path.exists(os.path.join(base_p, "vocab.txt"))
        else config.bert_pretrained_dir
    )

    # ----- Dataset -----
    print("📚 Building dataset...")
    datasets, num_labels = build_dataset(config, tok)

    # ----- Fisher importance (BEFORE PEFT wrapping) -----
    target_modules = list(getattr(config, 'lora_target_modules', ['query', 'key', 'value']))
    print(f"🎯 LoRA target modules: {target_modules}")
    print("🧮 Computing Fisher Information...")
    fisher_bs = int(getattr(config, 'fisher_batch_size', config.unlearn_batch_size))
    fisher_n = int(getattr(config, 'fisher_max_samples', 256))
    f_imp = compute_fisher_importance(
        model_og, DataLoader(datasets['forget'], batch_size=fisher_bs),
        device, target_modules, config, max_samples=fisher_n,
    )
    r_imp = compute_fisher_importance(
        model_og, DataLoader(datasets['retain'], batch_size=fisher_bs),
        device, target_modules, config, max_samples=fisher_n,
    )

    # Free fp32 model_og memory; reload as fp16 for training
    del model_og; torch.cuda.empty_cache()
    model_og = ImageTextModel.from_pretrained(base_p).to(device).half()
    for p in model_og.parameters(): p.requires_grad = False

    # ----- LoRA -----
    peft_cfg = LoraConfig(
        r=int(config.lora_r),
        lora_alpha=int(config.lora_alpha),
        target_modules=target_modules,
        lora_dropout=float(config.lora_dropout),
        bias='none',
    )
    model_unlearn = get_peft_model(model_unlearn, peft_cfg)
    apply_loku_soft_init(
        model_unlearn, f_imp, r_imp, target_modules,
        r=int(config.lora_r),
        init_scale=float(getattr(config, 'loku_init_scale', 0.05)),
    )

    # ----- Unfreeze classifier heads -----
    # Important: LoRA only touches BERT attention, but eval uses outputs[1] = img_logits
    # which comes from img_model.fc1. Without unfreezing, the image classifier never
    # changes during training and eval is invariant to all hyperparameters.
    if getattr(config, 'unfreeze_classifier_heads', True):
        try:
            n_before = sum(p.numel() for p in model_unlearn.parameters() if p.requires_grad)
            for p in model_unlearn.img_model.fc1.parameters():
                p.requires_grad = True
            for p in model_unlearn.text_model.classifier.parameters():
                p.requires_grad = True
            n_after = sum(p.numel() for p in model_unlearn.parameters() if p.requires_grad)
            print(f"🔓 Unfroze classifier heads: +{n_after - n_before:,} trainable params")
        except AttributeError as e:
            print(f"⚠️  Could not unfreeze classifier heads: {e}")

    trainable, total = count_params(model_unlearn)
    print(f"📊 Trainable: {trainable:,} / {total:,} ({100*trainable/total:.3f}%)")

    # ----- Fusion gates -----
    gates = {
        'ul_ret': Gate(768, 768).to(device),
        'ul_frg': Gate(768, 768).to(device),
        'og_rnd': Gate(768, 768).to(device),
        'og_ret': Gate(768, 768).to(device),
    }

    # ----- Optimizer -----
    params = [p for p in model_unlearn.parameters() if p.requires_grad]
    for g in gates.values():
        params += list(g.parameters())
    optimizer = AdamW(params, lr=float(config.learning_rate),
                      weight_decay=float(config.weight_decay))

    # ----- Weights -----
    alpha = float(config.get('alpha', 1.0))
    beta = float(config.get('beta', 1.0))
    theta = float(config.get('theta', 2.0))
    gamma = float(config.get('gamma', 2.0))
    eta = float(getattr(config, 'eta_re_anchor', 0.5))

    # ----- Train -----
    print(f"🚀 Unlearning ({config.unlearn_epochs} epochs max)...")
    elapsed_h = unlearn(config, out_dir, device,
                       model_og, model_unlearn, model_re,
                       gates, optimizer, datasets,
                       weights=(alpha, beta, theta, gamma, eta),
                       tracker=tracker)

    # ============================================================
    # FINAL EVALUATION
    # ============================================================
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION")
    print("=" * 60)

    # Free training-only memory before eval (model_og + optimizer + gates not needed)
    del model_og, optimizer, gates
    torch.cuda.empty_cache()

    # Merge LoRA back into base for clean evaluation
    print("🔀 Merging LoRA adapters into base model...")
    merged = model_unlearn.merge_and_unload()
    merged.eval()
    for p in merged.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()

    # Wrap MIA + perf eval in try/except so a single failure doesn't lose everything
    try:
        print("🛡️  Computing MIA score...")
        mia = run_mia(merged, datasets['retain'], datasets['test'], datasets['forget'],
                      device, config, batch_size=config.eval_batch_size,
                      seed=int(config.random_seed))
    except Exception as e:
        print(f"⚠️  MIA failed: {e}"); mia = float('nan')

    try:
        print("📐 Computing CosSim vs retrained model...")
        cossim_re = cosine_sim_models(merged, model_re, datasets['retain'], device,
                                      config, batch_size=config.eval_batch_size)
    except Exception as e:
        print(f"⚠️  CosSim failed: {e}"); cossim_re = float('nan')

    try:
        print("📊 Performance on test set...")
        test_m = perf_metrics(merged, datasets['test'], device, config,
                              batch_size=config.eval_batch_size)
    except Exception as e:
        print(f"⚠️  Test perf failed: {e}")
        test_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    try:
        print("📊 Performance on forget set...")
        forget_m = perf_metrics(merged, datasets['forget'], device, config,
                                batch_size=config.eval_batch_size)
    except Exception as e:
        print(f"⚠️  Forget perf failed: {e}")
        forget_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    gpu_peak = (torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0)

    results = {
        # ---- main metrics (compare to Forget-MI paper Table 1) ----
        'final/MIA':         round(mia, 3),
        'final/Df_AUC':      round(forget_m['AUC'], 3),
        'final/Df_F1':       round(forget_m['Macro_F1'], 3),
        'final/Dt_AUC':      round(test_m['AUC'], 3),
        'final/Dt_F1':       round(test_m['Macro_F1'], 3),
        'final/dist_vs_re':  round(1 - cossim_re, 3),
        'final/cossim_vs_re': round(cossim_re, 4),
        # ---- efficiency ----
        'efficiency/unlearn_time_hours': round(elapsed_h, 3),
        'efficiency/gpu_peak_GB':        round(gpu_peak, 2),
        'efficiency/trainable_params':   int(trainable),
        'efficiency/total_params':       int(total),
        'efficiency/trainable_ratio':    round(trainable / total, 5),
    }
    wandb.log(results)

    print("\n" + "─" * 60)
    print(" METRIC                    | VALUE   | PAPER TARGET ")
    print("─" * 60)
    print(f"  MIA           (↓)        | {mia:.3f}  | 0.571 (3%) / 0.615 (6%) / 0.810 (10%)")
    print(f"  Forget AUC    (↓)        | {forget_m['AUC']:.3f}  | 0.735 (3%) / 0.654 (6%) / 0.656 (10%)")
    print(f"  Forget Mac-F1 (↓)        | {forget_m['Macro_F1']:.3f}  | 0.393 (3%) / 0.328 (6%) / 0.313 (10%)")
    print(f"  Test AUC      (↑)        | {test_m['AUC']:.3f}  | 0.625 (3%) / 0.599 (6%) / 0.565 (10%)")
    print(f"  Test Mac-F1   (↑)        | {test_m['Macro_F1']:.3f}  | 0.250 (3%) / 0.270 (6%) / 0.252 (10%)")
    print(f"  1 - CosSim    (↓)        | {1-cossim_re:.3f}  |  ~0.4-0.5 (closer to retrained = better)")
    print("─" * 60)
    print(f"  Time (hours)             | {elapsed_h:.3f}  | Forget-MI paper: ~5h")
    print(f"  GPU peak (GB)            | {gpu_peak:.2f}   |")
    print(f"  Trainable params         | {trainable:,} ({100*trainable/total:.3f}%)")
    print("─" * 60)

    # Save CSV summary (easy to copy into thesis tables)
    csv_path = os.path.join(out_dir, "results_summary.csv")
    row = {**{k.split('/')[-1]: v for k, v in results.items()},
           'forget_pct': os.path.basename(config.forget_set_path),
           'seed': int(config.random_seed),
           'lora_r': int(config.lora_r),
           'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    new_file = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"\n💾 Saved row to {csv_path}")

    # ----- Auto-fill experiment MD file + INDEX (if --exp was passed) -----
    if tracker is not None:
        tracker_results = {
            'MIA':            float(mia) if mia == mia else float('nan'),
            'Df_AUC':         float(forget_m['AUC']),
            'Df_F1':          float(forget_m['Macro_F1']),
            'Dt_AUC':         float(test_m['AUC']),
            'Dt_F1':          float(test_m['Macro_F1']),
            'dist_vs_re':     float(1 - cossim_re) if cossim_re == cossim_re else float('nan'),
            'time_h':         float(elapsed_h),
            'gpu_gb':         float(gpu_peak),
            'trainable_pct':  100.0 * trainable / total,
        }
        try:
            tracker.finalize(tracker_results, elapsed_h)
        except Exception as e:
            print(f"⚠️  Tracker.finalize failed: {e}")

    wandb.finish()


if __name__ == '__main__':
    main()
