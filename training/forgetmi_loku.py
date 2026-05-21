#!/usr/bin/env python3
import os
import sys

# Đảm bảo Python tìm thấy thư mục gốc của dự án
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from datetime import datetime
import random
import pandas as pd
from tqdm import tqdm, trange
import logging
import numpy as np
import json
import sklearn
import time
import csv
from torch.utils.data import DataLoader, Dataset, Sampler, RandomSampler, SequentialSampler
import copy
import wandb
import re
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim import AdamW
from transformers import BertTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Forget-MI Shared Modules
import joint_img_txt.metrics as eval_metrics
from joint_img_txt import main_utils, parser
from training.joint_embedding import Gate, Outer, Attention
import joint_img_txt.loss as custom_loss
from joint_img_txt.model_utils import CXRImageTextDataset, EdemaClassificationProcessor, RandomTranslateCrop, CenterCrop, EdemaMultiLabelClassificationProcessor
from joint_img_txt.model import ImageTextModel
from joint_img_txt.convert_examples_to_features import convert_examples_to_features, convert_examples_to_features_multilabel

# --- UTILS ---

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def euclidean_distance(embed1, embed2):
    return torch.sqrt(torch.sum((embed1 - embed2)**2, dim=1))

class AlignedSampler(Sampler):
    def __init__(self, dataset_length, shuffle=False, seed=None):
        self.dataset_length = dataset_length
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        indices = list(range(self.dataset_length))
        if self.shuffle:
            if self.seed is not None:
                torch.manual_seed(self.seed) 
            indices = torch.randperm(len(indices)).tolist()
        return iter(indices)

    def __len__(self):
        return self.dataset_length

# --- DATA LOADING ---

def data_split(split_list_path, forget_ids_path, rand_ratio, validation_ratio=0.1):
    random.seed(0)
    train_labels, train_img_txt_ids = {}, {}
    test_labels, test_img_txt_ids = {}, {}
    rand_labels, rand_img_txt_ids = {}, {}
    forget_labels, forget_img_txt_ids = {}, {}

    with open(split_list_path, 'r') as train_label_file:
        forget_ids = pd.read_csv(forget_ids_path).astype(str).subject_id.values
        train_label_file_reader = csv.reader(train_label_file)
        header = next(train_label_file_reader)

        for row in train_label_file_reader:
            if row[3] == 'edeme_severity': continue
            try:
                severity = float(row[3])
            except ValueError: continue

            if row[0] in forget_ids:
                forget_labels[row[2]] = [severity]
                forget_img_txt_ids[row[2]] = row[1]
                rand_labels[row[2]] = [severity]
                rand_img_txt_ids[row[2]] = row[1]
            else:
                if row[-1] == 'TEST':
                    test_labels[row[2]] = [severity]
                    test_img_txt_ids[row[2]] = row[1]
                else:
                    train_labels[row[2]] = [severity]
                    train_img_txt_ids[row[2]] = row[1]

    from sklearn.model_selection import train_test_split
    train_ids = list(train_img_txt_ids.keys())
    train_values = list(train_img_txt_ids.values())
    train_labels_list = list(train_labels.values())

    tr_ids, val_ids, tr_vals, val_vals, tr_labs, val_labs = train_test_split(
        train_ids, train_values, train_labels_list, test_size=validation_ratio, random_state=42, stratify=train_labels_list)

    val_labels = dict(zip(val_ids, val_labs))
    val_img_txt_ids = dict(zip(val_ids, val_vals))
    
    # Retain set is training minus validation
    retain_labels = dict(zip(tr_ids, tr_labs))
    retain_img_txt_ids = dict(zip(tr_ids, tr_vals))

    return retain_labels, retain_img_txt_ids, val_labels, val_img_txt_ids, test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids, forget_labels, forget_img_txt_ids

def build_dataset(args, tokenizer, image_noise_params=None):
    processor = EdemaMultiLabelClassificationProcessor() if args.output_channel_encoding == 'multilabel' else EdemaClassificationProcessor()
    num_labels = len(processor.get_labels())
    get_features = convert_examples_to_features_multilabel if args.output_channel_encoding == 'multilabel' else convert_examples_to_features
    
    cached_features_file = os.path.join(args.text_data_dir, f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")
    cached_noisy_features_file = os.path.join(args.text_data_dir, f"cachednoisyfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}")

    if os.path.exists(cached_features_file) and not args.reprocess_input_data:
        features = torch.load(cached_features_file)
        noisy_features = torch.load(cached_noisy_features_file)
    else:
        synonyms_df = pd.read_csv(args.synonyms_dir)
        examples = processor.get_all_examples(args.text_data_dir)
        noisy_examples = processor.get_noisy_examples(args.text_data_dir, synonyms_df, args.text_noise_level)
        features = get_features(examples, processor.get_labels(), args.max_seq_length, tokenizer)
        noisy_features = get_features(noisy_examples, processor.get_labels(), args.max_seq_length, tokenizer)
        torch.save(features, cached_features_file)
        torch.save(noisy_features, cached_noisy_features_file)

    all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in features}
    noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in noisy_features}

    res = data_split(args.data_split_path, args.forget_set_path, args.random_point_ratio, args.validation_ratio)
    ret_l, ret_ids, val_l, val_ids, tst_l, tst_ids, rnd_l, rnd_ids, frg_l, frg_ids = res

    def extract(mapping, ids):
        tokens, masks, segs, labels = {}, {}, {}, {}
        for rid in ids:
            tokens[rid], masks[rid], segs[rid], labels[rid] = mapping[rid]
        return tokens, masks, segs, labels

    train_trans = RandomTranslateCrop(2048)
    eval_trans = CenterCrop(2048)

    datasets = {}
    for name, ids, labels, trans, txt_map in [
        ('retain', ret_ids, ret_l, train_trans, all_txt),
        ('validation', val_ids, val_l, eval_trans, all_txt),
        ('test', tst_ids, tst_l, eval_trans, all_txt),
        ('forget', frg_ids, frg_l, train_trans, all_txt),
        ('random', rnd_ids, rnd_l, train_trans, noisy_txt)
    ]:
        tk, mk, sg, lb = extract(txt_map, ids.values())
        datasets[name] = CXRImageTextDataset(args.id, tk, mk, sg, lb, ids, labels, args.img_data_dir, labels, transform=trans, output_channel_encoding=args.output_channel_encoding)
    
    return datasets, num_labels

def get_model_inputs(args, batch, device):
    image, label_raw, txt_ids, txt_mask, txt_segment_ids, label_onehot, report_id = batch
    inputs = {
        'input_img': image.to(device),
        'input_ids': txt_ids.to(device),
        'attention_mask': txt_mask.to(device),
        'token_type_ids': txt_segment_ids.to(device),
        'bert_pool_last_hidden': args.bert_pool_last_hidden,
        'bert_pool_use_img': args.bert_pool_use_img,
        'bert_pool_img_lowerlevel': args.bert_pool_img_lowerlevel
    }
    return inputs, label_raw.to(device), report_id.to(device)

# --- LoKU CORE: Importance & SVD ---

def compute_importance(model, dataloader, device, target_modules, max_samples=512):
    model.eval()
    importance = {name: torch.zeros_like(param.data) for name, param in model.named_parameters() if any(tm in name for tm in target_modules) and 'weight' in name}
    
    samples = 0
    for batch in tqdm(dataloader, desc="Computing Importance"):
        if samples >= max_samples: break
        inputs, labels, _ = get_model_inputs(None, batch, device) # Using None for args as only basic fields needed
        # Patching bert pool flags for importance pass
        inputs['bert_pool_last_hidden'] = False
        inputs['bert_pool_use_img'] = False
        inputs['bert_pool_img_lowerlevel'] = False

        model.zero_grad()
        outputs = model(**inputs)
        # Simple loss for gradient accumulation: sum of logits
        loss = outputs[1].sum() + outputs[3].sum() # img_logits + txt_logits
        loss.backward()

        for name, param in model.named_parameters():
            if name in importance and param.grad is not None:
                importance[name] += param.grad.data.pow(2)
        
        samples += batch[0].size(0)

    return {k: v / samples for k, v in importance.items()}

def apply_loku_svd(model, forget_importance, retain_importance, target_modules, r, lora_alpha):
    from functools import reduce
    def get_module(model, path):
        return reduce(getattr, path.split('.'), model)

    for name, f_imp in forget_importance.items():
        r_imp = retain_importance.get(name, torch.zeros_like(f_imp))
        # Relative importance
        imp = f_imp / (r_imp + 1e-8)
        
        module_path = name.replace('.weight', '')
        peft_module = get_module(model, module_path)
        
        # Access LoRA parts via peft structure
        # In peft, the original linear is base_layer
        base_layer = peft_module.base_layer
        lora_A = peft_module.lora_A['default']
        lora_B = peft_module.lora_B['default']
        scaling = peft_module.scaling['default']

        W = base_layer.weight.data.float()
        # Row-wise importance scaling for SVD
        row_imp = imp.mean(dim=1).sqrt()
        
        U, S, V = torch.svd_lowrank(row_imp[:, None] * W, q=r)
        
        # Initialize LoRA
        S_sqrt = torch.sqrt(S)
        new_A = (V * S_sqrt).t()
        new_B = (U * S_sqrt) / (row_imp[:, None] + 1e-8)
        
        # Knowledge Subtraction: W = W - (B @ A) * scaling
        subtraction = (new_B @ new_A) * scaling
        base_layer.weight.data -= subtraction.to(base_layer.weight.dtype)
        
        # Set LoRA weights
        lora_A.weight.data = new_A.to(lora_A.weight.dtype)
        lora_B.weight.data = new_B.to(lora_B.weight.dtype)

# --- TRAINING & CHECKPOINTING ---

def save_checkpoint(output_dir, epoch, model_ul, gates, optimizer, metrics):
    cp_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(cp_dir, exist_ok=True)
    
    # Save only LoRA and Gate params to keep it lean
    model_state = {k: v for k, v in model_ul.state_dict().items() if 'lora' in k}
    gate_states = {name: gate.state_dict() for name, gate in gates.items()}
    
    checkpoint = {
        'epoch': epoch,
        'model_state': model_state,
        'gate_states': gate_states,
        'optimizer_state': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, os.path.join(cp_dir, f"checkpoint_epoch_{epoch}.pt"))
    torch.save(checkpoint, os.path.join(cp_dir, "latest_checkpoint.pt"))

def load_latest_checkpoint(output_dir, model_ul, gates, optimizer):
    path = os.path.join(output_dir, "checkpoints", "latest_checkpoint.pt")
    if not os.path.exists(path): return -1
    
    print(f"Resuming from checkpoint: {path}")
    cp = torch.load(path)
    model_ul.load_state_dict(cp['model_state'], strict=False)
    for name, gate in gates.items():
        gate.load_state_dict(cp['gate_states'][name])
    optimizer.load_state_dict(cp['optimizer_state'])
    return cp['epoch']

def unlearn(args, output_dir, device, model_og, model_ul, model_re, gates, optimizer, scheduler, dataset, alpha, beta, theta, gamma):
    start_epoch = load_latest_checkpoint(output_dir, model_ul, gates, optimizer) + 1
    
    forget_dl = DataLoader(dataset['forget'], sampler=AlignedSampler(len(dataset['forget']), shuffle=True, seed=42), batch_size=args.unlearn_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)
    rand_dl = DataLoader(dataset['random'], sampler=AlignedSampler(len(dataset['random']), shuffle=True, seed=42), batch_size=args.unlearn_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)
    val_dl = DataLoader(dataset['validation'], sampler=SequentialSampler(dataset['validation']), batch_size=args.eval_batch_size)

    from evaluation.eval_unlearning import get_probability_measure
    
    best_loss = float('inf')
    patience_counter = 0
    patience = 5

    for epoch in range(start_epoch, args.unlearn_epochs):
        model_ul.train()
        for g in gates.values(): g.train()
        
        retain_dl = DataLoader(dataset['retain'], sampler=RandomSampler(dataset['retain']), batch_size=args.unlearn_batch_size, num_workers=args.num_cpu_workers)
        epoch_iterator = tqdm(zip(forget_dl, rand_dl, retain_dl), desc=f"Epoch {epoch}")

        epoch_metrics = {"MD": 0, "UU": 0, "MKR": 0, "UKR": 0, "Total": 0}
        steps = 0

        for f_batch, r_batch, ret_batch in epoch_iterator:
            # 1. Inputs
            ret_in, _, _ = get_model_inputs(args, ret_batch, device)
            f_in, _, _ = get_model_inputs(args, f_batch, device)
            rnd_in, _, _ = get_model_inputs(args, r_batch, device)

            # 2. Model Outputs
            og_ret_img_emb, _, og_ret_txt_emb, _ = model_og(**ret_in)[:4]
            ul_ret_img_emb, _, ul_ret_txt_emb, _ = model_ul(**ret_in)[:4]
            
            og_rnd_img_emb, _, og_rnd_txt_emb, _ = model_og(**rnd_in)[:4]
            
            ul_frg_img_emb, _, ul_frg_txt_emb, _ = model_ul(**f_in)[:4]

            # 3. Fusion
            ul_ret_joint = gates['ul_ret'](ul_ret_img_emb, ul_ret_txt_emb)
            ul_frg_joint = gates['ul_frg'](ul_frg_img_emb, ul_frg_txt_emb)
            og_rnd_joint = gates['og_rnd'](og_rnd_img_emb, og_rnd_txt_emb)
            og_ret_joint = gates['og_ret'](og_ret_img_emb, og_ret_txt_emb)

            # 4. Losses (Using Forget-MI Baseline Logic)
            ul_frg_concat = torch.cat((ul_frg_img_emb, ul_frg_txt_emb), dim=-1)
            og_rnd_concat = torch.cat((og_rnd_img_emb, og_rnd_txt_emb), dim=-1)
            L_uu = euclidean_distance(ul_frg_concat, og_rnd_concat).mean()
            if not args.use_noise: L_uu = -L_uu

            L_md = euclidean_distance(ul_frg_joint, og_rnd_joint).mean()
            if not args.use_noise: L_md = -L_md

            ul_ret_concat = torch.cat((ul_ret_img_emb, ul_ret_txt_emb), dim=-1)
            og_ret_concat = torch.cat((og_ret_img_emb, og_ret_txt_emb), dim=-1)
            L_ukr = euclidean_distance(ul_ret_concat, og_ret_concat).mean()
            L_mkr = euclidean_distance(ul_ret_joint, og_ret_joint).mean()

            # Hinge
            if epoch == 0 and steps == 0:
                args.margin_ukr = (L_ukr + 1).detach()
                args.margin_mkr = (L_mkr + 1).detach()
            
            if epoch > 0:
                L_ukr = torch.minimum(L_ukr, args.margin_ukr)
                L_mkr = torch.minimum(L_mkr, args.margin_mkr)

            loss = (alpha * L_ukr + beta * L_uu) + (theta * L_md + gamma * L_mkr)

            # 5. Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_metrics["MD"] += L_md.item()
            epoch_metrics["UU"] += L_uu.item()
            epoch_metrics["MKR"] += L_mkr.item()
            epoch_metrics["UKR"] += L_ukr.item()
            epoch_metrics["Total"] += loss.item()
            steps += 1

        avg_loss = epoch_metrics["Total"] / steps
        cosine_sim = get_probability_measure(args, copy.deepcopy(model_re).eval(), copy.deepcopy(model_ul).eval(), retain_dl, device)
        
        wandb.log({f"epoch": epoch, "loss": avg_loss, "cosine_sim": cosine_sim, **{k: v/steps for k, v in epoch_metrics.items()}})
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | CosSim: {cosine_sim:.4f}")

        save_checkpoint(output_dir, epoch, model_ul, gates, optimizer, epoch_metrics)

        # Early Stopping
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

def ensure_model_path(path, description="model"):
    if os.path.exists(os.path.join(path, "pytorch_model.bin")):
        return path
    print(f"⚠️ {description} not found at {path}. Searching for pytorch_model.bin...")
    import glob
    # Search in common Colab locations
    matches = glob.glob(f"/content/**/pytorch_model.bin", recursive=True)
    if matches:
        # Avoid picking up checkpoints or temp files
        valid_matches = [m for m in matches if "checkpoint" not in m and "extract_temp" not in m]
        if valid_matches:
            new_path = os.path.dirname(valid_matches[0])
            print(f"✅ Found {description} at: {new_path}")
            return new_path
    return path

def main():
    import yaml
    import argparse
    
    parser_arg = argparse.ArgumentParser()
    parser_arg.add_argument("--config", type=str, default="config.yaml")
    args_cli = parser_arg.parse_args()

    with open(args_cli.config, "r") as f:
        raw_config = yaml.safe_load(f)
    
    # Robust config parsing for both flat and sweep-style YAML
    if "parameters" in raw_config:
        my_config = {k: (v['value'] if isinstance(v, dict) and 'value' in v else (v['values'][0] if isinstance(v, dict) and 'values' in v else v)) for k, v in raw_config["parameters"].items()}
    else:
        my_config = raw_config
    
    wandb.init(project=my_config.get("project", "forget_exp"), entity=my_config.get("entity", "unlearning"), config=my_config)
    config = wandb.config
    set_seed(config.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load Original Model (With Auto-Discovery)
    base_path = ensure_model_path(config.base_model_path, "Base Model")
    model_og = ImageTextModel.from_pretrained(base_path).to(device)
    model_unlearn = copy.deepcopy(model_og)
    
    re_path = ensure_model_path(config.retrained_model_path, "Retrained Model")
    model_re = ImageTextModel.from_pretrained(re_path).to(device)
    
    # Tokenizer usually in the same dir as base model
    tokenizer = BertTokenizer.from_pretrained(base_path if os.path.exists(os.path.join(base_path, "vocab.txt")) else config.bert_pretrained_dir)

    # 2. Build Dataset
    dataset, num_labels = build_dataset(config, tokenizer)

    # 3. LoKU PRE-TRAINING PHASE
    target_modules = ["query", "value", "fc1"]
    
    # Calculate Importance
    f_imp = compute_importance(model_og, DataLoader(dataset['forget'], batch_size=config.unlearn_batch_size), device, target_modules)
    r_imp = compute_importance(model_og, DataLoader(dataset['retain'], batch_size=config.unlearn_batch_size), device, target_modules, max_samples=len(dataset['forget'])*2)

    # Apply PEFT
    peft_config = LoraConfig(
        r=config.lora_r, 
        lora_alpha=config.lora_alpha, 
        target_modules=target_modules, 
        lora_dropout=config.lora_dropout, 
        bias="none",
        task_type=TaskType.SEQ_CLS # Using a generic task type
    )
    model_unlearn = get_peft_model(model_unlearn, peft_config)

    # Apply SVD & Knowledge Subtraction
    apply_loku_svd(model_unlearn, f_imp, r_imp, target_modules, config.lora_r, config.lora_alpha)

    # 4. Fusion Gates (Persistent)
    gates = {
        'ul_ret': Gate(768, 768).to(device),
        'ul_frg': Gate(768, 768).to(device),
        'og_rnd': Gate(768, 768).to(device),
        'og_ret': Gate(768, 768).to(device)
    }

    # 5. Optimizer
    params = list(model_unlearn.parameters())
    for g in gates.values(): params += list(g.parameters())
    optimizer = AdamW(params, lr=float(config.learning_rate), weight_decay=config.weight_decay)
    
    # 6. Unlearn
    alpha = config.get("alpha", 1.0)
    beta = config.get("beta", 1.0)
    theta = config.get("theta", 2.0)
    gamma = config.get("gamma", 2.0)
    
    unlearn(config, config.output_dir, device, model_og, model_unlearn, model_re, gates, optimizer, None, dataset, alpha, beta, theta, gamma)

if __name__ == '__main__':
    main()
