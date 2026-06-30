#!/usr/bin/env python3
from datetime import datetime
import os
import random
import pandas as pd
from tqdm import tqdm, trange
from scipy.stats import logistic
from scipy.special import softmax
import logging
import numpy as np
import json
import sklearn
import time
import csv
from torch.utils.data import DataLoader, Dataset, Sampler
import copy
import wandb
import time
import re
import csv
from sklearn.model_selection import train_test_split
import random

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import torch.optim.lr_scheduler as lr_scheduler

import joint_img_txt.metrics as eval_metrics
from joint_img_txt import main_utils, parser
from training.joint_embedding import Gate, Outer, Attention


from transformers import BertTokenizer
from torch.optim import AdamW

import joint_img_txt.loss as custom_loss
from joint_img_txt.model_utils import CXRImageTextDataset, EdemaClassificationProcessor, RandomTranslateCrop, CenterCrop, EdemaMultiLabelClassificationProcessor
from joint_img_txt.model import ImageTextModel
from joint_img_txt.convert_examples_to_features import convert_examples_to_features, convert_examples_to_features_multilabel
from sklearn.model_selection import train_test_split

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def flatten_metrics(metrics, prefix=""):
    """
    Flattens nested metric dictionaries, appending index for list values.
    """
    flat_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, dict):
            # Recursively flatten dictionaries
            flat_metrics.update(flatten_metrics(value, prefix=f"{prefix}{key}."))
        elif isinstance(value, list):
            # Log each list element separately
            for i, v in enumerate(value):
                flat_metrics[f"{prefix}{key}_{i}"] = v
        else:
            flat_metrics[f"{prefix}{key}"] = value
    return flat_metrics

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
            # BUG FIX 2026-06-16: `torch.randperm(...)` không assign → shuffle vô hiệu.
            # Phải assign về `indices` để vòng iter thực sự shuffle.
            indices = torch.randperm(len(indices)).tolist()
        return iter(indices)

    def __len__(self):
        return self.dataset_length

def euclidean_distance(embed1, embed2):
        return torch.sqrt(torch.sum((embed1 - embed2)**2, dim=1))

def cosine_similarity_loss(embed1, embed2):
    norm1 = torch.norm(embed1, dim=1)
    norm2 = torch.norm(embed2, dim=1)
    return torch.sum(embed1 * embed2, dim=1) / (norm1 * norm2)

def build_dataset(args, tokenizer, image_noise_params=None):
    logger = logging.getLogger(__name__)

    '''
    Load text features if they have been pre-processed;
    otherwise pre-process the raw text and save the features
    '''
    processor = EdemaMultiLabelClassificationProcessor() \
        if args.output_channel_encoding == 'multilabel' \
        else EdemaClassificationProcessor()
    num_labels = len(processor.get_labels())

    if args.output_channel_encoding == 'multilabel':
        get_features = convert_examples_to_features_multilabel
    else:
        get_features = convert_examples_to_features
    cache_fname = f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}"
    cache_noisy_fname = f"cachednoisyfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}"
    cached_features_file = os.path.join(args.text_data_dir, cache_fname)
    cached_noisy_features_file = os.path.join(args.text_data_dir, cache_noisy_fname)

    def _regen_to(dst_dir):
        """Build features fresh from all_data.tsv and write to dst_dir.
        Raises FileNotFoundError if source tsv missing."""
        src_tsv = os.path.join(args.text_data_dir, 'all_data.tsv')
        if not os.path.exists(src_tsv):
            raise FileNotFoundError(
                f"Cannot regenerate cache: {src_tsv} missing. Upload it to the Kaggle "
                f"dataset (folder 'metadata/') or run with a populated text_data_dir."
            )
        os.makedirs(dst_dir, exist_ok=True)
        label_list = processor.get_labels()
        synonyms_df = pd.read_csv(args.synonyms_dir)
        examples = processor.get_all_examples(args.text_data_dir)
        noisy_examples = processor.get_noisy_examples(args.text_data_dir, synonyms_df, args.text_noise_level)
        feats = get_features(examples, label_list, args.max_seq_length, tokenizer)
        feats_n = get_features(noisy_examples, label_list, args.max_seq_length, tokenizer)
        torch.save(feats, os.path.join(dst_dir, cache_fname))
        torch.save(feats_n, os.path.join(dst_dir, cache_noisy_fname))
        return feats, feats_n

    def _probe_match(feats):
        """Sample 50 study_ids from split CSV, check how many resolve in cache keys.
        Returns (n_match, n_total, sample_cache_keys, sample_csv_ids)."""
        cache_keys = set()
        for f in feats[:2000]:
            cache_keys.add(f.report_id)
        csv_ids = []
        with open(args.data_split_path, 'r') as fp:
            rdr = csv.reader(fp); next(rdr, None)
            for row in rdr:
                if len(row) > 1:
                    csv_ids.append(row[1])
                if len(csv_ids) >= 50:
                    break
        n = sum(1 for r in csv_ids
                if r in cache_keys or f"s{r}" in cache_keys or str(r).replace('s', '') in cache_keys)
        return n, len(csv_ids), list(cache_keys)[:5], csv_ids[:5]

    if os.path.exists(cached_features_file) and os.path.exists(cached_noisy_features_file) and not args.reprocess_input_data:
        logger.info("Loading features from cached file %s", cached_features_file)
        # weights_only=False: PyTorch 2.6+ defaults to True, which breaks loading
        # of pickled InputFeatures objects (custom class). Same fix as forgetmi_loku.py.
        features = torch.load(cached_features_file, weights_only=False)
        noisy_features = torch.load(cached_noisy_features_file, weights_only=False)

        # Cache integrity probe. If keys don't match split CSV at all, the
        # downstream filter will wipe everything → useless training run.
        # Detect early + try regenerating to a writable dir (Kaggle input is read-only).
        n_match, n_total, sample_cache, sample_csv = _probe_match(features)
        if n_match == 0 and n_total > 0:
            print(f"⚠️  Cache 0/{n_total} match split CSV — STALE.")
            print(f"   sample cache keys : {sample_cache}")
            print(f"   sample split ids  : {sample_csv}")
            writable = '/kaggle/working/text_cache_regen' if os.path.isdir('/kaggle/working') else os.path.join(os.getcwd(), 'text_cache_regen')
            re_cached = os.path.join(writable, cache_fname)
            re_cached_noisy = os.path.join(writable, cache_noisy_fname)
            if os.path.exists(re_cached) and os.path.exists(re_cached_noisy):
                print(f"   ↳ found prior regen cache → loading {writable}")
                features = torch.load(re_cached, weights_only=False)
                noisy_features = torch.load(re_cached_noisy, weights_only=False)
            else:
                print(f"   ↳ regenerating to {writable} (1-time, ~5 min)...")
                features, noisy_features = _regen_to(writable)
                print(f"   ✅ Regenerated. Subsequent runs in this session will reuse {writable}.")
            n_match2, _, sc2, ci2 = _probe_match(features)
            if n_match2 == 0:
                raise RuntimeError(
                    f"After regenerate STILL 0 match. Cache: {sc2} vs split: {ci2}. "
                    f"Likely all_data.tsv is from a different MIMIC subset than {args.data_split_path}."
                )
            print(f"   ✅ Post-regen probe: {n_match2}/{n_total} match.")
    else:
        logger.info("Creating features from dataset file at %s", args.text_data_dir)
        # Try writing back to text_data_dir; fall back to /kaggle/working if read-only.
        try:
            features, noisy_features = _regen_to(args.text_data_dir)
            print(f"Saving features into cached file {cached_features_file}")
        except (PermissionError, OSError, RuntimeError):
            # torch.save trên read-only FS ném RuntimeError (PyTorchFileWriter C++),
            # KHÔNG phải OSError → phải bắt cả RuntimeError. (Kaggle /kaggle/input read-only.)
            writable = '/kaggle/working/text_cache_regen' if os.path.isdir('/kaggle/working') else os.path.join(os.getcwd(), 'text_cache_regen')
            print(f"⚠️  text_data_dir read-only → regenerating to {writable}")
            features, noisy_features = _regen_to(writable)

    all_txt_tokens = {f.report_id: f.input_ids for f in features}
    all_txt_masks = {f.report_id: f.input_mask for f in features}
    all_txt_segments = {f.report_id: f.segment_ids for f in features}
    all_txt_labels = {f.report_id: f.label_id for f in features}

    noisy_txt_tokens = {f.report_id: f.input_ids for f in noisy_features}
    noisy_txt_masks = {f.report_id: f.input_mask for f in noisy_features}
    noisy_txt_segments = {f.report_id: f.segment_ids for f in noisy_features}
    noisy_txt_labels = {f.report_id: f.label_id for f in noisy_features}

    retain_img_labels, retain_img_txt_ids, val_img_labels, val_img_txt_ids, test_img_labels, test_img_txt_ids, rand_img_labels, rand_img_txt_ids, \
        forget_img_labels, forget_img_txt_ids, n_retain, n_val, n_test, n_rand, n_forget = data_split(args.data_split_path, args.forget_set_path, args.random_point_ratio, args.validation_ratio)

    # Drop items whose study_id has no cached text features (else KeyError mid-DataLoader).
    # Same root cause + fix as forgetmi_loku.py build_dataset.
    def _filter_valid(ids_dict, labels_dict, txt_map, name):
        valid = []
        for d, r in ids_dict.items():
            if r in txt_map or f"s{r}" in txt_map or str(r).replace('s', '') in txt_map:
                valid.append(d)
        skipped = len(ids_dict) - len(valid)
        if skipped > 0:
            print(f"WARNING: {skipped}/{len(ids_dict)} items in '{name}' set skipped (text features missing in cache)")
        return {d: ids_dict[d] for d in valid}, {d: labels_dict[d] for d in valid}

    retain_img_txt_ids, retain_img_labels = _filter_valid(retain_img_txt_ids, retain_img_labels, all_txt_tokens,   'retain')
    val_img_txt_ids,    val_img_labels    = _filter_valid(val_img_txt_ids,    val_img_labels,    all_txt_tokens,   'validation')
    test_img_txt_ids,   test_img_labels   = _filter_valid(test_img_txt_ids,   test_img_labels,   all_txt_tokens,   'test')
    forget_img_txt_ids, forget_img_labels = _filter_valid(forget_img_txt_ids, forget_img_labels, all_txt_tokens,   'forget')
    rand_img_txt_ids,   rand_img_labels   = _filter_valid(rand_img_txt_ids,   rand_img_labels,   noisy_txt_tokens, 'random')

    # Loud error if filter wiped a set that SHOULD have had items (cache totally stale):
    # silent no-op would otherwise produce a trained-on-nothing model that still passes
    # all eval gates. Use the pre-filter counts so a legitimately-empty set is allowed:
    # og training uses an empty forget_set (forget=0), and random can be 0 by config.
    for name, dct, n_pre in [('retain', retain_img_txt_ids, n_retain),
                             ('validation', val_img_txt_ids, n_val),
                             ('test', test_img_txt_ids, n_test),
                             ('forget', forget_img_txt_ids, n_forget),
                             ('random', rand_img_txt_ids, n_rand)]:
        if n_pre > 0 and not dct:
            raise RuntimeError(
                f"After filter '{name}' set is EMPTY (had {n_pre} before filter) — cached "
                f"text features in {args.text_data_dir} don't match study_ids in "
                f"{args.data_split_path}. Set reprocess_input_data=true or delete the cache."
            )

    '''
    Specify the image pre-processing method
    depending on it's for training/evaluation
    '''
    if args.do_train:
        xray_transform = RandomTranslateCrop(2048)
    if args.do_eval:
        xray_transform = CenterCrop(2048)

    '''
    Instantiate the image-text dataset
    '''
    retain_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, retain_img_txt_ids, args.img_data_dir, 
                                  retain_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform,
                                  output_channel_encoding = args.output_channel_encoding)

    test_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, test_img_txt_ids, args.img_data_dir, 
                                  test_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    rand_dataset = CXRImageTextDataset(args.id, 
                                  noisy_txt_tokens, noisy_txt_masks, noisy_txt_segments, 
                                  noisy_txt_labels, rand_img_txt_ids, args.img_data_dir, 
                                  rand_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, perturb_img=True,
                                  noise_params=image_noise_params, output_channel_encoding = args.output_channel_encoding)

    forget_dataset = CXRImageTextDataset(args.id, 
                                  all_txt_tokens, all_txt_masks, all_txt_segments, 
                                  all_txt_labels, forget_img_txt_ids, args.img_data_dir, 
                                  forget_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                  output_channel_encoding = args.output_channel_encoding)

    val_dataset = CXRImageTextDataset(args.id, 
                                all_txt_tokens, all_txt_masks, all_txt_segments, 
                                all_txt_labels, val_img_txt_ids, args.img_data_dir, 
                                val_img_labels, dataset_split_path=args.data_split_path, transform=xray_transform, 
                                output_channel_encoding=args.output_channel_encoding)
                      
    print(f"Datasets: retain={len(retain_dataset)} val={len(val_dataset)} test={len(test_dataset)} forget={len(forget_dataset)} random={len(rand_dataset)}")

    dataset = {
        'retain': retain_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'random': rand_dataset,
        'forget': forget_dataset,
        'n_retain': n_retain,
        'n_val': n_val,  
        'n_test': n_test,
        'n_rand': n_rand,
        'n_forget': n_forget,
    }

    return dataset, num_labels
def data_split(split_list_path, forget_ids_path, rand_ratio, validation_ratio=0.1):

    random.seed(0)

    train_labels = {}
    train_img_txt_ids = {}
    test_labels = {}
    test_img_txt_ids = {}
    rand_labels = {}
    rand_img_txt_ids = {}
    forget_labels = {}
    forget_img_txt_ids = {}

    with open(split_list_path, 'r') as train_label_file:
        forget_ids = pd.read_csv(forget_ids_path)
        forget_ids = forget_ids.astype(str)
        forget_ids = forget_ids.subject_id.values

        train_label_file_reader = csv.reader(train_label_file)
        header = next(train_label_file_reader)

        for row in train_label_file_reader:
            if row == header or row[3] == 'edeme_severity':
                continue
            try:
                severity = float(row[3])
            except ValueError:
                continue
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

    # VALIDATION DATASET
    train_ids = list(train_img_txt_ids.keys())
    train_values = list(train_img_txt_ids.values())
    train_labels_list = list(train_labels.values())

    train_ids, val_ids, train_values, val_values, train_labels_split, val_labels_split = train_test_split(
        train_ids, train_values, train_labels_list, test_size=validation_ratio, random_state=42, stratify=train_labels_list)

    # Add validation labels and images back into training set
    train_labels.update(dict(zip(val_ids, val_labels_split)))
    train_img_txt_ids.update(dict(zip(val_ids, val_values)))

    val_labels = dict(zip(val_ids, val_labels_split))
    val_img_txt_ids = dict(zip(val_ids, val_values))

    train_labels.update(val_labels)
    train_img_txt_ids.update(val_img_txt_ids)

    n_train, n_val, n_rand, n_test, n_forget = len(train_img_txt_ids), len(val_img_txt_ids), len(rand_img_txt_ids), len(test_img_txt_ids), len(forget_img_txt_ids)

    print(f"Split: train={len(train_labels)} val={len(val_labels)} test={len(test_labels)} forget={len(forget_labels)} random={len(rand_labels)}")

    return train_labels, train_img_txt_ids, val_labels, val_img_txt_ids, test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids, forget_labels, forget_img_txt_ids, n_train, n_val, n_test, n_rand, n_forget

def get_model_inputs(args, dataset, device):

    image, label_raw, txt_ids, txt_mask, txt_segment_ids, label_onehot_or_ordinal, report_id = dataset

    image = image.to(device)
    label_raw = label_raw.to(device)
    txt_ids = txt_ids.to(device)
    txt_mask = txt_mask.to(device)
    txt_segment_ids = txt_segment_ids.to(device)
    label_onehot_or_ordinal = label_onehot_or_ordinal.to(device)
    report_id = report_id.to(device)

    inputs = {  
                'input_img':                image,
                'input_ids':                txt_ids,
                'attention_mask':           txt_mask,
                'token_type_ids':           txt_segment_ids,
                'labels':                   None,
                'bert_pool_last_hidden':    args.bert_pool_last_hidden,
                'bert_pool_use_img':        args.bert_pool_use_img,
                'bert_pool_img_lowerlevel': args.bert_pool_img_lowerlevel
            } 
    return inputs, label_raw, report_id

def unlearn(args, output_dir, device, model_og, model_ul, model_re, optimizer, optimizer_grouped_parameters, scheduler, tokenizer, dataset, num_labels, alpha, beta, theta, gamma):
    retain_set, val_set, rand_set, forget_set, test_set = (
        dataset['retain'], dataset['validation'], dataset['random'], dataset['forget'], dataset['test']
    )    
    n_retain, n_val, n_rand, n_forget, n_test = (
        dataset['n_retain'], dataset['n_val'], dataset['n_rand'], dataset['n_forget'], dataset['n_test']
    )

    aligned_sampler = AlignedSampler(len(forget_set), shuffle=True, seed=42)

    forget_dataloader = DataLoader(forget_set, sampler=aligned_sampler,
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers,
                                  pin_memory=True)
    val_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set),
                                batch_size=args.eval_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)
    rand_dataloader = DataLoader(rand_set, sampler=aligned_sampler,
                                  batch_size=args.unlearn_batch_size,
                                  num_workers=args.num_cpu_workers,
                                  pin_memory=True)
    test_dataloader = DataLoader(test_set, sampler=SequentialSampler(test_set),
                                  batch_size=args.eval_batch_size, num_workers=args.num_cpu_workers, pin_memory=True)
    print(f"Dataloaders: forget={len(forget_set)} val={len(val_set)} rand={len(rand_set)} test={len(test_set)} (batch={args.unlearn_batch_size})")

    print('Starting the Unlearning Process...')
    n_epochs = args.unlearn_epochs
    # disable=True: in non-TTY (Kaggle .log) tqdm spams a new line per iter; we print
    # one summary line per epoch below instead.
    unlearning_iterator = trange(int(n_epochs), desc="Epoch", disable=True)

    model_re.eval()
    model_ul.train()
    # FAITHFUL to paper code: model_og.train() (NOT .eval()) — keeps BN running stats
    # updating per-epoch as in the original Forget-MI repository. Weights still frozen
    # (requires_grad=False set earlier) — only BN buffers drift.
    model_og.train()

    unlearning_start_time = time.time()

    for epoch in unlearning_iterator:
        # ------------------------------------------- UNLEARNING ------------------------------------------- 
        total_loss = 0
        md_loss, uu_loss = 0, 0
        mkr_loss, ukr_loss = 0, 0

        retain_sampler = RandomSampler(retain_set)
        retain_dataloader = DataLoader(retain_set, sampler=retain_sampler, batch_size=args.unlearn_batch_size,
                                    num_workers=args.num_cpu_workers, pin_memory=True)
        # Same reason as outer trange: disable progress spam in non-TTY.
        epoch_iterator = tqdm(zip(forget_dataloader, rand_dataloader, retain_dataloader),
                              desc="Retain Set Iteration", disable=True)

        steps = 0
        for (forget_batch, rand_batch, retain_batch) in epoch_iterator:
            # ------------------------------------------- GET INPUTS FROM ORIGINAL AND UNLEARNING MODELS -------------------------------------------
            # model_og is FROZEN — wrap in no_grad so autograd doesn't build a graph
            # for activations (which it would otherwise hold until backward). This is
            # the dominant T4 OOM saver (~5-6 GB). Loss targets don't need a grad path
            # through the frozen model.
            retain_batch = tuple(t.to(device=device, non_blocking=True) for t in retain_batch)
            retain_inputs, retain_labels, retain_report_id = get_model_inputs(args, retain_batch, device)
            with torch.no_grad():
                original_retain_outputs = model_og(**retain_inputs)
            og_ret_img_emb, og_ret_img_log, og_ret_txt_emb, og_ret_txt_log = original_retain_outputs[:4]
            unlearn_retain_outputs = model_ul(**retain_inputs)
            ul_ret_img_emb, ul_ret_img_log, ul_ret_txt_emb, ul_ret_txt_log = unlearn_retain_outputs[:4]

            rand_batch = tuple(t.to(device=device, non_blocking=True) for t in rand_batch)
            rand_inputs, rand_labels, rand_report_id = get_model_inputs(args, rand_batch, device)
            with torch.no_grad():
                original_rand_outputs = model_og(**rand_inputs)
            unlearn_rand_outputs = model_ul(**rand_inputs)
            og_rand_img_emb, og_rand_img_log, og_rand_txt_emb, og_rand_txt_log = original_rand_outputs[:4]
            ul_rand_img_emb, ul_rand_img_log, ul_rand_txt_emb, ul_rand_txt_log = unlearn_rand_outputs[:4]

            forget_batch = tuple(t.to(device=device, non_blocking=True) for t in forget_batch)
            forget_inputs, forget_labels, forget_report_id = get_model_inputs(args, forget_batch, device)
            with torch.no_grad():
                original_forget_outputs = model_og(**forget_inputs)
            unlearn_forget_outputs = model_ul(**forget_inputs)
            ul_frgt_img_emb, ul_frgt_img_log, ul_frgt_txt_emb, ul_frgt_txt_log = unlearn_forget_outputs[:4]
            og_frgt_img_emb, og_frgt_img_log, og_frgt_txt_emb, og_frgt_txt_log = original_forget_outputs[:4]

            # ------------------------------------------- JOINT EMBEDDINGS -------------------------------------------
            gate_ul_ret =  Gate(inp1_size = ul_ret_img_emb.shape[1], inp2_size = ul_ret_txt_emb.shape[1]).to(device)
            ul_ret_joint_emb = gate_ul_ret(ul_ret_img_emb, ul_ret_txt_emb)

            gate_ul_frgt =  Gate(inp1_size = ul_frgt_img_emb.shape[1], inp2_size = ul_frgt_txt_emb.shape[1]).to(device)
            ul_frgt_joint_emb = gate_ul_frgt(ul_frgt_img_emb, ul_frgt_txt_emb)

            gate_og_frgt =  Gate(inp1_size = og_frgt_img_emb.shape[1], inp2_size = og_frgt_img_emb.shape[1]).to(device)
            og_frgt_joint_emb = gate_ul_frgt(og_frgt_img_emb, og_frgt_img_emb)

            gate_og_rand =  Gate(inp1_size = og_rand_img_emb.shape[1], inp2_size = og_rand_txt_emb.shape[1]).to(device)
            og_rand_joint_emb = gate_og_rand(og_rand_img_emb, og_rand_txt_emb)

            gate_og_ret =  Gate(inp1_size = og_ret_img_emb.shape[1], inp2_size = og_ret_txt_emb.shape[1]).to(device)
            og_ret_joint_emb = gate_og_ret(og_ret_img_emb, og_ret_txt_emb)

            # we use large noise therefore its minimization problem
            if args.use_noise:
                # ------------------------------------------- UU Loss -------------------------------------------
                ul_frgt_concat_emb = torch.cat((ul_frgt_img_emb, ul_frgt_txt_emb), dim=-1)
                og_rand_concat_emb = torch.cat((og_rand_img_emb, og_rand_txt_emb), dim=-1)

                L_uu = (euclidean_distance(ul_frgt_concat_emb, og_rand_concat_emb).mean())

                # ------------------------------------------- MD Loss -------------------------------------------
                L_md = euclidean_distance(ul_frgt_joint_emb, og_rand_joint_emb).mean()
            # don't use any noise, maximization problem
            else:
                # ------------------------------------------- UU Loss -------------------------------------------
                ul_frgt_concat_emb = torch.cat((ul_frgt_img_emb, ul_frgt_txt_emb), dim=-1)
                og_rand_concat_emb = torch.cat((og_rand_img_emb, og_rand_txt_emb), dim=-1)

                L_uu = (euclidean_distance(ul_frgt_concat_emb, og_rand_concat_emb).mean())
                L_uu = -L_uu
                # ------------------------------------------- MD Loss -------------------------------------------
                L_md = euclidean_distance(ul_frgt_joint_emb, og_rand_joint_emb).mean()
                L_md = -L_md

            # ------------------------------------------- UKR Loss -------------------------------------------
            ul_ret_concat_emb = torch.cat((ul_ret_img_emb, ul_ret_txt_emb), dim=-1)
            og_ret_concat_emb = torch.cat((og_ret_img_emb, og_ret_txt_emb), dim=-1)

            L_ukr = euclidean_distance(ul_ret_concat_emb, og_ret_concat_emb).mean()
    
            # ------------------------------------------- MKR Loss -------------------------------------------
            L_mkr = euclidean_distance(ul_ret_joint_emb, og_ret_joint_emb).mean()

            # ------------------------------------------- Hinge Loss -------------------------------------------
            if epoch == 0:
                margin_ukr = (L_ukr + 1).detach() 
                margin_mkr = (L_mkr + 1).detach()

            if epoch != 0:
                L_ukr = torch.minimum(L_ukr, margin_ukr)
                L_mkr = torch.minimum(L_mkr, margin_mkr)

            # ------------------------------------------- Total Loss -------------------------------------------
            loss = (alpha*L_ukr + beta*L_uu) + (theta*L_md + gamma*L_mkr)

            #------------------------------------------- Backpropagation -------------------------------------------
            md_loss += L_md.item()
            uu_loss += L_uu.item()
            mkr_loss += L_mkr.item()
            ukr_loss += L_ukr.item()
            total_loss += loss.item()
            
            if epoch != 0:
                loss.backward()

                if 'grad' in optimizer_grouped_parameters:
                    torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters, 
                                                args.max_grad_norm)

            steps += 1

        if epoch != 0:
            optimizer.step()
            optimizer.zero_grad()  

        # ----- Per-epoch CosSim eval (FAITHFUL to original Forget-MI implementation) -----
        # Runs FULL retain forward through model_ul AND model_re each epoch — this is the
        # primary contributor to the paper's reported ~5h training time per seed. We keep
        # it (despite being expensive) to reproduce the published baseline runtime profile.
        from evaluation.eval_unlearning import get_probability_measure
        try:
            cosine_similarity = get_probability_measure(
                args,
                copy.deepcopy(model_re).eval(),
                copy.deepcopy(model_ul).eval(),
                retain_dataloader,
                device
            )
        except Exception as e:
            print(f"⚠️  per-epoch CosSim eval failed at epoch {epoch}: {e}")
            cosine_similarity = float('nan')

        try:
            wandb.log({
                "Epoch": epoch,
                "MD Loss": md_loss / steps,
                "UU Loss": uu_loss / steps,
                "MKR Loss": mkr_loss / steps,
                "UKR Loss": ukr_loss / steps,
                "Total Loss": total_loss / steps,
                "Learning Rate": args.learning_rate,
                "Cosine Similarity": cosine_similarity,
            })
        except Exception:
            pass

        elapsed_min = (time.time() - unlearning_start_time) / 60
        eta_min = elapsed_min / (epoch + 1) * (n_epochs - epoch - 1)
        print(f"[E{epoch:02d}/{n_epochs}] loss={total_loss/steps:+.3f} "
              f"UKR={ukr_loss/steps:+.3f} UU={uu_loss/steps:+.3f} "
              f"MD={md_loss/steps:+.3f} MKR={mkr_loss/steps:+.3f} "
              f"CosSim={cosine_similarity:+.4f} | elapsed={elapsed_min:.1f}m ETA={eta_min:.1f}m")

        # ----- Per-epoch checkpoint save (FAITHFUL to original) -----
        # Original saves every epoch. ~450 MB × 30 = ~13.5 GB on /kaggle/working/.
        # Tight against 20 GB cap but fits; matches paper behavior.
        epoch_output_dir = os.path.join(output_dir, f"epoch_{epoch}")
        os.makedirs(epoch_output_dir, exist_ok=True)
        model_to_save = model_ul.module if hasattr(model_ul, 'module') else model_ul
        torch.save(model_to_save.state_dict(), os.path.join(epoch_output_dir, "model_state_dict.pth"))


    # ------------------------------------------- Log Unlearning Time Taken -------------------------------------------    
    unlearning_end_time = time.time()  
    unlearning_duration = (unlearning_end_time - unlearning_start_time) / 3600

    try:
        wandb.log({"time(hours)": unlearning_duration})
    except Exception:
        pass
    print(f"⏱  Unlearning training done in {unlearning_duration*60:.1f} min ({unlearning_duration:.2f}h)")

    return retain_dataloader, forget_dataloader, val_dataloader, test_dataloader
    
def _parse_cli():
    """CLI args added 2026-06-16 for Kaggle/multi-seed compatibility.

    Backward-compatible: nếu chạy `python forgetmi_partial.py` không args, sẽ tự đọc
    `config.yaml` ở CWD và behavior y nguyên bản gốc.
    """
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='config.yaml',
                   help='Path tới YAML config (default: config.yaml ở CWD)')
    p.add_argument('--seed', type=int, default=None,
                   help='Override random_seed (multi-seed runs)')
    p.add_argument('--override', default='',
                   help='Comma-separated k=v overrides, vd: forget_set_path=./data_splits/forget_set_6per.csv,id=test')
    p.add_argument('--exp', default=None,
                   help='Tên experiment cho auto-tracker MD file')
    p.add_argument('--hypothesis', default=None,
                   help='Hypothesis cho experiment tracker')
    p.add_argument('--fresh', action='store_true',
                   help='Xóa output_dir cũ trước khi train')
    return p.parse_args()


def _load_config(args_cli):
    """Load YAML config + apply --seed / --override. Returns dict."""
    import yaml
    with open(args_cli.config, 'r') as f:
        raw = yaml.safe_load(f)

    cfg = {}
    for k, v in raw.get('parameters', {}).items():
        if isinstance(v, dict):
            if 'value' in v:
                cfg[k] = v['value']
            elif 'values' in v:
                cfg[k] = v['values'][0]
        else:
            cfg[k] = v

    # CLI overrides
    if args_cli.seed is not None:
        cfg['random_seed'] = args_cli.seed
    for kv in (args_cli.override or '').split(','):
        kv = kv.strip()
        if not kv or '=' not in kv:
            continue
        k, v = kv.split('=', 1)
        # auto-cast: int → float → str
        try:
            cfg[k.strip()] = int(v)
        except ValueError:
            try:
                cfg[k.strip()] = float(v)
            except ValueError:
                cfg[k.strip()] = v.strip()
    return cfg


def _init_wandb(my_config):
    """Init WandB, fallback to SimpleNamespace nếu WANDB_MODE=disabled hoặc lỗi auth.

    Returns config object có attribute access (`config.alpha`, etc.).
    """
    wandb_mode = os.environ.get('WANDB_MODE', '').lower()
    if wandb_mode in ('disabled', 'offline'):
        print(f"ℹ️  WANDB_MODE={wandb_mode} → bỏ qua wandb.init, dùng SimpleNamespace config.")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**my_config)
        cfg.get = lambda k, d=None: getattr(cfg, k, d)
        return cfg

    try:
        wandb.init(
            project=my_config.get('wandb_project', 'unlearning-mbzuai'),
            entity=my_config.get('wandb_entity', None),  # None = dùng default user
            config=my_config,
        )
        return wandb.config
    except Exception as e:
        print(f"⚠️  wandb.init failed ({e}). Fallback sang SimpleNamespace.")
        from types import SimpleNamespace
        cfg = SimpleNamespace(**my_config)
        cfg.get = lambda k, d=None: getattr(cfg, k, d)
        return cfg


def _final_evaluation(config, output_dir, device, model_unlearn, model_retrained,
                      dataset, elapsed_h, trainable, total_params, tracker=None):
    """Final eval block — port từ forgetmi_loku.py để có đầy đủ 12 metrics cho luận văn.

    Tái sử dụng các helper trong forgetmi_loku (run_mia, cosine_sim_models, perf_metrics)
    để baseline và LoKU dùng CÙNG eval logic → so sánh fair.
    """
    print("\n" + "=" * 60)
    print("📈 FINAL EVALUATION (baseline Forget-MI)")
    print("=" * 60)

    # Import helpers từ forgetmi_loku
    from training.forgetmi_loku import (
        run_mia, cosine_sim_models, perf_metrics, _subsample_dataset
    )

    model_unlearn.eval()
    for p in model_unlearn.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()

    eval_max_retain = int(getattr(config, 'eval_max_retain', 512))
    bs = int(getattr(config, 'eval_batch_size', 16))
    paper_bs = int(getattr(config, 'mia_paper_batch_size', 32))

    # 1. MIA (per-sample + paper-style)
    try:
        print("🛡️  Computing MIA scores...")
        mia_res = run_mia(model_unlearn, dataset['retain'], dataset['test'], dataset['forget'],
                          device, config, batch_size=bs,
                          seed=int(config.random_seed), paper_batch_size=paper_bs)
        mia = mia_res['persample']
        mia_paper = mia_res['paper']
        forget_ce = mia_res.get('forget_ce', float('nan'))
        test_ce = mia_res.get('test_ce', float('nan'))
    except Exception as e:
        print(f"⚠️  MIA failed: {e}")
        mia = mia_paper = forget_ce = test_ce = float('nan')

    # 2. CosSim vs retrained
    try:
        print("📐 Computing CosSim vs retrained model...")
        cossim_ds = _subsample_dataset(dataset['retain'], eval_max_retain, int(config.random_seed))
        cossim_re = cosine_sim_models(model_unlearn, model_retrained, cossim_ds, device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  CosSim failed: {e}")
        cossim_re = float('nan')

    # 3. Test + Forget perf
    try:
        print("📊 Performance on test set...")
        test_m = perf_metrics(model_unlearn, dataset['test'], device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Test perf failed: {e}")
        test_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    try:
        print("📊 Performance on forget set...")
        forget_m = perf_metrics(model_unlearn, dataset['forget'], device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Forget perf failed: {e}")
        forget_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    gpu_peak = (torch.cuda.max_memory_allocated() / 1e9
                if torch.cuda.is_available() else 0.0)

    results = {
        'final/MIA':         round(mia, 3),
        'final/MIA_paper':   round(mia_paper, 3),
        'final/forget_ce':   round(forget_ce, 3),
        'final/test_ce':     round(test_ce, 3),
        'final/Df_AUC':      round(forget_m['AUC'], 3),
        'final/Df_F1':       round(forget_m['Macro_F1'], 3),
        'final/Dt_AUC':      round(test_m['AUC'], 3),
        'final/Dt_F1':       round(test_m['Macro_F1'], 3),
        'final/dist_vs_re':  round(1 - cossim_re, 3),
        'final/cossim_vs_re': round(cossim_re, 4),
        'efficiency/unlearn_time_hours': round(elapsed_h, 3),
        'efficiency/gpu_peak_GB':        round(gpu_peak, 2),
        'efficiency/trainable_params':   int(trainable),
        'efficiency/total_params':       int(total_params),
        'efficiency/trainable_ratio':    round(trainable / total_params, 5),
    }

    # Log to WandB if active
    try:
        wandb.log(results)
    except Exception:
        pass

    # Print summary
    print("\n" + "─" * 60)
    print(" METRIC                    | VALUE   | PAPER TARGET ")
    print("─" * 60)
    print(f"  MIA_persample (↓)        | {mia:.3f}  | (per-sample SVM)")
    print(f"  MIA_paper     (↓)        | {mia_paper:.3f}  | 0.571 (3%) / 0.615 (6%) / 0.810 (10%)")
    print(f"  Forget AUC    (↓)        | {forget_m['AUC']:.3f}  | 0.735 (3%) / 0.654 (6%) / 0.656 (10%)")
    print(f"  Forget Mac-F1 (↓)        | {forget_m['Macro_F1']:.3f}  | 0.393 (3%) / 0.328 (6%) / 0.313 (10%)")
    print(f"  Test AUC      (↑)        | {test_m['AUC']:.3f}  | 0.625 (3%) / 0.599 (6%) / 0.565 (10%)")
    print(f"  Test Mac-F1   (↑)        | {test_m['Macro_F1']:.3f}  | 0.250 (3%) / 0.270 (6%) / 0.252 (10%)")
    print(f"  1 - CosSim    (↓)        | {1-cossim_re:.3f}")
    print("─" * 60)
    print(f"  Time (hours)             | {elapsed_h:.3f}  | Forget-MI paper: ~5h")
    print(f"  GPU peak (GB)            | {gpu_peak:.2f}")
    print(f"  Trainable params         | {trainable:,} ({100*trainable/total_params:.3f}%)")
    print("─" * 60)

    # CSV summary (cùng format LoKU để Cell 5 notebook gộp được)
    # Save CSV at the PARENT of output_dir's parent → e.g. /kaggle/working/results_summary.csv
    # (not inside baseline_output/). This matches Cell 1's restore path and Cell 6's push path
    # in run_kaggle_baseline.ipynb, so the multi-seed checkpoint loop sees the same CSV.
    csv_path = os.path.join(output_dir, "..", "..", "results_summary.csv")
    csv_path = os.path.normpath(csv_path)
    # 'id' added 2026-06-18 — same fix as forgetmi_loku.py: notebooks filter by id
    # to detect completed configs. Without it, sweep/multi-seed re-runs everything.
    row = {**{k.split('/')[-1]: v for k, v in results.items()},
           'id': str(getattr(config, 'id', '')),
           'forget_pct': os.path.basename(config.forget_set_path),
           'seed': int(config.random_seed),
           'method': 'baseline_partial',
           'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    new_file = not os.path.exists(csv_path)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
    print(f"\n💾 CSV row saved: {csv_path}")

    # Auto-tracker MD (nếu --exp passed)
    if tracker is not None:
        tracker_results = {
            'MIA':           float(mia) if mia == mia else float('nan'),
            'MIA_paper':     float(mia_paper) if mia_paper == mia_paper else float('nan'),
            'Df_AUC':        float(forget_m['AUC']),
            'Df_F1':         float(forget_m['Macro_F1']),
            'Dt_AUC':        float(test_m['AUC']),
            'Dt_F1':         float(forget_m['Macro_F1']),
            'dist_vs_re':    float(1 - cossim_re) if cossim_re == cossim_re else float('nan'),
            'time_h':        float(elapsed_h),
            'gpu_gb':        float(gpu_peak),
            'trainable_pct': 100.0 * trainable / total_params,
        }
        try:
            tracker.finalize(tracker_results, elapsed_h)
        except Exception as e:
            print(f"⚠️  Tracker.finalize failed: {e}")

    return results


def main():
    """Forget-MI BASELINE main entrypoint.

    CLI usage:
        # Default (backward-compat):
        python training/forgetmi_partial.py

        # Multi-seed + override (Kaggle/Colab):
        WANDB_MODE=disabled python training/forgetmi_partial.py \\
            --config config_baseline_kaggle.yaml --seed 42 \\
            --override "forget_set_path=./data_splits/forget_set_6per.csv,id=baseline_6per" \\
            --exp baseline_6per_seed42 --hypothesis "Reproduce paper Table 2 6% Unimodal setting"
    """
    args_cli = _parse_cli()
    my_config = _load_config(args_cli)

    # ---- WandB (graceful fallback) ----
    config = _init_wandb(my_config)

    # ---- Normalize loss weights ----
    alpha = config.alpha
    beta = config.beta
    theta = config.theta
    gamma = config.gamma
    total = alpha + beta + theta + gamma
    alpha, beta, theta, gamma = (round(x / total, 2) for x in (alpha, beta, theta, gamma))

    image_noise_params = {"mean": config.noise_mean, "std": config.noise_std}
    pct_match = re.search(r'forget_set_(\d+)per\.csv', config.forget_set_path)
    unlearning_percentage = int(pct_match.group(1)) if pct_match else None

    try:
        wandb.log({
            "seed": config.random_seed, "unlearning_percentage": unlearning_percentage,
            "use_noise": config.use_noise, "learning_rate": config.learning_rate,
            "unlearn_epochs": config.unlearn_epochs,
            "alpha": alpha, "beta": beta, "theta": theta, "gamma": gamma,
        })
    except Exception:
        pass

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = (f"{alpha}_UKR_{beta}_UU_{gamma}_MD_{theta}_MKR_"
                f"mean{config.noise_mean}_std{config.noise_std}_{config.use_noise}_"
                f"{timestamp}_partials_hinge_euc")
    try:
        wandb.run.name = run_name
    except Exception:
        pass

    # ---- Seed & device ----
    set_seed(config.random_seed)
    torch.cuda.empty_cache()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if not torch.cuda.is_available():
        print("⚠️  KHÔNG có GPU — baseline sẽ cực kỳ chậm (~25× chậm hơn)")
    else:
        # Reset per-seed so gpu_peak_GB reflects THIS seed, not cumulative across multi-seed runs.
        torch.cuda.reset_peak_memory_stats()

    # ---- Output dir ----
    output_dir = os.path.join(config.output_dir, run_name)
    if args_cli.fresh and os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output: {output_dir}")

    # ---- Auto-tracker (nếu --exp) ----
    tracker = None
    if args_cli.exp:
        try:
            from scripts.exp_tracker import ExpTracker
            tracker = ExpTracker(name=args_cli.exp, hypothesis=args_cli.hypothesis)
        except Exception as e:
            print(f"⚠️  Tracker init failed ({e}). Tiếp tục không tracker.")

    # ---- Load models ----
    print(f"📦 Loading models...")
    model_og = ImageTextModel.from_pretrained(config.base_model_path).to(device)
    model_unlearn = copy.deepcopy(model_og)
    # Try to load retrained; on failure fallback to model_og as dummy so the script
    # can still run (per-epoch CosSim eval will be vs model_og — meaningless but doesn't
    # crash). Happens when forget% has no gold retrained model on disk (e.g., 6%/10%).
    try:
        model_retrained = ImageTextModel.from_pretrained(config.retrained_model_path).to(device)
    except Exception as e:
        print(f"⚠️  Cannot load retrained from {config.retrained_model_path}")
        print(f"   Reason: {type(e).__name__}: {str(e)[:200]}")
        print(f"   → Using model_og as DUMMY retrained — CosSim metrics will be INVALID (dist_vs_re ≈ 0)")
        model_retrained = copy.deepcopy(model_og).to(device)

    tokenizer = BertTokenizer.from_pretrained(config.bert_pretrained_dir)

    for p in model_og.parameters(): p.requires_grad = False
    for p in model_retrained.parameters(): p.requires_grad = False
    for p in model_unlearn.parameters(): p.requires_grad = True

    trainable = sum(p.numel() for p in model_unlearn.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model_unlearn.parameters())
    print(f"📊 Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.3f}%) — baseline full FT")

    # Gradient checkpointing on the BERT side of model_unlearn: trades ~20% slowdown
    # for ~30-40% less activation memory. Essential to fit baseline (113M trainable +
    # 3 stacked forwards) on a 14.5 GB T4 without dropping batch size further.
    try:
        model_unlearn.text_model.bert.gradient_checkpointing_enable()
        # use_cache must be off for grad-checkpointing — HF prints a noisy warning otherwise.
        model_unlearn.text_model.bert.config.use_cache = False
        print("✅ Gradient checkpointing enabled on model_unlearn.text_model.bert")
    except Exception as e:
        print(f"⚠️  Gradient checkpointing not enabled ({e}) — may OOM at batch_size > 8")

    # ---- Optimizer ----
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = list(model_unlearn.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': config.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=float(config.learning_rate))
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", threshold=1e-6)

    # ---- Build dataset ----
    dataset, num_labels = build_dataset(config, tokenizer, image_noise_params=image_noise_params)

    # Add n_* counts (legacy: unlearn() expects them)
    dataset['n_retain'] = len(dataset['retain'])
    dataset['n_val'] = len(dataset['validation'])
    dataset['n_test'] = len(dataset['test'])
    dataset['n_rand'] = len(dataset['random'])
    dataset['n_forget'] = len(dataset['forget'])

    # ---- Train ----
    train_start = time.time()
    unlearn(
        config, output_dir, device, model_og, model_unlearn, model_retrained,
        optimizer, optimizer_grouped_parameters, scheduler, tokenizer,
        dataset, num_labels, alpha, beta, theta, gamma
    )
    elapsed_h = (time.time() - train_start) / 3600

    # Free training-only memory
    del model_og, optimizer
    torch.cuda.empty_cache()

    # ---- Final evaluation ----
    _final_evaluation(config, output_dir, device, model_unlearn, model_retrained,
                      dataset, elapsed_h, trainable, total_params, tracker=tracker)

    # ---- Post-eval cleanup ----
    # Per-epoch checkpoints (~450MB × 30 = 13.5GB) would otherwise fill Kaggle's 20GB
    # /kaggle/working/ cap across multi-seed runs. Final metrics already in CSV +
    # tracker MD; only the LAST epoch's checkpoint is kept for audit/reload.
    # Environment-required fix — does NOT affect training, eval, or reported numbers.
    try:
        import glob as _glob, shutil as _shutil
        epoch_dirs = sorted(_glob.glob(os.path.join(output_dir, "epoch_*")),
                            key=lambda p: int(os.path.basename(p).split('_')[1]))
        for ep_dir in epoch_dirs[:-1]:                # keep last
            try:
                _shutil.rmtree(ep_dir)
            except Exception:
                pass
        if epoch_dirs:
            print(f"🧹 Cleaned {len(epoch_dirs)-1} per-epoch checkpoints (kept last: {os.path.basename(epoch_dirs[-1])})")
    except Exception as e:
        print(f"⚠️  Per-epoch cleanup failed ({e}); manual cleanup may be needed.")

    try:
        wandb.finish()
    except Exception:
        pass


if __name__ == '__main__':
    main()