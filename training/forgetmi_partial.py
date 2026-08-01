#!/usr/bin/env python3
"""forgetmi_partial.py — TÁI LẬP Forget-MI, dùng làm BASELINE đối chứng cho P3.

Quy tắc đối chiếu (áp dụng cho MỌI khác biệt paper ↔ code gốc):
  paper mô tả ĐẦY ĐỦ → theo paper;  paper bỏ ngỏ/mơ hồ → theo code gốc của tác giả;
  không cố tình sao chép lỗi lập trình hiển nhiên.

THEO PAPER (khác code gốc `Original-Code/.../forgetmi_partial.py`):
  - Eq.(1)-(2): L_UU, L_MU = ÂM khoảng cách Euclid tới tham chiếu NHIỄU. Code gốc để dấu
    DƯƠNG ở nhánh use_noise=true (kéo lại gần) — trái phương trình paper.
  - Eq.(3)-(4): L_UR, L_MR = khoảng cách Euclid THUẦN. Đã bỏ margin=(L+1) đo ở epoch 0 và
    torch.minimum(L, margin) của code gốc; KHÔNG thay bằng cơ chế nào khác.
  - Hệ quả: epoch 0 không còn là epoch "chỉ đo margin" → backward + cập nhật từ epoch đầu,
    30 epoch = 30 lần cập nhật (code gốc: 29).

THEO CODE GỐC (paper không quy định):
  - Cadence: gradient tích lũy qua các batch trong epoch, optimizer.step() MỘT lần cuối epoch.
  - Fusion Gate: tạo mới mỗi batch, riêng từng vai trò, train mode mặc định, KHÔNG vào optimizer.
  - Tập random luôn là bản nhiễu (ảnh Gaussian σ + text perturbation).

KHÔNG đưa thành phần của LoKU/P3 sang đây: không Fisher, FILA, LoRA, IHL, CE/KD bổ sung,
không đóng băng head, không checkpoint W*. Baseline vẫn là full fine-tuning.
"""
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
from torch.utils.data import DataLoader, Dataset, Sampler, Subset
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


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)

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
        forget_img_labels, forget_img_txt_ids, n_retain, n_val, n_test, n_rand, n_forget, \
        sel_img_labels, sel_img_txt_ids = data_split(
            args.data_split_path, args.forget_set_path,
            seed=int(getattr(args, 'random_seed', 42)),
            test_sel_splits=int(getattr(args, 'test_sel_splits', 4)),
            retain_heldout_splits=int(getattr(args, 'retain_heldout_splits', 10)))
    n_sel = len(sel_img_txt_ids)

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
    sel_img_txt_ids,    sel_img_labels    = _filter_valid(sel_img_txt_ids,    sel_img_labels,    all_txt_tokens,   'sel')

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

    # D_nm_val cho selector S1–S4 (25% test, og chưa từng thấy). KHÔNG dùng để train, chỉ
    # để đo nm_val_ce + utility → dùng CenterCrop (tất định) ĐÚNG NHƯ P3 dùng cho tập `sel`,
    # để trajectory CE của hai phương pháp so được với nhau.
    sel_dataset = CXRImageTextDataset(args.id,
                                all_txt_tokens, all_txt_masks, all_txt_segments,
                                all_txt_labels, sel_img_txt_ids, args.img_data_dir,
                                sel_img_labels, dataset_split_path=args.data_split_path,
                                transform=CenterCrop(2048),
                                output_channel_encoding=args.output_channel_encoding)

    print(f"Datasets: retain={len(retain_dataset)} val={len(val_dataset)} test={len(test_dataset)} "
          f"forget={len(forget_dataset)} random={len(rand_dataset)} sel={len(sel_dataset)}")

    dataset = {
        'retain': retain_dataset,
        'validation': val_dataset,
        'test': test_dataset,
        'sel': sel_dataset,
        'random': rand_dataset,
        'forget': forget_dataset,
        'n_retain': n_retain,
        'n_val': n_val,
        'n_test': n_test,
        'n_rand': n_rand,
        'n_forget': n_forget,
        'n_sel': n_sel,
    }

    return dataset, num_labels
def data_split(split_list_path, forget_ids_path, rand_ratio=None, validation_ratio=None,
               seed=42, test_sel_splits=4, retain_heldout_splits=10):
    """Chia dữ liệu cho baseline Forget-MI — DÙNG CHUNG protocol holdout với P3.

    Vì sao không giữ hàm split của code gốc:
      (1) LỖI LẬP TRÌNH rõ ràng — sau khi tách val_ids bằng train_test_split, code gốc lại
          `train_labels.update(...)` đưa validation NGƯỢC vào tập train (và làm HAI lần).
          Validation vì thế nằm trong tập được huấn luyện → val-CE không còn là tín hiệu
          held-out, chọn checkpoint theo nó là rò rỉ. Theo quy tắc đã chốt, không sao chép
          lỗi lập trình hiển nhiên.
      (2) CÔNG BẰNG — để so P3 vs Forget-MI có kiểm soát, hai bên phải đánh giá trên CÙNG
          các mẫu. Ở đây gọi thẳng `data_split_advanced` của P3 với cùng seed và cùng tham
          số chia, nên retain/forget/validation/test là TRÙNG KHỚP mẫu giữa hai phương pháp.

    Ánh xạ sang tên biến của baseline:
      train      ← retain      (D_r dùng để train UR/MR — KHÔNG chứa validation)
      val        ← r_heldout   (giữ ngoài cập nhật; dùng cho val-CE chọn checkpoint)
      test       ← test_final  (75% test, CHỈ dùng đánh giá cuối)
      forget     ← forget      (D_f)
      rand       ← random      (bản nhiễu của D_f)
      sel        ← sel         (25% test, og-unseen — D_nm_val cho selector S1–S4)

    ĐÂY CHỈ LÀ PROTOCOL DỮ LIỆU/ĐÁNH GIÁ, không phải thành phần phương pháp: baseline vẫn
    là Forget-MI full fine-tuning, không có Fisher/FILA/LoRA/IHL.

    rand_ratio / validation_ratio giữ trong chữ ký cho tương thích lời gọi cũ nhưng KHÔNG
    còn được dùng (rand_ratio vốn đã không được dùng ở bản gốc; tỉ lệ validation nay do
    retain_heldout_splits quyết định)."""
    from training.adv_common import data_split_advanced

    splits = data_split_advanced(split_list_path, forget_ids_path, seed=int(seed),
                                 test_sel_splits=int(test_sel_splits),
                                 retain_heldout_splits=int(retain_heldout_splits))

    train_img_txt_ids,  train_labels  = splits['retain']
    val_img_txt_ids,    val_labels    = splits['r_heldout']
    test_img_txt_ids,   test_labels   = splits['test_final']
    forget_img_txt_ids, forget_labels = splits['forget']
    rand_img_txt_ids,   rand_labels   = splits['random']
    sel_img_txt_ids,    sel_labels    = splits['sel']

    # Chốt chặn: validation KHÔNG được nằm trong train (bug của code gốc).
    leak = set(val_img_txt_ids) & set(train_img_txt_ids)
    assert not leak, f"Validation bị lẫn vào train ({len(leak)} mẫu) — split sai."

    n_train, n_val, n_rand, n_test, n_forget = (len(train_img_txt_ids), len(val_img_txt_ids),
                                                len(rand_img_txt_ids), len(test_img_txt_ids),
                                                len(forget_img_txt_ids))

    print(f"Split (protocol chung với P3, seed={seed}): train/retain={n_train} "
          f"val/r_heldout={n_val} test_final={n_test} forget={n_forget} random={n_rand} "
          f"sel/D_nm_val={len(sel_img_txt_ids)}")

    # sel trả về ở CUỐI để không đổi thứ tự 15 giá trị cũ (mọi lời gọi hiện có vẫn chạy).
    return train_labels, train_img_txt_ids, val_labels, val_img_txt_ids, test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids, forget_labels, forget_img_txt_ids, n_train, n_val, n_test, n_rand, n_forget, sel_labels, sel_img_txt_ids

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

def _write_perepoch_row(csv_path, epoch, args, m):
    """Append 1 dòng metrics/epoch vào CSV (header tự tạo lần đầu)."""
    import csv as _csv
    cols = ['id', 'seed', 'epoch', 'Df_AUC', 'Df_F1', 'Dt_AUC', 'Dt_F1',
            'MIA', 'MIA_paper', 'forget_ce', 'test_ce', 'dist_vs_re']
    row = {'id': getattr(args, 'id', ''), 'seed': getattr(args, 'random_seed', ''), 'epoch': epoch}
    for k in ['Df_AUC', 'Df_F1', 'Dt_AUC', 'Dt_F1', 'MIA', 'MIA_paper', 'forget_ce', 'test_ce', 'dist_vs_re']:
        row[k] = m.get(k, '')
    new = not os.path.exists(csv_path)
    d = os.path.dirname(csv_path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(csv_path, 'a', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)


def _append_row_csv(csv_path, row):
    """Ghi 1 dòng vào CSV, tự MIGRATE header khi row có cột mới (thêm vào cuối header) và
    LUÔN ghi theo thứ tự header THỰC của file → không bao giờ lệch cột. Row cũ thiếu cột
    mới thì điền ''. Dùng chung baseline + LoKU để CSV luôn hợp lệ khi schema đổi."""
    import csv as _csv
    row_keys = list(row.keys())
    header = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='') as f:
            header = next(_csv.reader(f), [])
    new_cols = [k for k in row_keys if k not in header]
    if header and new_cols:                       # migrate: rewrite với union header
        with open(csv_path, 'r', newline='') as f:
            old_rows = list(_csv.DictReader(f))
        header = header + new_cols
        with open(csv_path, 'w', newline='') as f:
            w = _csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in old_rows:
                w.writerow({k: r.get(k, '') for k in header})
    write_header = not header
    if write_header:
        header = row_keys
    with open(csv_path, 'a', newline='') as f:
        w = _csv.DictWriter(f, fieldnames=header, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, '') for k in header})


def _eval_epoch_metrics(args, device, model_ul, model_re, dataset):
    """Eval READ-ONLY tại epoch hiện tại → trả dict metrics (giống final, bỏ phần in).

    KHÔNG được làm lệch training: lưu & phục hồi train-mode, requires_grad, và TOÀN BỘ
    RNG state (torch/cuda/python/numpy) — nhờ vậy kết quả E29 y hệt bản chạy trung thực.
    """
    import random as _random
    import numpy as _np
    from training.forgetmi_loku import (
        run_mia, cosine_sim_models, perf_metrics, _subsample_dataset
    )
    was_training = model_ul.training
    grad_states = [p.requires_grad for p in model_ul.parameters()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    py_rng = _random.getstate()
    np_rng = _np.random.get_state()

    bs = int(getattr(args, 'eval_batch_size', 16))
    eval_max_retain = int(getattr(args, 'eval_max_retain', 512))
    paper_bs = int(getattr(args, 'mia_paper_batch_size', 32))
    out = {}
    try:
        model_ul.eval()
        with torch.no_grad():
            try:
                mia_res = run_mia(model_ul, dataset['retain'], dataset['test'], dataset['forget'],
                                  device, args, batch_size=bs,
                                  seed=int(args.random_seed), paper_batch_size=paper_bs)
                out['MIA'] = round(mia_res['persample'], 3)
                out['MIA_paper'] = round(mia_res['paper'], 3)
                out['forget_ce'] = round(mia_res.get('forget_ce', float('nan')), 3)
                out['test_ce'] = round(mia_res.get('test_ce', float('nan')), 3)
            except Exception as e:
                print(f"   ⚠️  per-epoch MIA failed: {e}")
            try:
                fm = perf_metrics(model_ul, dataset['forget'], device, args, batch_size=bs)
                tm = perf_metrics(model_ul, dataset['test'], device, args, batch_size=bs)
                out['Df_AUC'] = round(fm['AUC'], 3); out['Df_F1'] = round(fm['Macro_F1'], 3)
                out['Dt_AUC'] = round(tm['AUC'], 3); out['Dt_F1'] = round(tm['Macro_F1'], 3)
            except Exception as e:
                print(f"   ⚠️  per-epoch perf failed: {e}")
            try:
                if model_re is not None:
                    cds = _subsample_dataset(dataset['retain'], eval_max_retain, int(args.random_seed))
                    cs = cosine_sim_models(model_ul, model_re, cds, device, args, batch_size=bs)
                    out['dist_vs_re'] = round(1 - cs, 3)
            except Exception as e:
                print(f"   ⚠️  per-epoch cossim failed: {e}")
    finally:
        if was_training:
            model_ul.train()
        for p, g in zip(model_ul.parameters(), grad_states):
            p.requires_grad = g
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        _random.setstate(py_rng)
        _np.random.set_state(np_rng)
        torch.cuda.empty_cache()
    return out


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
    # Bóc tách thời gian theo CÙNG chuẩn với P3 (adv_common.CudaTimer, đồng bộ CUDA):
    #   T_core(Forget-MI) = T_train   (không có Fisher/FILA)
    #   T_selection       = val-CE + CE-selector  (giao thức, KHÔNG vào core)
    #   monitor/ckpt      = CosSim log + I/O, cũng không vào core
    from training.adv_common import CudaTimer, reset_gpu_peak, get_gpu_peak, _stat
    t_train = t_monitor = t_ckpt = t_selection = 0.0
    epoch_train_times = []
    tm_peaks = {}
    peak_train = peak_sel = (0.0, 0.0)
    # warm-up: loại chi phí khởi tạo CUDA khỏi phép đo (không đổi checkpoint/dữ liệu)
    try:
        _wb = next(iter(DataLoader(dataset['forget'], batch_size=2)))
        with torch.no_grad():
            _wi, _, _ = get_model_inputs(args, _wb, device)
            model_og(**_wi)
        del _wb, _wi
    except Exception as _e:
        print(f"   (bỏ qua warm-up: {_e})")

    dual_checkpoint_eval = _as_bool(getattr(args, 'evaluate_last_and_best', False))
    legacy_cossim_monitor = _as_bool(getattr(args, 'legacy_cossim_monitor', True))
    selection_max_validation = int(getattr(args, 'selection_max_validation', 400))
    selection_min_delta = float(getattr(args, 'selection_min_delta', 0.0))
    val_selection_set = (Subset(val_set, list(range(min(selection_max_validation, len(val_set)))))
                         if selection_max_validation > 0 else val_set)
    best_val_ce = float('inf')
    best_epoch = None
    best_state = None
    last_val_ce = float('nan')
    last_epoch = -1

    # Định nghĩa trước vòng lặp: khi unlearn_epochs=0 (eval-only θ_og/θ_re) vòng lặp
    # không chạy nên biến này (bình thường gán lại mỗi epoch) sẽ chưa tồn tại → return crash.
    retain_dataloader = None

    # ----- CE-crossing selector (gold-free, INLINE) — bật bằng --override ce_selector_out=<dir>.
    # KHÔNG đổi training/loss: chỉ eval read-only mỗi epoch (RNG-safe) + snapshot ứng viên. -----
    _ce_sel = None
    if getattr(args, 'ce_selector_out', None):
        from training.ce_selector_pilot import OnlineCESelector
        # Truyền THẲNG sel/test_final của protocol chung với P3. Nếu để selector tự cắt
        # dataset['test'] thành 25/75 (nhánh make_nmval_tfinal) thì nó sẽ cắt LẠI bên trong
        # test_final → D_nm_val thành tập CON của tập dùng chấm điểm (rò rỉ) và tập chấm
        # chỉ còn ~56% test, khác hẳn P3. Truyền sẵn ⇒ S1–S4 hai bên dùng ĐÚNG cùng mẫu.
        _ce_sel = OnlineCESelector(args, dataset, device, out_dir=str(args.ce_selector_out),
                                   split_seed=int(getattr(args, 'random_seed', 42)),
                                   nm_val_ds=dataset['sel'], tfinal_ds=dataset['test'])

    for epoch in unlearning_iterator:
        # T_train của epoch: CHỈ forward/backward/step, đồng bộ CUDA hai đầu.
        reset_gpu_peak()
        _tt = CudaTimer(); _tt.__enter__()
        model_ul.train()
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
        optimizer.zero_grad(set_to_none=True)   # bắt đầu tích lũy gradient cho epoch này
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

            # (Code gốc còn dựng `og_frgt_joint_emb` bằng gate_ul_frgt(img, img) — dùng nhầm
            #  gate và truyền ảnh hai lần. Biến này KHÔNG tham gia hàm mất mát nên đây là
            #  lỗi lập trình chứ không phải thiết kế phương pháp → không sao chép lại.)

            gate_og_rand =  Gate(inp1_size = og_rand_img_emb.shape[1], inp2_size = og_rand_txt_emb.shape[1]).to(device)
            og_rand_joint_emb = gate_og_rand(og_rand_img_emb, og_rand_txt_emb)

            gate_og_ret =  Gate(inp1_size = og_ret_img_emb.shape[1], inp2_size = og_ret_txt_emb.shape[1]).to(device)
            og_ret_joint_emb = gate_og_ret(og_ret_img_emb, og_ret_txt_emb)

            # ------------------------------------------- UU / MD Loss -------------------------------------------
            # ĐÚNG PAPER Eq.(1)-(2): L_UU = −Dist, L_MU = −Dist — đẩy F_ul(D_f) RA XA biểu
            # diễn mà F_og tạo trên BẢN NHIỄU (Ĩ_f, T̃_f).
            #
            # Code gốc có if/else theo `use_noise` chỉ để ĐỔI DẤU (use_noise=true → +Dist,
            # tức kéo LẠI GẦN — trái phương trình paper). Sau khi cả hai nhánh về −Dist thì
            # if/else thành nhánh chết, nên gộp lại làm một.
            #
            # LƯU Ý: `use_noise` KHÔNG điều khiển việc tập tham chiếu có nhiễu hay không.
            # Trong Forget-MI (cả bản gốc lẫn bản này), rand_dataset LUÔN là bản nhiễu
            # (perturb_img=True + noisy_txt, xem build_dataset) → cờ này hiện không còn
            # ảnh hưởng gì tới hàm mất mát.
            ul_frgt_concat_emb = torch.cat((ul_frgt_img_emb, ul_frgt_txt_emb), dim=-1)
            og_rand_concat_emb = torch.cat((og_rand_img_emb, og_rand_txt_emb), dim=-1)

            L_uu = -euclidean_distance(ul_frgt_concat_emb, og_rand_concat_emb).mean()
            L_md = -euclidean_distance(ul_frgt_joint_emb, og_rand_joint_emb).mean()

            # ------------------------------------------- UKR Loss -------------------------------------------
            ul_ret_concat_emb = torch.cat((ul_ret_img_emb, ul_ret_txt_emb), dim=-1)
            og_ret_concat_emb = torch.cat((og_ret_img_emb, og_ret_txt_emb), dim=-1)

            L_ukr = euclidean_distance(ul_ret_concat_emb, og_ret_concat_emb).mean()
    
            # ------------------------------------------- MKR Loss -------------------------------------------
            L_mkr = euclidean_distance(ul_ret_joint_emb, og_ret_joint_emb).mean()

            # ĐÚNG PAPER Eq.(3)-(4): UR/MR là khoảng cách Euclid THUẦN, KHÔNG cap.
            # (Code gốc đo margin=(L+1) ở epoch 0 rồi torch.minimum(L, margin) từ epoch sau
            #  → gradient triệt tiêu khi độ lệch vượt margin, không có trong phương trình
            #  paper. Đã bỏ hoàn toàn margin_ukr/margin_mkr; KHÔNG thay bằng cơ chế khác.)

            # ------------------------------------------- Total Loss -------------------------------------------
            loss = (alpha*L_ukr + beta*L_uu) + (theta*L_md + gamma*L_mkr)

            #------------------------------------------- Backpropagation -------------------------------------------
            md_loss += L_md.item()
            uu_loss += L_uu.item()
            mkr_loss += L_mkr.item()
            ukr_loss += L_ukr.item()
            total_loss += loss.item()

            # Gradient TÍCH LŨY qua các batch trong epoch (cadence của code gốc Forget-MI:
            # optimizer.step() nằm NGOÀI vòng batch). Backward chạy từ epoch ĐẦU TIÊN —
            # ngoại lệ "epoch 0 chỉ để đo margin, không cập nhật" đã mất lý do tồn tại
            # sau khi retention margin bị bỏ theo Eq.(3)-(4). Với 30 epoch → 30 lần cập nhật.
            loss.backward()

            steps += 1

        # ----- cuối epoch: clip (nếu bật) rồi cập nhật MỘT lần -----
        # LƯU Ý: điều kiện dưới đây giữ NGUYÊN như code gốc và thực tế luôn False
        # (optimizer_grouped_parameters là list các dict) → bản gốc KHÔNG clip. Giữ nguyên
        # để không tự ý thêm một cơ chế mà Forget-MI không dùng.
        if 'grad' in optimizer_grouped_parameters:
            torch.nn.utils.clip_grad_norm_(optimizer_grouped_parameters,
                                           args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        _tt.__exit__(None, None, None)
        epoch_train_s = _tt.elapsed
        epoch_train_times.append(epoch_train_s)
        t_train += epoch_train_s
        peak_train = tuple(max(a, b) for a, b in zip(peak_train, get_gpu_peak()))

        # ----- Per-epoch CosSim eval (FAITHFUL to original Forget-MI implementation) -----
        # Runs FULL retain forward through model_ul AND model_re each epoch. Chỉ để LOG
        # (không dùng cho unlearning/early-stop) → tính vào MONITOR, không phải train.
        _t_mon = time.time()
        last_epoch = epoch
        last_val_ce = float('nan')
        is_best_so_far = False
        if dual_checkpoint_eval:
            # val-CE là GIAO THỨC chọn checkpoint → tính vào T_selection, không vào T_core.
            from training.forgetmi_loku import per_sample_ce
            reset_gpu_peak()
            was_training = model_ul.training
            with CudaTimer() as _tsel:
                last_val_ce = float(per_sample_ce(
                    model_ul, val_selection_set, device, args, args.eval_batch_size
                ).mean())
            t_selection += _tsel.elapsed
            peak_sel = tuple(max(a, b) for a, b in zip(peak_sel, get_gpu_peak()))
            if was_training:
                model_ul.train()
            # Mọi epoch (kể cả epoch 0) đều đã cập nhật trọng số → đều là ứng viên hợp lệ.
            if last_val_ce < best_val_ce - selection_min_delta:
                best_val_ce = last_val_ce
                best_epoch = epoch
                is_best_so_far = True
                model_to_save = model_ul.module if hasattr(model_ul, 'module') else model_ul
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model_to_save.state_dict().items()
                }
        from evaluation.eval_unlearning import get_probability_measure
        try:
            cosine_similarity = (get_probability_measure(
                args,
                copy.deepcopy(model_re).eval(),
                copy.deepcopy(model_ul).eval(),
                retain_dataloader,
                device
            ) if legacy_cossim_monitor else float('nan'))
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
                "selection/val_ce": last_val_ce,
            })
        except Exception:
            pass

        elapsed_min = (time.time() - unlearning_start_time) / 60
        eta_min = elapsed_min / (epoch + 1) * (n_epochs - epoch - 1)
        monitor_text = (f"val_CE={last_val_ce:.4f}" if dual_checkpoint_eval
                        else f"CosSim={cosine_similarity:+.4f}")
        print(f"[E{epoch:02d}/{n_epochs}] loss={total_loss/steps:+.3f} "
              f"UKR={ukr_loss/steps:+.3f} UU={uu_loss/steps:+.3f} "
              f"MD={md_loss/steps:+.3f} MKR={mkr_loss/steps:+.3f} "
              f"{monitor_text} | {elapsed_min:.1f}m ETA {eta_min:.1f}m")
        epoch_monitor_s = time.time() - _t_mon
        t_monitor += epoch_monitor_s

        history_csv_path = getattr(args, 'history_csv_path', None)
        if history_csv_path:
            _append_row_csv(history_csv_path, {
                'id': str(getattr(args, 'id', '')),
                'method': 'forgetmi',
                'seed': int(getattr(args, 'random_seed', 0)),
                'forget_pct': os.path.basename(args.forget_set_path),
                'epoch': int(epoch + 1),
                # Sau khi bỏ ngoại lệ epoch 0 (đo margin), MỌI epoch đều có 1 lần cập nhật.
                'optimizer_update': 1,
                'val_ce': last_val_ce,
                'is_best_so_far': int(is_best_so_far),
                'best_epoch_so_far': (int(best_epoch + 1) if best_epoch is not None else ''),
                'total_loss': total_loss / steps,
                'ukr_loss': ukr_loss / steps,
                'uu_loss': uu_loss / steps,
                'md_loss': md_loss / steps,
                'mkr_loss': mkr_loss / steps,
                'epoch_train_hours': epoch_train_s / 3600,
                'epoch_monitor_hours': epoch_monitor_s / 3600,
                'cumulative_train_hours': t_train / 3600,
                'cumulative_monitor_hours': t_monitor / 3600,
            })

        # ----- Per-epoch checkpoint save (FAITHFUL to original) -----
        # Original saves every epoch. ~450 MB × 30 = ~13.5 GB on /kaggle/working/. I/O → tính CKPT.
        _t_ck = time.time()
        epoch_output_dir = (output_dir if dual_checkpoint_eval
                            else os.path.join(output_dir, f"epoch_{epoch}"))
        os.makedirs(epoch_output_dir, exist_ok=True)
        model_to_save = model_ul.module if hasattr(model_ul, 'module') else model_ul
        if not dual_checkpoint_eval:
            torch.save(model_to_save.state_dict(), os.path.join(epoch_output_dir, "model_state_dict.pth"))
        t_ckpt += time.time() - _t_ck

        # ----- Per-epoch eval (READ-ONLY, bật bằng --override eval_every_epoch=1) -----
        # Vẽ quỹ đạo Df_AUC/MIA... theo epoch để dò điểm khớp paper. KHÔNG đụng training:
        # _eval_epoch_metrics khôi phục train-mode/grad/RNG nên E29 vẫn y hệt bản trung thực.
        if getattr(args, 'eval_every_epoch', False):
            _t_mon2 = time.time()
            _pe = _eval_epoch_metrics(args, device, model_ul, model_re, dataset)
            _pe_csv = os.path.join(os.path.dirname(os.path.dirname(output_dir)),
                                   f"perepoch_{getattr(args, 'id', 'run')}.csv")
            _write_perepoch_row(_pe_csv, epoch, args, _pe)
            print(f"   📈 [eval E{epoch:02d}] Df_AUC={_pe.get('Df_AUC')} Dt_AUC={_pe.get('Dt_AUC')} "
                  f"MIA={_pe.get('MIA')} MIA_paper={_pe.get('MIA_paper')} "
                  f"fce={_pe.get('forget_ce')} tce={_pe.get('test_ce')}")
            t_monitor += time.time() - _t_mon2

        # ----- CE-crossing selector: eval CE + snapshot ứng viên epoch này (gold-free) -----
        if _ce_sel is not None:
            with CudaTimer() as _tcs:
                _ce_sel.step(epoch, model_ul)
            t_selection += _tcs.elapsed     # giao thức, KHÔNG vào T_core

    # sau vòng lặp: chọn epoch theo 4 cách + eval selected trên D_t_final (gold-free)
    if _ce_sel is not None:
        with CudaTimer() as _tcf:
            _ce_sel.finalize(model_ul)
        t_selection += _tcf.elapsed

    checkpoint_dir = None
    best_checkpoint = None
    last_checkpoint = None
    if dual_checkpoint_eval:
        if best_state is None:
            raise RuntimeError("No eligible val-best checkpoint was produced. Forget-MI needs at least 2 passes.")
        _t_ck = time.time()
        checkpoint_dir = os.path.join(output_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_to_save = model_ul.module if hasattr(model_ul, 'module') else model_ul
        last_checkpoint = os.path.join(checkpoint_dir, 'last.pt')
        best_checkpoint = os.path.join(checkpoint_dir, 'val_best.pt')
        torch.save({
            'model_state': model_to_save.state_dict(),
            'epoch': last_epoch,
            'val_ce': last_val_ce,
            'checkpoint_kind': 'last',
        }, last_checkpoint)
        torch.save({
            'model_state': best_state,
            'epoch': best_epoch,
            'val_ce': best_val_ce,
            'checkpoint_kind': 'val_best',
        }, best_checkpoint)
        t_ckpt += time.time() - _t_ck
        del best_state

    # ------------------------------------------- Timing breakdown -------------------------------------------
    wall_h = (time.time() - unlearning_start_time) / 3600
    tm_peaks['train'] = peak_train
    tm_peaks['selection'] = peak_sel
    _est = _stat(epoch_train_times)
    print(f"⏱  T_train = {t_train:.1f}s ({t_train/3600:.4f}h) · epoch: mean {_est['mean']:.2f}s "
          f"± {_est['std']:.2f} (min {_est['min']:.2f} / max {_est['max']:.2f})")
    print(f"⏱  T_selection = {t_selection:.1f}s — giao thức chọn checkpoint, KHÔNG vào core")
    print(f"⏱  monitor(CosSim log) {t_monitor:.1f}s · ckpt I/O {t_ckpt:.1f}s — cũng không vào core")
    timing = {'train_h': t_train / 3600, 'monitor_h': t_monitor / 3600,
              'ckpt_h': t_ckpt / 3600, 'wall_h': wall_h,
              # ---- chuẩn đo dùng chung với P3: core(FMI) = train (không Fisher/FILA) ----
              'fisher_seconds': 0.0, 'fila_seconds': 0.0,
              'train_seconds': float(t_train),
              'selection_seconds': float(t_selection),
              'eval_seconds': 0.0,                    # điền sau _final_evaluation
              'epoch_train_stat': _est, 'peaks': tm_peaks,
              'optimizer_updates': max(last_epoch + 1, 0),
              'best_checkpoint': best_checkpoint, 'last_checkpoint': last_checkpoint,
              'best_epoch': best_epoch, 'last_epoch': last_epoch,
              'best_val_ce': best_val_ce, 'last_val_ce': last_val_ce,
              'training_passes': max(last_epoch + 1, 0),
              # MỌI epoch đều cập nhật (ngoại lệ epoch 0 đã bỏ) → update_epochs = số epoch,
              # bằng training_passes. Trước đây trừ 1 vì epoch 0 chỉ đo margin.
              'update_epochs': max(last_epoch + 1, 0)}
    try:
        wandb.log({"time(hours)": wall_h, "time/train_h": timing['train_h'],
                   "time/monitor_h": timing['monitor_h']})
    except Exception:
        pass
    print(f"⏱  train {timing['train_h']:.3f}h + monitor {timing['monitor_h']:.3f}h "
          f"+ ckpt {timing['ckpt_h']:.3f}h = wall {wall_h:.3f}h")
    return timing
    
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
                      dataset, elapsed_h, trainable, total_params, tracker=None,
                      method='baseline_partial', timing=None, checkpoint_kind='last',
                      selected_epoch=None, selection_metric='val_ce', selection_value=None,
                      training_passes=None, update_epochs=None, csv_path=None,
                      row_extra=None, eval_helpers=None, finalize_tracker=True):
    """Final eval block — port từ forgetmi_loku.py để có đầy đủ metrics cho luận văn.

    Tái sử dụng các helper trong forgetmi_loku (run_mia, cosine_sim_models, perf_metrics)
    để baseline và LoKU dùng CÙNG eval logic → so sánh fair. `timing` = dict bóc tách thời
    gian từ unlearn() (train/monitor/ckpt/fisher/load); None → chỉ hiện elapsed_h.
    """
    _eval_start = time.time()
    if eval_helpers is None:
        from training.forgetmi_loku import (
            run_mia, cosine_sim_models, perf_metrics, _subsample_dataset
        )
    else:
        run_mia, cosine_sim_models, perf_metrics, _subsample_dataset = eval_helpers

    model_unlearn.eval()
    for p in model_unlearn.parameters():
        p.requires_grad = False
    torch.cuda.empty_cache()

    eval_max_retain = int(getattr(config, 'eval_max_retain', 512))
    bs = int(getattr(config, 'eval_batch_size', 16))
    paper_bs = int(getattr(config, 'mia_paper_batch_size', 32))

    # 1. MIA (per-sample + paper-style)
    try:
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
        cossim_ds = _subsample_dataset(dataset['retain'], eval_max_retain, int(config.random_seed))
        cossim_re = cosine_sim_models(model_unlearn, model_retrained, cossim_ds, device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  CosSim failed: {e}")
        cossim_re = float('nan')

    # 3. Test + Forget perf
    try:
        test_m = perf_metrics(model_unlearn, dataset['test'], device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Test perf failed: {e}")
        test_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    try:
        forget_m = perf_metrics(model_unlearn, dataset['forget'], device, config, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Forget perf failed: {e}")
        forget_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan')}

    measured_gpu_peak = (torch.cuda.max_memory_allocated() / 1e9
                         if torch.cuda.is_available() else 0.0)
    gpu_peak = (timing or {}).get('method_gpu_peak_gb', measured_gpu_peak)
    final_eval_h = (time.time() - _eval_start) / 3600

    t = timing or {}
    train_h   = t.get('train_h', elapsed_h)
    fisher_h  = t.get('fisher_h', 0.0)
    monitor_h = t.get('monitor_h', 0.0)
    ckpt_h    = t.get('ckpt_h', 0.0)
    load_h    = t.get('load_h', 0.0)
    wall_h    = t.get('wall_h', elapsed_h)
    adapter_init_h = t.get('adapter_init_h', 0.0)
    core_h = train_h + fisher_h + adapter_init_h
    selection_h = monitor_h + ckpt_h
    method_total_h = core_h + selection_h
    total_wall_h = load_h + method_total_h + final_eval_h

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
        'efficiency/unlearn_time_hours': round(method_total_h, 3),
        'efficiency/gpu_peak_GB':        round(gpu_peak, 2),
        'efficiency/trainable_params':   int(trainable),
        'efficiency/total_params':       int(total_params),
        'efficiency/trainable_ratio':    round(trainable / total_params, 5),
        # --- bóc tách thời gian (thêm ở CUỐI dict để CSV schema-migration an toàn) ---
        'efficiency/unlearn_core_hours': round(core_h, 3),
        'efficiency/unlearn_total_hours': round(method_total_h, 3),
        'efficiency/train_hours':        round(train_h, 3),
        'efficiency/fisher_hours':       round(fisher_h, 3),
        'efficiency/adapter_init_hours': round(adapter_init_h, 3),
        'efficiency/selection_hours':    round(selection_h, 3),
        'efficiency/monitor_hours':      round(monitor_h, 3),
        'efficiency/ckpt_hours':         round(ckpt_h, 3),
        'efficiency/final_eval_hours':   round(final_eval_h, 3),
        'efficiency/load_hours':         round(load_h, 3),
        'efficiency/total_wall_hours':   round(total_wall_h, 3),
    }
    try:
        wandb.log(results)
    except Exception:
        pass

    # ---- Compact summary (KHÔNG còn cột PAPER TARGET) ----
    tag = (f"{method} / {checkpoint_kind} / {os.path.basename(config.forget_set_path)} "
           f"/ seed {int(config.random_seed)}")
    print("\n" + "─" * 60)
    print(f" FINAL EVAL — {tag}")
    print("─" * 60)
    print(f"  Forget    AUC {forget_m['AUC']:.3f}(↓)   F1 {forget_m['Macro_F1']:.3f}   CE {forget_ce:.3f}")
    print(f"  Test      AUC {test_m['AUC']:.3f}(↑)   F1 {test_m['Macro_F1']:.3f}   CE {test_ce:.3f}")
    print(f"  MIA       persample {mia:.3f}(↓)   paper {mia_paper:.3f}(↓)")
    print(f"  1−CosSim(re) {1 - cossim_re:.3f}(↓)")
    print("─" * 60)
    print(f"  Selected  epoch={selected_epoch}  {selection_metric}={selection_value}")
    print(f"  Time      method(total) {method_total_h:.3f}h = core {core_h:.3f} + selection {selection_h:.3f}")
    print(f"            overhead: monitor {monitor_h:.3f}  ckpt {ckpt_h:.3f}  "
          f"load {load_h:.3f}  eval {final_eval_h:.3f}  → wall {wall_h:.3f}h")
    print(f"  Compute   GPU {gpu_peak:.2f} GB   Trainable {trainable:,} ({100 * trainable / total_params:.2f}%)")
    print("─" * 60)

    # CSV summary (cùng format LoKU để Cell 5 notebook gộp được). Lưu ở PARENT-của-parent
    # của output_dir → /kaggle/working/results_summary.csv (khớp restore/push của notebook).
    if csv_path is None:
        csv_path = os.path.normpath(os.path.join(output_dir, "..", "..", "results_summary.csv"))
    row = {**{k.split('/')[-1]: v for k, v in results.items()},
           'id': str(getattr(config, 'id', '')),
           'forget_pct': os.path.basename(config.forget_set_path),
           'seed': int(config.random_seed),
           'method': method,
           'checkpoint': checkpoint_kind,
           'selected_epoch': selected_epoch,
           'selection_metric': selection_metric,
           'selection_value': selection_value,
           'training_passes': training_passes,
           'update_epochs': update_epochs,
           'gpu_name': (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'),
           'torch_version': torch.__version__,
           'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    if row_extra:
        row.update(row_extra)
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    _append_row_csv(csv_path, row)              # tự migrate header + ghi đúng thứ tự cột
    print(f"💾 CSV: {csv_path}")

    # Auto-tracker MD (nếu --exp passed)
    if tracker is not None and finalize_tracker:
        tracker_results = {
            'MIA':           float(mia) if mia == mia else float('nan'),
            'MIA_paper':     float(mia_paper) if mia_paper == mia_paper else float('nan'),
            'Df_AUC':        float(forget_m['AUC']),
            'Df_F1':         float(forget_m['Macro_F1']),
            'Dt_AUC':        float(test_m['AUC']),
            'Dt_F1':         float(test_m['Macro_F1']),
            'dist_vs_re':    float(1 - cossim_re) if cossim_re == cossim_re else float('nan'),
            'time_h':        float(method_total_h),
            'gpu_gb':        float(gpu_peak),
            'trainable_pct': 100.0 * trainable / total_params,
        }
        try:
            tracker.finalize(tracker_results, method_total_h)
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
    # Khớp ĐÚNG hàm mất mát: loss = alpha*L_ukr + beta*L_uu + theta*L_md + gamma*L_mkr
    # (trước đây tên run gán nhầm gamma cho MD và theta cho MKR).
    run_name = (f"{alpha}_UKR_{beta}_UU_{theta}_MD_{gamma}_MKR_"
                f"mean{config.noise_mean}_std{config.noise_std}_{config.use_noise}_"
                f"{timestamp}_partials_euc")   # bỏ hậu tố 'hinge': UR/MR không còn cap
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
    _load_start = time.time()
    print(f"📦 Loading models...")
    model_og = ImageTextModel.from_pretrained(config.base_model_path).to(device)
    model_unlearn = copy.deepcopy(model_og)
    # Try to load retrained; on failure fallback to model_og as dummy so the script
    # can still run (per-epoch CosSim eval will be vs model_og — meaningless but doesn't
    # crash). Happens when forget% has no gold retrained model on disk (e.g., 6%/10%).
    try:
        model_retrained = ImageTextModel.from_pretrained(config.retrained_model_path).to(device)
        gold_retrained_available = (
            os.path.abspath(config.retrained_model_path) != os.path.abspath(config.base_model_path)
        )
    except Exception as e:
        print(f"⚠️  Cannot load retrained from {config.retrained_model_path}")
        print(f"   Reason: {type(e).__name__}: {str(e)[:200]}")
        print(f"   → Using model_og as DUMMY retrained — CosSim metrics will be INVALID (dist_vs_re ≈ 0)")
        model_retrained = copy.deepcopy(model_og).to(device)
        gold_retrained_available = False

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
    load_h = (time.time() - _load_start) / 3600

    # ---- Train ----
    timing = unlearn(
        config, output_dir, device, model_og, model_unlearn, model_retrained,
        optimizer, optimizer_grouped_parameters, scheduler, tokenizer,
        dataset, num_labels, alpha, beta, theta, gamma
    )
    timing['fisher_h'] = 0.0                  # baseline full-FT: không có Fisher/PEFT setup
    timing['load_h'] = load_h
    timing['method_gpu_peak_gb'] = (torch.cuda.max_memory_allocated() / 1e9
                                    if torch.cuda.is_available() else 0.0)
    elapsed_h = timing['wall_h']              # headline wall (giữ tương thích CSV cũ)

    # Free training-only memory
    del model_og, optimizer
    torch.cuda.empty_cache()

    # ---- Final evaluation: one training run, two checkpoint policies ----
    dual_checkpoint_eval = _as_bool(getattr(config, 'evaluate_last_and_best', False))
    common_eval = dict(
        config=config, output_dir=output_dir, device=device,
        model_retrained=model_retrained, dataset=dataset, elapsed_h=elapsed_h,
        trainable=trainable, total_params=total_params, timing=timing,
        training_passes=timing.get('training_passes'),
        update_epochs=timing.get('update_epochs'),
        row_extra={'gold_retrained_available': gold_retrained_available},
        csv_path=getattr(config, 'results_csv_path', None),
    )
    # T_eval: đánh giá cuối (last + val_best) — GIAO THỨC, không vào T_core.
    from training.adv_common import (CudaTimer as _CT, reset_gpu_peak as _rp,
                                     get_gpu_peak as _gp, save_timing_json as _sj)
    _rp()
    _tev = _CT(); _tev.__enter__()
    _final_evaluation(
        model_unlearn=model_unlearn, tracker=None if dual_checkpoint_eval else tracker,
        checkpoint_kind='last', selected_epoch=(timing.get('last_epoch', -1) + 1),
        selection_value=timing.get('last_val_ce'), finalize_tracker=not dual_checkpoint_eval,
        **common_eval,
    )

    if dual_checkpoint_eval:
        best_payload = torch.load(timing['best_checkpoint'], map_location='cpu',
                                  weights_only=False)
        model_unlearn.load_state_dict(best_payload['model_state'])
        best_epoch = int(best_payload['epoch'])
        best_val_ce = float(best_payload['val_ce'])
        del best_payload
        torch.cuda.empty_cache()
        _final_evaluation(
            model_unlearn=model_unlearn, tracker=tracker,
            checkpoint_kind='val_best', selected_epoch=(best_epoch + 1),
            selection_value=best_val_ce, finalize_tracker=True,
            **common_eval,
        )

    # ---- chốt T_eval + ghi timing_result.json (cùng chuẩn với P3) ----
    _tev.__exit__(None, None, None)
    timing['eval_seconds'] = _tev.elapsed
    timing.setdefault('peaks', {})['eval'] = _gp()
    print(f"⏱  T_eval = {_tev.elapsed:.1f}s ({_tev.elapsed/3600:.4f}h) — giao thức, KHÔNG vào core")
    _rid = str(getattr(config, 'id', 'forgetmi'))
    _sj(os.path.join(output_dir, f'timing_baseline_partial_{_rid}.json'),
        'baseline_partial', config, timing, trainable, total_params,
        extra={'run_id': _rid,
               'core_definition': 'T_core(Forget-MI) = T_train (khong co Fisher/FILA)',
               'monitor_seconds': float(timing.get('monitor_h', 0.0)) * 3600,
               'ckpt_io_seconds': float(timing.get('ckpt_h', 0.0)) * 3600,
               'best_epoch': timing.get('best_epoch'),
               'last_epoch': timing.get('last_epoch')})

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
