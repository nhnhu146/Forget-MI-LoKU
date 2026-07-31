#!/usr/bin/env python3
"""
adv_common.py — Nền dùng CHUNG cho nhánh ablation nâng cao (P3, P4, P5, P6, main).
================================================================================
File này ĐỘC LẬP với ``forgetmi_loku.py`` (LoKU cũ giữ nguyên 100%). Mọi phương
pháp mới (``forgetmi_p3/p4/p5/p6/main.py``) import từ đây để bảo đảm:

  * data split / holdout / seed / metric / selector GIỐNG HỆT nhau → ablation
    công bằng, và đủ MỌI metric mà Forget-MI báo cáo (thêm được, không thiếu);
  * so sánh với Forget-MI ở CÙNG protocol: 30 epoch, cùng data, cùng eval.

Protocol chốt (xem trao đổi thiết kế):
  - 30 epoch (ngân sách của Forget-MI, KHÔNG phải "30 epoch kiểu LoKU"); LoKU
    dùng ngân sách/tiêu chí dừng khác giữa TDEC và TOFU.
  - cadence cập nhật theo code gốc Forget-MI: gradient TÍCH LŨY qua các batch trong
    epoch, optimizer.step() MỘT lần ở cuối epoch → total_optimizer_steps = số epoch.
  - evaluate_last_and_best: 'selected' (chọn bằng validation) = kết quả chính,
    'last' (E29) = phân tích ổn định / over-forget.
  - use_noise=true (μ=0, σ=0.1): tham chiếu og_rnd là bản NHIỄU thật sự.

BÁM SÁT HAI BÀI GỐC (không được lệch):
  - Forget-MI Eq.(1)-(2): L_UU = −Dist([F_ul(I_f),F_ul(T_f)], [F_og(Ĩ_f),F_og(T̃_f)]),
    L_MU = −Dist(F_ul((I_f,T_f)), F_og((Ĩ_f,T̃_f))) — DẤU ÂM của khoảng cách Euclid tới
    bản NHIỄU, KHÔNG dùng hinge có chặn / margin.
  - Forget-MI Eq.(3)-(4): L_UR, L_MR = khoảng cách Euclid trên D_r, KHÔNG cap min(d,m),
    KHÔNG margin, KHÔNG "lực kéo nhẹ".
  - LoKU Eq.(2): empirical Fisher = trung bình BÌNH PHƯƠNG GRADIENT TỪNG MẪU.
  - LoKU Sec.3.4: F_rel = F_f/(F_r+ε); trọng số hàng = sqrt(row-wise SUM) (không phải mean);
    W* = W − B*A*.
  - LoKU Sec.3.5: CHỈ θ_FILA (LoRA A,B) được tối ưu → classifier heads đóng băng.
  - Fusion gate: paper Forget-MI KHÔNG quy định gate dùng chung hay riêng, train hay eval
    → theo CODE GỐC: tạo mới mỗi batch, riêng từng vai trò, train mode, KHÔNG vào optimizer.

Holdout cố định (seed cố định, chung cho mọi phương pháp):
  D_f            : tập cần quên (subject trong forget_set_*.csv)
  D_r            = train − D_f, chia:
      r_heldout    : member giữ ngoài cập nhật unlearning → lớp member của MIA
                     (D_r_attack) + val_ce (chỉ dùng cho ablation selector)
      retain       : D_r_unlearning → UR/MR/CE/KD
  test           : chia theo BỆNH NHÂN + phân tầng nhãn:
      sel          : 25% (og-unseen) → D_nm_val (non-member MIA) VÀ
                     D_utility_val (AUC/F1 og-vs-ul) khi CHỌN checkpoint
      test_final   : 75% → CHỈ báo cáo cuối (MIA/AUC/F1...)
  random         : bản nhiễu của D_f (ảnh Gaussian σ + text synonym)

Selector S_val (PROTOCOL RIÊNG CỦA LUẬN VĂN — không phải thành phần của Forget-MI hay
LoKU; dùng CHUNG cho P3–P6 và main, chọn nhỏ nhất):
  S_val = w_f·G_forget + w_p·G_privacy + w_u·G_utility ,   mỗi G ∈ [0,1]
  G_forget  = ⅓(IHL/2 + d⁰_u/(d_u+d⁰_u) + d⁰_m/(d_m+d⁰_m))   (đánh giá trên D_f, no-grad)
      d_u,d_m = khoảng cách Euclid đơn/đa phương thức tới tham chiếu og(nhiễu);
      d⁰_u,d⁰_m = CHUẨN HÓA tham chiếu đo tại epoch 0 (chỉ để đưa G về [0,1] cho
      selector — KHÔNG phải margin, KHÔNG xuất hiện trong hàm mất mát).
  G_privacy = MIA_val  (SVM member=D_r_attack, non-member=D_nm_val, đoán D_f; nhánh ảnh)
  G_utility = ½[clip(g_A,0,1)+clip(g_F,0,1)],  g_A=max(0,AUC_og−AUC_ul−δ_A)/(1−δ_A)
  → 0 là tốt nhất ở cả ba thành phần.

Ghi chú luận văn: MIA & G_privacy dựa trên NHÁNH ẢNH (đồng nhất repo); báo cáo
phải nói rõ đây là đánh giá privacy dựa trên nhánh ảnh, chưa đủ chứng minh privacy
đa phương thức.
"""
import os
import sys
import csv
import random
from functools import reduce

# Cho phép import package gốc khi chạy từ repo root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Sampler, Subset

from joint_img_txt.model_utils import (
    CXRImageTextDataset, EdemaClassificationProcessor, EdemaMultiLabelClassificationProcessor,
    RandomTranslateCrop, CenterCrop,
)
from joint_img_txt.convert_examples_to_features import (
    convert_examples_to_features, convert_examples_to_features_multilabel,
)


# ============================================================================
# Utilities cơ bản (sao y forgetmi_loku.py — đã kiểm chứng)
# ============================================================================

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {'1', 'true', 'yes', 'on'}
    return bool(value)


def euclidean_distance(a, b):
    """Khoảng cách Euclid theo TỪNG MẪU (không .mean()) — [N]."""
    return torch.sqrt(torch.sum((a - b) ** 2, dim=1) + 1e-8)


class AlignedSampler(Sampler):
    """Giữ thứ tự căn chỉnh giữa forget/rand (giống Forget-MI)."""
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
    """Auto-cast input floating để khớp dtype model (tránh lỗi fp16/fp32 ngoài autocast)."""
    dtype = _model_dtype(model)
    casted = {
        k: (v.to(dtype) if torch.is_tensor(v) and v.is_floating_point() and v.dtype != dtype else v)
        for k, v in inputs.items()
    }
    return model(**casted)


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


def resolve_image_targets(model, last_k_blocks, include_fc1=False, total_layers=7):
    """Tên đầy đủ các Conv2d ở k residual-stage cuối của image encoder (+ fc1 tùy chọn).
    Dùng cho cả PEFT target_modules (exact) lẫn Fisher/FILA (substring)."""
    targets = []
    if last_k_blocks and last_k_blocks > 0:
        sel = set(range(max(1, total_layers - last_k_blocks + 1), total_layers + 1))
        prefixes = tuple(f"img_model.layer{n}." for n in sel)
        for name, mod in model.named_modules():
            if isinstance(mod, torch.nn.Conv2d) and name.startswith(prefixes):
                targets.append(name)
    if include_fc1:
        targets.append('img_model.fc1')
    return targets


# ============================================================================
# DATA SPLIT — 4 tập holdout (theo bệnh nhân + phân tầng nhãn)
# ============================================================================

def _grouped_stratified_holdout(keys, groups, labels, n_splits, seed):
    """Trả về (heldout_keys, rest_keys) — 1 fold làm heldout, phần còn lại là rest.
    Ưu tiên StratifiedGroupKFold (grouped + stratified). Fallback GroupShuffleSplit
    (chỉ grouped) rồi cuối cùng train_test_split (chỉ stratified) kèm cảnh báo.

    keys/groups/labels: list cùng độ dài. n_splits: 4 → ~25% heldout; 10 → ~10%."""
    keys = list(keys); groups = list(groups); labels = list(labels)
    idx = np.arange(len(keys))
    try:
        from sklearn.model_selection import StratifiedGroupKFold
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        # .split yields (train_idx, test_idx); test_idx = 1 fold ≈ 1/n_splits → làm heldout
        held_idx = None
        for _train_idx, test_idx in sgkf.split(idx, labels, groups):
            held_idx = test_idx
            break
    except Exception as e:
        try:
            from sklearn.model_selection import GroupShuffleSplit
            print(f"⚠️  StratifiedGroupKFold không dùng được ({e}) → GroupShuffleSplit (grouped, chưa stratify).")
            gss = GroupShuffleSplit(n_splits=1, test_size=1.0 / n_splits, random_state=seed)
            _rest, held_idx = next(gss.split(idx, labels, groups))
        except Exception as e2:
            from sklearn.model_selection import train_test_split
            print(f"⚠️  GroupShuffleSplit lỗi ({e2}) → train_test_split (stratify, KHÔNG group theo bệnh nhân).")
            rest_i, held_i = train_test_split(idx, test_size=1.0 / n_splits,
                                              random_state=seed, stratify=labels)
            held_idx = held_i
    held_set = set(int(i) for i in held_idx)
    heldout = [keys[i] for i in range(len(keys)) if i in held_set]
    rest = [keys[i] for i in range(len(keys)) if i not in held_set]
    return heldout, rest


def data_split_advanced(split_list_path, forget_ids_path, seed=42,
                        test_sel_splits=4, retain_heldout_splits=10):
    """Đọc split CSV gốc và tạo các tập holdout theo protocol nhánh nâng cao.

    Cột CSV (giống Forget-MI): [subject_id(0), dicom_id(1), study_id/report_id(2),
    edeme_severity(3), ..., split(-1)].

    Trả về dict map {report_id: value/label} cho các tập:
      retain      (= D_r_unlearning)   : dùng cho UR/MR/CE/KD
      r_heldout   (= D_r_attack/val)   : member giữ ngoài cập nhật unlearning
      sel         (= D_nm_val/util)    : 25% test, og-unseen, dùng khi CHỌN checkpoint
      test_final                        : 75% test, chỉ báo cáo cuối
      forget (D_f) / random (nhiễu D_f)
    """
    random.seed(0)
    forget_set = pd.read_csv(forget_ids_path).astype(str).subject_id.values

    # gom raw theo vai trò; giữ subject_id để group theo bệnh nhân
    train_l, train_ids, train_grp = {}, {}, {}
    test_l, test_ids, test_grp = {}, {}, {}
    forget_l, forget_ids = {}, {}
    rand_l, rand_ids = {}, {}

    with open(split_list_path, 'r') as f:
        reader = csv.reader(f); hdr = next(reader) or []
        ix = {str(n).strip().lower(): i for i, n in enumerate(hdr)}      # đọc theo TÊN cột (bền IU)
        i_sub = ix.get('subject_id', 0); i_key = ix.get('dicom_id', 2)   # key = dicom_id (ảnh)
        i_val = ix.get('study_id', ix.get('report_id', 1))               # value = study/text id
        i_lab = ix.get('edeme_severity', ix.get('label', ix.get('severity', 3)))
        i_spl = ix.get('fold', ix.get('split', len(hdr) - 1))
        for row in reader:
            if len(row) <= max(i_sub, i_key, i_val, i_lab, i_spl):
                continue
            try:
                sev = float(row[i_lab])
            except ValueError:
                continue
            rid = row[i_key]; subj = row[i_sub]
            if subj in forget_set:
                forget_l[rid] = [sev]; forget_ids[rid] = row[i_val]
                rand_l[rid] = [sev];   rand_ids[rid] = row[i_val]
            elif str(row[i_spl]).strip().upper() == 'TEST':
                test_l[rid] = [sev]; test_ids[rid] = row[i_val]; test_grp[rid] = subj
            else:
                train_l[rid] = [sev]; train_ids[rid] = row[i_val]; train_grp[rid] = subj

    # ---- chia TEST → sel (25%) + test_final (75%) theo bệnh nhân + phân tầng ----
    t_keys = list(test_ids.keys())
    sel_keys, tf_keys = _grouped_stratified_holdout(
        t_keys, [test_grp[k] for k in t_keys], [test_l[k][0] for k in t_keys],
        n_splits=test_sel_splits, seed=seed)

    # ---- chia D_r → r_heldout (10%) + retain (D_r_unlearning) ----
    r_keys = list(train_ids.keys())
    held_keys, ret_keys = _grouped_stratified_holdout(
        r_keys, [train_grp[k] for k in r_keys], [train_l[k][0] for k in r_keys],
        n_splits=retain_heldout_splits, seed=seed)

    def _submap(keys, id_map, lab_map):
        return ({k: id_map[k] for k in keys}, {k: lab_map[k] for k in keys})

    retain_id, retain_lab = _submap(ret_keys, train_ids, train_l)
    heldout_id, heldout_lab = _submap(held_keys, train_ids, train_l)
    sel_id, sel_lab = _submap(sel_keys, test_ids, test_l)
    tf_id, tf_lab = _submap(tf_keys, test_ids, test_l)

    print(f"[split] retain(unlearn)={len(retain_id)}  r_heldout(member)={len(heldout_id)}  "
          f"forget={len(forget_ids)}  test_sel(25%)={len(sel_id)}  test_final(75%)={len(tf_id)}")
    return {
        'retain':     (retain_id, retain_lab),
        'r_heldout':  (heldout_id, heldout_lab),
        'sel':        (sel_id, sel_lab),
        'test_final': (tf_id, tf_lab),
        'forget':     (forget_ids, forget_l),
        'random':     (rand_ids, rand_l),
    }


# ---- text-feature cache (sao y forgetmi_loku.py: xử lý Kaggle read-only + regen khi mismatch) ----

def _cache_names(args):
    cf = f"cachedfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}"
    cnf = f"cachednoisyfeatures_train_seqlen-{args.max_seq_length}_{args.output_channel_encoding}"
    return cf, cnf


def _writable_cache_dir():
    return ('/kaggle/working/text_cache_regen' if os.path.isdir('/kaggle/working')
            else os.path.join(os.getcwd(), 'text_cache_regen'))


def _make_processor(args):
    return (EdemaMultiLabelClassificationProcessor()
            if args.output_channel_encoding == 'multilabel'
            else EdemaClassificationProcessor())


def _regen_text_features(args, tokenizer, dst_dir):
    """Rebuild features từ all_data.tsv vào dst_dir (phải writable). Trả (feats, noisy_feats)."""
    processor = _make_processor(args)
    get_features = (convert_examples_to_features_multilabel
                    if args.output_channel_encoding == 'multilabel'
                    else convert_examples_to_features)
    cache_fname, cache_noisy_fname = _cache_names(args)
    src_tsv = os.path.join(args.text_data_dir, 'all_data.tsv')
    if not os.path.exists(src_tsv):
        raise FileNotFoundError(
            f"Thiếu {src_tsv} để regenerate cache. Upload all_data.tsv vào 'metadata/'.")
    os.makedirs(dst_dir, exist_ok=True)
    synonyms = pd.read_csv(args.synonyms_dir)
    examples = processor.get_all_examples(args.text_data_dir)
    noisy_examples = processor.get_noisy_examples(args.text_data_dir, synonyms, args.text_noise_level)
    feats = get_features(examples, processor.get_labels(), args.max_seq_length, tokenizer)
    noisy_feats = get_features(noisy_examples, processor.get_labels(), args.max_seq_length, tokenizer)
    torch.save(feats, os.path.join(dst_dir, cache_fname))
    torch.save(noisy_feats, os.path.join(dst_dir, cache_noisy_fname))
    return feats, noisy_feats


def _build_text_features(args, tokenizer):
    processor = _make_processor(args)
    cache_fname, cache_noisy_fname = _cache_names(args)
    cached = os.path.join(args.text_data_dir, cache_fname)
    cached_noisy = os.path.join(args.text_data_dir, cache_noisy_fname)
    if os.path.exists(cached) and os.path.exists(cached_noisy) and not args.reprocess_input_data:
        print("📂 Loading cached features...")
        features = torch.load(cached, weights_only=False)
        noisy_features = torch.load(cached_noisy, weights_only=False)
    else:
        print("🔨 Generating features (lần đầu)...")
        try:
            features, noisy_features = _regen_text_features(args, tokenizer, args.text_data_dir)
        except (PermissionError, OSError, RuntimeError):
            wd = _writable_cache_dir()
            print(f"⚠️  text_data_dir read-only → regenerating to {wd}")
            features, noisy_features = _regen_text_features(args, tokenizer, wd)
    return processor, features, noisy_features


def build_dataset(args, tokenizer):
    """Xây dựng các dataset theo protocol 4-tập holdout. Trả (datasets, num_labels)."""
    processor, features, noisy_features = _build_text_features(args, tokenizer)
    num_labels = len(processor.get_labels())

    all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in features}
    noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in noisy_features}

    def _resolve(r, txt_map):
        # dung sai prefix 's' + int/str: cache report_id có thể là int trong khi split là str.
        s = str(r).replace('s', '')
        cands = [r, str(r), f"s{r}", s]
        if s.isdigit():
            try:
                cands.append(int(s))
            except ValueError:
                pass
        for cand in cands:
            if cand in txt_map:
                return cand
        return None

    splits = data_split_advanced(
        args.data_split_path, args.forget_set_path,
        seed=int(getattr(args, 'random_seed', 42)),
        test_sel_splits=int(getattr(args, 'test_sel_splits', 4)),
        retain_heldout_splits=int(getattr(args, 'retain_heldout_splits', 10)),
    )

    # cache integrity check (giống loku) trên tập retain: nếu cache report_ids KHÔNG khớp
    # split → cache stale → REGENERATE từ all_data.tsv sang writable dir (Kaggle input read-only).
    retain_ids = splits['retain'][0]
    sample_ids = list(retain_ids.values())[:50]
    matches = sum(1 for rid in sample_ids if _resolve(rid, all_txt))
    if sample_ids and matches == 0:
        wd = _writable_cache_dir()
        cache_fname, cache_noisy_fname = _cache_names(args)
        re_cached = os.path.join(wd, cache_fname)
        re_cached_noisy = os.path.join(wd, cache_noisy_fname)
        if os.path.exists(re_cached) and os.path.exists(re_cached_noisy):
            print(f"⚠️  Cache mismatch (0/{len(sample_ids)}) → dùng lại regen cache tại {wd}")
            features = torch.load(re_cached, weights_only=False)
            noisy_features = torch.load(re_cached_noisy, weights_only=False)
        else:
            print(f"⚠️  Cache mismatch (0/{len(sample_ids)}) → regenerate từ all_data.tsv sang {wd} (~5 phút)...")
            features, noisy_features = _regen_text_features(args, tokenizer, wd)
        all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in features}
        noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id) for f in noisy_features}
        matches = sum(1 for rid in sample_ids if _resolve(rid, all_txt))
        if matches == 0:
            raise RuntimeError(
                f"Sau regenerate vẫn 0/{len(sample_ids)} match. Kiểm tra data_split="
                f"{args.data_split_path} và {args.text_data_dir}/all_data.tsv.")
        print(f"   ✅ Sau regen: {matches}/{len(sample_ids)} match.")

    def extract(mapping, ids):
        t, m, s, l = {}, {}, {}, {}
        for rid in ids:
            key = _resolve(rid, mapping)
            t[rid], m[rid], s[rid], l[rid] = mapping[key]
        return t, m, s, l

    train_trans = RandomTranslateCrop(2048)
    eval_trans = CenterCrop(2048)

    # use_noise: tập 'random' là BẢN NHIỄU (Ĩ_f, T̃_f) của D_f dùng làm THAM CHIẾU cho UU/MU.
    #   use_noise=true  → og_rnd = og(ảnh nhiễu Gaussian σ + text perturbation)
    #                     [SETTING CHÍNH — đúng Forget-MI Eq.(1)-(2), mặc định]
    #   use_noise=false → og_rnd = og(D_f SẠCH hoàn toàn)   [chỉ dùng cho ABLATION 'no_noise']
    rand_perturb = as_bool(getattr(args, 'use_noise', True))
    rand_txt = noisy_txt if rand_perturb else all_txt
    # (tên_dataset, key_split, transform, txt_map, perturb_img)
    spec = [
        ('retain',     'retain',     train_trans, all_txt,   False),
        ('r_heldout',  'r_heldout',  eval_trans,  all_txt,   False),
        ('sel',        'sel',        eval_trans,  all_txt,   False),
        ('test_final', 'test_final', eval_trans,  all_txt,   False),
        ('forget',     'forget',     train_trans, all_txt,   False),
        ('random',     'random',     train_trans, rand_txt,  rand_perturb),
    ]
    image_noise_params = {'mean': float(getattr(args, 'noise_mean', 0.0)),
                          'std': float(getattr(args, 'noise_std', 0.1))}
    datasets = {}
    for name, key, trans, txt_map, perturb in spec:
        ids_map, lab_map = splits[key]
        valid = [d for d, r in ids_map.items() if _resolve(r, txt_map)]
        if len(valid) < len(ids_map):
            print(f"⚠️  {len(ids_map) - len(valid)} item ở '{name}' bị bỏ (không thấy text)")
        f_ids = {d: ids_map[d] for d in valid}
        f_labels = {d: lab_map[d] for d in valid}
        tk, mk, sg, lb = extract(txt_map, f_ids.values())
        datasets[name] = CXRImageTextDataset(
            args.id, tk, mk, sg, lb, f_ids, args.img_data_dir, f_labels,
            args.data_split_path, transform=trans,
            perturb_img=perturb, noise_params=(image_noise_params if perturb else None),
            output_channel_encoding=args.output_channel_encoding,
        )
        print(f"  [{name}] {len(datasets[name])} samples")
    # Fail-fast rõ ràng nếu tập cốt lõi rỗng (thay vì lỗi 'need at least one array' sâu ở eval).
    empty = [n for n in ('retain', 'forget', 'sel', 'test_final') if len(datasets[n]) == 0]
    if empty:
        raise RuntimeError(
            f"Dataset rỗng {empty} (0 sample) — text/cache KHÔNG khớp split. "
            f"Kiểm tra: (1) {args.text_data_dir}/all_data.tsv có tồn tại để regen? "
            f"(2) cache report_id có khớp study_id trong {args.data_split_path}? "
            f"Cache regen tại {_writable_cache_dir()}.")
    return datasets, num_labels


# ============================================================================
# Fisher Information & FILA (LoKU) — hỗ trợ rank không đồng nhất cho P6
# ============================================================================

def compute_fisher_importance(model, dataloader, device, target_modules, args,
                              max_samples=256):
    """Empirical Fisher ĐÚNG LoKU Eq.(2) — trung bình BÌNH PHƯƠNG GRADIENT TỪNG MẪU:

        F̂ = (1/N) · Σ_{i=1..N} ( ∂ℓ_i/∂θ )² ,   ℓ_i = CE_img,i + CE_txt,i

    (KHÔNG phải ( ∂/∂θ mean_i ℓ_i )² như bản cũ — bình phương của gradient trung bình
    batch làm mất phương sai giữa các mẫu và KHÔNG phải Fisher. Cũng KHÔNG dùng mẹo
    fisher_batch_size=1 để lách công thức: forward vẫn theo batch, chỉ gradient là
    per-sample qua autograd.grad trên đồ thị dùng chung.)"""
    model.eval()
    params = {name: p for name, p in model.named_parameters()
              if any(tm in name for tm in target_modules) and 'weight' in name and p.requires_grad}
    names = list(params.keys())
    plist = [params[n] for n in names]
    importance = {name: torch.zeros_like(params[name].data) for name in names}
    if not names:
        print("⚠️  Không có tham số mục tiêu cho Fisher. Trả zeros.")
        return importance
    n = 0
    for batch in tqdm(dataloader, desc="Fisher(per-sample)"):
        if n >= max_samples:
            break
        inputs, labels, _ = get_model_inputs(args, batch, device)
        labels = labels.long().view(-1)
        outputs = model(**inputs)
        img_logits, txt_logits = outputs[1], outputs[3]
        bs = int(batch[0].size(0))
        for i in range(bs):
            if n >= max_samples:
                break
            # ℓ_i của RIÊNG mẫu i (không lấy trung bình batch)
            loss_i = (F.cross_entropy(img_logits[i:i + 1], labels[i:i + 1])
                      + F.cross_entropy(txt_logits[i:i + 1], labels[i:i + 1]))
            grads = torch.autograd.grad(loss_i, plist, retain_graph=(i < bs - 1),
                                        allow_unused=True)
            for name, g in zip(names, grads):
                if g is not None:
                    importance[name] += g.detach().pow(2)
            n += 1
        del outputs, img_logits, txt_logits
    if n == 0:
        print("⚠️  Không có sample cho Fisher. Trả zeros.")
        return importance
    print(f"   Fisher: {n} mẫu (gradient per-sample, F̂ = Σ g_i²/N)")
    return {k: v / n for k, v in importance.items()}


def _fila_decompose(W2d, rel2d, r, target_r, peft_scaling=1.0):
    """Phân rã low-rank có trọng số Fisher theo hàng (row-wise WLRA của LoKU Sec. 3.4).
    Trả (A_pad, B_pad, sub2d). target_r = số slot rank PEFT cấp; effective rank clamp
    cho ma trận nhỏ.

    Trọng số hàng ĐÚNG LoKU: s_i = sqrt( Σ_j F_rel[i,j] )  ("square-root of the row-wise
    SUM of F_rel"). Bản cũ dùng mean(dim=1) → sai hệ số 1/√(in_dim) và làm phẳng chênh
    lệch giữa các hàng khi in_dim khác nhau.

    CHIA S CHO peft_scaling (=α/r) TRƯỚC khi khai căn — đúng `S = S / scaling['default']`
    của code gốc LoKU (TOFU/forget.py:279). PEFT nhân α/r vào nhánh LoRA lúc forward, nên
    nếu để nguyên S thì thành phần thực sự đưa vào adapter là (α/r)·B*A* chứ không phải
    B*A*. Sau khi chia:
        peft_scaling · (B_raw @ A_raw) = B*A* = ΔW_FILA
    tức đúng lượng mà paper ký hiệu B*A*, không bị nhân dư α/r."""
    out_dim, in_dim = W2d.shape
    row_imp = rel2d.sum(dim=1).sqrt() + 1e-6
    Wp = row_imp[:, None] * W2d
    r_eff = max(1, min(r, min(out_dim, in_dim) - 1))
    U, S, V = torch.svd_lowrank(Wp, q=r_eff)
    S = S / max(float(peft_scaling), 1e-12)          # <-- bước của code gốc LoKU
    S_sqrt = torch.sqrt(S + 1e-8)
    A_raw = (V * S_sqrt).t()
    B_raw = (U * S_sqrt) / (row_imp[:, None] + 1e-6)
    sub2d = B_raw @ A_raw                            # = ΔW_FILA / peft_scaling
    A_pad = W2d.new_zeros(target_r, in_dim);  A_pad[:r_eff] = A_raw
    B_pad = W2d.new_zeros(out_dim, target_r); B_pad[:, :r_eff] = B_raw
    return A_pad, B_pad, sub2d


def apply_loku_soft_init(model, forget_imp, retain_imp, target_modules,
                         r, init_scale=0.05, subtract_scale=0.0,
                         image_target_names=None, image_subtract_scale=None,
                         rank_pattern=None):
    """FILA init (Linear + Conv2d). subtract_scale>0 = TRUE-LoKU (trừ hướng quên khỏi
    base, LoRA giữ hướng đó → train đẩy LoRA đi = lộ phần đã trừ = unlearning).
    subtract_scale=0 = soft init (base nguyên vẹn, LoRA nhỏ).

    rank_pattern (P6): dict {substring_tên_module: rank_hiệu_dụng}. Với mỗi lớp, dùng
    rank khớp pattern (mặc định = r). Rank hiệu dụng chỉ ảnh hưởng phân rã FILA;
    số slot thực do PEFT cấp (lora_A.shape[0]) — nếu lệch thì clamp về target_r.

    TRẢ VỀ: fila_subtraction = {module_path: tensor B*A* đã trừ khỏi base} (CPU).
    Cần cho việc dựng lại checkpoint 'selected' ĐÚNG reparameterization W* = W − B*A*
    (nếu dựng lại từ pretrained mà không trừ bù thì base sai → checkpoint vô nghĩa)."""
    matched, skipped = 0, 0
    fila_subtraction = {}
    image_set = set(image_target_names) if image_target_names else set()

    def _rank_for(path):
        if rank_pattern:
            for pat, rk in rank_pattern.items():
                if pat in path:
                    return int(rk)
        return int(r)

    for name, f_imp in forget_imp.items():
        r_imp = retain_imp.get(name, torch.zeros_like(f_imp))
        rel = (f_imp / (r_imp + 1e-6)).clamp(0, 1e3)
        path = name.replace('.weight', '')
        try:
            mod = get_module(model, path)
            base = mod.base_layer
            lora_A = mod.lora_A['default']
            lora_B = mod.lora_B['default']
            peft_scaling = mod.scaling['default']
        except (AttributeError, KeyError):
            skipped += 1
            continue

        is_image = path in image_set
        scale = image_subtract_scale if (is_image and image_subtract_scale is not None) else subtract_scale
        r_layer = _rank_for(path)

        W = base.weight.data.float()
        target_r = lora_A.weight.shape[0]
        is_conv = (W.dim() == 4)
        if is_conv:
            out_dim, in_dim, kh, kw = W.shape
            W2d = W.reshape(out_dim, -1)
            rel2d = rel.reshape(out_dim, -1)
        else:
            W2d, rel2d = W, rel

        try:
            A_pad, B_pad, sub2d = _fila_decompose(W2d, rel2d, r_layer, target_r,
                                                  peft_scaling=peft_scaling)
        except Exception:
            skipped += 1
            continue

        if scale > 0:
            # sub2d = ΔW_FILA/(α/r) ⇒ phần trừ = peft_scaling·sub2d·γ = γ·B*A* (đúng LoKU:
            # new_residual = W − scaling · B @ A), không còn dư hệ số α/r.
            subtraction = sub2d * peft_scaling * scale
            if is_conv:
                subtraction = subtraction.reshape(out_dim, in_dim, kh, kw)
            subtraction = subtraction.to(base.weight.dtype)
            base.weight.data -= subtraction
            fila_subtraction[path] = subtraction.detach().cpu().clone()
            sq = scale ** 0.5
            A_set, B_set = A_pad * sq, B_pad * sq
        else:
            A_set, B_set = A_pad * init_scale, B_pad * init_scale

        if is_conv:
            lora_A.weight.data = A_set.reshape(target_r, in_dim, kh, kw).to(lora_A.weight.dtype)
            lora_B.weight.data = B_set.reshape(out_dim, target_r, 1, 1).to(lora_B.weight.dtype)
        else:
            lora_A.weight.data = A_set.to(lora_A.weight.dtype)
            lora_B.weight.data = B_set.to(lora_B.weight.dtype)
        matched += 1
    mode = f"TRUE-SUBTRACT(txt={subtract_scale},img={image_subtract_scale})" \
        if (subtract_scale > 0 or (image_subtract_scale or 0) > 0) else f"soft({init_scale})"
    print(f"✅ FILA init [{mode}] áp dụng {matched} lớp ({skipped} bỏ qua)"
          + (f" | rank_pattern={rank_pattern}" if rank_pattern else ""))
    if fila_subtraction:
        nb = sum(t.numel() * t.element_size() for t in fila_subtraction.values()) / 1e6
        print(f"   💾 lưu FILA subtraction cho {len(fila_subtraction)} lớp ({nb:.0f} MB CPU) "
              f"→ dựng lại checkpoint 'selected' vẫn giữ W* = W − B*A*")
    return fila_subtraction


def apply_fila_subtraction(model, fila_subtraction):
    """Áp lại phép trừ bù W* = W − B*A* lên model vừa dựng từ pretrained (đã bọc PEFT),
    TRƯỚC khi nạp state LoRA đã học. Không có bước này thì base là W chứ không phải W*,
    tức checkpoint 'selected' không còn đúng reparameterization của LoKU."""
    applied, missing = 0, 0
    for path, sub in fila_subtraction.items():
        try:
            base = get_module(model, path).base_layer
        except (AttributeError, KeyError):
            missing += 1
            continue
        base.weight.data -= sub.to(base.weight.device, base.weight.dtype)
        applied += 1
    print(f"♻️  Khôi phục FILA subtraction: {applied} lớp"
          + (f" ({missing} lớp KHÔNG khớp — cảnh báo!)" if missing else ""))
    return applied


# ============================================================================
# Thang chuẩn hóa d⁰ tại epoch 0 — CHỈ dùng cho selector S_val của luận văn
# (KHÔNG phải margin; hàm mất mát UU/MU/UR/MR không dùng biên — xem Forget-MI Eq.(1)-(4))
# ============================================================================

@torch.no_grad()
def compute_epoch0_scales(model_ul, model_og, forget_dl, rand_dl, args, device,
                          max_batches=None):
    """Đo khoảng cách TRUNG BÌNH tại epoch 0 (trước khi unlearning) làm THANG CHUẨN HÓA
    cho G_forget của selector S_val:
      d⁰_u = mean_i ||[F_ul_img,F_ul_txt](D_f) − [F_og_img,F_og_txt](nhiễu)||
      d⁰_m = mean_i ||gate(F_ul(D_f)) − gate(F_og(nhiễu))||
    Vì Euclid phụ thuộc số chiều/thang đo embedding, cần một mốc quy chiếu để đưa
    G_forget về [0,1]. ĐÂY KHÔNG PHẢI MARGIN: không có hằng số này trong loss, loss dùng
    đúng −Dist của Forget-MI.

    Gate được tạo MỚI theo từng batch (như code gốc Forget-MI), nên d⁰_m có thành phần
    ngẫu nhiên của phép hợp nhất; lấy trung bình trên toàn tập D_f để ổn định."""
    # Cùng regime với lúc train: model_ul.train() nhưng BN giữ eval.
    model_ul.train(); set_bn_eval(model_ul); model_og.eval()
    du_all, dm_all = [], []
    it = zip(forget_dl, rand_dl)
    for bi, (fb, rb) in enumerate(it):
        if max_batches is not None and bi >= max_batches:
            break
        f_in, _, _ = get_model_inputs(args, fb, device)
        rnd_in, _, _ = get_model_inputs(args, rb, device)
        # model_ul FP32 (không autocast) — ĐỒNG NHẤT với forward lúc train/selector.
        ul_i, _, ul_t, _ = model_ul(**f_in)[:4]
        with torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            og_i, _, og_t, _ = safe_forward(model_og, rnd_in)[:4]
        ul_i = ul_i.float(); ul_t = ul_t.float(); og_i = og_i.float(); og_t = og_t.float()
        ul_c = torch.cat((ul_i, ul_t), dim=-1)
        og_c = torch.cat((og_i, og_t), dim=-1)
        du_all.append(euclidean_distance(ul_c, og_c).cpu().numpy())
        # Gate MỚI theo batch, riêng từng vai trò — như code gốc Forget-MI
        g = new_batch_gates(ul_i.shape[1], ul_t.shape[1], device, roles=('ul_frg', 'og_rnd'))
        ul_j = g['ul_frg'](ul_i, ul_t)
        og_j = g['og_rnd'](og_i, og_t)
        dm_all.append(euclidean_distance(ul_j, og_j).cpu().numpy())
    du = np.concatenate(du_all) if du_all else np.array([1.0])
    dm = np.concatenate(dm_all) if dm_all else np.array([1.0])
    d_u0 = max(float(du.mean()), 1e-6)
    d_m0 = max(float(dm.mean()), 1e-6)
    print(f"📏 Thang chuẩn hóa selector tại epoch 0: d⁰_u={d_u0:.4f} "
          f"(d_u∈[{du.min():.3f},{du.max():.3f}])  d⁰_m={d_m0:.4f} "
          f"(d_m∈[{dm.min():.3f},{dm.max():.3f}])")
    return d_u0, d_m0


# ============================================================================
# Eval: MIA, CosSim, perf (AUC + Macro-F1 + Pairwise-AUC + F1) — đủ metric Forget-MI
# ============================================================================

def _eval_autocast():
    return torch.autocast(device_type='cuda', enabled=torch.cuda.is_available())


def subsample_dataset(dataset, max_n, seed=42):
    if not max_n or max_n <= 0 or len(dataset) <= max_n:
        return dataset
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(dataset), int(max_n), replace=False)
    return Subset(dataset, idx.tolist())


@torch.no_grad()
def per_sample_ce(model, dataset, device, args, batch_size=32):
    """CE per-sample trên img_logits (đặc trưng MIA nhánh ảnh)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    out = []
    model.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, labels, _ = get_model_inputs(args, batch, device)
        with _eval_autocast():
            outputs = safe_forward(model, inputs)
        logits = outputs[1].float()
        losses = F.cross_entropy(logits, labels.long().view(-1), reduction='none')
        out.append(losses.cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def _batch_means(losses_1d, bs):
    bs = max(1, int(bs))
    return np.array([losses_1d[i:i + bs].mean() for i in range(0, len(losses_1d), bs)])


def run_mia(model, member_ds, nonmember_ds, forget_ds, device, args, batch_size=32,
            seed=42, paper_batch_size=32):
    """MIA (nhánh ảnh). member/non-member huấn luyện SVM, đoán forget.
      'persample' : per-SAMPLE CE, member/non-member cân bằng 1:1 (SVC C=3 rbf).
      'paper'     : per-BATCH-mean CE, không cân bằng (replicate eval_unlearning.py gốc).
    Cả hai = tỉ lệ forget bị đoán 'member'. THẤP = tốt (0=gold, 1=original)."""
    from sklearn.svm import SVC
    ml = per_sample_ce(model, member_ds, device, args, batch_size)
    nl = per_sample_ce(model, nonmember_ds, device, args, batch_size)
    fl = per_sample_ce(model, forget_ds, device, args, batch_size)
    diag = {'member_ce': float(ml.mean()) if len(ml) else float('nan'),
            'nonmember_ce': float(nl.mean()) if len(nl) else float('nan'),
            'forget_ce': float(fl.mean()) if len(fl) else float('nan')}

    ml2, nl2, fl2 = ml.reshape(-1, 1), nl.reshape(-1, 1), fl.reshape(-1, 1)
    n = min(len(ml2), len(nl2))
    rng = np.random.default_rng(seed)
    m_sub = ml2[rng.choice(len(ml2), n, replace=False)]
    n_sub = nl2[rng.choice(len(nl2), n, replace=False)]
    clf = SVC(C=3, kernel='rbf', gamma='auto', random_state=seed)
    clf.fit(np.vstack([m_sub, n_sub]), np.concatenate([np.ones(n), np.zeros(n)]))
    mia_persample = float(clf.predict(fl2).mean())

    try:
        rb = _batch_means(ml, paper_batch_size).reshape(-1, 1)
        tb = _batch_means(nl, paper_batch_size).reshape(-1, 1)
        fb = _batch_means(fl, paper_batch_size).reshape(-1, 1)
        clf2 = SVC(C=3, kernel='rbf', gamma='auto')
        clf2.fit(np.vstack([rb, tb]), np.concatenate([np.ones(len(rb)), np.zeros(len(tb))]))
        mia_paper = float(clf2.predict(fb).mean())
    except Exception as e:
        print(f"   [MIA paper warn] {e}"); mia_paper = float('nan')

    return {'persample': mia_persample, 'paper': mia_paper, **diag}


@torch.no_grad()
def val_mia(model, member_ds, nonmember_ds, forget_ds, device, args, batch_size=32, seed=42):
    """G_privacy tại lúc CHỌN checkpoint: SVM(member=D_r_attack, non-member=D_nm_val),
    đoán D_f. Trả tỉ lệ forget bị đoán 'member' ∈ [0,1] (0 tốt nhất). Nhánh ảnh, cân bằng."""
    res = run_mia(model, member_ds, nonmember_ds, forget_ds, device, args,
                  batch_size=batch_size, seed=seed)
    return float(res['persample'])


@torch.no_grad()
def cosine_sim_models(model_a, model_b, dataset, device, args, batch_size=32):
    """CosSim trung bình của img_logits giữa hai model (1−CosSim vs retrained = metric Forget-MI)."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    sims = []
    model_a.eval(); model_b.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, _, _ = get_model_inputs(args, batch, device)
        with _eval_autocast():
            la = safe_forward(model_a, inputs)[1].float()
            lb = safe_forward(model_b, inputs)[1].float()
        sims.append(F.cosine_similarity(la, lb, dim=1).cpu().numpy())
    return float(np.concatenate(sims).mean()) if sims else float('nan')


@torch.no_grad()
def perf_metrics(model, dataset, device, args, batch_size=32, num_classes=4):
    """AUC + Macro-F1 + F1 + Accuracy + PAIRWISE-AUC (đủ metric Forget-MI).
    Forget-MI báo cáo cả Pairwise-AUC (dict 0v1..2v3) → ta surface lại (mean + dict)."""
    from joint_img_txt.metrics import compute_auc, get_acc_f1
    from scipy.special import softmax
    from scipy.stats import logistic
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_logits, all_labels_oh = [], []
    model.eval()
    for batch in loader:
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, label_raw, _ = get_model_inputs(args, batch, device)
        label_oh = batch[5]
        with _eval_autocast():
            outputs = safe_forward(model, inputs)
        all_logits.append(outputs[1].float().cpu().numpy())
        oh_np = label_oh.cpu().numpy()
        if oh_np.ndim == 1 or oh_np.shape[-1] == 1:
            raw = oh_np.flatten().astype(int)
            oh_np = np.eye(num_classes)[np.clip(raw, 0, num_classes - 1)]
        all_labels_oh.append(oh_np.astype(int))
    logits = np.concatenate(all_logits, axis=0)
    labels_oh = np.concatenate(all_labels_oh, axis=0)
    probs = (logistic.cdf(logits) if args.output_channel_encoding == 'multilabel'
             else softmax(logits, axis=1))
    try:
        auc, pairwise = compute_auc(labels_oh.tolist(), probs.tolist(),
                                    output_channel_encoding=args.output_channel_encoding)
        valid = [a for a in auc if not (isinstance(a, float) and np.isnan(a))]
        auc_mean = float(np.mean(valid)) if valid else float('nan')
        pw_valid = [v for v in pairwise.values() if not (isinstance(v, float) and np.isnan(v))]
        pairwise_mean = float(np.mean(pw_valid)) if pw_valid else float('nan')
    except Exception as e:
        print(f"   [auc warn] {e}"); auc_mean = pairwise_mean = float('nan'); pairwise = {}
    try:
        f1m, _, _ = get_acc_f1(labels_oh, probs, args.output_channel_encoding)
        macro_f1 = float(f1m['macro_f1']); f1 = float(f1m['f1']); accuracy = float(f1m['accuracy'])
    except Exception as e:
        print(f"   [f1 warn] {e}"); macro_f1 = f1 = accuracy = float('nan')
    return {'AUC': auc_mean, 'Macro_F1': macro_f1, 'F1': f1, 'Accuracy': accuracy,
            'Pairwise_AUC': pairwise_mean, 'Pairwise_AUC_dict': pairwise}


# ============================================================================
# Selector S_val (dùng CHUNG cho P3–P6 + main)
# ============================================================================

@torch.no_grad()
def eval_forget_losses(model_ul, model_og, forget_dl, rand_dl, args, device):
    """Đánh giá IHL và các KHOẢNG CÁCH quên d_u, d_m trên D_f (KHÔNG backprop) — cho
    G_forget của selector. P4 dù tắt loss quên ở GĐ2 vẫn gọi hàm này để bắt 'nhớ lại'.
    d_u/d_m là khoảng cách Euclid tới tham chiếu og(nhiễu) đúng Forget-MI Eq.(1)-(2);
    không có hinge/margin ở đây nữa. Gate tạo mới theo batch (như code gốc Forget-MI)."""
    model_ul.eval(); model_og.eval()
    ihl_s, du_s, dm_s, n = 0.0, 0.0, 0.0, 0
    for fb, rb in zip(forget_dl, rand_dl):
        f_in, f_lbl, _ = get_model_inputs(args, fb, device)
        rnd_in, _, _ = get_model_inputs(args, rb, device)
        # model_ul forward ở FP32 (KHÔNG autocast) — khớp forward lúc TRAIN. IHL đẩy logit
        # text trên forget rất lớn; dưới fp16 autocast logit >65504 → inf → softmax → NAN
        # (làm G_forget/S_val = nan từ ~E11, selector mù các epoch sau). fp32 → softmax ổn định.
        ul_i, ul_il, ul_t, ul_tl = model_ul(**f_in)[:4]
        with _eval_autocast():
            og_i, _, og_t, _ = safe_forward(model_og, rnd_in)[:4]
        ul_i = ul_i.float(); ul_t = ul_t.float(); og_i = og_i.float(); og_t = og_t.float()
        ul_il = ul_il.float(); ul_tl = ul_tl.float()
        g = new_batch_gates(ul_i.shape[1], ul_t.shape[1], device, roles=('ul_frg', 'og_rnd'))
        ul_j = g['ul_frg'](ul_i, ul_t); og_j = g['og_rnd'](og_i, og_t)
        _, _, st = forget_distance(ul_i, ul_t, ul_j, og_i, og_t, og_j)
        # IHL = 1 + p(true) − max_{v≠true} p(v)  (bounded [0,2])
        f_y = f_lbl.long().view(-1)
        ihl = inverted_hinge_loss(ul_il, ul_tl, f_y)
        bs = fb[0].size(0)
        ihl_s += float(ihl) * bs
        du_s += st['d_u_mean'] * bs; dm_s += st['d_m_mean'] * bs; n += bs
    n = max(n, 1)
    return {'ihl': ihl_s / n, 'd_u': du_s / n, 'd_m': dm_s / n}


def compute_S_val(forget_losses, mia_val, ul_perf, og_perf, ref_scales, weights, deltas):
    """S_val = w_f·G_forget + w_p·G_privacy + w_u·G_utility (mỗi G∈[0,1], nhỏ=tốt).
    forget_losses: {ihl,d_u,d_m}; mia_val∈[0,1]; *_perf: {AUC,Macro_F1};
    ref_scales=(d⁰_u,d⁰_m) đo tại epoch 0; weights=(w_f,w_p,w_u); deltas=(δ_A,δ_F).

    G_forget dùng ánh xạ giảm-đơn-điệu, bị chặn trong (0,1]:
        ĝ(d) = d⁰/(d + d⁰)   → d=0 ⇒ 1 (chưa quên, tệ nhất); d=d⁰ ⇒ 0.5; d→∞ ⇒ 0.
    d⁰ CHỈ là hằng số chuẩn hóa của selector (protocol luận văn), KHÔNG phải margin và
    KHÔNG có mặt trong hàm mất mát."""
    d_u0, d_m0 = ref_scales; w_f, w_p, w_u = weights; d_A, d_F = deltas
    eps = 1e-6

    def _safe01(x):
        """clip [0,1]; nan/inf → 1.0 (tệ nhất) để S_val luôn hữu hạn, selector không bị mù."""
        x = float(x)
        if not np.isfinite(x):
            return 1.0
        return float(np.clip(x, 0.0, 1.0))

    g_forget = (1.0 / 3.0) * (forget_losses['ihl'] / 2.0
                              + d_u0 / (forget_losses['d_u'] + d_u0 + eps)
                              + d_m0 / (forget_losses['d_m'] + d_m0 + eps))
    g_forget = _safe01(g_forget)
    g_privacy = _safe01(mia_val)
    g_A = max(0.0, og_perf['AUC'] - ul_perf['AUC'] - d_A) / max(1e-6, 1.0 - d_A)
    g_F = max(0.0, og_perf['Macro_F1'] - ul_perf['Macro_F1'] - d_F) / max(1e-6, 1.0 - d_F)
    g_utility = 0.5 * (_safe01(g_A) + _safe01(g_F))
    s = w_f * g_forget + w_p * g_privacy + w_u * g_utility
    return {'S_val': float(s), 'G_forget': g_forget, 'G_privacy': g_privacy,
            'G_utility': g_utility}


# ============================================================================
# Losses dùng chung (IHL) — sao y LoKU (bounded [0,2])
# ============================================================================

def inverted_hinge_loss(img_logits, txt_logits, y):
    """L_IHL = 1 + p(true) − max_{v≠true} p(v), lấy trung bình img+txt. Bounded [0,2].
    (LoKU multiclass_hinge_loss dạng crammer-singer, clamp(1+margin,0).)"""
    def _one(logits):
        n_cls = logits.size(-1)
        probs = F.softmax(logits, dim=-1)
        mask = F.one_hot(y, n_cls).bool()
        p_true = probs.gather(1, y.unsqueeze(1)).squeeze(1)
        p_other = probs.masked_fill(mask, -1.0).max(dim=-1)[0]
        return (1.0 + p_true - p_other).mean()
    return 0.5 * (_one(img_logits) + _one(txt_logits))


def forget_distance(ul_i, ul_t, ul_j, og_i, og_t, og_j):
    """QUÊN biểu diễn — ĐÚNG Forget-MI Eq.(1)-(2): ÂM của khoảng cách Euclid tới biểu
    diễn mà mô hình GỐC tạo ra trên BẢN NHIỄU (Ĩ_f, T̃_f):

        L_UU = − mean_i || [z^img_ul, z^txt_ul](I_f,T_f) − [z^img_og, z^txt_og](Ĩ_f,T̃_f) ||
        L_MU = − mean_i || z^joint_ul(I_f,T_f) − z^joint_og(Ĩ_f,T̃_f) ||

    KHÔNG dùng hinge có chặn ReLU(m−d) và KHÔNG có margin: bài gốc tối thiểu hóa −d,
    tức càng xa càng tốt (việc kiểm soát quên quá đà nằm ở các thành phần giữ lại và ở
    bước chọn checkpoint, không nằm trong định nghĩa của L_UU/L_MU).
    Trả (L_uu, L_mu, stats={d_u_mean,d_m_mean})."""
    d_u = euclidean_distance(torch.cat((ul_i, ul_t), -1), torch.cat((og_i, og_t), -1))  # [N]
    d_m = euclidean_distance(ul_j, og_j)                                                 # [N]
    L_uu = -d_u.mean()
    L_mu = -d_m.mean()
    with torch.no_grad():
        stats = {'d_u_mean': float(d_u.mean()), 'd_m_mean': float(d_m.mean())}
    return L_uu, L_mu, stats


def combined_batch_loss(cfg, ctx, fb, rb, retb, device, weights,
                        forget_active=True, guard_ihl=0.0):
    """LÕI mất mát dùng chung cho P3 / P4-GĐ1 / P5 / main.
      forget_active=True  → đủ: λ_UR·UR + λ_MR·MR + λ_UU·UU + λ_MU·MU + λ_CE·CE + λ_KD·KD + λ_IHL·IHL
      forget_active=False → chỉ giữ (P4-GĐ2): UR + MR + CE + KD (+ guard_ihl·IHL nếu có fb)
    weights=(λ_UR,λ_UU,λ_MU,λ_MR,λ_CE,λ_KD,λ_IHL).

    QUÊN (Forget-MI Eq.(1)-(2)): UU = −d_u, MU = −d_m tới tham chiếu og(nhiễu).
    GIỮ (Forget-MI Eq.(3)-(4)): UR = mean d_ur, MR = mean d_mr — khoảng cách Euclid THUẦN,
      KHÔNG cap min(d,m), KHÔNG margin, KHÔNG lực kéo nhẹ.
    KD teacher=og (honest). Fusion gate: TẠO MỚI theo từng batch, riêng từng vai trò,
    không nằm trong optimizer — bám cấu trúc triển khai của code gốc Forget-MI."""
    lam_ur, lam_uu, lam_mu, lam_mr, lam_ce, lam_kd, lam_ihl = weights
    model_ul = ctx['model_unlearn']; model_og = ctx['model_og']

    ret_in, ret_lbl, _ = get_model_inputs(cfg, retb, device)
    with torch.no_grad(), torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
        og_ret_i, og_ret_il, og_ret_t, og_ret_tl = model_og(**ret_in)[:4]
    og_ret_i = og_ret_i.float(); og_ret_t = og_ret_t.float()
    og_ret_il = og_ret_il.float(); og_ret_tl = og_ret_tl.float()

    ul_ret_i, ul_ret_il, ul_ret_t, ul_ret_tl = model_ul(**ret_in)[:4]
    gates = new_batch_gates(ul_ret_i.shape[1], ul_ret_t.shape[1], device)
    ul_ret_j = gates['ul_ret'](ul_ret_i, ul_ret_t)
    og_ret_j = gates['og_ret'](og_ret_i, og_ret_t)

    # ----- RETAIN repr (Forget-MI Eq.(3)-(4)): khoảng cách Euclid THUẦN, không cap -----
    #   L_UR = Dist([F_ul(I_r),F_ul(T_r)], [F_og(I_r),F_og(T_r)])
    #   L_MR = Dist(F_ul((I_r,T_r)), F_og((I_r,T_r)))
    # Bản cũ dùng torch.minimum(d, margin) → gradient = 0 khi d vượt margin, tức càng lệch
    # xa mô hình gốc thì lực giữ càng biến mất (ngược hẳn ý nghĩa retention của bài gốc).
    L_ur = euclidean_distance(torch.cat((ul_ret_i, ul_ret_t), -1),
                              torch.cat((og_ret_i, og_ret_t), -1)).mean()
    L_mr = euclidean_distance(ul_ret_j, og_ret_j).mean()

    # ----- CE + KD trên retain -----
    ret_y = ret_lbl.long().view(-1)
    L_ce = 0.5 * (F.cross_entropy(ul_ret_il, ret_y) + F.cross_entropy(ul_ret_tl, ret_y))
    T = float(getattr(cfg, 'distill_temperature', 2.0))
    def _kl(student, teacher):
        return F.kl_div(F.log_softmax(student / T, dim=-1),
                        F.softmax(teacher / T, dim=-1).detach(),
                        reduction='batchmean') * (T * T)
    L_kd = 0.5 * (_kl(ul_ret_il, og_ret_il) + _kl(ul_ret_tl, og_ret_tl))

    loss = lam_ur * L_ur + lam_mr * L_mr + lam_ce * L_ce + lam_kd * L_kd
    comp = {'UR': L_ur.item(), 'MR': L_mr.item(), 'CE': L_ce.item(), 'KD': L_kd.item(),
            'UU': 0.0, 'MU': 0.0, 'IHL': 0.0}

    # ----- FORGET (nếu bật, hoặc chỉ guard IHL) -----
    need_forget = (forget_active or guard_ihl > 0) and fb is not None and rb is not None
    if need_forget:
        f_in, f_lbl, _ = get_model_inputs(cfg, fb, device)
        rnd_in, _, _ = get_model_inputs(cfg, rb, device)
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=torch.cuda.is_available()):
            og_rnd_i, _, og_rnd_t, _ = model_og(**rnd_in)[:4]
        og_rnd_i = og_rnd_i.float(); og_rnd_t = og_rnd_t.float()
        ul_frg_i, ul_frg_il, ul_frg_t, ul_frg_tl = model_ul(**f_in)[:4]
        ul_frg_j = gates['ul_frg'](ul_frg_i, ul_frg_t)
        og_rnd_j = gates['og_rnd'](og_rnd_i, og_rnd_t)
        f_y = f_lbl.long().view(-1)
        L_ihl = inverted_hinge_loss(ul_frg_il, ul_frg_tl, f_y)
        comp['IHL'] = L_ihl.item()
        if forget_active:
            L_uu, L_mu, hstats = forget_distance(ul_frg_i, ul_frg_t, ul_frg_j,
                                                 og_rnd_i, og_rnd_t, og_rnd_j)
            comp['d_u_mean'] = hstats['d_u_mean']; comp['d_m_mean'] = hstats['d_m_mean']
            # ABLATION 'A': ablate_uu_mu=true → KHÔNG cộng UU/MU vào loss (vẫn log để đối chiếu).
            # Chặn ở TẦNG LOSS vì P5/main lấy trọng số từ preset, KHÔNG đọc lambda_uu/lambda_mu
            # → override lambda_uu=0 vô tác dụng với main (bug đã gặp: noUUMU trùng hệt main).
            if as_bool(getattr(cfg, 'ablate_uu_mu', False)):
                comp['UU'] = 0.0; comp['MU'] = 0.0
                loss = loss + lam_ihl * L_ihl
            else:
                comp['UU'] = L_uu.item(); comp['MU'] = L_mu.item()
                loss = loss + lam_uu * L_uu + lam_mu * L_mu + lam_ihl * L_ihl
        else:
            loss = loss + guard_ihl * L_ihl        # P4-GĐ2: guard nhỏ

    comp['Total'] = loss.item()
    return loss, comp


# ============================================================================
# Scheduler: warmup (10% global step) → ReduceLROnPlateau (per-epoch, theo S_val)
# ============================================================================

class WarmupThenPlateau:
    """Warmup tuyến tính trong warmup_steps lần CẬP NHẬT đầu; sau đó
    ReduceLROnPlateau step MỘT LẦN mỗi epoch theo S_val (mode='min').

    LR luôn được đặt TRƯỚC lần cập nhật dùng nó: constructor đặt LR cho update #1, và
    mỗi batch_step() (gọi sau optimizer.step()) đặt LR cho update KẾ TIẾP. Với
    warmup_steps=3, dãy LR thực sự được dùng là ⅓·LR → ⅔·LR → LR → LR …
    (bản cũ gọi batch_step sau step nhưng đặt LR cho update vừa chạy, nên update đầu
    tiên đã dùng nguyên LR rồi mới tụt về ⅓ — tức warmup bị lệch một nhịp.)

    Sau khi hết warmup, batch_step KHÔNG đụng vào LR nữa để không ghi đè các lần giảm
    LR của ReduceLROnPlateau."""
    def __init__(self, optimizer, base_lr, warmup_steps, plateau_kwargs=None):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = max(1, int(warmup_steps))
        self.updates_done = 0
        self.in_warmup = True
        pk = dict(mode='min', factor=0.5, patience=2, threshold=1e-4)
        if plateau_kwargs:
            pk.update(plateau_kwargs)
        self.plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **pk)
        self._set_lr_for_update(1)          # LR cho lần cập nhật ĐẦU TIÊN

    def _set_lr(self, lr):
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def _set_lr_for_update(self, k):
        """Đặt LR sẽ dùng cho lần cập nhật thứ k (đánh số từ 1)."""
        return self._set_lr(self.base_lr * min(k, self.warmup_steps) / self.warmup_steps)

    def batch_step(self):
        """Gọi SAU mỗi optimizer.step(): đặt LR cho lần cập nhật KẾ TIẾP.
        Chỉ tác dụng trong pha warmup."""
        self.updates_done += 1
        if not self.in_warmup:
            return
        if self.updates_done >= self.warmup_steps:
            self.in_warmup = False
            self._set_lr(self.base_lr)      # chốt LR gốc rồi nhường quyền cho plateau
        else:
            self._set_lr_for_update(self.updates_done + 1)

    def epoch_step(self, s_val):
        """Gọi cuối mỗi epoch (sau khi hết warmup) — plateau theo S_val."""
        if not self.in_warmup and s_val is not None and np.isfinite(s_val):
            self.plateau.step(s_val)


# ============================================================================
# Checkpoint
# ============================================================================

def trainable_state_dict(model):
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}
    mutable_names = trainable_names | {name for name, _ in model.named_buffers()}
    return {k: v.detach().cpu() for k, v in model.state_dict().items() if k in mutable_names}


def save_ckpt(out_dir, epoch, model, optimizer, metrics, filename='latest.pt',
              val_ce=None, s_val=None, include_optimizer=True):
    """Gate KHÔNG được lưu: nó là module tạm tạo lại mỗi batch (như code gốc Forget-MI),
    không mang trạng thái học được nên không thuộc checkpoint."""
    cp = os.path.join(out_dir, "checkpoints")
    os.makedirs(cp, exist_ok=True)
    ts = trainable_state_dict(model)
    payload = {
        'epoch': epoch,
        'trainable_state': ts,
        'lora_state': {k: v for k, v in ts.items() if 'lora' in k.lower()},
        'metrics': metrics, 'val_ce': val_ce, 's_val': s_val,
    }
    if include_optimizer:
        payload['optimizer'] = optimizer.state_dict()
    torch.save(payload, os.path.join(cp, filename))


def load_ckpt(out_dir, model, optimizer):
    p = os.path.join(out_dir, "checkpoints", "latest.pt")
    if not os.path.exists(p):
        return -1
    try:
        cp = torch.load(p, map_location='cpu', weights_only=False)
        state = cp.get('trainable_state', cp.get('lora_state'))
        model.load_state_dict(state, strict=False)
        if 'optimizer' in cp and optimizer is not None:
            optimizer.load_state_dict(cp['optimizer'])
        print(f"📂 Resumed from epoch {cp['epoch']}")
        return cp['epoch']
    except Exception as e:
        print(f"⚠️  Checkpoint incompatible ({e}); starting fresh.")
        return -1


# ============================================================================
# CSV / report
# ============================================================================

def append_row_csv(csv_path, row):
    """Ghi 1 dòng, tự migrate header khi thêm cột (union), ghi theo thứ tự header thực."""
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    existing_rows, header = [], []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', newline='', encoding='utf-8') as fh:
            reader = csv.DictReader(fh)
            header = list(reader.fieldnames or [])
            existing_rows = list(reader)
    new_header = list(header)
    for k in row.keys():
        if k not in new_header:
            new_header.append(k)
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=new_header)
        writer.writeheader()
        for r in existing_rows:
            writer.writerow({k: r.get(k, '') for k in new_header})
        writer.writerow({k: row.get(k, '') for k in new_header})


def append_history_row(csv_path, row):
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    write_header = not os.path.exists(csv_path)
    with open(csv_path, 'a', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ============================================================================
# Model path discovery (nested-zip / Kaggle mount) — sao y loku
# ============================================================================

def ensure_model_path(path, desc="model"):
    import glob
    WEIGHT_FILES = ("pytorch_model.bin", "model.safetensors")
    for fname in WEIGHT_FILES:
        if os.path.exists(os.path.join(path, fname)):
            return path
    if os.path.exists(os.path.join(path, "config.json")):
        return path
    for fname in WEIGHT_FILES:
        nested = glob.glob(os.path.join(path, "**", fname), recursive=True)
        if nested:
            return os.path.dirname(nested[0])
    base = os.path.basename(path.rstrip('/'))
    for root in ("/content", "/kaggle/input", "/kaggle/working"):
        for fname in WEIGHT_FILES:
            for m in glob.glob(f"{root}/**/{fname}", recursive=True):
                if base in m and "checkpoint" not in m:
                    return os.path.dirname(m)
    print(f"❌ Không tìm thấy {desc} tại {path}")
    return path


# ============================================================================
# Config: load YAML (dạng wandb 'parameters'), override, lưu resolved
# ============================================================================

def load_config(config_path):
    import yaml
    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f)
    if 'parameters' in raw:
        cfg = {k: (v['value'] if isinstance(v, dict) and 'value' in v
                   else (v['values'][0] if isinstance(v, dict) and 'values' in v else v))
               for k, v in raw['parameters'].items()}
    else:
        cfg = raw
    return cfg


def flatten_method_config(cfg, method):
    """Config chung có 'common' + 'methods'.{p3,p4,...}. Trả dict phẳng:
    common + methods[method] (methods ghi đè common). Nếu không có cấu trúc này,
    trả cfg nguyên (tương thích config phẳng)."""
    if 'common' in cfg or 'methods' in cfg:
        out = dict(cfg.get('common', {}))
        methods = cfg.get('methods', {})
        out.update(methods.get(method, {}))
        # giữ lại các khóa top-level khác (paths...) nếu người dùng để ngoài common
        for k, v in cfg.items():
            if k not in ('common', 'methods') and k not in out:
                out[k] = v
        return out
    return dict(cfg)


def apply_overrides(cfg, override_str):
    if not override_str:
        return cfg
    for kv in override_str.split(','):
        if '=' not in kv:
            continue
        k, v = kv.split('=', 1); k = k.strip(); v = v.strip()
        try:
            v2 = int(v)
        except ValueError:
            try:
                v2 = float(v)
            except ValueError:
                v2 = v
        cfg[k] = v2
        print(f"🔧 Override: {k} = {v2}")
    return cfg


def save_resolved_config(cfg, out_dir, method, run_id):
    import yaml
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"resolved_config_{method}_{run_id}.yaml")
    dumpable = {k: (v if isinstance(v, (int, float, str, bool, list, dict, type(None))) else str(v))
                for k, v in dict(cfg).items()}
    with open(path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(dumpable, f, allow_unicode=True, sort_keys=True)
    print(f"🗎 Resolved config saved: {path}")
    return path


# ============================================================================
# DRIVER dùng chung: dựng model nền, gates, selection, final eval
# ============================================================================

class Cfg:
    """Bọc dict config thành object có .attr + .get() (thay cho wandb.config).
    Attr thiếu → raise AttributeError để getattr(cfg, k, default) lấy được default,
    và truy cập trực tiếp khóa bắt buộc bị thiếu thì fail-fast."""
    def __init__(self, d):
        self.__dict__.update(dict(d))
    def get(self, k, default=None):
        return self.__dict__.get(k, default)
    def __getattr__(self, k):
        raise AttributeError(f"Config thiếu khóa '{k}'")


def build_base_models(cfg, device):
    """Load base (og fp32 cho Fisher, unlearn fp32 cho LoRA) + retrained (fp16, chỉ để
    ĐÁNH GIÁ hậu nghiệm CosSim — KHÔNG dùng cho selection). Trả context dict."""
    from transformers import BertTokenizer
    from joint_img_txt.model import ImageTextModel
    base_p = ensure_model_path(cfg.base_model_path, "base")
    re_p = ensure_model_path(cfg.retrained_model_path, "retrained")
    model_og = ImageTextModel.from_pretrained(base_p).to(device)
    model_unlearn = ImageTextModel.from_pretrained(base_p).to(device)
    try:
        model_re = ImageTextModel.from_pretrained(re_p).to(device).half()
        gold_available = os.path.abspath(re_p) != os.path.abspath(base_p)
    except Exception as e:
        print(f"⚠️  retrained model unavailable ({e}); dùng base làm placeholder eval.")
        model_re = ImageTextModel.from_pretrained(base_p).to(device).half()
        gold_available = False
    tok = BertTokenizer.from_pretrained(
        base_p if os.path.exists(os.path.join(base_p, "vocab.txt")) else cfg.bert_pretrained_dir)
    return {'base_p': base_p, 're_p': re_p, 'model_og': model_og,
            'model_unlearn': model_unlearn, 'model_re': model_re,
            'tokenizer': tok, 'gold_available': gold_available}


def reload_og_fp16(base_p, device):
    from joint_img_txt.model import ImageTextModel
    m = ImageTextModel.from_pretrained(base_p).to(device).half()
    for p in m.parameters():
        p.requires_grad = False
    return m


def new_batch_gates(img_dim, txt_dim, device, roles=('ul_ret', 'ul_frg', 'og_rnd', 'og_ret')):
    """Tạo MỚI một Gate cho TỪNG VAI TRÒ ở MỖI BATCH — bám đúng code gốc Forget-MI
    (`forgetmi_partial.py`: gate_ul_ret / gate_ul_frgt / gate_og_rand / gate_og_ret
    được khởi tạo bên trong vòng lặp batch).

    Vì sao làm vậy (quy tắc paper↔code): Forget-MI chỉ nói joint embedding được dựng
    bằng multimodal adaptation gate, KHÔNG quy định một gate dùng chung hay nhiều gate
    riêng, cũng không nói gate ở train() hay eval(). Chi tiết paper bỏ ngỏ → theo code gốc.

    Gate ở train mode mặc định (Dropout hoạt động) và KHÔNG được đưa vào optimizer,
    nên tham số gate không bao giờ được cập nhật; gradient vẫn truyền qua phép gate
    về embedding đầu vào (và từ đó xuống LoRA).

    KHÔNG sao chép lỗi lập trình hiển nhiên của code gốc (dòng 403-404 dùng nhầm gate
    và truyền ảnh hai lần thay vì ảnh+văn bản)."""
    from training.joint_embedding import Gate
    return {r: Gate(inp1_size=int(img_dim), inp2_size=int(txt_dim)).to(device) for r in roles}


def build_peft(model_unlearn, cfg, target_modules, rank_pattern=None, alpha_pattern=None):
    """Bọc PEFT LoRA. rank_pattern/alpha_pattern (P6) = dict {tên_module: rank/alpha}.
    Giữ α/r không đổi: nếu chỉ truyền rank_pattern mà thiếu alpha_pattern, tự suy
    alpha = rank × (lora_alpha/lora_r) để α/r cố định."""
    from peft import LoraConfig, get_peft_model
    base_r = int(cfg.lora_r); base_alpha = int(cfg.lora_alpha)
    ratio = base_alpha / max(1, base_r)
    if rank_pattern and not alpha_pattern:
        alpha_pattern = {k: int(round(v * ratio)) for k, v in rank_pattern.items()}
        print(f"   α/r={ratio:.2f} giữ nguyên → alpha_pattern={alpha_pattern}")
    kwargs = dict(r=base_r, lora_alpha=base_alpha, target_modules=target_modules,
                  lora_dropout=float(cfg.lora_dropout), bias='none')
    if rank_pattern:
        kwargs['rank_pattern'] = rank_pattern
    if alpha_pattern:
        kwargs['alpha_pattern'] = alpha_pattern
    peft_cfg = LoraConfig(**kwargs)
    model = get_peft_model(model_unlearn, peft_cfg)
    return model, peft_cfg


def verify_lora_ranks(model, desired_full_ranks):
    """KIỂM CHỨNG rank thực tế PEFT đã cấp cho từng module (đúng yêu cầu 'verify Conv-LoRA
    + rank_pattern'). PEFT khớp rank_pattern bằng regex `.*\\.{key}$` — có thể lệch giữa
    module tuyến tính (SciBERT) và Conv2d (ResNet). Hàm đọc lora_A.shape[0] thực, so với
    mong muốn, IN BẢNG + CẢNH BÁO nếu lệch (không im lặng chạy sai). Trả (found, mismatches)."""
    found = {}
    for name, mod in model.named_modules():
        la = getattr(mod, 'lora_A', None)
        if la is not None and hasattr(la, 'keys') and 'default' in la:
            clean = name.replace('base_model.model.', '')
            try:
                found[clean] = int(la['default'].weight.shape[0])
            except Exception:
                pass
    mismatches = [(m, want, found.get(m)) for m, want in desired_full_ranks.items()
                  if found.get(m) != want]
    n_ok = len(desired_full_ranks) - len(mismatches)
    print(f"🔎 verify_lora_ranks: {n_ok}/{len(desired_full_ranks)} module đúng rank mong muốn")
    by_rank = {}
    for m, r in found.items():
        by_rank[r] = by_rank.get(r, 0) + 1
    print(f"   phân bố rank thực tế: {dict(sorted(by_rank.items()))}")
    if mismatches:
        print(f"   ⚠️  {len(mismatches)} module LỆCH rank (PEFT rank_pattern không khớp như ý!):")
        for m, want, got in mismatches[:10]:
            print(f"      - {m}: muốn r={want}, thực r={got}")
        print("   → rank_pattern regex có thể không áp cho nhánh này. Kiểm tra key/phiên bản PEFT.")
    return found, mismatches


def enforce_lora_only(model):
    """LoKU Sec. 3.5: giữ NGUYÊN trọng số pretrained, CHỈ cập nhật các thừa số hạng thấp
    A, B. Vì vậy classifier heads (img_model.fc1, text_model.classifier) phải đóng băng —
    nếu mở băng thì phần cập nhật unlearning bị hấp thụ ở head, claim 'LoRA-only' của
    phương pháp không còn đúng (lỗi 9).

    Gradient vẫn chảy QUA head (đóng băng ≠ chặn gradient) xuống các adapter LoRA."""
    n_frozen = 0
    for name, p in model.named_parameters():
        if p.requires_grad and 'lora_' not in name:
            p.requires_grad = False
            n_frozen += p.numel()
    leftover = [n for n, p in model.named_parameters() if p.requires_grad and 'lora_' not in n]
    assert not leftover, f"Còn tham số trainable KHÔNG phải LoRA: {leftover[:5]}"
    if n_frozen:
        print(f"🔒 LoRA-only: đóng băng thêm {n_frozen:,} tham số ngoài LoRA (classifier heads…)")
    else:
        print("🔒 LoRA-only: mọi tham số trainable đều là LoRA ✓")


def make_optimizer(model, cfg):
    """CHỈ tối ưu tham số LoRA (LoKU Sec. 3.5). Fusion gate KHÔNG vào optimizer (lỗi 10):
    Forget-MI dùng gate để dựng joint embedding chứ không tối ưu nó — trong code gốc gate
    là biến cục bộ tạo lại mỗi batch nên không thể nằm trong optimizer; LoKU chỉ tối ưu A,B.
    → gate không được là nơi hấp thụ cập nhật unlearning."""
    from torch.optim import AdamW
    params = [p for n, p in model.named_parameters() if p.requires_grad and 'lora_' in n]
    print(f"⚙️  Optimizer: {len(params)} tensor LoRA "
          f"({sum(p.numel() for p in params):,} tham số) — không có gate/head")
    return AdamW(params, lr=float(cfg.learning_rate), weight_decay=float(cfg.weight_decay))


def set_bn_eval(model):
    """Giữ MỌI BatchNorm ở EVAL (running-stats, KHÔNG cập nhật) ngay cả khi model.train().
    Vì sao BẮT BUỘC: base bị đóng băng (chỉ LoRA/head train). Nếu để BN train-mode, ảnh
    encoder dùng batch-stat của forget set NHỎ (16 mẫu) → embedding nhảy loạn (d_u eval≈0.02
    nhưng train≈3.6) → (1) margin tính ở eval KHÔNG khớp loss ở train ⇒ UU/MU=0;
    (2) ul (train-BN) lệch og (eval-BN) trên cùng input. Đóng băng BN sửa cả hai + đúng chuẩn
    fine-tune trên base đóng băng."""
    import torch.nn as nn
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.eval(); n += 1
    return n


def setup_experiment(cfg, device, rank_alloc_fn=None):
    """Dựng TOÀN BỘ context dùng chung cho P3/P4/P5/P6/main:
    models → dataset (4-tập) → Fisher → (rank_alloc_fn) → PEFT+FILA → LoRA-only →
    optimizer → dataloaders → thang d⁰ → og-perf → scheduler.

    rank_alloc_fn(f_imp, r_imp, target_modules, cfg) → (target_modules, rank_pattern,
    alpha_pattern): dùng cho P6 (phân rank theo Fisher, có thể LOẠI bớt module).

    Fusion gate KHÔNG được dựng ở đây: bám code gốc Forget-MI, gate được tạo mới trong
    từng batch (xem new_batch_gates) và không bao giờ nằm trong optimizer."""
    import time as _time
    ctx = build_base_models(cfg, device)
    print("📚 Building dataset (4-tập holdout)...")
    _t = _time.time()
    datasets, num_labels = build_dataset(cfg, ctx['tokenizer'])
    ctx['datasets'] = datasets; ctx['num_labels'] = num_labels
    ctx['load_h'] = (_time.time() - _t) / 3600

    target_modules = list(getattr(cfg, 'lora_target_modules', ['query', 'key', 'value']))
    img_last_k = int(getattr(cfg, 'lora_image_last_k_blocks', 0))
    img_inc_fc1 = bool(getattr(cfg, 'lora_image_include_fc1', False))
    image_targets = resolve_image_targets(ctx['model_og'], img_last_k, img_inc_fc1)
    if image_targets:
        target_modules = target_modules + image_targets
    ctx['image_targets'] = image_targets

    _t = _time.time()
    print(f"🎯 LoRA targets (trước phân rank): {target_modules}")
    fisher_bs = int(getattr(cfg, 'fisher_batch_size', cfg.unlearn_batch_size))
    fisher_n = int(getattr(cfg, 'fisher_max_samples', 256))
    f_imp = compute_fisher_importance(ctx['model_og'],
        DataLoader(datasets['forget'], batch_size=fisher_bs), device, target_modules, cfg, fisher_n)
    r_imp = compute_fisher_importance(ctx['model_og'],
        DataLoader(datasets['retain'], batch_size=fisher_bs), device, target_modules, cfg, fisher_n)
    ctx['fisher_h'] = (_time.time() - _t) / 3600
    ctx['f_imp'], ctx['r_imp'] = f_imp, r_imp

    rank_pattern = alpha_pattern = None
    if rank_alloc_fn is not None:
        target_modules, rank_pattern, alpha_pattern = rank_alloc_fn(
            f_imp, r_imp, target_modules, cfg)
        print(f"🎯 LoRA targets (sau phân rank): {target_modules}")
    ctx['target_modules'] = target_modules
    ctx['rank_pattern'] = rank_pattern; ctx['alpha_pattern'] = alpha_pattern

    del ctx['model_og']; torch.cuda.empty_cache()
    ctx['model_og'] = reload_og_fp16(ctx['base_p'], device)

    model, peft_cfg = build_peft(ctx['model_unlearn'], cfg, target_modules,
                                 rank_pattern=rank_pattern, alpha_pattern=alpha_pattern)
    ctx['peft_cfg'] = peft_cfg
    img_sub = float(getattr(cfg, 'loku_image_subtract_scale',
                            getattr(cfg, 'loku_subtract_scale', 0.0)))
    fila_subtraction = {}
    if getattr(cfg, 'loku_random_init', False):
        print("🎲 ABLATION: loku_random_init=True → BỎ Fisher/FILA init")
    else:
        fila_subtraction = apply_loku_soft_init(model, f_imp, r_imp, target_modules,
            r=int(cfg.lora_r), init_scale=float(getattr(cfg, 'loku_init_scale', 0.05)),
            subtract_scale=float(getattr(cfg, 'loku_subtract_scale', 0.0)),
            image_target_names=set(image_targets), image_subtract_scale=img_sub,
            rank_pattern=rank_pattern)
    # LoKU Sec. 3.5: pretrained frozen, chỉ A,B được cập nhật (classifier heads ĐÓNG BĂNG)
    enforce_lora_only(model)
    ctx['fila_subtraction'] = fila_subtraction
    ctx['model_unlearn'] = model
    ctx['trainable'], ctx['total'] = count_params(model)
    print(f"📊 Trainable: {ctx['trainable']:,}/{ctx['total']:,} "
          f"({100*ctx['trainable']/ctx['total']:.3f}%)")

    print("🚪 Fusion gate: tạo MỚI mỗi batch, riêng từng vai trò, KHÔNG vào optimizer "
          "(bám code gốc Forget-MI)")
    ctx['optimizer'] = make_optimizer(model, cfg)

    nw = min(int(getattr(cfg, 'num_cpu_workers', 2)), 2)
    ctx['nw'] = nw
    ctx['forget_dl'] = DataLoader(datasets['forget'],
        sampler=AlignedSampler(len(datasets['forget']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw, pin_memory=False)
    ctx['rand_dl'] = DataLoader(datasets['random'],
        sampler=AlignedSampler(len(datasets['random']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw, pin_memory=False)

    # thang chuẩn hóa d⁰ CHỈ cho selector S_val (không tham gia loss — xem compute_S_val)
    ctx['ref_scales'] = compute_epoch0_scales(model, ctx['model_og'],
        ctx['forget_dl'], ctx['rand_dl'], cfg, device)
    ctx['og_perf'] = precompute_og_perf(ctx['model_og'], datasets['sel'], device, cfg)

    # CADENCE (bám code gốc Forget-MI): gradient TÍCH LŨY qua các batch trong epoch,
    # optimizer.step() MỘT LẦN ở cuối epoch → tổng số cập nhật = số epoch.
    total_steps = int(cfg.unlearn_epochs)
    warmup_steps = max(1, int(float(getattr(cfg, 'warmup_proportion', 0.1)) * total_steps))
    ctx['scheduler'] = WarmupThenPlateau(ctx['optimizer'], float(cfg.learning_rate), warmup_steps)
    ctx['total_planned_steps'] = total_steps
    ctx['forget_batches'] = max(1, len(ctx['forget_dl']))
    return ctx


@torch.no_grad()
def precompute_og_perf(model_og, sel_dataset, device, cfg):
    """AUC/Macro-F1 của model gốc trên D_utility_val (=sel) — dùng cho G_utility (cache)."""
    perf = perf_metrics(model_og, sel_dataset, device, cfg,
                        batch_size=int(cfg.eval_batch_size))
    print(f"📌 og perf trên sel (utility ref): AUC={perf['AUC']:.3f} MacroF1={perf['Macro_F1']:.3f}")
    return perf


def selection_metrics(model_ul, model_og, datasets, forget_dl, rand_dl,
                      ref_scales, og_perf, cfg, device):
    """Tính S_val cho checkpoint hiện tại (KHÔNG dùng test_final/retrained).
    Trả dict {S_val, G_forget, G_privacy, G_utility, val_ce, ...}."""
    weights = (float(cfg.get('sel_w_forget', 1.0)), float(cfg.get('sel_w_privacy', 1.0)),
               float(cfg.get('sel_w_utility', 1.0)))
    deltas = (float(cfg.get('sel_delta_auc', 0.02)), float(cfg.get('sel_delta_f1', 0.02)))
    bs = int(cfg.eval_batch_size)
    # G_forget (đánh giá IHL + khoảng cách quên d_u/d_m trên D_f, no-grad)
    flosses = eval_forget_losses(model_ul, model_og, forget_dl, rand_dl, cfg, device)
    # G_privacy = MIA_val (member=r_heldout, non-member=sel, đoán forget)
    mia_v = val_mia(model_ul, datasets['r_heldout'], datasets['sel'], datasets['forget'],
                    device, cfg, batch_size=bs, seed=int(cfg.random_seed))
    # G_utility (ul perf trên sel vs og perf)
    ul_perf = perf_metrics(model_ul, datasets['sel'], device, cfg, batch_size=bs)
    S = compute_S_val(flosses, mia_v, ul_perf, og_perf, ref_scales, weights, deltas)
    # val_ce (member held-out) — chỉ dùng cho ablation selector
    val_ce = float(per_sample_ce(model_ul, datasets['r_heldout'], device, cfg, bs).mean())
    S.update({'val_ce': val_ce, 'mia_val': mia_v,
              'ul_sel_AUC': ul_perf['AUC'], 'ul_sel_MacroF1': ul_perf['Macro_F1'],
              **{f'floss_{k}': v for k, v in flosses.items()}})
    return S


@torch.no_grad()
def eval_on_test_final(model_ul, model_re, datasets, cfg, device, light=False):
    """Eval model SỐNG (LoRA active, eval-mode) trên D_t_final + D_f + MIA + 1−Sim(gold).
    Dùng để 'CHỌN EPOCH TỐT NHẤT' như cách tái lập Forget-MI (mỗi epoch ghi 1 dòng, rồi dò
    epoch gần GOLD nhất). MIA: member=retain(subsample), non-member=D_t_final, đoán D_f.
    light=True: BỎ cosine_sim(gold) (forward gold 2×, nặng nhất) — dist_vs_re=nan. Dùng cho
    vòng eval-mỗi-epoch (Cell 6 chọn epoch bằng Df-AUC/MIA/Dt, KHÔNG cần dist_vs_re) → nhanh hơn."""
    bs = int(cfg.eval_batch_size)
    emr = int(cfg.get('eval_max_retain', 512))
    member = subsample_dataset(datasets['retain'], emr, int(cfg.random_seed))
    try:
        mia = run_mia(model_ul, member, datasets['test_final'], datasets['forget'], device, cfg,
                      batch_size=bs, seed=int(cfg.random_seed),
                      paper_batch_size=int(cfg.get('mia_paper_batch_size', 32)))
    except Exception:
        mia = {'persample': float('nan'), 'paper': float('nan'),
               'forget_ce': float('nan'), 'nonmember_ce': float('nan')}
    tm = perf_metrics(model_ul, datasets['test_final'], device, cfg, batch_size=bs)
    fm = perf_metrics(model_ul, datasets['forget'], device, cfg, batch_size=bs)
    if light:
        cs = float('nan')          # bỏ forward gold 2× cho vòng eval-mỗi-epoch
    else:
        try:
            cs = cosine_sim_models(model_ul, model_re, member, device, cfg, batch_size=bs)
        except Exception:
            cs = float('nan')
    return {'Df_AUC': fm['AUC'], 'Df_F1': fm['Macro_F1'], 'Df_PairAUC': fm.get('Pairwise_AUC'),
            'Dt_AUC': tm['AUC'], 'Dt_F1': tm['Macro_F1'], 'Dt_PairAUC': tm.get('Pairwise_AUC'),
            'MIA': mia['persample'], 'MIA_paper': mia['paper'],
            'forget_ce': mia['forget_ce'], 'test_ce': mia['nonmember_ce'],
            'dist_vs_re': (1 - cs) if np.isfinite(cs) else float('nan')}


def run_training(cfg, ctx, device, method, weight_fn, select_by='S_val'):
    """Vòng train dùng chung (P3/P4/P5/P6/main). weight_fn(cfg, ctx, epoch, global_frac)
    → (weights, forget_active, guard_ihl) quyết định trọng số/giai đoạn TỪNG BATCH.
    Luôn zip(forget, rand, retain) để step/epoch nhất quán; combined_batch_loss tự bỏ
    forward forget khi forget_active=False & guard=0.

    CADENCE (bám code gốc Forget-MI, xem forgetmi_partial.py): trong mỗi epoch, các batch
    chỉ backward để TÍCH LŨY gradient; cuối epoch mới clip rồi optimizer.step() MỘT lần.
    Không có epoch hiệu chỉnh không-cập-nhật như code gốc, vì retention margin đã bị bỏ
    theo Eq.(3)-(4) của paper.

    select_by ∈ {'S_val','val_ce'} (đều nhỏ=tốt): tiêu chí chọn checkpoint. Mặc định S_val;
    'val_ce' chỉ dùng cho ABLATION SELECTOR của main. Trả (timing, best, total_steps)."""
    import time as _time
    model_ul = ctx['model_unlearn']
    optimizer = ctx['optimizer']; sched = ctx['scheduler']; datasets = ctx['datasets']
    grad_clip = float(getattr(cfg, 'grad_clip', 1.0))
    history_csv = getattr(cfg, 'history_csv_path', None)
    trainable_params = [p for p in model_ul.parameters() if p.requires_grad]   # chỉ LoRA

    best = {'s_val': 1e9, 'sel_value': 1e9, 'select_by': select_by,
            'epoch': None, 'state': None, 'metrics': None}
    total_steps = 0
    total_planned = max(1, int(ctx.get('total_planned_steps', 1)))
    t_train = t_sel = 0.0
    wall0 = _time.time()

    # ----- CE-crossing selector gold-free (S1-S4) — bật bằng ce_selector=1. Dùng sel(=D_nm_val)
    # + test_final CÓ SẴN, eval bằng hàm của adv_common, snapshot LoRA (trainable_state, ~2MB). -----
    _ce_sel = None
    if as_bool(getattr(cfg, 'ce_selector', False)):
        from training.ce_selector_pilot import OnlineCESelector
        _ce_sel = OnlineCESelector(
            cfg, {'forget': datasets['forget'], 'retain': datasets['retain']}, device,
            os.path.join(ctx['out_dir'], f'checkpoint_selection_{method}'),   # riêng mỗi method
            label=str(method), split_seed=int(cfg.random_seed),
            nm_val_ds=datasets['sel'], tfinal_ds=datasets['test_final'],
            ce_fn=per_sample_ce, perf_fn=perf_metrics, mia_fn=run_mia, subsample_fn=subsample_dataset,
            state_fn=lambda m: {k: v.detach().cpu().clone() for k, v in trainable_state_dict(m).items()},
            load_fn=lambda m, s: m.load_state_dict(s, strict=False))

    for epoch in range(int(cfg.unlearn_epochs)):
        _t = _time.time()
        model_ul.train()
        _nbn = set_bn_eval(model_ul)     # BN giữ eval (running-stats) — đúng chuẩn LoRA base đóng băng
        if epoch == 0:
            print(f"   🧊 giữ {_nbn} BatchNorm ở eval-mode trong lúc train (base đóng băng)")
        retain_dl = DataLoader(datasets['retain'], sampler=torch.utils.data.RandomSampler(datasets['retain']),
                               batch_size=cfg.unlearn_batch_size, num_workers=ctx['nw'])
        n_batches = max(1, int(ctx.get('forget_batches', 1)))
        n_ep = max(1, int(cfg.unlearn_epochs))
        agg = {}; steps = 0
        optimizer.zero_grad()                 # gradient TÍCH LŨY suốt epoch
        for bi, (fb, rb, retb) in enumerate(zip(ctx['forget_dl'], ctx['rand_dl'], retain_dl)):
            # tiến trình mượt trong epoch để lịch trọng số của P5 không bị bậc thang
            global_frac = min(1.0, (epoch + bi / n_batches) / n_ep)
            weights, forget_active, guard_ihl = weight_fn(cfg, ctx, epoch, global_frac)
            loss, comp = combined_batch_loss(cfg, ctx, fb, rb, retb, device, weights,
                                             forget_active=forget_active, guard_ihl=guard_ihl)
            loss.backward()                   # KHÔNG step ở đây
            steps += 1
            for k, v in comp.items():
                agg[k] = agg.get(k, 0.0) + v
        # ---- cuối epoch: clip rồi cập nhật MỘT lần (cadence của code gốc Forget-MI) ----
        if grad_clip and trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        optimizer.zero_grad()
        sched.batch_step()
        total_steps += 1
        avg = {k: v / max(steps, 1) for k, v in agg.items()}
        t_train += _time.time() - _t

        _t = _time.time()
        S = selection_metrics(model_ul, ctx['model_og'], datasets,
                              ctx['forget_dl'], ctx['rand_dl'], ctx['ref_scales'],
                              ctx['og_perf'], cfg, device)
        sched.epoch_step(S['S_val'])
        t_sel += _time.time() - _t

        d_u0, d_m0 = ctx['ref_scales']
        print(f"[{method} E{epoch:02d}] loss={avg.get('Total', 0):+.3f} "
              f"UU={avg.get('UU', 0):+.4f} MU={avg.get('MU', 0):+.4f} "
              f"(d̄ {avg.get('d_u_mean', 0):.3f}/{avg.get('d_m_mean', 0):.2f} "
              f"vs d⁰ {d_u0:.3f}/{d_m0:.2f}) "
              f"CE={avg.get('CE', 0):.3f} KD={avg.get('KD', 0):.4f} IHL={avg.get('IHL', 0):.3f} "
              f"| S_val={S['S_val']:.4f} (Gf={S['G_forget']:.3f} Gp={S['G_privacy']:.3f} "
              f"Gu={S['G_utility']:.3f}) val_ce={S['val_ce']:.3f}")

        # ---- 'CHỌN EPOCH TỐT NHẤT': eval mỗi epoch trên D_t_final (như tái lập Forget-MI) ----
        if as_bool(getattr(cfg, 'eval_test_every_epoch', False)):
            _t2 = _time.time()
            te = eval_on_test_final(model_ul, ctx['model_re'], datasets, cfg, device,
                                    light=as_bool(getattr(cfg, 'eval_test_light', True)))
            t_sel += _time.time() - _t2
            print(f"    📊 [test E{epoch:02d}] Df-AUC {te['Df_AUC']:.3f} Df-F1 {te['Df_F1']:.3f}  "
                  f"Dt-AUC {te['Dt_AUC']:.3f} Dt-F1 {te['Dt_F1']:.3f}  "
                  f"MIA {te['MIA']:.3f}/{te['MIA_paper']:.3f}  "
                  f"fce/tce {te['forget_ce']:.2f}/{te['test_ce']:.2f}")
            tcsv = getattr(cfg, 'test_history_csv_path', None)
            if tcsv:
                append_history_row(tcsv, {'method': method, 'id': str(getattr(cfg, 'id', '')),
                    'seed': int(cfg.random_seed), 'epoch': epoch + 1,
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in te.items()}})

        save_ckpt(ctx['out_dir'], epoch, model_ul, optimizer, avg,
                  filename='latest.pt', val_ce=S['val_ce'], s_val=S['S_val'])
        sel_value = S[select_by]
        # isfinite guard: checkpoint có S_val/val_ce = nan/inf KHÔNG BAO GIỜ được chọn.
        if np.isfinite(sel_value) and sel_value < best['sel_value']:
            best.update({'sel_value': sel_value, 's_val': S['S_val'], 'epoch': epoch,
                         'state': {k: v.clone() for k, v in trainable_state_dict(model_ul).items()},
                         'metrics': dict(S)})
        if history_csv:
            append_history_row(history_csv, {
                'method': method, 'id': str(getattr(cfg, 'id', '')), 'seed': int(cfg.random_seed),
                'epoch': epoch + 1, 'S_val': S['S_val'], 'G_forget': S['G_forget'],
                'G_privacy': S['G_privacy'], 'G_utility': S['G_utility'], 'val_ce': S['val_ce'],
                'mia_val': S['mia_val'], 'total_loss': avg.get('Total', 0),
                'ihl': avg.get('IHL', 0), 'uu': avg.get('UU', 0), 'mu': avg.get('MU', 0),
                'd_u_mean': avg.get('d_u_mean', 0), 'd_m_mean': avg.get('d_m_mean', 0),
                'ref_d_u': d_u0, 'ref_d_m': d_m0,
                'sel_d_u': S.get('floss_d_u', 0), 'sel_d_m': S.get('floss_d_m', 0),
                'ur': avg.get('UR', 0), 'mr': avg.get('MR', 0), 'ce': avg.get('CE', 0),
                'kd': avg.get('KD', 0), 'cum_optimizer_steps': total_steps,
            })

        # ----- CE-crossing selector: eval CE + snapshot ứng viên epoch này (gold-free) -----
        if _ce_sel is not None:
            _ce_sel.step(epoch, model_ul)

    # sau vòng lặp: chọn epoch theo 4 cách + eval selected trên D_t_final (gold-free)
    if _ce_sel is not None:
        _ce_sel.finalize(model_ul)

    if best['epoch'] is not None:
        cp = os.path.join(ctx['out_dir'], 'checkpoints')
        os.makedirs(cp, exist_ok=True)
        torch.save({'epoch': best['epoch'], 'trainable_state': best['state'],
                    's_val': best['s_val']},
                   os.path.join(cp, 'selected.pt'))

    timing = {'train_h': t_train / 3600, 'sel_h': t_sel / 3600,
              'wall_h': (_time.time() - wall0) / 3600, 'fisher_h': ctx.get('fisher_h', 0.0),
              'load_h': ctx.get('load_h', 0.0), 'last_epoch': int(cfg.unlearn_epochs) - 1}
    return timing, best, total_steps


@torch.no_grad()
def _probe_logits(model, dataset, device, cfg, n_batches=2):
    """Lấy chữ ký img_logits trên vài batch cố định để đối chiếu hai model (verify)."""
    loader = DataLoader(dataset, batch_size=int(cfg.eval_batch_size), shuffle=False)
    model.eval()
    out = []
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t for t in batch)
        inputs, _, _ = get_model_inputs(cfg, batch, device)
        out.append(safe_forward(model, inputs)[1].float().cpu())
    return torch.cat(out) if out else torch.empty(0)


def finalize_and_eval(cfg, ctx, device, method, run_id, timing, best, total_steps,
                      csv_path, extra_row=None):
    """Eval LAST(E29) + SELECTED(S_val min) trên D_t_final; ghi CSV cả hai.

    SELECTED phải giữ ĐÚNG reparameterization của LoKU: base = W* = W − B*A*. Dựng lại
    từ pretrained rồi chỉ nạp state LoRA sẽ cho base = W (MẤT phép trừ bù) → checkpoint
    sai hoàn toàn. Vì vậy ta áp lại ctx['fila_subtraction'] trước khi nạp adapter, rồi
    ĐỐI CHIẾU output với model sống ở cùng trạng thái best (sai số phải ~0)."""
    from joint_img_txt.model import ImageTextModel
    from peft import get_peft_model
    extra_row = dict(extra_row or {})
    extra_row.update({'ref_d_u': round(float(ctx['ref_scales'][0]), 4),
                      'ref_d_m': round(float(ctx['ref_scales'][1]), 4)})

    # ---- chữ ký tham chiếu: model SỐNG (base đã có FILA subtraction) ở trạng thái best ----
    ref_probe, probe_ds = None, ctx['datasets']['test_final']
    if best['epoch'] is not None:
        live = ctx['model_unlearn']
        last_state = {k: v.detach().cpu().clone() for k, v in trainable_state_dict(live).items()}
        live.load_state_dict({k: v.to(device) for k, v in best['state'].items()}, strict=False)
        ref_probe = _probe_logits(live, probe_ds, device, cfg)
        live.load_state_dict({k: v.to(device) for k, v in last_state.items()}, strict=False)

    # LAST (model sống → base đã đúng W*)
    last_model = ctx['model_unlearn'].merge_and_unload()
    final_evaluation(last_model, ctx['model_re'], ctx['datasets'], cfg, device, method, run_id,
        checkpoint_kind='last', selected_epoch=timing['last_epoch'] + 1, timing=timing,
        trainable=ctx['trainable'], total=ctx['total'], total_optimizer_steps=total_steps,
        csv_path=csv_path, gold_available=ctx['gold_available'], extra_row=extra_row)
    del last_model; torch.cuda.empty_cache()

    # SELECTED (dựng lại: pretrained → PEFT → ÁP LẠI W* = W − B*A* → nạp LoRA đã học)
    if best['epoch'] is not None:
        base = ImageTextModel.from_pretrained(ctx['base_p']).to(device)
        peft = get_peft_model(base, ctx['peft_cfg'])
        apply_fila_subtraction(peft, ctx.get('fila_subtraction', {}))
        peft.load_state_dict({k: v.to(device) for k, v in best['state'].items()}, strict=False)
        er = dict(extra_row); er['best_s_val'] = round(float(best['s_val']), 4)
        # verify: selected dựng lại phải trùng model sống ở cùng trạng thái best
        if ref_probe is not None and ref_probe.numel():
            diff = float((_probe_logits(peft, probe_ds, device, cfg) - ref_probe).abs().max())
            er['selected_rebuild_maxdiff'] = round(diff, 6)
            ok = diff < 1e-2
            print(f"{'✅' if ok else '❌'} verify selected-rebuild vs live: max|Δlogit|={diff:.6f}")
            if not ok:
                print("   ⚠️  LỆCH LỚN → checkpoint 'selected' có thể mất phép trừ bù FILA. "
                      "Kiểm tra ctx['fila_subtraction'] và peft_cfg.")
        sel_model = peft.merge_and_unload()
        final_evaluation(sel_model, ctx['model_re'], ctx['datasets'], cfg, device, method, run_id,
            checkpoint_kind='selected', selected_epoch=best['epoch'] + 1, timing=timing,
            trainable=ctx['trainable'], total=ctx['total'], total_optimizer_steps=total_steps,
            csv_path=csv_path, gold_available=ctx['gold_available'], extra_row=er)
        del sel_model; torch.cuda.empty_cache()


def final_evaluation(merged_model, model_re, datasets, cfg, device, method, run_id,
                     checkpoint_kind, selected_epoch, timing, trainable, total,
                     total_optimizer_steps, csv_path, gold_available, extra_row=None):
    """Đánh giá 1 model (đã merge, frozen) trên D_t_final + báo cáo ĐỦ metric Forget-MI:
      1−Sim(re), MIA (persample+paper), Forget/Test × {AUC, Macro-F1, Pairwise-AUC, F1}
    + extra (forget_ce/test_ce, efficiency, total_optimizer_steps). Ghi CSV, in gọn.
    MIA cuối: member=retain(subsample), non-member=D_t_final(75% test), đoán D_f."""
    import time as _time
    from datetime import datetime
    merged_model.eval()
    for p in merged_model.parameters():
        p.requires_grad = False
    bs = int(cfg.eval_batch_size)
    eval_max_retain = int(cfg.get('eval_max_retain', 512))
    t0 = _time.time()

    member_ds = subsample_dataset(datasets['retain'], eval_max_retain, int(cfg.random_seed))
    try:
        mia = run_mia(merged_model, member_ds, datasets['test_final'], datasets['forget'],
                      device, cfg, batch_size=bs, seed=int(cfg.random_seed),
                      paper_batch_size=int(cfg.get('mia_paper_batch_size', 32)))
    except Exception as e:
        print(f"⚠️  MIA cuối lỗi: {e}")
        mia = {'persample': float('nan'), 'paper': float('nan'),
               'forget_ce': float('nan'), 'nonmember_ce': float('nan')}
    forget_ce = mia.get('forget_ce', float('nan'))
    test_ce = mia.get('nonmember_ce', float('nan'))

    try:
        cs_ds = subsample_dataset(datasets['retain'], eval_max_retain, int(cfg.random_seed))
        cossim = cosine_sim_models(merged_model, model_re, cs_ds, device, cfg, batch_size=bs)
    except Exception as e:
        print(f"⚠️  CosSim lỗi: {e}"); cossim = float('nan')

    try:
        test_m = perf_metrics(merged_model, datasets['test_final'], device, cfg, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Test perf lỗi: {e}")
        test_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan'),
                  'Pairwise_AUC': float('nan')}
    try:
        forget_m = perf_metrics(merged_model, datasets['forget'], device, cfg, batch_size=bs)
    except Exception as e:
        print(f"⚠️  Forget perf lỗi: {e}")
        forget_m = {'AUC': float('nan'), 'Macro_F1': float('nan'), 'F1': float('nan'),
                    'Pairwise_AUC': float('nan')}

    gpu_peak = (torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0)
    final_eval_h = (_time.time() - t0) / 3600
    t = timing or {}
    core_h = float(t.get('train_h', 0.0)) + float(t.get('fisher_h', 0.0))

    def _r(x, n=3):
        try:
            return round(float(x), n)
        except Exception:
            return x

    results = {
        'method': method, 'checkpoint_kind': checkpoint_kind,
        'selected_epoch': selected_epoch,
        # --- metric Forget-MI ---
        'MIA': _r(mia['persample']), 'MIA_paper': _r(mia['paper']),
        '1_minus_Sim': _r(1 - cossim), 'cossim_vs_re': _r(cossim, 4),
        'Forget_AUC': _r(forget_m['AUC']), 'Forget_Macro_F1': _r(forget_m['Macro_F1']),
        'Forget_Pairwise_AUC': _r(forget_m.get('Pairwise_AUC')), 'Forget_F1': _r(forget_m.get('F1')),
        'Test_AUC': _r(test_m['AUC']), 'Test_Macro_F1': _r(test_m['Macro_F1']),
        'Test_Pairwise_AUC': _r(test_m.get('Pairwise_AUC')), 'Test_F1': _r(test_m.get('F1')),
        # --- extra (nhiều hơn Forget-MI) ---
        'forget_ce': _r(forget_ce), 'test_ce': _r(test_ce),
        'gold_retrained_available': gold_available,
        # --- efficiency ---
        'unlearn_core_hours': _r(core_h), 'train_hours': _r(t.get('train_h', 0.0)),
        'fisher_hours': _r(t.get('fisher_h', 0.0)), 'wall_hours': _r(t.get('wall_h', 0.0)),
        'final_eval_hours': _r(final_eval_h), 'gpu_peak_GB': _r(gpu_peak, 2),
        'trainable_params': int(trainable), 'total_params': int(total),
        'trainable_ratio': _r(trainable / max(1, total), 5),
        'total_optimizer_steps': int(total_optimizer_steps),
        # --- meta ---
        'id': str(cfg.get('id', '')), 'forget_pct': os.path.basename(cfg.forget_set_path),
        'seed': int(cfg.random_seed), 'run_id': run_id,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_row:
        results.update(extra_row)

    tag = f"{method} · {checkpoint_kind} · {os.path.basename(cfg.forget_set_path)} · seed {int(cfg.random_seed)}"
    print("\n" + "─" * 64)
    print(f" FINAL EVAL — {tag}  (E{selected_epoch})")
    print("─" * 64)
    print(f"  Forget  AUC {forget_m['AUC']:.3f}(↓)  MacroF1 {forget_m['Macro_F1']:.3f}  "
          f"PairAUC {forget_m.get('Pairwise_AUC', float('nan')):.3f}  CE {forget_ce:.3f}")
    print(f"  Test    AUC {test_m['AUC']:.3f}(↑)  MacroF1 {test_m['Macro_F1']:.3f}  "
          f"PairAUC {test_m.get('Pairwise_AUC', float('nan')):.3f}  CE {test_ce:.3f}")
    print(f"  MIA     persample {mia['persample']:.3f}(↓)  paper {mia['paper']:.3f}(↓)   "
          f"1−Sim(re) {1 - cossim:.3f}(↓)")
    print(f"  Time    core {core_h:.3f}h (train {t.get('train_h', 0):.3f}+fisher {t.get('fisher_h', 0):.3f}) "
          f"wall {t.get('wall_h', 0):.3f}h  steps {total_optimizer_steps}")
    print(f"  Compute GPU {gpu_peak:.2f}GB  Trainable {trainable:,} ({100*trainable/max(1,total):.2f}%)")
    print("─" * 64)

    append_row_csv(csv_path, results)
    print(f"💾 CSV: {csv_path}")
    return results
