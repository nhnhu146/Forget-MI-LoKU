#!/usr/bin/env python3
"""
adv_e30.py — LÕI E30: phân hoạch theo code gốc Forget-MI, KHÔNG selector
========================================================================
File này CHÉP RIÊNG đúng 5 thứ cần đổi so với ``adv_common.py`` và IMPORT nguyên phần
còn lại. KHÔNG hoán tên hàm (monkeypatch) ở bất cứ đâu — bản trước dùng monkeypatch và
đã dính hai lỗi runtime chỉ lộ ra khi chạy thật trên Kaggle:

  1. bỏ mất khối kiểm tra/tái tạo cache text  → mọi tập lọc còn 0 mẫu;
  2. ``precompute_og_perf(..., datasets['sel'], ...)`` — đối số được tính TRƯỚC lời gọi
     nên thay hàm bằng no-op vẫn ``KeyError: 'sel'``.

Cả hai đều là hệ quả của việc "sửa từ xa" một hàm mình không nhìn thấy toàn bộ. Ở đây
mọi thứ chạy đều nằm ngay trong file này, đọc là thấy.

--------------------------------------------------------------------------------
CODE GỐC DÙNG DỮ LIỆU NHƯ THẾ NÀO
--------------------------------------------------------------------------------
Đọc từ ``Forget-MI-main/training/forgetmi_partial.py::data_split`` bản tác giả phát
hành và ``evaluation/eval_unlearning.py``:

  forget   = MỌI dòng của subject_id trong forget_set_*.csv
  test     = fold == 'TEST' và không thuộc forget      → TOÀN BỘ 531 cặp
  retain   = phần còn lại                              → TOÀN BỘ D_r
  random   = bản nhiễu của forget
  val      = tách 10% rồi NHẬP NGƯỢC vào train (dòng 270-278) → retain không mất mẫu
             nào, và ``val_dataloader`` tạo ra rồi KHÔNG BAO GIỜ được dùng.
  LR       = ``ReduceLROnPlateau`` được tạo nhưng KHÔNG BAO GIỜ ``.step()`` → hằng số.

Tức code gốc chỉ có BỐN tập: retain / forget / random / test. Không tập chọn checkpoint.

--------------------------------------------------------------------------------
NĂM THỨ ĐỔI SO VỚI adv_common
--------------------------------------------------------------------------------
  1. ``data_split_original``   thay ``data_split_advanced``  — bỏ 2 lát cắt holdout
  2. ``build_dataset_e30``     spec còn 4 tập (bỏ r_heldout, sel)
  3. ``WarmupThenConstant``    thay ``WarmupThenPlateau``    — bỏ plateau
  4. ``setup_experiment_e30``  bỏ dòng ``precompute_og_perf`` (chỉ nuôi G_utility)
  5. ``run_training_e30``      bỏ selector S_val + CE-selector + theo dõi ``best``
     ``finalize_and_eval_e30`` bỏ nhánh dựng lại checkpoint 'selected'

                        adv_common (cũ)        adv_e30 (file này)
    retain              5.410 (90% D_r)        6.010  (TOÀN BỘ D_r)
    test_final            398 (75% D_t)          531  (TOÀN BỘ D_t)
    r_heldout             600                    — không tồn tại —
    sel                   133                    — không tồn tại —
    learning rate       giảm theo S_val        hằng số sau warmup
    hàng CSV            last + selected        CHỈ last
                                                 (số của mức 3%)

Mọi thứ khác — Fisher, FILA, LoRA, hàm mất mát, metric, MIA, ghi CSV, đo thời gian —
import thẳng từ ``adv_common``, KHÔNG chép, nên sửa ở đó thì bản này hưởng ngay.

--------------------------------------------------------------------------------
CHẶN LỆCH ÂM THẦM
--------------------------------------------------------------------------------
Năm hàm trên là bản chép, nên nếu ai sửa CHÍNH chúng trong ``adv_common`` thì bản E30
sẽ âm thầm chạy logic cũ. ``assert_core_unchanged()`` băm mã nguồn của chúng bên
``adv_common`` và so với hash chốt dưới đây; lệch thì DỪNG kèm tên hàm. Gọi nó ở đầu
mỗi driver E30.

Sau khi cố ý đồng bộ một thay đổi từ adv_common sang đây, chạy:
    python training/adv_e30.py --update-hashes
để cập nhật lại bảng hash.
"""
import os
import sys
import csv
import ast
import hashlib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd

# Phần lõi KHÔNG đổi: dùng nguyên của adv_common (một nguồn duy nhất).
# `import *` kéo theo cả os/np/torch/DataLoader/... mà adv_common đã import sẵn.
from training.adv_common import *                    # noqa: F401,F403
from training.adv_common import (                    # tên _private không vào qua `*`
    _stat, _build_text_features, _cache_names, _writable_cache_dir, _regen_text_features,
)
import training.adv_common as C

import numpy as np
import torch
from torch.utils.data import DataLoader

from joint_img_txt.model_utils import CXRImageTextDataset, RandomTranslateCrop, CenterCrop


# ============================================================================
# 0. Chặn lệch âm thầm giữa 5 hàm chép và bản gốc trong adv_common
# ============================================================================

# hash SHA-256 (12 ký tự đầu) của mã nguồn 5 hàm bên adv_common tại thời điểm chép.
CORE_HASHES = {
    'build_dataset':        '59a8950d424a',
    'WarmupThenPlateau':    '7ef801d7d6f4',
    'setup_experiment':     '628c00272102',
    'run_training':         '51a46924f2e8',
    'finalize_and_eval':    '2c34ec6e519c',
}


def _hash_defs(path):
    """{tên hàm/lớp cấp module: hash mã nguồn}."""
    src = open(path, encoding='utf-8').read()
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            seg = ast.get_source_segment(src, node) or ''
            out[node.name] = hashlib.sha256(seg.encode('utf-8')).hexdigest()[:12]
    return out


def assert_core_unchanged(strict=True):
    """DỪNG nếu một trong 5 hàm nguồn bên adv_common đã đổi kể từ lúc chép.

    Nếu không có bước này, sửa adv_common rồi chạy E30 sẽ ra một bảng số trông hợp lý
    nhưng chạy logic cũ — kiểu hỏng im lặng, nguy hiểm hơn hẳn một cái crash."""
    cur = _hash_defs(os.path.join(project_root, 'training', 'adv_common.py'))
    drift = [(n, h, cur.get(n, '<mất>')) for n, h in CORE_HASHES.items()
             if cur.get(n) != h]
    if not drift:
        print(f"✅ adv_e30: {len(CORE_HASHES)} hàm nguồn bên adv_common không đổi.")
        return True
    print('=' * 78)
    print('❌ adv_common ĐÃ ĐỔI ở hàm mà adv_e30 có bản chép riêng:')
    for n, old, new in drift:
        print(f'   {n:22} chép lúc {old}  →  hiện tại {new}')
    print('   Đồng bộ thay đổi sang adv_e30 rồi chạy:')
    print('     python training/adv_e30.py --update-hashes')
    print('=' * 78)
    if strict:
        raise SystemExit('adv_e30: lõi lệch — dừng để khỏi ra số sai.')
    return False


# ============================================================================
# 1. Phân hoạch — bám ``forgetmi_partial.py::data_split`` của tác giả
# ============================================================================

def data_split_original(split_list_path, forget_ids_path):
    """Bốn tập của code gốc: retain / test_final / forget / random.

    KHÔNG cắt r_heldout, KHÔNG cắt sel, KHÔNG dựng validation (code gốc có dựng nhưng
    nhập ngược lại vào retain rồi bỏ không dùng — xem docstring đầu file).

    Trả dict {tên: (id_map, label_map)} — cùng định dạng ``data_split_advanced``.
    """
    forget_set = set(pd.read_csv(forget_ids_path).astype(str).subject_id.values)

    retain_l, retain_id = {}, {}
    test_l,   test_id   = {}, {}
    forget_l, forget_id = {}, {}
    rand_l,   rand_id   = {}, {}

    with open(split_list_path, 'r') as f:
        reader = csv.reader(f)
        hdr = next(reader) or []
        # đọc theo TÊN cột (bền với IU) — giống data_split_advanced
        ix = {str(n).strip().lower(): i for i, n in enumerate(hdr)}
        i_sub = ix.get('subject_id', 0)
        i_key = ix.get('dicom_id', 2)                                  # key = ảnh
        i_val = ix.get('study_id', ix.get('report_id', 1))             # value = text
        i_lab = ix.get('edeme_severity', ix.get('label', ix.get('severity', 3)))
        i_spl = ix.get('fold', ix.get('split', len(hdr) - 1))
        for row in reader:
            if len(row) <= max(i_sub, i_key, i_val, i_lab, i_spl):
                continue
            try:
                sev = float(row[i_lab])
            except ValueError:
                continue                                   # bỏ header lặp / dòng hỏng
            rid, subj = row[i_key], row[i_sub]
            # THỨ TỰ NÀY BÁM ĐÚNG CODE GỐC: forget xét TRƯỚC, nên bệnh nhân cần quên bị
            # gỡ khỏi CẢ train LẪN test. (Thực tế không bệnh nhân nào của forget_set nằm
            # trong fold TEST, nhưng giữ đúng thứ tự để khỏi lệch.)
            if subj in forget_set:
                forget_l[rid] = [sev]; forget_id[rid] = row[i_val]
                rand_l[rid]   = [sev]; rand_id[rid]   = row[i_val]
            elif str(row[i_spl]).strip().upper() == 'TEST':
                test_l[rid] = [sev]; test_id[rid] = row[i_val]
            else:
                retain_l[rid] = [sev]; retain_id[rid] = row[i_val]

    print(f"[split-e30] retain={len(retain_id)} (TOÀN BỘ D_r)  forget={len(forget_id)}  "
          f"test_final={len(test_id)} (TOÀN BỘ D_t)  — không cắt r_heldout/sel")
    return {
        'retain':     (retain_id, retain_l),
        'test_final': (test_id,   test_l),
        'forget':     (forget_id, forget_l),
        'random':     (rand_id,   rand_l),
    }


# ============================================================================
# 2. build_dataset — chép từ adv_common, spec còn 4 tập
# ============================================================================

def build_dataset_e30(args, tokenizer):
    """Như ``adv_common.build_dataset`` nhưng dùng phép chia code gốc và chỉ 4 tập.

    GIỮ NGUYÊN khối kiểm tra + tái tạo cache text: cache trong /kaggle/input là
    read-only và report_id trong đó có thể KHÔNG khớp study_id của split → mọi tập lọc
    còn 0 mẫu. Bỏ khối này chính là lỗi ở lần chạy ref_m3 đầu tiên."""
    processor, features, noisy_features = _build_text_features(args, tokenizer)
    num_labels = len(processor.get_labels())

    all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
               for f in features}
    noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
                 for f in noisy_features}

    def _resolve(r, txt_map):
        """Dung sai prefix 's' + int/str: cache report_id có thể là int trong khi split là str."""
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

    splits = data_split_original(args.data_split_path, args.forget_set_path)

    # ---- cache integrity: 0 khớp → REGENERATE sang writable dir ----
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
            print(f"⚠️  Cache mismatch (0/{len(sample_ids)}) → regenerate từ all_data.tsv "
                  f"sang {wd} (~5 phút)...")
            features, noisy_features = _regen_text_features(args, tokenizer, wd)
        all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
                   for f in features}
        noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
                     for f in noisy_features}
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

    rand_perturb = as_bool(getattr(args, 'use_noise', True))
    rand_txt = noisy_txt if rand_perturb else all_txt
    # (tên_dataset, key_split, transform, txt_map, perturb_img) — 4 tập, không r_heldout/sel
    spec = [
        ('retain',     'retain',     train_trans, all_txt,  False),
        ('test_final', 'test_final', eval_trans,  all_txt,  False),
        ('forget',     'forget',     train_trans, all_txt,  False),
        ('random',     'random',     train_trans, rand_txt, rand_perturb),
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
            num_labels=num_labels,
        )
        print(f"  [{name}] {len(datasets[name])} samples")

    empty = [n for n in ('retain', 'forget', 'test_final') if len(datasets[n]) == 0]
    if empty:
        raise RuntimeError(
            f"Dataset rỗng {empty} (0 sample) — text/cache KHÔNG khớp split. "
            f"Kiểm tra {args.text_data_dir}/all_data.tsv và report_id trong "
            f"{args.data_split_path}.")
    return datasets, num_labels


# ============================================================================
# 3. Scheduler — warmup rồi HẰNG SỐ (code gốc không bao giờ step plateau)
# ============================================================================

class WarmupThenConstant:
    """Warmup tuyến tính trong ``warmup_steps`` lần cập nhật đầu, sau đó LR = base.

    Chép từ ``WarmupThenPlateau``, BỎ ``ReduceLROnPlateau``. Plateau ở bản cũ được nuôi
    bằng ``S_val`` — mà S_val là selector. Giữ plateau mà không có S_val thì nó không
    bao giờ step; bỏ hẳn cho khỏi hiểu nhầm là LR còn thích ứng.

    Kết quả: LR hằng số sau warmup — trùng chế độ của baseline Forget-MI (nó tạo
    scheduler rồi không bao giờ gọi ``.step()``), nên hai phía cùng một chế độ tối ưu.

    LR luôn được đặt TRƯỚC lần cập nhật dùng nó: constructor đặt LR cho update #1, và
    mỗi ``batch_step()`` (gọi sau ``optimizer.step()``) đặt LR cho update KẾ TIẾP.
    Với warmup_steps=3, dãy LR thực dùng là ⅓·LR → ⅔·LR → LR → LR …"""

    def __init__(self, optimizer, base_lr, warmup_steps):
        self.opt = optimizer
        self.base_lr = float(base_lr)
        self.warmup_steps = max(1, int(warmup_steps))
        self.updates_done = 0
        self.in_warmup = True
        self._set_lr_for_update(1)

    def _set_lr(self, lr):
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        return lr

    def _set_lr_for_update(self, k):
        return self._set_lr(self.base_lr * min(k, self.warmup_steps) / self.warmup_steps)

    def batch_step(self):
        self.updates_done += 1
        if not self.in_warmup:
            return
        if self.updates_done >= self.warmup_steps:
            self.in_warmup = False
            self._set_lr(self.base_lr)
        else:
            self._set_lr_for_update(self.updates_done + 1)

    def epoch_step(self, *_a, **_k):
        """Không làm gì — giữ chữ ký để code gọi nhầm không nổ."""
        return None


# ============================================================================
# 4. setup_experiment — chép từ adv_common, BỎ precompute_og_perf
# ============================================================================

def setup_experiment_e30(cfg, device, rank_alloc_fn=None):
    """Dựng context E30: models → dataset (4 tập) → Fisher → (rank_alloc_fn) →
    PEFT+FILA → LoRA-only → optimizer → dataloaders → thang d⁰ → scheduler.

    Khác ``adv_common.setup_experiment`` đúng hai chỗ:
      * gọi ``build_dataset_e30`` (phép chia code gốc);
      * BỎ ``ctx['og_perf'] = precompute_og_perf(..., datasets['sel'], ...)`` — hàm đó
        chỉ nuôi G_utility của S_val, và tập 'sel' không còn tồn tại.
    ``compute_epoch0_scales`` giữ lại: rẻ, và ``finalize_and_eval`` ghi ref_d_u/ref_d_m
    vào CSV nên bỏ đi sẽ mất hai cột."""
    import time as _time
    if as_bool(getattr(cfg, 'ce_selector', False)):
        raise SystemExit("adv_e30 không hỗ trợ ce_selector=1 (cần tập 'sel'). Đặt ce_selector=0.")

    print('=' * 78)
    print('adv_e30: PHÂN HOẠCH THEO CODE GỐC')
    print('  retain = TOÀN BỘ D_r · test_final = TOÀN BỘ D_t')
    print('  không r_heldout, không sel, không S_val, LR hằng số sau warmup')
    print("  CSV chỉ có hàng 'last' (E30) — CHỈ so được với run khác cũng chạy adv_e30")
    print('=' * 78)

    ctx = build_base_models(cfg, device)
    print("📚 Building dataset (4 tập code gốc)...")
    _t = _time.time()
    datasets, num_labels = build_dataset_e30(cfg, ctx['tokenizer'])
    ctx['datasets'] = datasets; ctx['num_labels'] = num_labels
    ctx['load_h'] = (_time.time() - _t) / 3600

    target_modules = list(getattr(cfg, 'lora_target_modules', ['query', 'key', 'value']))
    extra = str(getattr(cfg, 'lora_extra_target_modules', '') or '').strip()
    if extra:
        add = [m.strip() for m in extra.split('|') if m.strip()]
        target_modules = target_modules + [m for m in add if m not in target_modules]
        print(f"➕ Mở rộng LoRA target (+{len(add)}): {add}")
    img_last_k = int(getattr(cfg, 'lora_image_last_k_blocks', 0))
    img_inc_fc1 = bool(getattr(cfg, 'lora_image_include_fc1', False))
    image_targets = resolve_image_targets(ctx['model_og'], img_last_k, img_inc_fc1)
    if image_targets:
        target_modules = target_modules + image_targets
    ctx['image_targets'] = image_targets

    # warm-up: loại chi phí khởi tạo CUDA khỏi phép đo (không đổi checkpoint/dữ liệu)
    try:
        _wb = next(iter(DataLoader(datasets['forget'], batch_size=2)))
        with torch.no_grad():
            _wi, _, _ = get_model_inputs(cfg, _wb, device)
            safe_forward(ctx['model_og'], _wi)
        del _wb, _wi
    except Exception as _e:
        print(f"   (bỏ qua warm-up: {_e})")
    ctx['peaks'] = {}

    print(f"🎯 LoRA targets (trước phân rank): {target_modules}")
    fisher_bs = int(getattr(cfg, 'fisher_batch_size', cfg.unlearn_batch_size))
    fisher_n = int(getattr(cfg, 'fisher_max_samples', 256))
    _skip_fisher = as_bool(getattr(cfg, 'loku_random_init', False)) and rank_alloc_fn is None
    reset_gpu_peak()
    if _skip_fisher:
        print("🎲 ABLATION w/o Fisher/FILA: BỎ luôn bước ước lượng Fisher (T_fisher = 0)")
        f_imp = r_imp = {}
        ctx['fisher_seconds'] = 0.0
        ctx['peaks']['fisher'] = (0.0, 0.0)
        ctx['fisher_h'] = 0.0
    else:
        with CudaTimer() as _tf:
            f_imp = compute_fisher_importance(ctx['model_og'],
                DataLoader(datasets['forget'], batch_size=fisher_bs), device, target_modules, cfg, fisher_n)
            r_imp = compute_fisher_importance(ctx['model_og'],
                DataLoader(datasets['retain'], batch_size=fisher_bs), device, target_modules, cfg, fisher_n)
        ctx['fisher_seconds'] = _tf.elapsed
        ctx['peaks']['fisher'] = get_gpu_peak()
        ctx['fisher_h'] = _tf.elapsed / 3600
        print(f"⏱  T_fisher = {_tf.elapsed:.1f}s  (peak alloc {ctx['peaks']['fisher'][0]:.2f} GB)")
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

    img_sub = float(getattr(cfg, 'loku_image_subtract_scale',
                            getattr(cfg, 'loku_subtract_scale', 0.0)))
    fila_subtraction = {}
    reset_gpu_peak()
    with CudaTimer() as _tl:
        model, peft_cfg = build_peft(ctx['model_unlearn'], cfg, target_modules,
                                     rank_pattern=rank_pattern, alpha_pattern=alpha_pattern)
        if getattr(cfg, 'loku_random_init', False):
            print("🎲 ABLATION: loku_random_init=True → BỎ Fisher/FILA init")
        else:
            fila_subtraction = apply_loku_soft_init(model, f_imp, r_imp, target_modules,
                r=int(cfg.lora_r), init_scale=float(getattr(cfg, 'loku_init_scale', 0.05)),
                subtract_scale=float(getattr(cfg, 'loku_subtract_scale', 0.0)),
                image_target_names=set(image_targets), image_subtract_scale=img_sub,
                rank_pattern=rank_pattern)
        enforce_lora_only(model)
    ctx['fila_seconds'] = _tl.elapsed
    ctx['peaks']['fila'] = get_gpu_peak()
    print(f"⏱  T_fila = {_tl.elapsed:.1f}s  (peak alloc {ctx['peaks']['fila'][0]:.2f} GB)")
    ctx['peft_cfg'] = peft_cfg
    ctx['fila_subtraction'] = fila_subtraction
    ctx['model_unlearn'] = model
    ctx['trainable'], ctx['total'] = count_params(model)
    print(f"📊 Trainable: {ctx['trainable']:,}/{ctx['total']:,} "
          f"({100*ctx['trainable']/ctx['total']:.3f}%)")

    _gm = str(getattr(cfg, 'gate_mode', 'per_batch')).lower()
    if _gm == 'fixed_shared':
        print("🚪 Fusion gate: fixed_shared — 1 Gate cố định dùng chung [CẢI TIẾN khóa luận]")
    else:
        print("🚪 Fusion gate: per_batch — tạo MỚI mỗi batch, riêng từng vai trò, "
              "KHÔNG vào optimizer (bám code gốc Forget-MI)")
    ctx['gate_mode'] = _gm
    ctx['optimizer'] = make_optimizer(model, cfg)

    nw = min(int(getattr(cfg, 'num_cpu_workers', 2)), 2)
    ctx['nw'] = nw
    _pin = dl_pin_memory(cfg)
    ctx['pin_memory'] = _pin
    ctx['forget_dl'] = DataLoader(datasets['forget'],
        sampler=AlignedSampler(len(datasets['forget']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw, pin_memory=_pin)
    ctx['rand_dl'] = DataLoader(datasets['random'],
        sampler=AlignedSampler(len(datasets['random']), shuffle=True, seed=42),
        batch_size=cfg.unlearn_batch_size, num_workers=nw, pin_memory=_pin)

    # d⁰ chỉ để in tiến trình + ghi ref_d_u/ref_d_m vào CSV (KHÔNG tham gia loss)
    ctx['ref_scales'] = compute_epoch0_scales(model, ctx['model_og'],
        ctx['forget_dl'], ctx['rand_dl'], cfg, device, ctx=ctx)
    ctx['og_perf'] = None            # E30: không có 'sel' → không có G_utility

    total_steps = int(cfg.unlearn_epochs)
    warmup_steps = max(1, int(float(getattr(cfg, 'warmup_proportion', 0.1)) * total_steps))
    ctx['scheduler'] = WarmupThenConstant(ctx['optimizer'], float(cfg.learning_rate), warmup_steps)
    print(f"📉 LR: warmup {warmup_steps} update rồi HẰNG SỐ {float(cfg.learning_rate):g} "
          f"(không plateau — không có S_val)")
    ctx['total_planned_steps'] = total_steps
    ctx['forget_batches'] = max(1, len(ctx['forget_dl']))
    print(f"✅ adv_e30 ctx sẵn sàng: retain={len(datasets['retain'])} "
          f"forget={len(datasets['forget'])} test_final={len(datasets['test_final'])}")
    return ctx


# ============================================================================
# 5. run_training — chép từ adv_common, BỎ selector
# ============================================================================

def run_training_e30(cfg, ctx, device, method, weight_fn):
    """Vòng train E30. weight_fn(cfg, ctx, epoch, global_frac) → (weights, forget_active,
    guard_ihl) quyết định trọng số TỪNG BATCH.

    CADENCE giữ nguyên code gốc Forget-MI: các batch chỉ backward để TÍCH LŨY gradient;
    cuối epoch mới clip rồi ``optimizer.step()`` MỘT lần.

    Khác ``adv_common.run_training``: KHÔNG tính S_val, KHÔNG gọi ce_selector, KHÔNG
    theo dõi ``best``. Chỉ lưu ``latest.pt`` ở epoch cuối. Trả (timing, total_steps)."""
    import time as _time
    model_ul = ctx['model_unlearn']
    optimizer = ctx['optimizer']; sched = ctx['scheduler']; datasets = ctx['datasets']
    grad_clip = float(getattr(cfg, 'grad_clip', 1.0))
    history_csv = getattr(cfg, 'history_csv_path', None)
    trainable_params = [p for p in model_ul.parameters() if p.requires_grad]   # chỉ LoRA
    epoch_train_s = []
    peaks = ctx.setdefault('peaks', {})
    peak_train = (0.0, 0.0)
    total_steps = 0
    t_train = t_ckpt = t_diag = 0.0
    wall0 = _time.time()
    n_epochs = int(cfg.unlearn_epochs)

    # ---- RESET RNG NGAY TRƯỚC VÒNG TRAIN (bắt buộc cho ablation) ----
    # Gate tạo MỚI mỗi batch từ RNG toàn cục và chạy ở train-mode, nên quỹ đạo huấn
    # luyện phụ thuộc mạnh vào trạng thái RNG lúc bắt đầu train. Giai đoạn khởi tạo tiêu
    # tốn RNG khác nhau giữa các biến thể (FILA gọi svd_lowrank, ablation thì không).
    set_seed(int(cfg.random_seed))
    for epoch in range(n_epochs):
        model_ul.train()
        _nbn = set_bn_eval(model_ul)     # BN giữ eval — đúng chuẩn LoRA base đóng băng
        if epoch == 0:
            print(f"   🧊 giữ {_nbn} BatchNorm ở eval-mode trong lúc train (base đóng băng)")
        n_batches = max(1, int(ctx.get('forget_batches', 1)))
        n_ep = max(1, n_epochs)
        agg = {}; steps = 0
        reset_gpu_peak()
        _tt = CudaTimer(); _tt.__enter__()
        # retain_dl dựng TRONG timer: Forget-MI cũng dựng lại mỗi epoch bên trong khoảng
        # đo của nó, nên chi phí spawn worker + pin_memory phải tính ở cả hai bên.
        retain_dl = DataLoader(datasets['retain'],
                               sampler=torch.utils.data.RandomSampler(datasets['retain']),
                               batch_size=cfg.unlearn_batch_size, num_workers=ctx['nw'],
                               pin_memory=ctx.get('pin_memory', True))
        optimizer.zero_grad(set_to_none=True)   # gradient TÍCH LŨY suốt epoch
        for bi, (fb, rb, retb) in enumerate(zip(ctx['forget_dl'], ctx['rand_dl'], retain_dl)):
            global_frac = min(1.0, (epoch + bi / n_batches) / n_ep)
            weights, forget_active, guard_ihl = weight_fn(cfg, ctx, epoch, global_frac)
            loss, comp = combined_batch_loss(cfg, ctx, fb, rb, retb, device, weights,
                                             forget_active=forget_active, guard_ihl=guard_ihl)
            loss.backward()                   # KHÔNG step ở đây
            steps += 1
            for k, v in comp.items():
                agg[k] = agg.get(k, 0.0) + v
        if grad_clip and trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        sched.batch_step()
        total_steps += 1
        avg = {k: v / max(steps, 1) for k, v in agg.items()}
        _tt.__exit__(None, None, None)
        epoch_train_s.append(_tt.elapsed)
        t_train += _tt.elapsed
        peak_train = tuple(max(a, b) for a, b in zip(peak_train, get_gpu_peak()))

        d_u0, d_m0 = ctx['ref_scales']
        print(f"[{method} E{epoch:02d}] loss={avg.get('Total', 0):+.3f} "
              f"UU={avg.get('UU', 0):+.4f} MU={avg.get('MU', 0):+.4f} "
              f"(d̄ {avg.get('d_u_mean', 0):.3f}/{avg.get('d_m_mean', 0):.2f} "
              f"vs d⁰ {d_u0:.3f}/{d_m0:.2f}) "
              f"CE={avg.get('CE', 0):.3f} KD={avg.get('KD', 0):.4f} "
              f"IHL={avg.get('IHL', 0):.3f} | E30 (không selector)")

        # ---- chẩn đoán tuỳ chọn: eval mỗi epoch trên D_t_final ----
        if as_bool(getattr(cfg, 'eval_test_every_epoch', False)):
            with CudaTimer() as _t2:
                te = eval_on_test_final(model_ul, ctx['model_re'], datasets, cfg, device,
                                        light=as_bool(getattr(cfg, 'eval_test_light', True)))
            t_diag += _t2.elapsed
            print(f"    📊 [test E{epoch:02d}] Df-AUC {te['Df_AUC']:.3f} Df-F1 {te['Df_F1']:.3f}  "
                  f"Dt-AUC {te['Dt_AUC']:.3f} Dt-F1 {te['Dt_F1']:.3f}  "
                  f"MIA {te['MIA']:.3f}/{te['MIA_paper']:.3f}  "
                  f"fce/tce {te['forget_ce']:.2f}/{te['test_ce']:.2f}")
            tcsv = getattr(cfg, 'test_history_csv_path', None)
            if tcsv:
                append_history_row(tcsv, {'method': method, 'id': str(getattr(cfg, 'id', '')),
                    'seed': int(cfg.random_seed), 'epoch': epoch + 1,
                    **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in te.items()}})

        # ---- checkpoint: CHỈ epoch cuối (không có 'selected' để chọn) ----
        if epoch == n_epochs - 1:
            with CudaTimer() as _tc:
                save_ckpt(ctx['out_dir'], epoch, model_ul, optimizer, avg,
                          filename='latest.pt', val_ce=None, s_val=None)
            t_ckpt += _tc.elapsed

        if history_csv:
            append_history_row(history_csv, {
                'method': method, 'id': str(getattr(cfg, 'id', '')), 'seed': int(cfg.random_seed),
                'epoch': epoch + 1,
                'total_loss': avg.get('Total', 0),
                'ihl': avg.get('IHL', 0), 'uu': avg.get('UU', 0), 'mu': avg.get('MU', 0),
                'd_u_mean': avg.get('d_u_mean', 0), 'd_m_mean': avg.get('d_m_mean', 0),
                'ref_d_u': d_u0, 'ref_d_m': d_m0,
                'ur': avg.get('UR', 0), 'mr': avg.get('MR', 0), 'ce': avg.get('CE', 0),
                'kd': avg.get('KD', 0), 'cum_optimizer_steps': total_steps,
            })

    peaks['train'] = peak_train
    est = _stat(epoch_train_s)
    print(f"⏱  T_train = {t_train:.1f}s ({t_train/3600:.4f}h) · epoch: mean {est['mean']:.2f}s "
          f"± {est['std']:.2f} (min {est['min']:.2f} / max {est['max']:.2f})")
    if t_diag:
        print(f"⏱  chẩn đoán (eval mỗi epoch) {t_diag:.1f}s — không vào core")
    timing = {
        'fisher_seconds': float(ctx.get('fisher_seconds', 0.0)),
        'fila_seconds': float(ctx.get('fila_seconds', 0.0)),
        'train_seconds': float(t_train),
        'selection_seconds': 0.0,                  # E30: không có khâu chọn checkpoint
        'ckpt_seconds': float(t_ckpt),
        'diagnostic_seconds': float(t_diag),
        'selector': 'none(e30)',
        'checkpoint_policy': 'last',
        'split_protocol': 'e30_original',
        'eval_seconds': 0.0,                       # điền ở finalize_and_eval_e30
        'epoch_train_stat': est,
        'peaks': peaks,
        'optimizer_updates': int(total_steps),
        # ---- khóa cũ (giữ để CSV/notebook không vỡ) ----
        'train_h': t_train / 3600, 'sel_h': 0.0,
        'wall_h': (_time.time() - wall0) / 3600,
        'fisher_h': ctx.get('fisher_h', 0.0),
        'load_h': ctx.get('load_h', 0.0),
        'last_epoch': n_epochs - 1,
    }
    return timing, total_steps


# ============================================================================
# 6. finalize_and_eval — chép từ adv_common, BỎ nhánh 'selected'
# ============================================================================

def finalize_and_eval_e30(cfg, ctx, device, method, run_id, timing, total_steps,
                          csv_path, extra_row=None):
    """Eval checkpoint CUỐI (E30) trên D_t_final rồi ghi CSV.

    Khác ``adv_common.finalize_and_eval``: bỏ toàn bộ nhánh dựng lại checkpoint
    'selected' (~40 dòng) — E30 không bao giờ có checkpoint đó."""
    extra_row = dict(extra_row or {})
    extra_row.update({'ref_d_u': round(float(ctx['ref_scales'][0]), 4),
                      'ref_d_m': round(float(ctx['ref_scales'][1]), 4),
                      'split_protocol': 'e30_original'})
    reset_gpu_peak()
    _te = CudaTimer(); _te.__enter__()

    last_model = ctx['model_unlearn'].merge_and_unload()
    final_evaluation(last_model, ctx['model_re'], ctx['datasets'], cfg, device, method, run_id,
        checkpoint_kind='last', selected_epoch=timing['last_epoch'] + 1, timing=timing,
        trainable=ctx['trainable'], total=ctx['total'], total_optimizer_steps=total_steps,
        csv_path=csv_path, gold_available=ctx['gold_available'], extra_row=extra_row)
    del last_model; torch.cuda.empty_cache()

    _te.__exit__(None, None, None)
    timing['eval_seconds'] = _te.elapsed
    timing.setdefault('peaks', {})['eval'] = get_gpu_peak()
    print(f"⏱  T_eval = {_te.elapsed:.1f}s ({_te.elapsed/3600:.4f}h) — giao thức, KHÔNG vào core")
    timing['precision'] = describe_precision({'model_ul': ctx.get('model_unlearn'),
                                              'model_og': ctx.get('model_og'),
                                              'model_re': ctx.get('model_re')})
    timing['pin_memory'] = bool(ctx.get('pin_memory', dl_pin_memory(cfg)))
    save_timing_json(os.path.join(ctx['out_dir'], f'timing_{method}_{run_id}.json'),
                     method, cfg, timing, ctx['trainable'], ctx['total'],
                     extra={'run_id': run_id, 'selected_epoch': None,
                            'split_protocol': 'e30_original',
                            'load_seconds': float(ctx.get('load_h', 0.0)) * 3600,
                            'lora_target_modules': list(ctx.get('target_modules', [])),
                            **{k: v for k, v in (extra_row or {}).items()
                               if isinstance(v, (int, float, str, bool))}})


# ============================================================================
# CLI phụ: cập nhật bảng hash sau khi cố ý đồng bộ thay đổi từ adv_common
# ============================================================================

def _update_hashes():
    me = os.path.join(project_root, 'training', 'adv_e30.py')
    cur = _hash_defs(os.path.join(project_root, 'training', 'adv_common.py'))
    src = open(me, encoding='utf-8').read()
    for name in CORE_HASHES:
        new = cur.get(name)
        if new is None:
            raise SystemExit(f"adv_common không còn hàm '{name}' — xem lại adv_e30.")
        old_line = f"'{name}':"
        i = src.index(old_line)
        j = src.index('\n', i)
        pad = ' ' * max(1, 21 - len(name))
        src = src[:i] + f"'{name}':{pad}'{new}'," + src[j:]
    open(me, 'w', encoding='utf-8').write(src)
    print('Đã cập nhật CORE_HASHES:')
    for name in CORE_HASHES:
        print(f'   {name:22} {cur[name]}')


if __name__ == '__main__':
    if '--update-hashes' in sys.argv:
        _update_hashes()
    else:
        assert_core_unchanged(strict=False)
