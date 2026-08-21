#!/usr/bin/env python3
"""
adv_e30.py — PHÂN HOẠCH DỮ LIỆU THEO ĐÚNG CODE GỐC FORGET-MI (bản E30, KHÔNG selector)
=====================================================================================
File này ĐỘC LẬP với ``adv_common.py`` — bản cũ giữ nguyên 100%, mọi kết quả đã chạy
vẫn tái lập được. Dùng file này khi muốn chạy 30 epoch với dữ liệu được sử dụng GIỐNG
CODE GỐC nhất, và không dính bất kỳ dấu vết nào của selector S_val.

--------------------------------------------------------------------------------
1. CODE GỐC DÙNG DỮ LIỆU NHƯ THẾ NÀO
--------------------------------------------------------------------------------
Đọc từ ``Forget-MI-main/training/forgetmi_partial.py::data_split`` (bản tác giả phát
hành) và ``evaluation/eval_unlearning.py``:

  forget   = MỌI dòng của các subject_id nằm trong forget_set_*.csv
  test     = fold == 'TEST' và không thuộc forget          → TOÀN BỘ 531 cặp
  retain   = phần còn lại                                  → TOÀN BỘ D_r
  random   = bản nhiễu của forget (ảnh Gaussian + text đồng nghĩa)
  val      = train_test_split(retain, test_size=0.1) NHƯNG được nhập NGƯỢC lại vào
             retain ngay sau đó (dòng 270-278 bản gốc) → retain KHÔNG mất mẫu nào,
             và ``val_dataloader`` được tạo ra rồi KHÔNG BAO GIỜ ĐƯỢC DÙNG.

  MIA (eval_unlearning.py): member = retain ĐẦY ĐỦ, non-member = test ĐẦY ĐỦ,
  SVC(C=3, rbf, gamma='auto') suy đoán trên forget. Không lấy mẫu con, không cân bằng.

  Learning rate: ``ReduceLROnPlateau`` được TẠO ra nhưng KHÔNG BAO GIỜ ``.step()``
  → LR là hằng số. Không có warmup (``warmup_proportion`` được truyền vào nhưng
  không hàm nào đọc).

Kết luận: code gốc KHÔNG hề có tập validation thật, KHÔNG có tập chọn checkpoint.
Chỉ có bốn tập: retain / forget / random / test.

--------------------------------------------------------------------------------
2. FILE NÀY LÀM GÌ
--------------------------------------------------------------------------------
``data_split_original()`` tái lập đúng bốn tập trên — BỎ HẲN hai lát cắt của nhánh
nâng cao (``r_heldout`` 10% của D_r và ``sel`` 25% của D_t):

                        adv_common (cũ)        adv_e30 (file này)
    retain              5.410 (90% D_r)        6.010  (TOÀN BỘ D_r)
    test_final            398 (75% D_t)          531  (TOÀN BỘ D_t)
    r_heldout             600                    — không tồn tại —
    sel                   133                    — không tồn tại —
                                                 (số của mức 3%)

Tên khoá ``retain`` / ``test_final`` / ``forget`` / ``random`` được GIỮ NGUYÊN để mọi
hàm phía sau của ``adv_common`` (vòng train, loss, metric, MIA, ghi CSV) chạy y hệt,
không phải sửa gì. Chỉ khác: nội dung của chúng giờ là tập đầy đủ.

--------------------------------------------------------------------------------
3. VÌ SAO KHÔNG CẦN VIẾT LẠI VÒNG TRAIN
--------------------------------------------------------------------------------
Đã kiểm chứng trong ``adv_common.run_training``: khi ``skip_selection=1`` thì

  * ``selection_metrics()`` KHÔNG được gọi  → không đọc r_heldout/sel, không tiêu RNG;
  * ``sched.epoch_step(S_val)`` KHÔNG được gọi → ReduceLROnPlateau không bao giờ hạ LR,
    LR giữ nguyên base sau warmup (giống chế độ LR hằng số của code gốc);
  * ``best`` không bao giờ cập nhật → không có checkpoint 'selected', CSV chỉ ra hàng
    'last' = đúng trọng số sau epoch cuối.

Hai hàm ``compute_epoch0_scales()`` và ``precompute_og_perf()`` chạy TRƯỚC
``set_seed()`` của vòng train, nên dù có chạy cũng không làm lệch quỹ đạo huấn luyện.
Dù vậy ``precompute_og_perf()`` vẫn đọc ``sel`` — tập không còn tồn tại — nên file này
vô hiệu hoá nó (xem ``setup_experiment_e30``).

Vì vậy khác biệt DUY NHẤT về kết quả giữa bản này và bản cũ chạy ``SKIP_SEL=1`` là
kích thước ``retain`` và ``test_final``. Đó chính là điều ta muốn.

--------------------------------------------------------------------------------
4. CÁCH DÙNG
--------------------------------------------------------------------------------
    import training.adv_e30 as E
    ctx = E.setup_experiment_e30(cfg, device)          # thay cho C.setup_experiment
    timing, best, steps = C.run_training(cfg, ctx, device, METHOD, weight_fn)
    C.finalize_and_eval(cfg, ctx, device, METHOD, run_id, timing, best, steps, csv)

``setup_experiment_e30`` tự bật ``skip_selection`` và báo lỗi nếu ``ce_selector`` bật.
"""
import os
import sys
import csv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
import torch

import training.adv_common as C
from joint_img_txt.model_utils import CXRImageTextDataset, RandomTranslateCrop, CenterCrop

# Giữ tham chiếu GỐC ngay lúc import. ``forgetmi_e30.py`` hoán ``C.setup_experiment``
# thành ``setup_experiment_e30``; nếu hàm dưới đây gọi ``C.setup_experiment`` theo tên
# thì sẽ tự gọi lại chính nó → đệ quy vô hạn. Bind sớm để tránh.
_ORIG_SETUP_EXPERIMENT = C.setup_experiment


# ============================================================================
# 1. Phân hoạch — bám ``forgetmi_partial.py::data_split`` của tác giả
# ============================================================================

def data_split_original(split_list_path, forget_ids_path, **_ignored):
    """Bốn tập của code gốc: retain / test_final / forget / random.

    KHÔNG cắt r_heldout, KHÔNG cắt sel, KHÔNG dựng validation (code gốc có dựng
    nhưng nhập ngược lại vào retain rồi bỏ không dùng — xem docstring đầu file).

    ``**_ignored`` nuốt ``test_sel_splits`` / ``retain_heldout_splits`` để chữ ký
    tương thích với ``data_split_advanced``, nhưng chúng KHÔNG có tác dụng gì ở đây.

    Trả dict {tên: (id_map, label_map)} — cùng định dạng ``data_split_advanced``.
    """
    for k, v in _ignored.items():
        if k in ('test_sel_splits', 'retain_heldout_splits'):
            print(f"   (adv_e30: bỏ qua {k}={v} — bản E30 không cắt holdout)")

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
            # THỨ TỰ NÀY BÁM ĐÚNG CODE GỐC: forget được xét TRƯỚC, nên một bệnh nhân
            # cần quên bị gỡ khỏi CẢ train LẪN test (thực tế không bệnh nhân nào của
            # forget_set nằm trong fold TEST, nhưng giữ đúng thứ tự để khỏi lệch).
            if subj in forget_set:
                forget_l[rid] = [sev]; forget_id[rid] = row[i_val]
                rand_l[rid]   = [sev]; rand_id[rid]   = row[i_val]
            elif str(row[i_spl]).strip().upper() == 'TEST':
                test_l[rid] = [sev]; test_id[rid] = row[i_val]
            else:
                retain_l[rid] = [sev]; retain_id[rid] = row[i_val]

    print(f"[split-e30] retain={len(retain_id)} (TOÀN BỘ D_r)  "
          f"forget={len(forget_id)}  test_final={len(test_id)} (TOÀN BỘ D_t)  "
          f"— không cắt r_heldout/sel")
    return {
        'retain':     (retain_id, retain_l),
        'test_final': (test_id,   test_l),
        'forget':     (forget_id, forget_l),
        'random':     (rand_id,   rand_l),
    }


# ============================================================================
# 2. Dựng dataset — sao cấu trúc ``adv_common.build_dataset`` với spec 4 tập
# ============================================================================

def build_dataset_e30(args, tokenizer):
    """Như ``adv_common.build_dataset`` nhưng chỉ dựng 4 tập của code gốc.

    Mọi bước nặng (đọc/regen cache text, resolve report_id) đều gọi lại hàm của
    ``adv_common`` để hai nhánh không bao giờ lệch nhau ở khâu xử lý văn bản.
    """
    processor, features, noisy_features = C._build_text_features(args, tokenizer)
    num_labels = len(processor.get_labels())

    all_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
               for f in features}
    noisy_txt = {f.report_id: (f.input_ids, f.input_mask, f.segment_ids, f.label_id)
                 for f in noisy_features}

    def _resolve(r, txt_map):
        """Dung sai prefix 's' + int/str — sao y adv_common.build_dataset."""
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

    # ---- KIỂM TRA + TÁI TẠO CACHE TEXT (sao y adv_common.build_dataset) ----
    # BẮT BUỘC phải có. Cache trên Kaggle input là read-only và report_id trong đó có
    # thể KHÔNG khớp study_id của split đang dùng → mọi tập lọc còn 0 mẫu.
    # Bỏ khối này đi thì build_dataset_e30 chết với "Dataset rỗng" trong khi phép chia
    # hoàn toàn đúng (đã dính đúng lỗi này ở lần chạy ref_m3 đầu tiên).
    retain_ids = splits['retain'][0]
    sample_ids = list(retain_ids.values())[:50]
    matches = sum(1 for rid in sample_ids if _resolve(rid, all_txt))
    if sample_ids and matches == 0:
        wd = C._writable_cache_dir()
        cache_fname, cache_noisy_fname = C._cache_names(args)
        re_cached = os.path.join(wd, cache_fname)
        re_cached_noisy = os.path.join(wd, cache_noisy_fname)
        if os.path.exists(re_cached) and os.path.exists(re_cached_noisy):
            print(f"⚠️  Cache mismatch (0/{len(sample_ids)}) → dùng lại regen cache tại {wd}")
            features = torch.load(re_cached, weights_only=False)
            noisy_features = torch.load(re_cached_noisy, weights_only=False)
        else:
            print(f"⚠️  Cache mismatch (0/{len(sample_ids)}) → regenerate từ all_data.tsv "
                  f"sang {wd} (~5 phút)...")
            features, noisy_features = C._regen_text_features(args, tokenizer, wd)
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

    # rand_txt phải lấy SAU khối trên: nếu regen thì noisy_txt đã được thay bằng bản mới.
    def extract(mapping, ids):
        t, m, s, l = {}, {}, {}, {}
        for rid in ids:
            key = _resolve(rid, mapping)
            t[rid], m[rid], s[rid], l[rid] = mapping[key]
        return t, m, s, l

    train_trans = RandomTranslateCrop(2048)
    eval_trans = CenterCrop(2048)

    rand_perturb = C.as_bool(getattr(args, 'use_noise', True))
    rand_txt = noisy_txt if rand_perturb else all_txt
    # (tên_dataset, key_split, transform, txt_map, perturb_img)
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
# 3. Dựng context — tái dùng adv_common, chỉ vô hiệu hoá phần selector
# ============================================================================

def _no_og_perf(*_a, **_k):
    """Thay ``precompute_og_perf`` — hàm đó đọc ``sel`` (không còn tồn tại) và chỉ
    phục vụ G_utility của S_val. Trả None: với skip_selection=1 giá trị này không
    bao giờ được đọc."""
    print("📌 (adv_e30) BỎ precompute_og_perf — không có tập 'sel', không có S_val")
    return None


def setup_experiment_e30(cfg, device, rank_alloc_fn=None):
    """``adv_common.setup_experiment`` nhưng dùng phép chia của code gốc và không có
    một mảnh nào của selector.

    Cài đặt: tạm hoán đổi hai tên ở cấp module của ``adv_common`` trong đúng phạm vi
    lời gọi, rồi trả lại nguyên trạng ở ``finally``. Làm vậy để KHÔNG phải sao chép
    ~140 dòng dựng Fisher/FILA/LoRA — phần lõi của phương pháp chỉ nên tồn tại ở MỘT
    nơi, nếu sau này sửa thì cả hai nhánh cùng hưởng.
    """
    # --- ép tắt selector, và chặn cấu hình mâu thuẫn ngay từ đầu ---
    if C.as_bool(getattr(cfg, 'ce_selector', False)):
        raise SystemExit(
            "adv_e30 không hỗ trợ ce_selector=1 (selector S1-S4 cần tập 'sel'). "
            "Đặt ce_selector=0.")
    try:
        cfg['skip_selection'] = 1                       # Cfg kiểu dict-like
    except (TypeError, AttributeError):
        setattr(cfg, 'skip_selection', 1)
    if not C.as_bool(getattr(cfg, 'skip_selection', False)):
        raise SystemExit("adv_e30: không ép được skip_selection=1 — kiểm tra lớp Cfg.")

    print('=' * 78)
    print('adv_e30: PHÂN HOẠCH THEO CODE GỐC — retain = TOÀN BỘ D_r, test = TOÀN BỘ D_t')
    print('  - không cắt r_heldout (10%), không cắt sel (25%)')
    print('  - skip_selection ép =1: không S_val, không ReduceLROnPlateau, LR hằng số')
    print('  - CSV chỉ có hàng "last" = trọng số sau epoch cuối (E30)')
    print('  - CHỈ so sánh được với run khác cũng chạy bằng adv_e30')
    print('=' * 78)

    _saved_build = C.build_dataset
    _saved_ogperf = C.precompute_og_perf
    C.build_dataset = build_dataset_e30
    C.precompute_og_perf = _no_og_perf
    try:
        ctx = _ORIG_SETUP_EXPERIMENT(cfg, device, rank_alloc_fn)
    finally:
        C.build_dataset = _saved_build
        C.precompute_og_perf = _saved_ogperf

    # Chốt hạ: hai tập của selector phải KHÔNG tồn tại. Nếu về sau có ai thêm một
    # lời gọi đọc chúng thì sẽ nổ KeyError ngay, thay vì âm thầm dùng dữ liệu sai.
    for k in ('sel', 'r_heldout'):
        ctx['datasets'].pop(k, None)
    assert 'sel' not in ctx['datasets'] and 'r_heldout' not in ctx['datasets']
    print(f"✅ adv_e30 ctx sẵn sàng: retain={len(ctx['datasets']['retain'])} "
          f"forget={len(ctx['datasets']['forget'])} "
          f"test_final={len(ctx['datasets']['test_final'])}")
    return ctx
