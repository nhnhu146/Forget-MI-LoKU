#!/usr/bin/env python3
"""
eval_multimodal.py — Đánh giá trên CẢ HAI nhánh (ảnh + văn bản) và nhánh GỘP
==============================================================================
VÌ SAO CẦN: mọi độ đo trong `adv_common.py` chỉ đọc `outputs[1] = logits_img`
(`per_sample_ce`, `perf_metrics`), đúng như bản cài đặt gốc của Forget-MI
(`evaluation/eval_unlearning.py:68` lấy `img_loss`, `evaluation/eval_utils.py:57` lấy
`outputs[1]`). Bài báo KHÔNG nêu dùng nhánh nào — phần "Đánh giá" chỉ viết "các loss từ
mẫu giữ lại/kiểm thử" và "macro-F1 và AUC". Với một phương pháp gỡ bỏ ĐA PHƯƠNG THỨC,
đo bằng một nhánh là đo thiếu: kẻ tấn công thật có cả hai đầu ra.

File này KHÔNG sửa gì trong đường tính cũ. Nó chạy độc lập, chỉ FORWARD, và xuất
ba khung nhìn cho cùng một checkpoint:

    img   — nhánh ảnh (logits_img).  PHẢI trùng số đã báo cáo → dùng làm phép tự kiểm.
    txt   — nhánh văn bản (logits_txt), chưa từng được đo.
    fuse  — gộp hai nhánh.

ĐỊNH NGHĨA GỘP (chốt TRƯỚC khi nhìn kết quả, không có tham số tự do):
    CE_fuse    = CE_img + CE_txt
                 (đúng cách bài báo gốc cộng loss phân loại: `CombinedLoss` trong
                  `evaluation/eval_unlearning.py` = img_loss + txt_loss + joint_loss)
    prob_fuse  = (softmax(logits_img) + softmax(logits_txt)) / 2   → AUC, F1
                 (late fusion, tất định, không thêm trọng số nào)
    MIA_fuse   = SVM trên VECTOR 2 CHIỀU [CE_img, CE_txt]
                 (khác nhánh đơn ở chỗ kẻ tấn công thấy cả hai; BẮT BUỘC có
                  StandardScaler vì hai đặc trưng khác thang, mà kernel RBF thì dựa
                  trên khoảng cách nên đặc trưng thang lớn sẽ át đặc trưng kia)

⚠️ StandardScaler KHÔNG trung tính với SVM-RBF: nó đổi khoảng cách nên đổi cả kết quả
   (đo được ~0.004 trong `tools/test_eval_multimodal.py` mục 5). Vì vậy hai khung nhìn
   1 chiều img/txt CHẠY KHÔNG SCALER — chỉ như thế cột img mới tái lập số cũ.

Bộ phân loại MIA giữ NGUYÊN của bài gốc: SVC(C=3, kernel='rbf', gamma='auto').
  MIA        = per-sample CE, member/non-member cân bằng 1:1 (như `adv_common.run_mia`)
  MIA_paper  = trung bình CE theo LÔ 32, KHÔNG cân bằng (replicate `eval_unlearning.py`)

Ba loại checkpoint, chọn bằng --model_type:
  pretrained  thư mục có `pytorch_model.bin`            → θ_og, θ_re
  state_dict  `.pth` đầy đủ                             → Forget-MI, NegGrad+, CF-k, EU-k
  p3_lora     `latest.pt`/`selected.pt` CHỈ chứa LoRA   → P3 và 3 biến thể ablation
              (phải chạy lại Fisher + FILA để tái tạo W*, ~4-5 phút/checkpoint —
               dùng lại nguyên logic + các chốt kiểm tra của `tools/diagnose_logits.py`)

Cách chạy (chỉ forward, không train):
  python training/eval_multimodal.py --config config_advanced_kaggle.yaml \
      --label og --model_type pretrained --model_path <duong/dan/model> \
      --out_csv /kaggle/working/results_multimodal.csv \
      --override forget_set_path=...,base_model_path=...,...

Tự kiểm: cột view='img' phải trùng khít Df-AUC/Dt-AUC/MIA đã báo cáo cho cùng
checkpoint. Lệch = có lỗi ở file này, KHÔNG phải phát hiện khoa học.
"""
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

VIEWS = ('img', 'txt', 'fuse')
MIA_PAPER_BATCH = 32


# ============================================================================
# Phần THUẦN MẢNG — không cần model/GPU nên tự kiểm được ở máy local
# ============================================================================

def per_sample_ce_np(logits, labels):
    """CE từng mẫu, ổn định số học (log-sum-exp). logits [N,C] float64, labels [N] int."""
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    z = z - z.max(axis=1, keepdims=True)
    logZ = np.log(np.exp(z).sum(axis=1))
    return logZ - z[np.arange(len(y)), y]


def softmax_np(logits):
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def batch_means(x, bs=MIA_PAPER_BATCH):
    """Trung bình theo lô, giữ nguyên thứ tự — giống `adv_common._batch_means`.
    x: [N] hoặc [N,D] → [ceil(N/bs)] hoặc [ceil(N/bs), D]."""
    x = np.asarray(x, dtype=np.float64)
    bs = max(1, int(bs))
    return np.array([x[i:i + bs].mean(axis=0) for i in range(0, len(x), bs)])


def _fit_predict_svc(X_mem, X_non, X_frg, seed, scale):
    """SVC(C=3, rbf, gamma='auto') — y hệt bài gốc. Trả tỉ lệ forget bị đoán 'member'."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    X = np.vstack([X_mem, X_non])
    y = np.concatenate([np.ones(len(X_mem)), np.zeros(len(X_non))])
    F = np.asarray(X_frg, dtype=np.float64)
    if scale:
        sc = StandardScaler().fit(X)
        X, F = sc.transform(X), sc.transform(F)
    clf = SVC(C=3, kernel='rbf', gamma='auto', random_state=seed).fit(X, y)
    return float(clf.predict(F).mean())


def mia_scores(ce_mem, ce_non, ce_frg, seed=42, scale=False):
    """MIA + MIA_paper từ ba mảng đặc trưng CE.

    ce_*: [N] (một nhánh) hoặc [N,2] (gộp [CE_img, CE_txt]).
    Nhánh cân bằng 1:1 sao chép ĐÚNG `adv_common.run_mia` (cùng thứ tự gọi RNG) để cột
    view='img' tái lập được số cũ."""
    m = np.asarray(ce_mem, dtype=np.float64).reshape(len(ce_mem), -1)
    n = np.asarray(ce_non, dtype=np.float64).reshape(len(ce_non), -1)
    f = np.asarray(ce_frg, dtype=np.float64).reshape(len(ce_frg), -1)

    k = min(len(m), len(n))
    rng = np.random.default_rng(seed)
    m_sub = m[rng.choice(len(m), k, replace=False)]
    n_sub = n[rng.choice(len(n), k, replace=False)]
    persample = _fit_predict_svc(m_sub, n_sub, f, seed, scale)

    try:
        paper = _fit_predict_svc(batch_means(m), batch_means(n), batch_means(f), seed, scale)
    except Exception as e:                                   # pragma: no cover
        print(f"   [MIA paper warn] {e}")
        paper = float('nan')

    return {'MIA': persample, 'MIA_paper': paper,
            'member_ce': float(m.mean(axis=0).sum()),        # SỐ THỨ BA còn thiếu
            'nonmember_ce': float(n.mean(axis=0).sum()),
            'forget_ce': float(f.mean(axis=0).sum()),
            'n_member': int(len(m)), 'n_nonmember': int(len(n)), 'n_forget': int(len(f)),
            'n_forget_batches': int(np.ceil(len(f) / MIA_PAPER_BATCH))}


def perf_from_probs(probs, labels_oh, encoding='multiclass'):
    """AUC (trung bình theo kênh) + Macro-F1 + F1 + Accuracy + Pairwise-AUC.
    Cùng công thức `adv_common.perf_metrics`, chỉ khác là nhận MẢNG thay vì model —
    kể cả việc tách try/except cho AUC-theo-kênh và Pairwise (bản vá ZeroDivisionError)."""
    from joint_img_txt.metrics import compute_auc, get_acc_f1
    auc_mean = pairwise_mean = float('nan')
    pairwise = {}
    try:
        auc, pairwise = compute_auc(np.asarray(labels_oh).tolist(),
                                    np.asarray(probs).tolist(),
                                    output_channel_encoding=encoding)
        valid = [a for a in auc if not (isinstance(a, float) and np.isnan(a))]
        auc_mean = float(np.mean(valid)) if valid else float('nan')
    except Exception as e:
        print(f"   [auc warn] {e}")
    try:
        pw = [v for v in pairwise.values() if not (isinstance(v, float) and np.isnan(v))]
        pairwise_mean = float(np.mean(pw)) if pw else float('nan')
    except Exception as e:
        print(f"   [pairwise warn] {e}")
    try:
        f1m, _, _ = get_acc_f1(np.asarray(labels_oh), np.asarray(probs), encoding)
        macro_f1, f1, acc = float(f1m['macro_f1']), float(f1m['f1']), float(f1m['accuracy'])
    except Exception as e:
        print(f"   [f1 warn] {e}")
        macro_f1 = f1 = acc = float('nan')
    return {'AUC': auc_mean, 'Macro_F1': macro_f1, 'F1': f1, 'Accuracy': acc,
            'Pairwise_AUC': pairwise_mean}


def view_features(arrays, view):
    """Từ dict mảng của MỘT tập → (đặc trưng CE cho MIA, xác suất cho AUC/F1).

    img : CE_img                       · softmax(logits_img)
    txt : CE_txt                       · softmax(logits_txt)
    fuse: [CE_img, CE_txt] (2 chiều)   · trung bình hai softmax
    """
    ce_i, ce_t = arrays['ce_img'], arrays['ce_txt']
    if view == 'img':
        return ce_i.reshape(-1, 1), softmax_np(arrays['logits_img'])
    if view == 'txt':
        return ce_t.reshape(-1, 1), softmax_np(arrays['logits_txt'])
    if view == 'fuse':
        return (np.stack([ce_i, ce_t], axis=1),
                0.5 * (softmax_np(arrays['logits_img']) + softmax_np(arrays['logits_txt'])))
    raise ValueError(f"view phải thuộc {VIEWS}, nhận '{view}'")


# ============================================================================
# Phần cần model — một lần forward, dùng lại cho cả ba khung nhìn
# ============================================================================

def collect_arrays(model, dataset, device, cfg, batch_size=32, num_classes=4):
    """MỘT lần forward → logits ảnh + logits văn bản + nhãn. Ba khung nhìn dùng chung."""
    import torch
    from torch.utils.data import DataLoader
    import training.adv_common as C

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    li, lt, lab, oh = [], [], [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = tuple(t.to(device, non_blocking=True) if torch.is_tensor(t) else t
                          for t in batch)
            inputs, label_raw, _ = C.get_model_inputs(cfg, batch, device)
            label_oh = batch[5]
            with C._eval_autocast():
                out = C.safe_forward(model, inputs)
            li.append(out[1].float().cpu().numpy())
            lt.append(out[3].float().cpu().numpy())     # <-- nhánh văn bản, trước nay bỏ qua
            lab.append(label_raw.long().view(-1).cpu().numpy())
            o = label_oh.cpu().numpy()
            if o.ndim == 1 or o.shape[-1] == 1:
                o = np.eye(num_classes)[np.clip(o.flatten().astype(int), 0, num_classes - 1)]
            oh.append(o.astype(int))

    logits_img = np.concatenate(li, axis=0)
    logits_txt = np.concatenate(lt, axis=0)
    labels = np.concatenate(lab, axis=0)
    return {'logits_img': logits_img, 'logits_txt': logits_txt,
            'labels': labels, 'labels_oh': np.concatenate(oh, axis=0),
            'ce_img': per_sample_ce_np(logits_img, labels),
            'ce_txt': per_sample_ce_np(logits_txt, labels)}


def evaluate_all_views(model, datasets, cfg, device, batch_size=None):
    """Trả list dòng (mỗi khung nhìn 1 dòng) cho MỘT checkpoint.
    Tập dùng y hệt `final_evaluation`: member = retain lấy mẫu `eval_max_retain`,
    non-member = D_t_final, mục tiêu = D_f."""
    import training.adv_common as C

    bs = int(batch_size or cfg.eval_batch_size)
    seed = int(cfg.random_seed)
    enc = str(getattr(cfg, 'output_channel_encoding', 'multiclass'))
    member_ds = C.subsample_dataset(datasets['retain'], int(cfg.get('eval_max_retain', 512)), seed)

    print("   forward: retain(member) / D_t_final(non-member) / D_f ...")
    A_mem = collect_arrays(model, member_ds, device, cfg, bs)
    A_non = collect_arrays(model, datasets['test_final'], device, cfg, bs)
    A_frg = collect_arrays(model, datasets['forget'], device, cfg, bs)

    rows = []
    for view in VIEWS:
        f_mem, _ = view_features(A_mem, view)
        f_non, p_non = view_features(A_non, view)
        f_frg, p_frg = view_features(A_frg, view)
        # scaler CHỈ bật cho khung nhìn gộp (2 chiều, hai đặc trưng khác thang).
        # Khung nhìn 1 chiều phải để tắt: bật vào là lệch số cũ (xem test mục 5).
        mia = mia_scores(f_mem, f_non, f_frg, seed=seed, scale=(f_frg.shape[1] > 1))
        forget_m = perf_from_probs(p_frg, A_frg['labels_oh'], enc)
        test_m = perf_from_probs(p_non, A_non['labels_oh'], enc)
        rows.append({
            'view': view,
            'Df_AUC': forget_m['AUC'], 'Df_F1': forget_m['Macro_F1'],
            'Dt_AUC': test_m['AUC'], 'Dt_F1': test_m['Macro_F1'],
            'Df_Pairwise_AUC': forget_m['Pairwise_AUC'],
            'Dt_Pairwise_AUC': test_m['Pairwise_AUC'],
            **mia,
        })
    return rows


def rebuild_p3_from_lora(ckpt_path, cfg_d, device, infer=True):
    """Dựng lại model P3 từ checkpoint CHỈ-CHỨA-LORA (`latest.pt` / `selected.pt`).

    Không nạp thẳng được như `.pth` của Forget-MI: checkpoint P3 chỉ có tham số LoRA,
    còn W* (nền đã trừ FILA) phải TÁI TẠO bằng cách chạy lại Fisher + FILA. Toàn bộ logic
    và các chốt kiểm tra lấy nguyên từ `tools/diagnose_logits.py` — chỗ đã dùng để cứu
    ô P3/IU/E30, không viết lại đường mới.

    Trả (model đã merge, ctx) — dùng luôn `ctx['datasets']` để khỏi dựng dataset hai lần.
    """
    import torch
    import training.adv_common as C
    from tools.diagnose_logits import infer_lora_cfg

    print(f"📦 Nạp checkpoint LoRA: {ckpt_path}")
    payload = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state = payload.get('trainable_state', payload.get('lora_state', payload))
    print(f"   epoch trong checkpoint: {payload.get('epoch')}  |  {len(state)} tensor")

    if infer:
        txt_extra, img_k = infer_lora_cfg(list(state.keys()))
        if txt_extra:
            cfg_d['lora_extra_target_modules'] = '|'.join(txt_extra)
        if img_k:
            cfg_d['lora_image_last_k_blocks'] = img_k
        print(f"🔎 Suy từ checkpoint: lora_extra_target_modules="
              f"{cfg_d.get('lora_extra_target_modules', '(không có)')}  "
              f"lora_image_last_k_blocks={cfg_d.get('lora_image_last_k_blocks')}")

    cfg = C.Cfg(cfg_d)
    C.set_seed(int(cfg.random_seed))
    print("🔧 Dựng lại pipeline P3 (Fisher + FILA) để tái tạo W* — mất ~4-5 phút...")
    ctx = C.setup_experiment(cfg, device)

    from joint_img_txt.model import ImageTextModel
    from peft import get_peft_model
    ctx.pop('model_unlearn', None)                 # bớt một model FP32 trên GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    base = ImageTextModel.from_pretrained(ctx['base_p']).to(device)
    peft = get_peft_model(base, ctx['peft_cfg'])
    # ⚠️ Biến thể ablation `w/o Fisher/FILA` train với loku_random_init=True nên KHÔNG có
    # phép trừ nào. Dựng lại mà quên cờ đó thì fila_subtraction khác rỗng ⇒ nền W* lệch
    # hẳn so với lúc train, và strict=False sẽ không hé một lời. In ra để soi được.
    _sub = ctx.get('fila_subtraction', {})
    print(f"   FILA subtraction: {len(_sub)} mô-đun"
          + ("  (rỗng — đúng cho abl_fila / loku_random_init=1)" if not _sub else ""))
    C.apply_fila_subtraction(peft, _sub)
    inc = peft.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=False)

    # Hai chốt BẮT BUỘC (strict=False nuốt im lặng mọi khóa lệch → số sẽ sai mà không báo)
    unexpected = list(getattr(inc, 'unexpected_keys', []) or [])
    if unexpected:
        for k in unexpected[:6]:
            print(f"      {k}")
        raise SystemExit(
            f"❌ {len(unexpected)} khóa trong checkpoint KHÔNG có chỗ trong model → cấu hình "
            f"LoRA dựng lại KHÁC lúc train. Với P3-NoKD-More cần:\n"
            f"  lora_extra_target_modules=attention.output.dense|intermediate.dense|output.dense\n"
            f"  lora_image_last_k_blocks=2")
    n_ck = sum(1 for k in state if '.lora_' in k)
    n_md = sum(1 for n_, _ in peft.named_parameters() if '.lora_' in n_)
    if n_ck != n_md:
        raise SystemExit(f"❌ Số tensor LoRA lệch ({n_ck} checkpoint vs {n_md} model dựng lại).")
    print(f"   ✅ nạp {len(state)} tensor · LoRA khớp {n_ck}/{n_md} vị trí")
    return peft.merge_and_unload(), ctx


def _fmt(rows, label):
    head = (f"{'view':>5} | {'Df-AUC':>7} {'Df-F1':>7} | {'Dt-AUC':>7} {'Dt-F1':>7} | "
            f"{'MIA':>6} {'MIA_pap':>7} | {'retain-CE':>9} {'test-CE':>8} {'forget-CE':>9}")
    out = [f"\n===== {label} =====", head, '-' * len(head)]
    for r in rows:
        out.append(f"{r['view']:>5} | {r['Df_AUC']:7.3f} {r['Df_F1']:7.3f} | "
                   f"{r['Dt_AUC']:7.3f} {r['Dt_F1']:7.3f} | {r['MIA']:6.3f} {r['MIA_paper']:7.3f} | "
                   f"{r['member_ce']:9.3f} {r['nonmember_ce']:8.3f} {r['forget_ce']:9.3f}")
    r0 = rows[0]
    out.append(f"\n|member|={r0['n_member']}  |non-member|={r0['n_nonmember']}  "
               f"|D_f|={r0['n_forget']}  → MIA_paper trung bình trên {r0['n_forget_batches']} lô "
               f"⇒ bước nhỏ nhất {1 / max(1, r0['n_forget_batches']):.3f}")
    out.append("view='img' phải TRÙNG số đã báo cáo cho cùng checkpoint — nếu lệch là lỗi file này.")
    return '\n'.join(out)


def main():
    import argparse
    import torch
    import training.adv_common as C
    from training.forgetmi_eval_only import load_eval_model

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, default='config_advanced_kaggle.yaml')
    ap.add_argument('--label', type=str, required=True, help='og | re | forgetmi | p3 | ...')
    ap.add_argument('--model_path', type=str, required=True)
    ap.add_argument('--model_type', type=str, default='pretrained',
                    choices=['pretrained', 'state_dict', 'p3_lora'],
                    help="pretrained = thư mục có pytorch_model.bin (og/re) · "
                         "state_dict = .pth đầy đủ (Forget-MI, NegGrad, CF-k, EU-k) · "
                         "p3_lora = latest.pt/selected.pt chỉ chứa LoRA (P3 + ablation)")
    ap.add_argument('--method', type=str, default='multimodal_eval')
    ap.add_argument('--seed', type=int, default=None)
    ap.add_argument('--override', type=str, default=None)
    ap.add_argument('--out_csv', type=str, default=None)
    ap.add_argument('--no-infer', action='store_true',
                    help='p3_lora: tắt suy cấu hình LoRA từ checkpoint, dùng nguyên override')
    cli = ap.parse_args()

    cfg_d = C.flatten_method_config(C.load_config(cli.config), 'p3')
    if cli.seed is not None:
        cfg_d['random_seed'] = int(cli.seed)
    cfg_d = C.apply_overrides(cfg_d, cli.override)

    C.set_seed(int(cfg_d.get('random_seed', 42)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and os.environ.get('FORGETMI_ALLOW_CPU') != '1':
        raise SystemExit("Không có GPU. Bật GPU hoặc set FORGETMI_ALLOW_CPU=1.")
    print(f"🟢 EVAL ĐA PHƯƠNG THỨC — label={cli.label}  type={cli.model_type}  device={device}")

    if cli.model_type == 'p3_lora':
        # rebuild_p3_from_lora có thể SỬA cfg_d (suy lora_extra/image_last_k từ checkpoint)
        # nên phải dựng Cfg SAU nó; dataset lấy luôn từ ctx, không dựng lại.
        model, ctx = rebuild_p3_from_lora(cli.model_path, cfg_d, device, infer=not cli.no_infer)
        cfg, datasets = C.Cfg(cfg_d), ctx['datasets']
    else:
        cfg = C.Cfg(cfg_d)
        from transformers import BertTokenizer
        base_p = C.ensure_model_path(cfg.base_model_path, "base")
        tok = BertTokenizer.from_pretrained(
            base_p if os.path.exists(os.path.join(base_p, "vocab.txt")) else cfg.bert_pretrained_dir)
        print("📚 Building dataset (CÙNG 4-tập holdout, cùng seed)...")
        datasets, _ = C.build_dataset(cfg, tok)
        model = load_eval_model(cli.model_path, cli.model_type, base_p, device)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    rows = evaluate_all_views(model, datasets, cfg, device)
    print(_fmt(rows, cli.label))

    csv_path = cli.out_csv or str(getattr(cfg, 'results_csv_path',
                                          os.path.join(cfg.output_dir, 'results_multimodal.csv')))
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)) or '.', exist_ok=True)
    for r in rows:
        C.append_row_csv(csv_path, {
            'label': cli.label, 'method': cli.method, 'seed': int(cfg.random_seed),
            'forget_set': os.path.basename(str(cfg.forget_set_path)),
            'model_type': cli.model_type, 'model_path': cli.model_path,
            **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()},
        })
    print(f"\n✅ Ghi {len(rows)} dòng (img/txt/fuse) → {csv_path}")


if __name__ == '__main__':
    main()
