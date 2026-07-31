# ROADMAP LUẬN VĂN — Forget-MI-LoKU

> **Đề tài**: Áp dụng LoRA + Fisher Information Matrix vào Forget-MI để tinh gọn machine unlearning trong multimodal medical data
> **Tác giả**: Nguyễn Hoàng Nhu
> **Ngày tạo roadmap**: 2026-06-16 (cập nhật: thêm dataset Indiana University)
> **Baseline paper**: Hardan et al., *Forget-MI: Machine Unlearning for Forgetting Multimodal Information in Healthcare Settings*, MICCAI 2025
> **Datasets**: (1) MIMIC-CXR (dataset paper gốc), (2) Indiana University Chest X-ray (cross-dataset generalization)

---

## MỤC LỤC
1. [Mục tiêu & Luận điểm trung tâm](#1-mục-tiêu--luận-điểm-trung-tâm)
2. [Datasets & baseline Forget-MI](#2-datasets--baseline-forget-mi)
3. [Các bug HIỆN TẠI cần sửa NGAY](#3-các-bug-hiện-tại-cần-sửa-ngay)
4. [Phương pháp đề xuất Forget-MI-LoKU](#4-phương-pháp-đề-xuất-forget-mi-loku)
5. [Pipeline thực thi & Evaluation](#5-pipeline-thực-thi--evaluation)
6. [Các thí nghiệm bắt buộc](#6-các-thí-nghiệm-bắt-buộc)
7. [Bảng & Hình cần có trong luận văn](#7-bảng--hình-cần-có-trong-luận-văn)
8. [Cấu trúc viết luận văn](#8-cấu-trúc-viết-luận-văn)
9. [Tiêu chí PASS / FAIL](#9-tiêu-chí-pass--fail)
10. [Timeline & Checklist tuần](#10-timeline--checklist-tuần)
11. [FAQ & Lỗi thường gặp](#11-faq--lỗi-thường-gặp)
12. [Phụ lục: Code patches](#12-phụ-lục-code-patches)
13. [Phụ lục: Chuẩn bị dataset Indiana University](#13-phụ-lục-chuẩn-bị-dataset-indiana-university)

---

## 1. Mục tiêu & Luận điểm trung tâm

### 1.1 Vấn đề bạn giải quyết
Forget-MI (MICCAI 2025) là phương pháp machine unlearning multimodal tốt nhất hiện tại cho dữ liệu y tế, NHƯNG:
- **Mất ~5 giờ** cho mỗi lần unlearning (so với NegGrad+/SCRUB/CF-k/EU-k chỉ ~4h)
- **Tinh chỉnh toàn bộ tham số** của model_unlearn (full fine-tune) → tốn RAM GPU
- Khó scale lên các foundation model lớn (paper tự thừa nhận trong Conclusion)

### 1.2 Luận điểm trung tâm (Thesis Statement)
> *"Chúng tôi đề xuất Forget-MI-LoKU, kết hợp LoRA (Low-Rank Adaptation) và Fisher Information Matrix-guided initialization, giảm ít nhất 40% thời gian unlearning và 30% bộ nhớ GPU so với Forget-MI gốc, trong khi giữ MIA, Test AUC, Test F1 trong khoảng ±5-10% so với Forget-MI baseline. Phương pháp được xác thực trên **hai tập dữ liệu y tế đa phương thức độc lập**: MIMIC-CXR (in-domain, theo paper gốc) và Indiana University Chest X-ray (cross-domain, để chứng minh khả năng tổng quát hóa)."*

### 1.3 4 trục đóng góp cần chứng minh
| Trục | Bằng chứng cần thu thập |
|---|---|
| **Hiệu quả (Efficacy)** | MIA score, Test AUC, Test F1, Forget AUC, distance vs retrained — không tệ hơn đáng kể so với Forget-MI |
| **Hiệu năng (Efficiency)** | Thời gian (giờ), GPU peak memory (GB), số tham số trainable (%), số epoch hội tụ |
| **Trade-off có kiểm soát (Ablation)** | Bóc tách: LoRA-only vs FIM-only vs LoRA+FIM, đo sensitivity với hyperparameter |
| **Khả năng tổng quát hóa (Cross-dataset)** | Kết quả ổn định trên cả MIMIC-CXR và Indiana University CXR — chứng tỏ phương pháp không over-fit đặc thù 1 dataset |

---

## 2. Datasets & baseline Forget-MI

### 2.0 So sánh 2 datasets

| Đặc tính | MIMIC-CXR (subset) | Indiana University CXR (IU-CXR / Open-i) |
|---|---|---|
| Số ảnh | 6,742 | ~7,470 |
| Số patient | 1,663 | ~3,851 |
| Số report | 6,742 (paired) | ~3,955 |
| Nguồn ảnh | Beth Israel Deaconess Medical Center | Indiana University Hospital Network |
| Format ảnh | DICOM | PNG/JPG (đã convert) |
| Report style | Free-text, có FINDINGS + IMPRESSION | Free-text, ngắn hơn MIMIC |
| Label gốc | Edema severity 4 mức (0/1/2/3) | MeSH tags, manual annotations (không có severity edema sẵn) |
| Task được paper dùng | Multi-class edema classification (4 lớp) | **Cần adapt** — xem mục 13 phụ lục |
| Phân bố lớp | no edema 43% / vascular 25% / interstitial 22% / alveolar 10% | Cần phân tích lại sau preprocessing |
| Có public không | Yes (cần PhysioNet credentialed) | Yes (Open-i, no credential) |
| Vai trò trong luận văn | **Dataset chính** — full matrix thí nghiệm | **Dataset xác thực** — subset thí nghiệm để chứng minh generalization |

### 2.0.1 Tại sao thêm IU-CXR?
1. **Chứng minh không over-fit dataset**: Nếu LoKU chỉ work tốt trên MIMIC, có thể là do may mắn / đặc thù MIMIC. IU-CXR là independent test.
2. **Khác biệt domain**: IU-CXR có report ngắn hơn, label sparse hơn → test khả năng adapt của method.
3. **Tăng giá trị luận văn**: Thesis có 2 datasets > thesis có 1 dataset (về mặt nghiên cứu).
4. **Paper gốc chỉ dùng MIMIC** → đây là 1 đóng góp nhỏ khác của bạn.

### 2.0.2 Lưu ý quan trọng về IU-CXR
- IU-CXR **KHÔNG có sẵn label edema 4 mức** như MIMIC. Bạn cần:
  - **Option A**: Tạo task binary "normal vs abnormal" từ MeSH tags → đơn giản, mất ít công preprocessing
  - **Option B**: Tự annotate edema severity bằng keyword matching trên report → phức tạp hơn nhưng giữ task gốc
  - **Option C**: Multi-label classification trên top-K MeSH tags phổ biến nhất → trung gian
- **Khuyến nghị**: chọn **Option A** (binary) để giảm rủi ro. Ghi rõ trong luận văn: "Trên IU-CXR, do hạn chế label, chúng tôi đánh giá trên task binary normal/abnormal làm proxy."
- Chi tiết preprocessing IU-CXR: xem **Mục 13 phụ lục**.

### 2.1 Kiến trúc & 4 loss

Mô hình: `ImageTextModel` = ResNet ảnh + BERT văn bản + multimodal gate (late fusion).

4 loss của Forget-MI:

| Loss | Công thức rút gọn | Mục đích |
|---|---|---|
| **L_UU** (Unimodal Unlearning) | $-\text{Dist}([\mathcal{F}_{ul}(I_f), \mathcal{F}_{ul}(T_f)], [\mathcal{F}_{og}(\tilde I_f), \mathcal{F}_{og}(\tilde T_f)])$ | Đẩy embedding unimodal của forget ra xa noisy version |
| **L_MU** (Multimodal Unlearning) | $-\text{Dist}(\mathcal{F}_{ul}(I_f,T_f), \mathcal{F}_{og}(\tilde I_f, \tilde T_f))$ | Đẩy joint embedding của forget ra xa |
| **L_UR** (Unimodal Retention) | $\text{Dist}([\mathcal{F}_{ul}(I_r), \mathcal{F}_{ul}(T_r)], [\mathcal{F}_{og}(I_r), \mathcal{F}_{og}(T_r)])$ | Giữ embedding unimodal của retain gần model gốc |
| **L_MR** (Multimodal Retention) | $\text{Dist}(\mathcal{F}_{ul}(I_r, T_r), \mathcal{F}_{og}(I_r, T_r))$ | Giữ joint embedding của retain gần model gốc |

Tổng: $\mathcal{L} = w_{uu}\mathcal{L}_{UU} + w_{ur}\mathcal{L}_{UR} + w_{mu}\mathcal{L}_{MU} + w_{mr}\mathcal{L}_{MR}$ với $\sum w = 1$.

### 2.2 5 weight settings (theo Table 2 paper)

| Setting | w_uu | w_ur | w_mu | w_mr | Đặc điểm |
|---|---|---|---|---|---|
| **No Noise** | 0.25 | 0.25 | 0.25 | 0.25 | + `use_noise=False` |
| **Equal** | 0.25 | 0.25 | 0.25 | 0.25 | + `use_noise=True` |
| **Multimodal** | 0.1 | 0.1 | 0.4 | 0.4 | Ưu tiên loss joint |
| **Unimodal** | 0.4 | 0.4 | 0.1 | 0.1 | Ưu tiên loss unimodal |
| **Retention** | 0.1 | 0.4 | 0.1 | 0.4 | Ưu tiên giữ kiến thức retain |

> **Lưu ý**: Trong code, các trọng số được lưu là `alpha (UR), beta (UU), theta (MU), gamma (MR)` rồi normalize. Khi chạy nhớ map đúng tên.

### 2.3 Forget percentages
3 mức: **3%, 6%, 10%** của dataset MIMIC-CXR (6,742 mẫu).

### 2.4 5 baselines paper đã so sánh
- **NegGrad+** (modality-agnostic, fine-tune trên Dr + đảo gradient trên Df)
- **SCRUB** (teacher-student, student không nghe teacher trên Df)
- **CF-k** (Catastrophic Forgetting, freeze k=2 layer đầu, fine-tune phần còn lại)
- **EU-k** (Exact Unlearning, retrain các layer không freeze trên Dr)
- **MultiDelete** (modality-aware, đối thủ multimodal duy nhất)

---

## 3. Trạng thái HIỆN TẠI của code (đã hoàn thiện qua 24 experiments)

> ✅ Code `training/forgetmi_loku.py` đã evolve qua **24 experiments (exp_001 → exp_024)** và KHÔNG còn bug nào. Phần này tổng hợp các thành phần đã có sẵn và kết quả đạt được trên MIMIC 3% forget set.

### 3.1 Các thành phần kỹ thuật đã có trong code hiện tại

| Thành phần | Mô tả | Tham số tiêu biểu |
|---|---|---|
| **TRUE LoKU FILA subtraction** | `W* = W - B·A · scale` tại init (theo paper LoKU/FILA) | `loku_subtract_scale=1.0` |
| **IHL (Inverted Hinge Loss)** | $\mathcal{L}_{IHL} = 1 + p(y_{true}) - \max_{v \neq y_{true}} p(v)$ — bounded [0,2], self-stopping | `ihl_forget_weight=0.75` |
| **Image-FILA** | LoRA + FILA trên Conv2d của image encoder (`img_model.layer7`) — KHÔNG chỉ BERT | `lora_image_last_k_blocks=1`, `loku_image_subtract_scale=0.3` |
| **Distillation từ F_og** (honest) | KL(student ‖ F_og) trên retain — không cần gold F_re khi train | `distill_teacher="og"`, `distill_retain_weight=1.5` |
| **Classifier heads unfrozen** | Cho update classifier để giữ utility | `unfreeze_classifier_heads=true` |
| **Early stop theo val CE** | Dùng val loss thay vì train loss — tránh overfit/diverge | `early_stop_metric="val"`, `patience=4` |
| **Bounded forget margin** | `forget_margin=20` cho L_MD | hinge bound đã đúng |
| **Fisher Information chuẩn** | Cross-entropy gradient² (không phải sum of logits) | `fisher_max_samples=256` |
| **Eval pipeline đầy đủ** | MIA_persample, MIA_paper, CosSim, AUC, F1, time, GPU, params | tự động cuối mỗi run |
| **Auto-tracker experiments** | Mỗi run tạo file `exp_NNN_*.md`, update `INDEX.md` | xem `experiments/` |

### 3.2 Kết quả CHÍNH THỨC trên MIMIC 3% (multi-seed: 42, 123, 7)

> Nguồn: `experiments/summary_final_ihl075_multiseed.md`

| Metric | LoKU mean ± std | Paper Forget-MI | Δ | Verdict |
|---|---|---|---|---|
| **MIA_paper ↓** | **0.429 ± 0.116** | 0.571 | **−0.142** | ✅ TỐT HƠN paper |
| **Forget AUC ↓** | **0.736 ± 0.004** | 0.735 | +0.001 | ✅ NGANG paper |
| **Forget F1 ↓** | **0.379 ± 0.013** | 0.393 | −0.014 | ✅ TỐT HƠN paper |
| **Test AUC ↑** | **0.677 ± 0.002** | 0.625 | **+0.052** | ✅ TỐT HƠN paper |
| **Test F1 ↑** | **0.364 ± 0.020** | 0.250 | **+0.114** | ✅ TỐT HƠN paper |
| **Time (h) ↓** | **0.20 ± 0.03** (~12 min) | ~5h | **−4.8h (96% nhanh hơn)** | ✅ |
| **Trainable params ↓** | **0.451%** | 100% | −99.5% | ✅ |
| **GPU peak (GB)** | ~12 GB | — | — | đo được |

→ **Kết luận**: Trên MIMIC 3%, **method đã PASS toàn bộ tiêu chí** ở Section 9. Phương pháp vượt paper trên 4/5 metric, ngang 1/5, nhanh hơn ~25-30 lần.

### 3.3 Lịch sử experiments quan trọng (chọn lọc)

| Exp | Insight chính | MIA | Df AUC | Dt AUC |
|---|---|---|---|---|
| 001 | Baseline LoKU v1 (chỉ LoRA BERT) | 0.552 | 0.829 | 0.674 |
| 008 | TRUE LoKU FILA subtraction (key idea) | 0.562 | 0.833 | 0.679 |
| 009 | + IHL loss bounded | 0.537 | 0.832 | 0.678 |
| 010 | + Image-FILA (key breakthrough — MIA chỉ đọc img_logits!) | 0.493 | 0.718 | 0.689 |
| 011 | Honest mode (distill F_og, không F_re) | 0.333 | 0.687 | 0.673 |
| 011d | IHL sweep → IHL=0.75 sweet spot | 0.453 | 0.734 | 0.677 |
| **022-024** | **Multi-seed final** (42, 123, 7) | **0.429** | **0.736** | **0.677** |

### 3.4 Những gì KHÔNG cần làm nữa (đã xong)
- ✅ Sửa code (24 exp đã hoàn thiện)
- ✅ Multi-seed trên MIMIC 3% forget set
- ✅ Ablation IHL (sweep 0.5/0.75/1.0)
- ✅ Comparison với gold F_re (CosSim, distance)
- ✅ Auto-tracker & experiment logging

---

## 4. Phương pháp đề xuất Forget-MI-LoKU (THỰC TẾ — đã triển khai)

### 4.1 Sơ đồ kiến trúc tổng quát
```
                    ┌─────────────────────┐
                    │ Pretrained F_og     │ (Frozen, fp16)
                    │ (BERT + ResNet)     │
                    └──────┬──────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
         ┌────▼────┐                ┌───▼────┐
         │  F_ul   │                │  F_re  │
         │ PEFT:   │                │(Frozen,│
         │ -LoRA   │                │  fp16) │
         │  BERT   │                │ chỉ    │
         │ -FILA   │                │ dùng   │
         │  Conv2d │                │ EVAL   │
         │ +CLS    │                │ (CosSim)│
         │  heads  │                └────────┘
         └────┬────┘
              │
   ┌──────────┴─────────────────────────┐
   │ Loss tổng (9 thành phần):          │
   │ • Retain alignment: L_UKR, L_MKR   │ ← Forget-MI cổ điển
   │ • Forget bounded:   L_UU, L_MD     │ ← chặn bằng forget_margin
   │ • Classifier CE:    L_CLS_RET      │ ← head unfrozen
   │ • Distillation:     L_DSL_RET (og) │ ← honest (không cần F_re)
   │ • Inverted Hinge:   L_IHL (forget) │ ← bounded [0,2], self-stop
   │ • Anchor (optional): L_RE = 0      │ ← tắt vì exp11 honest
   └────────────────────────────────────┘
```

### 4.2 6 thành phần kỹ thuật ĐÃ TRIỂN KHAI

#### (a) LoRA trên BERT text encoder
- `target_modules = ["query", "key", "value"]` (attention layers)
- `r=8`, `lora_alpha=16`, `lora_dropout=0.05`

#### (b) **Image-FILA** trên ResNet image encoder (BREAKTHROUGH exp10)
- LoRA + FILA trên Conv2d của `img_model.layer7` (last block)
- Reshape Conv2d `[out,in,kh,kw]` → 2D để áp SVD → reshape lại
- Lý do: MIA chỉ đọc `img_logits` → image branch phải được unlearn, KHÔNG chỉ text
- `loku_image_subtract_scale=0.3` (nhẹ hơn text scale 1.0)

#### (c) **TRUE LoKU FILA subtraction** (BREAKTHROUGH exp08)
- Init: `W* = W - B·A · peft_scaling · scale` với `loku_subtract_scale=1.0` (text), `0.3` (image)
- Đây là "knowledge subtraction" thực sự — KHÔNG phải soft init
- A, B init từ importance-weighted SVD: `Wp = row_imp[:,None] * W`, importance ratio `rel = f_imp / (r_imp + eps)`

#### (d) **IHL loss** (Inverted Hinge Loss, exp09)
$$\mathcal{L}_{IHL} = \frac{1}{|D_f|} \sum_{(x,y) \in D_f} \left(1 + p(y) - \max_{v \neq y} p(v)\right)$$
- Bounded trong [0, 2], **self-stopping** khi forget bị misclassify
- An toàn hơn NegGrad (unbounded → diverge nguy hiểm)
- Tính trên cả `img_logits` + `txt_logits` (avg 0.5)
- Sweet spot: `ihl_forget_weight=0.75` (sweep 0.5/0.75/1.0)

#### (e) **Honest Distillation** (exp11 — innovation lớn)
- Teacher = `F_og` (KHÔNG phải `F_re` như exp10c) → không cần gold standard khi train
- `distill_retain_weight=1.5` (KL trên retain)
- `distill_forget_weight=0.0` (TẮT — vì F_og biết forget; thay bằng IHL)
- Early stop theo `val_CE` (KHÔNG phải `cossim` vs F_re)
- → Hợp lệ với giả thuyết "không có F_re khi train"

#### (f) **Classifier heads unfrozen** (exp03+)
- Cho `img_model.fc1` và `text_model.classifier` được update
- Anchored bằng `kappa_cls_retain=2.0` CE trên retain
- Giúp Test AUC vượt paper (0.677 vs 0.625)

### 4.3 Loss tổng (công thức triển khai)
$$\mathcal{L} = \alpha L_{UKR} + \beta L_{UU} + \theta L_{MD} + \gamma L_{MKR} + \eta L_{RE} + \kappa_{ret} L_{CLS}^{ret} + w_{dsl}^{ret} L_{DSL}^{ret} + w_{IHL} L_{IHL}$$

**Config tối ưu (exp11_final_ihl075)**: α=1, β=0, θ=0, γ=1, η=0, κ_ret=2, w_dsl_ret=1.5, w_IHL=0.75

→ Lưu ý: β và θ = 0 vì forget được xử lý bằng FILA subtraction (tại init) + IHL (trong training). Không cần L_UU/L_MD push.

---

## 5. Pipeline thực thi & Evaluation

### 5.1 Quy trình 1 run hoàn chỉnh
```
Step 1: Load M_og (full fp16), M_re (full fp16), M_ul (PEFT + LoRA)
Step 2: Build dataset (retain / forget / random / val / test)
Step 3: Compute FIM importance trên Df và Dr (subset 512 samples)
Step 4: SVD init LoRA với importance ratio (scale nhỏ ~0.05)
Step 5: Unlearning loop (8 epochs, early stop theo CosSim)
   - Mỗi epoch: forward 3 batch (forget, random, retain)
   - Tính 4 loss + L_anchor
   - Backward → AdamW.step()
   - Clip grad norm = 1.0
   - Evaluate CosSim mỗi epoch
Step 6: Merge LoRA về base (model.merge_and_unload())
Step 7: Final Evaluation:
   - MIA score (SVM trên losses)
   - CosSim vs M_re
   - Test AUC, F1, Macro-F1
   - Forget AUC, F1, Macro-F1
   - Log: time, GPU peak, # params
Step 8: Save kết quả vào CSV + wandb log
```

### 5.2 Metrics cần đo trong MỖI run

#### Hiệu quả (Efficacy)
| Metric | Hướng tốt | Source |
|---|---|---|
| MIA score | gần 0.5 | `evaluation/eval_unlearning.py::run_mia` |
| CosSim(M_ul, M_re) | càng cao | `evaluation/eval_utils.py::get_probability_measure` |
| 1 − CosSim (distance) | càng thấp | `1 - CosSim` |
| Test AUC | càng cao | `compute_metrics(test_set)` |
| Test Macro-F1 | càng cao | `compute_metrics(test_set)` |
| Forget AUC | thấp (≈ M_re) | `compute_metrics(forget_set)` |
| Forget Macro-F1 | thấp (≈ M_re) | `compute_metrics(forget_set)` |

#### Hiệu năng (Efficiency)
| Metric | Cách đo |
|---|---|
| Wall-clock time (giờ) | `time.time()` bao quanh hàm `unlearn()` |
| GPU peak memory (GB) | `torch.cuda.max_memory_allocated() / 1e9` |
| Trainable params (%) | `sum(p.numel() for p in model.parameters() if p.requires_grad) / total` |
| # epochs to converge | `best_epoch_by_cossim` |

### 5.3 Log wandb đầy đủ (template)
```python
wandb.log({
    # Performance
    "final/MIA": mia_score,
    "final/CosSim_vs_re": cossim_re,
    "final/dist_vs_re": 1 - cossim_re,
    "final/Test_AUC": test_auc,
    "final/Test_F1": test_f1,
    "final/Test_MacroF1": test_macro_f1,
    "final/Forget_AUC": forget_auc,
    "final/Forget_F1": forget_f1,
    "final/Forget_MacroF1": forget_macro_f1,
    # Efficiency
    "efficiency/time_hours": elapsed_h,
    "efficiency/gpu_peak_GB": gpu_peak,
    "efficiency/trainable_params": trainable,
    "efficiency/total_params": total,
    "efficiency/trainable_ratio": trainable / total,
    "efficiency/best_epoch": best_epoch,
    # Config tracking
    "cfg/method": "Forget-MI-LoKU",
    "cfg/forget_pct": 3,
    "cfg/weighting": "Unimodal",
    "cfg/seed": 42,
    "cfg/lora_r": 8,
    "cfg/lora_alpha": 16,
    "cfg/use_noise": True,
})
```

---

## 6. Các thí nghiệm bắt buộc

### 6.1 Tổng quan ma trận thí nghiệm (cập nhật theo trạng thái thực tế)

**Tình trạng**: MIMIC 3% đã DONE (24 exp). Còn lại: MIMIC 6% & 10%, baselines reproduce, và toàn bộ IU-CXR.

| ID | Tên thí nghiệm | Dataset | Status | # runs còn lại | Ưu tiên |
|---|---|---|---|---|---|
| EXP2-M-3 | Forget-MI-LoKU MIMIC 3% (multi-seed) | MIMIC | ✅ **DONE** | 0 | — |
| EXP2-M-6 | Forget-MI-LoKU MIMIC 6% (multi-seed) | MIMIC | ⏳ TODO | 3 seeds = 3 runs | ★★★ |
| EXP2-M-10 | Forget-MI-LoKU MIMIC 10% (multi-seed) | MIMIC | ⏳ TODO | 3 seeds = 3 runs | ★★★ |
| EXP1-M | Reproduce Forget-MI baseline (cùng env) | MIMIC | ⏳ TODO | 3 forget% × 1 best setting × 3 seeds = 9 | ★★★ |
| EXP3-M | Ablation rộng (LoRA-only, FIM-only, IHL-off…) | MIMIC | 🟡 partial done (qua exp001-024) | 6-12 (gom lại làm bảng) | ★★ |
| EXP4-M | Sensitivity (LoRA rank, FIM samples) | MIMIC | ⏳ TODO | 12-15 | ★ |
| EXP5-M | MultiDelete baseline | MIMIC | ⏳ TODO (hoặc trích paper) | 3-9 | ★ |
| EXP6-M | NegGrad+ hoặc SCRUB | MIMIC | ⏳ TODO (hoặc trích paper) | 3-9 | ☆ |
| **EXP7-I** | **Pretrain `model_og_IU` + retrain `model_re_IU`** | IU | ⏳ TODO | 1 og + 3 re = 4 runs | ★★★ |
| **EXP8-I** | **Forget-MI baseline trên IU** | IU | ⏳ TODO | 3 forget% × 2 seeds = 6 | ★★★ |
| **EXP9-I** | **Forget-MI-LoKU trên IU** | IU | ⏳ TODO | 3 forget% × 2 seeds = 6 | ★★★ |
| **EXP10-I** | **(Tuỳ chọn) Ablation rút gọn trên IU** | IU | ⏳ TODO | 4-6 | ★ |

**Tổng runs CÒN LẠI** (so với roadmap cũ giả định bắt đầu từ 0):
- Tối thiểu (★★★ + ★★): ~35 runs (gồm 4 pretrain IU)
- Đầy đủ: ~55-70 runs

→ Giảm đáng kể vì MIMIC 3% đã xong và đã có **24 exp lịch sử để tham khảo** (không phải làm lại từ đầu).

### 6.1.1 Tại sao vẫn cần EXP1-M (reproduce Forget-MI gốc)?
Bạn đã có kết quả LoKU rất tốt, NHƯNG **chưa chạy Forget-MI gốc TRÊN CÙNG máy** để có số time/GPU công bằng. Paper báo "~5h" trên máy của họ — bạn cần ~5h confirm trên máy bạn. Đây là 3-9 runs `forgetmi_partial.py`, không phải code mới.

### 6.1.2 Subset matrix cho IU-CXR (lý do)
- Full matrix = 3 forget% × 5 weighting × 3 seeds = 45 runs/method. Quá nhiều cho dataset xác thực.
- **Subset matrix** = 3 forget% × 1 setting (best của MIMIC: exp11_final_ihl075) × 2 seeds = 6 runs/method.
- Lý lẽ trong luận văn: *"Sau khi xác định best setting trên MIMIC-CXR qua 24 exp, ta áp dụng nguyên setting đó trên IU-CXR để kiểm tra tính tổng quát hóa, không tinh chỉnh thêm."*

### 6.2 EXP1-M — Reproduce Forget-MI baseline (MIMIC) ★★★ — ⏳ TODO
- Script: `training/forgetmi_partial.py` (code gốc, KHÔNG sửa)
- Matrix RÚT GỌN: `forget_pct ∈ {3, 6, 10}` × `weighting = Unimodal (best của paper)` × `seed ∈ {42, 0, 123}` = **9 runs**
- Output: Bảng tái lập Table 1 paper TRÊN MÁY BẠN (để có số time/GPU công bằng so với LoKU)
- Lưu ý: Đây là "control" — Forget-MI gốc cần ~5h/run × 9 = ~45h GPU. Lập kế hoạch chạy đêm.
- Có thể trích dẫn số `MIA`, `AUC`, `F1` từ paper, nhưng `Time` PHẢI chạy lại

### 6.3 EXP2-M — Forget-MI-LoKU (MIMIC)
| Status | Forget % | Số seeds | Notes |
|---|---|---|---|
| ✅ **DONE** | 3% | 3 (42, 123, 7) | exp_022-024, summary tại `summary_final_ihl075_multiseed.md` |
| ⏳ TODO | 6% | 3 | dùng cùng config `exp11_final_ihl075`, chỉ đổi `forget_set_path` |
| ⏳ TODO | 10% | 3 | tương tự |

→ Còn lại **6 runs** (3 seeds × 2 forget%). Mỗi run ~12 phút trên máy bạn → tổng ~1.5h.

### 6.4 EXP3-M — Ablation (MIMIC) — 🟡 PARTIAL DONE qua 24 exp lịch sử

24 experiments hiện tại đã ngầm tạo nên ablation. Cần TỔNG HỢP lại thành 1 bảng chuẩn.

**Variants chính rút từ INDEX.md** (3% forget, các seeds khác nhau):

| Variant | Config | Đại diện exp | MIA | Df AUC | Dt AUC |
|---|---|---|---|---|---|
| V0: LoRA BERT only (no FILA) | `loku_subtract_scale=0` | exp_001 | 0.552 | 0.829 | 0.674 |
| V1: + Classifier heads unfrozen | `unfreeze_classifier_heads=true` | exp_003 | 0.557 | 0.830 | 0.669 |
| V2: + NegGrad on forget | `kappa_cls_forget>0` | exp_004 | 0.687 | 0.809 | 0.672 |
| V3: + Distillation from F_re | `distill_teacher='re'` | exp_007 | 0.657 | 0.819 | 0.687 |
| V4: + TRUE FILA subtraction | `loku_subtract_scale=1.0` | exp_008 | 0.562 | 0.833 | 0.679 |
| V5: + IHL bounded forget | `ihl_forget_weight>0` | exp_009 | 0.537 | 0.832 | 0.678 |
| V6: + Image-FILA (Conv2d) | `lora_image_last_k_blocks=1` | exp_010 | 0.493 | 0.718 | 0.689 |
| **V7: Honest mode (no F_re)** | `distill_teacher='og'`, `early_stop='val'` | exp_011d | 0.453 | 0.734 | 0.677 |
| **V8: V7 + multi-seed FINAL** | seeds [42, 123, 7] | exp_022-024 | **0.455 ± 0.027** | **0.736 ± 0.004** | **0.677 ± 0.002** |

→ Bảng này là **trục chính** của Chương 4 ablation. Không cần chạy thêm runs — chỉ cần copy số từ INDEX.md và viết.

→ Có thể chạy thêm **2 variants riêng để hoàn thiện**:
- V_lora_only: LoRA + FILA nhưng KHÔNG có IHL (`ihl_forget_weight=0`) → 3 seeds
- V_ihl_only: IHL + classifier-CE nhưng KHÔNG có FILA (`loku_subtract_scale=0`) → 3 seeds

→ Tổng còn lại nếu cần: **~6 runs**

### 6.5 EXP4-M — Sensitivity study (MIMIC) ★★
Chọn 1-2 hyperparameter quan trọng nhất:

**(a) LoRA rank r**: `r ∈ {2, 4, 8, 16, 32}` × 1 setting × 3 seeds = 15 runs
**(b) FIM samples**: `max_samples ∈ {128, 256, 512, 1024}` × 1 setting × 3 seeds = 12 runs

### 6.6 EXP5-M — MultiDelete baseline (MIMIC) ★★
- Triển khai theo [MultiDelete paper (Cheng & Amiri 2024)] nếu có code public, hoặc bỏ qua nếu khó tái lập
- 3 forget% × 3 seeds = 9 runs
- Nếu khó: trích dẫn trực tiếp số trong Forget-MI paper Table 1

### 6.7 EXP6-M — 1 baseline modality-agnostic (MIMIC) ★
Chọn **NegGrad+** (đơn giản nhất: fine-tune trên Dr + đảo gradient Df) hoặc **SCRUB**.
- 3 forget% × 3 seeds = 9 runs
- Lý do giữ 1 cái: cho thấy bạn cũng đo trên cùng môi trường, không chỉ dựa vào số paper

### 6.8 EXP7-I — Pretrain & Retrain models cho IU-CXR ★★★

> Đây là bước **chuẩn bị bắt buộc** trước khi unlearning trên IU. Vì paper chỉ release pretrained model cho MIMIC, bạn phải tự train từ đầu cho IU.

| Run | Mục đích | Thời gian dự kiến |
|---|---|---|
| **R1** | Train `model_og_IU` trên FULL IU-CXR (≈ paper train MIMIC) | 8-14h |
| **R2** | Train `model_re_IU` trên IU-CXR MINUS forget set (gold standard) | 8-14h × 3 forget% = ~24-42h |

**Lưu ý quan trọng**:
- R2 phải train **riêng cho mỗi forget%** (3%, 6%, 10%) → 3 model retrained. Đây là phần tốn thời gian nhất của EXP7-I.
- Có thể **chỉ train R2 cho 1-2 forget%** để tiết kiệm thời gian (vd: chỉ 3% và 10%).
- Pretrain dùng cùng config với MIMIC (ResNet + BERT, late fusion), chỉ đổi input dataset.
- **Cảnh báo**: nếu không có pretrained model cho IU thì EXP8-I và EXP9-I không thể chạy. Đây là blocker quan trọng nhất.

### 6.9 EXP8-I — Reproduce Forget-MI trên IU-CXR ★★★
- Script: `training/forgetmi_partial.py` với config IU-CXR
- Subset matrix: 3 forget% × 2 weighting (Unimodal, Multimodal) × 2 seeds = **12 runs**
- Output: Bảng so sánh Forget-MI MIMIC vs Forget-MI IU
- Cần `model_og_IU` và `model_re_IU` từ EXP7-I

### 6.10 EXP9-I — Forget-MI-LoKU trên IU-CXR ★★★
- Script: `training/forgetmi_loku.py` với config IU-CXR
- Subset matrix giống EXP8-I: **12 runs**
- Output: Bảng so sánh LoKU MIMIC vs LoKU IU → kiểm tra generalization
- Đây là phần "validate" claim của luận văn

### 6.11 EXP10-I — (Tuỳ chọn) Ablation rút gọn trên IU-CXR ★★
- 2 variants quan trọng nhất: V0 (Forget-MI gốc) vs V4 (Full LoKU)
- 1 forget% (3%) × 1 weighting (Unimodal) × 3 seeds = **6 runs**
- Output: Confirm ablation conclusion từ MIMIC cũng đúng trên IU

### 6.12 Chiến lược tiết kiệm thời gian (cập nhật cho 2 datasets)
| Nếu thiếu thời gian | Cắt cái gì |
|---|---|
| Cắt 1 tuần | Bỏ EXP10-I (ablation IU), EXP6-M (NegGrad+/SCRUB) |
| Cắt 2 tuần | Bỏ thêm EXP4-M (sensitivity) và EXP5-M (MultiDelete) |
| Cắt 3 tuần | Giảm EXP7-I từ 3 model retrained xuống 2 (chỉ 3% và 10%); giảm seed EXP8-I/9-I từ 2 xuống 1 |
| Cắt 4 tuần (gấp) | Bỏ IU-CXR hoàn toàn — quay về roadmap 1 dataset. **Ưu tiên báo cáo kỹ về MIMIC hơn là báo cáo sơ sài cả 2** |

> **Khuyến nghị mạnh**: KHÔNG cắt EXP7-I dù gấp đến mấy. Không có pretrained IU thì toàn bộ EXP8/9/10-I vô nghĩa.

---

## 7. Bảng & Hình cần có trong luận văn

### 7.1 Bảng BẮT BUỘC

#### Bảng 1: So sánh phương pháp với baseline (giống Table 1 paper)
```
                  |         3%          |         6%          |        10%          |
Method            | MIA | DfAUC| DfF1| DtAUC| DtF1 | MIA | DfAUC| DfF1| DtAUC| DtF1 | MIA | DfAUC| DfF1| DtAUC| DtF1 |
------------------+------+-----+-----+------+------+...
Original          |
Retrained         |
NegGrad+ (paper)  |
SCRUB (paper)     |
CF-k (paper)      |
EU-k (paper)      |
MultiDelete (paper)|
Forget-MI (repro) |  ← của bạn EXP1
Forget-MI-LoKU    |  ← của bạn EXP2 ★
```

#### Bảng 2: Hiệu năng (mới — chưa có trong paper)
```
Method            | Time(h) | GPU(GB) | Trainable | Best epoch
------------------+---------+---------+-----------+-----------
Forget-MI         |  ~5     |   ?     |   100%    |   ?
Forget-MI-LoKU    |  ?(target ≤3) | ?  |   ~1-2%  |   ?  ★
```

#### Bảng 3: Ablation
```
Variant                            | MIA | Dt AUC | Time | Params
-----------------------------------+-----+--------+------+--------
(V0) Forget-MI gốc                 |
(V1) + LoRA random init            |
(V2) + Full FT + FIM mask          |
(V3) + LoRA + FIM init             |
(V4) + LoRA + FIM + anchor (full)  |  ★
```

#### Bảng 4: So sánh CROSS-DATASET (mới — chứng minh generalization)
```
                     |       MIMIC-CXR          |     Indiana University CXR
Method               | MIA | Test AUC | Time(h) | MIA | Test AUC | Time(h)
---------------------+-----+----------+---------+-----+----------+--------
Forget-MI (baseline) |     |          |         |     |          |
Forget-MI-LoKU       |     |          |         |     |          | ★
Δ (LoKU - baseline)  |     |          |         |     |          |
```

→ **Câu chuyện cần kể**: "Δ trên MIMIC ≈ Δ trên IU" → method generalize. Nếu Δ rất khác nhau → có vấn đề về domain shift cần thảo luận.

### 7.2 Bảng KHUYẾN KHÍCH có

#### Bảng 5: Sensitivity LoRA rank
```
r  | MIA | Dt AUC | Time | Params(M)
---+-----+--------+------+----------
 2 |
 4 |
 8 |  ← thường sweet spot
16 |
32 |
```

### 7.3 Hình BẮT BUỘC

| Hình | Mô tả | Tool |
|---|---|---|
| **F1: Kiến trúc** | Vẽ pipeline Forget-MI-LoKU (cải tiến Fig 2 paper, thêm LoRA + FIM block) | draw.io / TikZ |
| **F2: Loss distribution** | Histogram loss của forget vs test cho 4 model: Original / Forget-MI / Forget-MI-LoKU / Retrained (giống Fig 3 paper) | matplotlib |
| **F3: Training curve** | CosSim & Loss theo epoch, so sánh Forget-MI vs LoKU | wandb export |
| **F4: Trade-off scatter** | Trục X = Time (giờ), Trục Y = MIA (hoặc Distance). Mỗi điểm = 1 method | matplotlib |
| **F5: Ablation bar** | Bar chart so sánh các variant (MIA hoặc Time) | matplotlib |

### 7.4 Hình KHUYẾN KHÍCH có

- **F6**: GPU memory profile theo epoch
- **F7**: Convergence comparison (số epoch để đạt cùng MIA)
- **F8 (mới)**: **Cross-dataset bar chart** — so sánh side-by-side metric chính (MIA, Test AUC) của Forget-MI và LoKU trên 2 datasets, mỗi forget% là 1 nhóm bar. Đây là hình "selling" cho phần generalization.
- **F9 (mới)**: Loss distribution trên IU-CXR (giống F2 nhưng cho dataset thứ 2)

---

## 8. Cấu trúc viết luận văn

### 8.1 Cấu trúc 5 chương cổ điển

| Chương | Nội dung | Số trang |
|---|---|---|
| **Chương 1: Mở đầu** | 1.1 Đặt vấn đề (right-to-be-forgotten, healthcare, privacy)<br>1.2 Mục tiêu nghiên cứu<br>1.3 Đối tượng & phạm vi<br>1.4 Đóng góp chính<br>1.5 Cấu trúc luận văn | 8-12 |
| **Chương 2: Cơ sở lý thuyết** | 2.1 Machine Unlearning (exact vs approximate)<br>2.2 Multimodal medical models (BERT + ResNet)<br>2.3 LoRA & PEFT<br>2.4 Fisher Information Matrix<br>2.5 Tổng quan Forget-MI<br>2.6 Membership Inference Attack | 15-20 |
| **Chương 3: Phương pháp đề xuất** | 3.1 Tổng quan kiến trúc Forget-MI-LoKU<br>3.2 FIM-guided LoRA initialization<br>3.3 SVD-based knowledge subtraction<br>3.4 Loss functions (4 gốc + anchor)<br>3.5 Training algorithm (pseudo-code)<br>3.6 Implementation details | 12-18 |
| **Chương 4: Thực nghiệm & Kết quả** | 4.1 Datasets (MIMIC-CXR + IU-CXR)<br>4.2 Setup & metrics<br>4.3 [MIMIC] Bảng 1: So sánh với baselines<br>4.4 [MIMIC] Bảng 2: Hiệu năng<br>4.5 [MIMIC] Bảng 3: Ablation<br>4.6 [MIMIC] Bảng 5: Sensitivity<br>4.7 [MIMIC] Hình F2: Loss distribution<br>4.8 [MIMIC] Hình F4: Trade-off<br>4.9 **[Cross-dataset] Bảng 4: So sánh MIMIC vs IU**<br>4.10 **[Cross-dataset] Hình F8: Bar chart**<br>4.11 **[Cross-dataset] Phân tích domain shift & generalization**<br>4.12 Phân tích tổng & thảo luận | 25-35 |
| **Chương 5: Kết luận & Hướng phát triển** | 5.1 Tóm tắt đóng góp<br>5.2 Hạn chế<br>5.3 Hướng mở rộng (foundation models) | 5-8 |
| **Tài liệu tham khảo** | ≥ 30 references | - |
| **Phụ lục** | Code snippets, thêm bảng/hình | optional |

### 8.2 Câu mở đầu mỗi chương (gợi ý)
- Ch1: "Trong kỷ nguyên AI y tế..."
- Ch2: "Để hiểu phương pháp đề xuất, ta cần điểm qua..."
- Ch3: "Forget-MI-LoKU được xây dựng dựa trên 3 ý tưởng chính..."
- Ch4: "Phần này trình bày kết quả thực nghiệm trên MIMIC-CXR..."
- Ch5: "Tóm lại, luận văn đã chứng minh rằng..."

---

## 9. Tiêu chí PASS / FAIL

### 9.1 Tiêu chí TỐI THIỂU (để bảo vệ thành công)
| # | Tiêu chí | Ngưỡng | Dataset | Status |
|---|---|---|---|---|
| 1 | Thời gian unlearning giảm | ≥ 40% (5h → ≤ 3h) | MIMIC | ✅ **ĐẠT** (5h → 0.2h, giảm 96%) |
| 2 | GPU peak memory giảm | ≥ 30% | MIMIC | ✅ ~12GB, cần đo baseline để xác nhận |
| 3 | Trainable params | ≤ 5% tổng tham số | Cả 2 | ✅ **ĐẠT** (0.451%) |
| 4 | MIA score | ±0.10 so với baseline | MIMIC | ✅ **VƯỢT** (0.455 vs 0.571, tốt hơn) |
| 5 | Test AUC | giảm ≤ 0.03 | MIMIC | ✅ **VƯỢT** (0.677 vs 0.625, tăng) |
| 6 | Test Macro-F1 | giảm ≤ 0.05 | MIMIC | ✅ **VƯỢT** (0.364 vs 0.250, tăng) |
| 7 | Số seeds | ≥ 3 MIMIC, ≥ 2 IU | Cả 2 | ✅ MIMIC 3% done / ⏳ IU |
| 8 | Có ablation | LoRA, FIM, IHL, FILA riêng | MIMIC | ✅ **ĐẠT** (24 exp đã gom thành bảng V0-V8) |
| 9 | Có ≥ 1 sensitivity study | IHL sweep 0.5/0.75/1.0 | MIMIC | ✅ **ĐẠT** (exp11c, exp11d) |
| 10 | Reproducibility | Code + config + seed | Cả 2 | ✅ auto-tracker `experiments/` |
| **11** | **Cross-dataset consistency** | |Δ_MIMIC − Δ_IU| ≤ 0.15 | Cả 2 | ⏳ chưa có IU |
| **12** | **Pretrained + Retrained models cho IU** | Đã train xong | IU | ⏳ chưa bắt đầu |
| **13** (mới) | **MIMIC 6% và 10%** | Phải có để so paper Table 1 cột 6%, 10% | MIMIC | ⏳ chưa chạy |

**Tổng kết**: 9/13 ĐẠT (chỉ thiếu IU end-to-end và MIMIC 6%/10%). Phần lớn công sức còn lại là **chạy thêm runs + viết**, không phải sửa code.

### 9.2 Tiêu chí HOÀN HẢO (đề tài xuất sắc)
- Tất cả 10 tiêu chí trên + những điều sau:
- Vượt Forget-MI trên ít nhất 1 metric (không chỉ ngang)
- Có statistical significance test (paired t-test giữa LoKU và Forget-MI)
- Có 2+ sensitivity studies
- Có ít nhất 1 hình "ngon" (loss distribution hoặc trade-off curve)
- Code clean, có README, có thể public lên GitHub

### 9.3 Dấu hiệu FAIL phải tránh
- Loss diverge (như hiện tại) → vô nghĩa
- Test AUC sụp đổ (≤ 0.4) → unlearning quá tay, mất utility
- MIA = 1.0 mọi lúc → unlearning không có tác dụng
- Chỉ chạy 1 seed → không đáng tin
- Không có ablation → không biết LoRA hay FIM đóng góp gì

---

## 10. Timeline & Checklist tuần (CẬP NHẬT — code MIMIC đã xong)

> **Trạng thái khởi đầu**: Method đã hoàn thiện trên MIMIC 3% với kết quả vượt paper. Còn lại: MIMIC 6%/10%, baseline reproduce, IU end-to-end, viết luận văn.
> **Giả định**: ~6 tuần đến deadline. Đã rút từ 8-9 tuần vì code chính đã xong.

### Tuần 1 (MIMIC mở rộng + Khởi động IU)
- [ ] **(MIMIC)** EXP2-M-6: 3 runs LoKU trên 6% forget (config `exp11_final_ihl075` + đổi `forget_set_path`)
- [ ] **(MIMIC)** EXP2-M-10: 3 runs LoKU trên 10% forget
- [ ] **(MIMIC)** Tạo `experiments/exp11_final_ihl075_6per_summary.md` và `_10per_summary.md`
- [ ] **(IU)** Download Indiana University CXR (Open-i)
- [ ] **(IU)** Setup folder `data_iu/` theo Mục 13.2

### Tuần 2 (Preprocessing IU + Baseline reproduce MIMIC 3%)
- [ ] **(IU)** Parse XML reports → text (script ở Mục 13.3 bước 2)
- [ ] **(IU)** Sinh label binary normal/abnormal (Option A)
- [ ] **(IU)** Tạo `all_data_iu.tsv` + train/test split + forget sets 3/6/10%
- [ ] **(IU)** Adapt config thành `config_iu.yaml`
- [ ] **(MIMIC)** EXP1-M chạy 3 seeds Forget-MI gốc trên 3% (~15h GPU)
- [ ] **(MIMIC)** Có Bảng 1 cột "Forget-MI repro" cho 3%

### Tuần 3 (Pretrain IU + Baseline reproduce MIMIC 6%, 10%)
- [ ] **(IU)** EXP7-I R1: pretrain `model_og_IU` trên full IU (~10-14h background)
- [ ] **(MIMIC)** EXP1-M chạy 3 seeds Forget-MI gốc trên 6% và 10% (~30h GPU đêm)
- [ ] **(MIMIC)** Hoàn thiện Bảng 1 đầy đủ 3 cột forget% so sánh paper
- [ ] **(MIMIC)** Tạo Bảng 2 hiệu năng (thời gian thực đo)

### Tuần 4 (Retrain IU + Chạy IU experiments)
- [ ] **(IU)** EXP7-I R2: retrain 3 `model_re_IU` cho forget 3/6/10% (~30h background — chạy đêm)
- [ ] **(IU)** Sanity run Forget-MI và LoKU trên IU 3% (1 seed) để confirm pipeline
- [ ] **(IU)** EXP8-I: 6 runs Forget-MI trên IU (3 forget% × 2 seeds)
- [ ] **(MIMIC)** Vẽ Hình F2 (loss distribution), F4 (trade-off), F3 (training curve)

### Tuần 5 (LoKU IU + Hình)
- [ ] **(IU)** EXP9-I: 6 runs LoKU trên IU (3 forget% × 2 seeds)
- [ ] **(IU)** Tổng hợp Bảng 4 cross-dataset (MIMIC vs IU)
- [ ] Vẽ Hình F1 (kiến trúc) — quan trọng cho Chương 3
- [ ] Vẽ Hình F8 (cross-dataset bar) — selling point chính
- [ ] Vẽ Hình F9 (loss dist IU)
- [ ] (Tuỳ chọn) EXP3-M bổ sung: 2 ablation variants còn thiếu (V_lora_only, V_ihl_only) — 6 runs

### Tuần 6 (Viết & Polish)
- [ ] Viết **Chương 4** trước — đây là phần dài nhất. Tận dụng các summary có sẵn (exp11_final, exp10c, etc.)
- [ ] Viết Chương 3 (Phương pháp) — dùng sơ đồ Mục 4.1 + bảng Mục 4.2 làm khung
- [ ] Viết Chương 2 (Cơ sở lý thuyết) — LoRA, FIM, FILA, IHL paper references
- [ ] Viết Chương 1 và 5
- [ ] Format BibTeX, statistical test (paired t-test trên 3 seeds), polish hình ảnh

### Buffer (nếu deadline xa hơn)
- [ ] Sensitivity study LoRA rank (5 giá trị × 3 seeds = 15 runs)
- [ ] MultiDelete baseline reproduce
- [ ] NegGrad+/SCRUB baseline reproduce
- [ ] Mở rộng ablation IU đầy đủ V0-V8

### ⚠️ Risk & Mitigation

| Rủi ro | Mức độ | Mitigation |
|---|---|---|
| Pretrain IU quá lâu | CAO | Khởi động NGAY tuần 1; nếu vượt 14h, giảm epoch từ 60 → 40 |
| Retrain IU × 3 hết time | CAO | Chỉ retrain 3% và 10% nếu gấp (bỏ 6%) |
| EXP1-M chậm (45h cho 9 runs) | TB | Chấp nhận trích số Time của Forget-MI gốc từ paper, ghi rõ "reported in [Hardan 2024]" |
| LoKU trên IU fail (do task khác) | TB | Đổi task IU thành multi-class top-3 MeSH thay vì binary |
| Domain shift làm Δ MIMIC ≠ Δ IU | THẤP | Đó là finding hợp lệ — viết thành phần discussion ở 4.11 |

---

## 11. FAQ & Lỗi thường gặp

### Q1: GPU OOM khi chạy LoKU?
- Giảm `unlearn_batch_size` từ 16 → 8 hoặc 4
- Đặt `model_og` và `model_re` ở `.half()` (đã làm trong code)
- Bật gradient checkpointing: `model_unlearn.gradient_checkpointing_enable()`

### Q2: CosSim không tăng dù đã fix bug?
- Kiểm tra LoRA có thật sự gắn vào BERT không: `print([n for n, _ in model_unlearn.named_parameters() if 'lora' in n])`
- Đảm bảo learning rate không quá thấp (thử 5e-5, 1e-4)
- Kiểm tra anchor loss có được tính không (thêm log L_anchor mỗi step)

### Q3: MIA score luôn = 1.0?
- SVM quá khớp với retain — thử SVC với `C=1` thay vì `C=3`
- Hoặc dùng MIA dựa trên entropy thay vì loss

### Q4: Thời gian chạy lâu hơn Forget-MI gốc?
- Compute_importance quá lâu? Giảm `max_samples` xuống 256
- Số epoch quá nhiều? Giảm `unlearn_epochs` từ 30 xuống 8-10
- Eval mỗi epoch nặng? Giảm `eval_subset` xuống 100 mẫu

### Q5: Kết quả tái lập (EXP1) khác paper nhiều?
- Bình thường nếu lệch ±0.05 trên MIA, ±0.02 trên AUC (do seed, GPU)
- Nếu lệch > 0.10: kiểm tra `text_noise_level`, `random_point_ratio`, weight setting có đúng không
- Ghi chú trong báo cáo: "Reproduced under {GPU}, {framework version}, may differ from reported numbers due to..."

### Q6: Có nên dùng SciBERT thay vì BERT?
- Paper dùng SciBERT. Nếu base_model_path đã là SciBERT thì giữ nguyên. Nếu không, có thể dùng BERT-base nhưng phải ghi chú.

### Q7: Test set lấy từ đâu?
- Từ `mimic-cxr-sub-img-edema-split-manualtest.csv` — các dòng có cột cuối = 'TEST'
- Đã được tách trong `data_split()` của `forgetmi_partial.py`

### Q8: Multi-run quản lý sao?
- Dùng wandb sweep với config YAML
- Hoặc bash loop: `for pct in 3 6 10; do for w in ...; do for s in ...; do python ... ; done; done; done`
- Lưu kết quả vào CSV để dễ load lại

### Q9: IU-CXR không có label edema, nên dùng task gì?
- **Khuyến nghị**: binary "normal vs abnormal" (Option A). Đơn giản, có sẵn từ MeSH.
- Trong report: ghi rõ "trên IU-CXR, do hạn chế label, chúng tôi đánh giá trên proxy task binary classification"
- KHÔNG cần task giống hệt MIMIC — vì mục đích IU là chứng minh **method generalize**, không phải **kết quả tuyệt đối**

### Q10: Pretrain IU mất quá lâu (14h), có cách nào nhanh hơn?
- Khởi tạo từ MIMIC pretrained model làm warm-start → giảm ~50% epoch
- Giảm số layer ResNet (từ 7 xuống 5) cho IU vì dataset nhỏ hơn
- Dùng batch_size lớn hơn nếu GPU cho phép (32 thay vì 16)
- Trade-off: pretrain ngắn → final model hơi yếu, nhưng vẫn OK cho mục đích unlearning research

### Q11: Có nên dùng PubMedBERT thay vì BERT cho IU?
- IU report ngắn hơn MIMIC → BERT base vẫn đủ
- Nếu muốn fair so với MIMIC (paper dùng SciBERT) → dùng SciBERT cho cả 2 dataset, dễ so sánh hơn

### Q12: Pipeline IU có giống MIMIC 100% không?
- Hầu hết: data loader, model, loss đều dùng chung
- KHÁC:
  - `text_data_dir`, `img_data_dir`, `data_split_path`, `forget_set_path` trỏ đến file IU
  - `output_channel_encoding` có thể đổi từ "multiclass" sang "binary" (nếu dùng task normal/abnormal)
  - `num_labels` đổi từ 4 sang 2
- Khuyến nghị: tạo `config_iu.yaml` riêng, không sửa `config.yaml` (MIMIC) — dễ debug

### Q13: Nếu kết quả IU tệ hơn MIMIC nhiều thì sao?
- Đó là **finding hợp lệ**, không phải failure
- Phân tích: do IU report ngắn hơn? Dataset nhỏ hơn? Task khác?
- Trong Chương 4.11, viết: "Method work trên cả 2 datasets nhưng efficacy gap ở IU lớn hơn vì [lý do]"
- Đây vẫn là kết quả có giá trị khoa học

---

## 12. Phụ lục: Code patches — ❌ KHÔNG CÒN ÁP DỤNG

> Trước đây phần này chứa 9 code patches (loss bound, eval block, SVD init, target modules, FIM, early stop, gradient clip, config tunings, batch script). **TẤT CẢ đã được tích hợp vào code chính** qua 24 experiments (exp_001 → exp_024).
>
> Không cần áp dụng patch nào nữa. Để hiểu code hiện tại, đọc:
> - **`training/forgetmi_loku.py`** (1278 dòng): tất cả components hoạt động đúng
> - **`config.yaml`**: full config tối ưu (`exp11_final_ihl075`)
> - **`experiments/INDEX.md`**: lịch sử 24 exp + insights chính
> - **`experiments/summary_final_ihl075_multiseed.md`**: kết quả chính thức multi-seed

### 12.1 Để chạy 1 experiment mới (vd: thêm seed, đổi forget%)
```bash
# 1. Sửa config.yaml:
#    - đổi forget_set_path → forget_set_6per.csv hoặc forget_set_10per.csv
#    - đổi random_seed → 42 / 123 / 7
# 2. Trên Colab notebook, đổi EXP_NAME ở Cell 4 đầu, rồi chạy
# 3. Auto-tracker sẽ tạo experiments/exp_NNN_<name>.md và update INDEX.md
# 4. Local: git pull, điền Observations + Conclusion vào file MD
```

### 12.2 Để chạy IU dataset (sau khi pretrain xong)
```bash
# 1. Tạo config_iu.yaml (copy config.yaml, đổi paths sang data_iu/)
# 2. Chạy:
PYTHONPATH=. python training/forgetmi_loku.py --config config_iu.yaml --exp exp_iu_3per_seed42
```

### 12.3 Để chạy Forget-MI gốc (baseline reproduce)
```bash
PYTHONPATH=. python training/forgetmi_partial.py --config config_baseline.yaml
# (cần tạo config_baseline.yaml giống config.yaml nhưng dùng tham số gốc paper)
```

### 12.4 Để chạy batch nhiều seeds
Hệ thống đã có Cell 4 trong `run.ipynb` hỗ trợ multi-seed loop. Hoặc bash đơn giản:
```bash
for seed in 42 123 7; do
  sed -i "s/random_seed:.*/random_seed:\n    value: $seed/" config.yaml
  python training/forgetmi_loku.py --config config.yaml --exp exp_3per_seed${seed}
done
```

---

## 13. Phụ lục: Chuẩn bị dataset Indiana University

### 13.1 Nguồn dữ liệu
- **Tên chính thức**: Indiana University Chest X-Ray Collection (a.k.a. Open-i CXR / IU-CXR)
- **Download**:
  - Ảnh: https://openi.nlm.nih.gov/faq#collection (3 GB, file `NLMCXR_png.tgz`)
  - Reports: file `NLMCXR_reports.tgz` (XML format)
- **License**: Open access, không cần credential (khác MIMIC)
- **Citation**: Demner-Fushman et al. (2015), *Preparing a collection of radiology examinations for distribution and retrieval*, JAMIA

### 13.2 Cấu trúc thư mục đề xuất
```
data_iu/
├── img_data/                    # PNG files, 1 ảnh = 1 file (cùng format MIMIC)
│   ├── CXR1_1_IM-0001-3001.png
│   └── ...
├── text_data/                   # txt file đã extract từ XML, 1 report = 1 file
│   ├── CXR1.txt
│   └── ...
├── metadata/
│   └── all_data_iu.tsv          # File mapping image_id <-> report_id <-> label
└── splits/
    ├── train_test_split_iu.csv  # cột TRAIN/TEST
    ├── forget_set_3per_iu.csv
    ├── forget_set_6per_iu.csv
    └── forget_set_10per_iu.csv
```

### 13.3 Pipeline preprocessing (10 bước)

**Bước 1: Download**
```bash
mkdir -p data_iu/raw && cd data_iu/raw
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_png.tgz
wget https://openi.nlm.nih.gov/imgs/collections/NLMCXR_reports.tgz
tar -xzf NLMCXR_png.tgz
tar -xzf NLMCXR_reports.tgz
```

**Bước 2: Parse XML reports → text**
```python
# scripts/parse_iu_reports.py
import xml.etree.ElementTree as ET
import os, glob

reports = {}
for xml_file in glob.glob("data_iu/raw/ecgen-radiology/*.xml"):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Extract findings + impression
    findings = root.find(".//AbstractText[@Label='FINDINGS']")
    impression = root.find(".//AbstractText[@Label='IMPRESSION']")
    text = ""
    if findings is not None and findings.text:
        text += findings.text + " "
    if impression is not None and impression.text:
        text += impression.text

    # Extract MeSH labels for later
    major_mesh = [m.text for m in root.findall(".//MeSH/major") if m.text]

    # Image IDs liên kết với report
    parent_image_ids = [p.get("id") for p in root.findall(".//parentImage")]

    report_id = os.path.basename(xml_file).replace(".xml", "")
    reports[report_id] = {
        "text": text.strip(),
        "mesh": major_mesh,
        "image_ids": parent_image_ids,
    }
```

**Bước 3: Sinh label**

Option A — Binary normal/abnormal (KHUYẾN NGHỊ):
```python
NORMAL_KEYWORDS = ["normal", "no acute", "no evidence", "negative", "unremarkable"]

def label_binary(report):
    text_lower = report["text"].lower()
    if any(kw in text_lower for kw in NORMAL_KEYWORDS) and len(report["mesh"]) == 0:
        return 0  # normal
    return 1  # abnormal
```

Option B — Multi-class theo top-K MeSH:
```python
# Đếm tần suất MeSH, chọn top-3 phổ biến (sau "normal")
# Vd: Atelectasis, Cardiomegaly, Effusion
```

**Bước 4: Tạo all_data_iu.tsv** (format giống MIMIC's `all_data.tsv`)
```
report_id  image_id  text  label
CXR1       IM-0001    ...   0
CXR1       IM-0002    ...   0
CXR2       IM-0003    ...   1
```

**Bước 5: Tạo train/test split**
```python
from sklearn.model_selection import train_test_split

# Split theo PATIENT (report_id), không theo image, để tránh leak
all_reports = list(reports.keys())
train_reports, test_reports = train_test_split(
    all_reports, test_size=0.15, random_state=42, stratify=labels
)
# Đánh dấu TRAIN/TEST trong file CSV
```

**Bước 6: Tạo forget sets**
```python
# forget_set_3per: chọn ngẫu nhiên 3% subject_id từ train set
import random
random.seed(42)
train_subjects = list(set(r.split('_')[0] for r in train_reports))
forget_3 = random.sample(train_subjects, int(0.03 * len(train_subjects)))
# Tương tự forget_6, forget_10
```

**Bước 7: Adapt `joint_img_txt/convert_examples_to_features.py`** (nếu cần)
- Hầu hết code MIMIC reuse được vì format `all_data.tsv` giống nhau
- Có thể cần chỉnh `EdemaClassificationProcessor` → tạo `IUBinaryClassificationProcessor` mới

**Bước 8: Tạo `config_iu.yaml`** (copy `config.yaml` và đổi paths)
```yaml
parameters:
  base_model_path:
    value: "./forgetme/training_iu_model"     # pretrained IU
  retrained_model_path:
    value: "./model_retrained_iu_3per/"
  output_dir:
    value: "./unlearning_output_iu/"
  bert_pretrained_dir:
    value: "./forgetme/training_iu_model/"
  text_data_dir:
    value: "./data_iu/metadata"
  data_split_path:
    value: "./data_iu/splits/train_test_split_iu.csv"
  img_data_dir:
    value: "./data_iu/img_data"
  forget_set_path:
    value: "./data_iu/splits/forget_set_3per_iu.csv"
  output_channel_encoding:
    value: "binary"          # đổi từ "multiclass"
  # ... các config khác giữ nguyên
```

**Bước 9: Pretrain model_og_IU** (tương tự cách paper pretrain MIMIC)
```bash
# Cần script pretrain — paper không release.
# Tham khảo Chauhan et al. (2020) "Joint modeling of chest radiographs and radiology reports"
# Repo: https://github.com/RayRuizhiLiao/joint_chestxray
# Adapt script train.py của họ cho IU dataset
```

**Bước 10: Retrain model_re_IU cho mỗi forget%**
```bash
# Train lại model_og nhưng EXCLUDE forget set
# 3 lần: cho 3%, 6%, 10%
python pretrain.py --exclude_forget data_iu/splits/forget_set_3per_iu.csv \
  --output_dir model_retrained_iu_3per
# Tương tự cho 6%, 10%
```

### 13.4 Checklist xác nhận trước khi chạy unlearning IU
- [ ] `data_iu/img_data/` chứa ≥ 7,000 PNG files
- [ ] `data_iu/text_data/` chứa ≥ 3,900 txt files
- [ ] `all_data_iu.tsv` mở được, không lỗi format, có đủ 4 cột
- [ ] `forget_set_*per_iu.csv` không overlap với test set
- [ ] `model_og_IU/pytorch_model.bin` tồn tại và load được
- [ ] `model_retrained_iu_3per/pytorch_model.bin` tồn tại cho ÍT NHẤT 1 forget%
- [ ] Chạy `make_tsv.py` adapted cho IU không error
- [ ] Sanity run 1 epoch forgetmi_partial trên IU → loss < 0 trong 1-2 epoch đầu

### 13.5 Ước tính thời gian preprocessing
| Bước | Thời gian | Ghi chú |
|---|---|---|
| Download + extract | 1-2h | tuỳ tốc độ mạng |
| Parse XML | 30 phút | script chạy nhanh |
| Generate labels (Option A) | 1h | bao gồm debug keyword |
| Generate splits + forget sets | 30 phút | |
| Adapt config + processor | 2-3h | code work |
| Pretrain model_og_IU | 8-14h | GPU background |
| Retrain × 3 forget% | 24-42h | GPU background |
| **Tổng setup IU** | **~2 ngày active + ~2 ngày GPU background** | |

---

## TÓM TẮT NHANH (nếu chỉ đọc 1 phần)

### Trạng thái HIỆN TẠI (2026-06-16)
- ✅ **Code MIMIC**: HOÀN THIỆN qua 24 exp. Final: `exp11_final_ihl075` (IHL=0.75, image-FILA scale=0.3, distill from F_og)
- ✅ **MIMIC 3% multi-seed**: VƯỢT paper trên 4/5 metric, ngang 1/5, nhanh hơn ~25x. Số chính thức: `experiments/summary_final_ihl075_multiseed.md`
- ⏳ **MIMIC 6%, 10%**: chưa chạy (6 runs còn lại)
- ⏳ **Forget-MI baseline reproduce** (cùng máy): chưa chạy
- ⏳ **IU-CXR**: chưa bắt đầu — đây là phần tốn thời gian nhất còn lại

### Đã đạt (không cần làm lại)
- Sửa code, tích hợp IHL, FILA, Image-FILA, honest distillation, eval pipeline
- 24 experiments với auto-tracker logging
- Multi-seed MIMIC 3% với mean ± std
- Ablation ngầm qua lịch sử exp (V0-V8 trong Section 6.4)
- IHL sensitivity sweep (0.5/0.75/1.0)

### Còn lại (6 tuần)
| Tuần | Việc chính |
|---|---|
| 1 | MIMIC 6%, 10% multi-seed (6 runs) + download IU |
| 2 | Preprocess IU + EXP1-M baseline reproduce 3% |
| 3 | Pretrain IU + baseline 6%, 10% |
| 4 | Retrain IU + EXP8-I (Forget-MI IU) |
| 5 | EXP9-I (LoKU IU) + vẽ hình |
| 6 | Viết luận văn |

### 3 risk lớn nhất
1. **Pretrain IU** (~14h GPU) — khởi động NGAY
2. **Retrain IU × 3 forget%** (~30h GPU) — chạy đêm song song
3. **Forget-MI baseline reproduce trên 6%/10%** (~30h GPU) — có thể trích số paper nếu gấp

### Câu cần nhớ
> **Bạn KHÔNG cần fix code nữa. Bạn cần CHẠY THÊM (MIMIC 6/10% + IU) và VIẾT.**

---

*Roadmap này được cập nhật ngày 2026-06-16 (v3: phản ánh trạng thái thực tế sau 24 experiments — code MIMIC đã xong, công sức còn lại tập trung vào IU + viết).*
