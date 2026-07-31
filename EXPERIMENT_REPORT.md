# Forget-MI-LoKU — Báo cáo thực nghiệm

> **Đối tượng**: Khóa luận tốt nghiệp — Machine Unlearning cho dữ liệu y tế đa phương thức
> **Ngày cập nhật**: 2025
> **Tác giả thực hiện**: Nguyễn Hoàng Như
> **Mô hình tham chiếu (baseline)**: Forget-MI (Hardan et al., MICCAI 2025)

---

## 1. Mục tiêu nghiên cứu

Đề xuất phương pháp **Forget-MI-LoKU** kết hợp **LoRA (Low-Rank Adaptation)** và **Fisher Information Matrix (FIM)** vào pipeline Forget-MI gốc nhằm:

1. **Giảm thời gian unlearning** so với Forget-MI baseline (paper báo ~5h)
2. **Giảm số tham số trainable** từ 100% xuống dưới 1%
3. **Giảm tài nguyên tính toán** (GPU memory, FLOPs)
4. **Bảo toàn** các chỉ số chính: MIA, Test AUC, Test F1 ≈ baseline
5. Chấp nhận **trade-off nhỏ** trên forget set performance

---

## 2. Bộ dữ liệu

| Mục | Giá trị |
|---|---|
| Dataset | **MIMIC-CXR** (subset từ [Chauhan 2020]) |
| Số bệnh nhân | 1,663 subjects |
| Số mẫu (image + report) | 6,742 |
| Số class | 4 (no edema / vascular congestion / interstitial edema / alveolar edema) |
| Phân bố class | 43% / 25% / 22% / 10% (imbalanced) |
| Forget percentage thử nghiệm | **3%** (paper test 3%, 6%, 10%) |

### Split sau preprocessing (forget 3%)

| Split | Số mẫu | Vai trò |
|---|---|---|
| Retain (D_r) | 5,409 | Train + giữ lại |
| Validation | 601 | Theo dõi training |
| Test (D_t) | 531 | Đánh giá utility |
| Forget (D_f) | 201 | Cần "quên" |
| Random (D̃_f) | 201 | Forget + noise (cho UU/MD loss) |

---

## 3. Kiến trúc mô hình

### 3.1 Backbone (kế thừa từ Forget-MI gốc)

**ImageTextModel** — kiến trúc late-fusion multimodal:

```
┌─────────────────────┐         ┌────────────────────────┐
│  CXR Image (1024×)  │         │  Radiology Report      │
└──────────┬──────────┘         └───────────┬────────────┘
           │                                │
    ┌──────▼──────┐                  ┌─────▼─────┐
    │  ResNet-18  │                  │   BERT    │
    │  (img_model)│                  │ (text_model)│
    └──────┬──────┘                  └─────┬─────┘
           │                                │
        z_img (768)                     z_txt (768)
           │                                │
    ┌──────▼──────┐                  ┌─────▼─────┐
    │  fc1 (img   │                  │ classifier│
    │  classifier)│                  │  (txt cls)│
    └──────┬──────┘                  └─────┬─────┘
           │                                │
       logits_img                       logits_txt
```

| Component | Kiểu | Hidden size | Trainable trong baseline |
|---|---|---|---|
| img_model (ResNet) | CNN 7 layers | 768 | ✓ (≈40M params) |
| text_model (BERT) | Transformer 12 layers | 768 | ✓ (≈110M params) |
| Gates (Joint Embedding) | Linear gates | 768 → 768 | ✓ |
| **Tổng tham số** | | | **~113.7M** |

### 3.2 Đóng góp LoKU — Cấu hình LoRA + FIM

**LoRA injection vào BERT attention**:

| Component | Giá trị |
|---|---|
| Target modules | `["query", "key", "value"]` (trong mọi BERT layer) |
| LoRA rank `r` | 8 |
| LoRA alpha `α` | 16 (scaling = α/r = 2) |
| LoRA dropout | 0.05 |
| Số layer LoRA inject | 36 (12 BERT layers × 3 modules) |
| **Trainable params** | **442,368 (0.389% tổng)** |

**Fisher Information Matrix (FIM) initialization**:

Thay vì init LoRA ngẫu nhiên, FIM dẫn đường init:

1. Tính importance F(θ) ≈ E[(∂ log p(y|x)/∂θ)²] cho mỗi tham số target
2. Trên forget set: `F_f` (Fisher trên D_f, 256 samples)
3. Trên retain set: `F_r` (Fisher trên D_r, 256 samples)
4. Relative importance: `imp = F_f / (F_r + ε)` — cao ở chỗ forget cần thay đổi, retain không
5. Init LoRA bằng SVD có trọng số theo imp:
   - `W_imp = sqrt(imp) ⊙ W_base`
   - `U, S, V = SVD_lowrank(W_imp, r)`
   - `lora_A = (V √S)^T × scale`, `lora_B = (U √S) × scale`
6. Init scale = 0.05 → BA ≈ 0 ban đầu → mô hình hành xử như F_og lúc khởi tạo

---

## 4. Hàm mất mát (Loss function)

### 4.1 Định nghĩa các thành phần

Gọi `D[a,b]` là khoảng cách Euclidean giữa 2 embedding.

**L_UU (Unimodal Unlearning)** — đẩy unimodal embedding của forget data ra xa noisy random data:
```
d_uu = D([F_ul(I_f); F_ul(T_f)], [F_og(Ĩ_f); F_og(T̃_f)])
L_UU = ReLU(margin_forget − d_uu)        ← bounded hinge
```

**L_MD (Multimodal Disassociation)** — tương tự nhưng trên joint embedding:
```
d_md = D(F_ul(I_f, T_f), F_og(Ĩ_f, T̃_f))
L_MD = ReLU(margin_forget − d_md)        ← bounded hinge
```

**L_UKR (Unimodal Knowledge Retention)** — giữ unimodal retain gần F_og:
```
d_ukr = D([F_ul(I_r); F_ul(T_r)], [F_og(I_r); F_og(T_r)])
L_UKR = ReLU(d_ukr − margin_retain) + 0.1 × d_ukr
```

**L_MKR (Multimodal Knowledge Retention)** — tương tự cho joint:
```
d_mkr = D(F_ul(I_r, T_r), F_og(I_r, T_r))
L_MKR = ReLU(d_mkr − margin_retain) + 0.1 × d_mkr
```

**L_RE (Anchor toward Retrained — đóng góp mới của LoKU)**:
```
L_RE = D([F_ul(I_r); F_ul(T_r)], [F_re(I_r); F_re(T_r)])
```

### 4.2 Tổng loss

```
L_total = α·L_UKR + β·L_UU + θ·L_MD + γ·L_MKR + η·L_RE
```

### 4.3 So sánh loss design

| Aspect | Forget-MI gốc | Forget-MI-LoKU (ta) |
|---|---|---|
| L_UU/L_MD | `−D[…]` (unbounded → diverge về −∞) | `ReLU(margin − D)` (bounded) |
| L_UKR/L_MKR | `min(L, margin)` (clamp, mất gradient khi vượt) | `ReLU(L − margin) + 0.1·L` (luôn có gradient) |
| Anchor về F_re | ❌ Không có | ✅ Thêm L_RE |
| Early stopping | Theo loss (loss âm → không bao giờ stop) | Theo CosSim(F_ul, F_re) ↑ |
| Gradient clip | ❌ | ✅ max_norm=1.0 |

---

## 5. Hyperparameter

| Tham số | Giá trị | Ghi chú |
|---|---|---|
| Learning rate | 5e-4 | LoRA chịu được lr cao hơn full FT |
| Weight decay | 0.01 | AdamW |
| Batch size (unlearn) | 16 | |
| Batch size (eval) | 16 | |
| Số epoch tối đa | 8 | LoRA hội tụ nhanh hơn |
| Early stop patience | 3 | Trên CosSim |
| Grad clip | 1.0 | |
| Random seed | 42 | |
| forget_margin | 8.0 | Ngưỡng "đủ quên" |
| α (UKR weight) | 1.0 | |
| β (UU weight) | 1.0 | |
| θ (MD weight) | 0.5 | |
| γ (MKR weight) | 0.5 | |
| η (RE-anchor weight) | 0.5 | Đóng góp LoKU |
| Fisher max samples | 256 | |
| LoKU init scale | 0.05 | |
| Precision | mixed (fp32 LoRA, fp16 frozen) | autocast |

---

## 6. Quy trình thực thi

```
┌─────────────────────────────────────────────────────────┐
│ 1. Load 3 models: F_og (fp32 → fp16), F_re (fp16),     │
│    F_ul = clone of F_og (fp32 cho training)             │
├─────────────────────────────────────────────────────────┤
│ 2. Build dataset: retain (5409), val (601), test (531), │
│    forget (201), random (201)                           │
├─────────────────────────────────────────────────────────┤
│ 3. Compute Fisher Information (FIM)                     │
│    - F_f trên forget set (256 samples)                  │
│    - F_r trên retain set (256 samples)                  │
├─────────────────────────────────────────────────────────┤
│ 4. Apply LoRA wrapping với target_modules               │
│    + FIM-guided SVD soft init (scale=0.05)             │
│    → Trainable: 442K/113.7M (0.389%)                    │
├─────────────────────────────────────────────────────────┤
│ 5. Training loop (tối đa 8 epoch):                      │
│    - Forward 3 set: forget, random, retain              │
│    - Compute 4+1 losses (UU, MD, UKR, MKR, RE)         │
│    - Backward chỉ qua LoRA params                       │
│    - Early stop nếu CosSim plateau                      │
├─────────────────────────────────────────────────────────┤
│ 6. Merge LoRA vào base model                            │
├─────────────────────────────────────────────────────────┤
│ 7. Final evaluation:                                     │
│    - MIA (SVM trên loss distribution)                   │
│    - CosSim vs F_re                                     │
│    - AUC + Macro-F1 trên test set                       │
│    - AUC + Macro-F1 trên forget set                     │
│    - Time, GPU peak, trainable params                   │
├─────────────────────────────────────────────────────────┤
│ 8. Append kết quả vào results_summary.csv               │
└─────────────────────────────────────────────────────────┘
```

---

## 7. Các chỉ số đánh giá

### 7.1 Bảng tham chiếu nhanh

| Chỉ số | Hướng | Đo cái gì | F_og | F_re | Forget-MI paper | **LoKU (ta)** |
|---|---|---|---|---|---|---|
| **MIA** | ↓ | Attacker phân biệt được forget khỏi training set không | 1.000 | 0.000 | 0.571 | **0.552** |
| **Forget AUC** (D_f) | ↓ | Model còn nhận ra forget data không | 0.999 | 0.566 | 0.735 | **0.829** |
| **Forget F1** (D_f) | ↓ | Tương tự nhưng macro F1 | 0.965 | 0.310 | 0.393 | **0.597** |
| **Test AUC** (D_t) | ↑ | Utility trên data mới | 0.677 | 0.626 | 0.625 | **0.674** |
| **Test F1** (D_t) | ↑ | Tương tự với F1 | 0.388 | 0.362 | 0.250 | **0.387** |
| **1 − CosSim** | ↓ | Khoảng cách F_ul vs F_re | — | 0.000 | ~0.4-0.5 | **0.371** |
| **Time (h)** | ↓ | Wall-clock unlearning | 14h (train từ đầu) | 14h | 5h | **0.094h** |
| **Trainable params** | ↓ | Số tham số phải tối ưu | 100% | 100% | 100% | **0.389%** |
| **GPU peak (GB)** | ↓ | RAM GPU max | — | — | (không báo) | **11.76 GB** |

### 7.2 Định nghĩa chi tiết

**MIA (Membership Inference Attack)**:
- Train SVM(RBF) phân biệt loss của retain (=1) vs test (=0)
- Predict trên forget → trả về tỉ lệ forget bị classify là "member"
- Ideal: 0.5 (attacker không phân biệt được)

**Forget AUC / F1**: Hiệu năng phân loại trên forget set. Cao = model nhớ; thấp = model quên.

**Test AUC / F1**: Hiệu năng phân loại trên test set. Cao = utility tốt.

**CosSim**: Mean cosine similarity giữa img_logits của F_ul và F_re trên retain set. Cao = giống F_re.

---

## 8. Kết quả thực nghiệm (run cuối)

### 8.1 Output console
```
Epoch 0: 13it [01:05, 5.07s/it]
[E00] loss=+16.684  UU=+5.052  MD=+0.000  UKR=+0.300  MKR=+1.855  RE=+20.810  | CosSim=0.6220
[E01] loss=+16.292  UU=+4.845  MD=+0.000  UKR=+0.325  MKR=+1.458  RE=+20.785  | CosSim=0.6217
[E02] loss=+15.891  UU=+4.268  MD=+0.000  UKR=+0.517  MKR=+1.399  RE=+20.814  | CosSim=0.6217
[E03] loss=+15.786  UU=+3.585  MD=+0.000  UKR=+1.050  MKR=+1.380  RE=+20.923  | CosSim=0.6217
⏹ Early stop at epoch 3 (best CosSim=0.6220)
```

### 8.2 Bảng kết quả cuối cùng (đã fix metrics bug)

```
────────────────────────────────────────────────────────────
 METRIC                    | VALUE   | PAPER TARGET 
────────────────────────────────────────────────────────────
  MIA           (↓)        | 0.552  | 0.571 (3%) / 0.615 (6%) / 0.810 (10%)
  Forget AUC    (↓)        | 0.829  | 0.735 (3%) / 0.654 (6%) / 0.656 (10%)
  Forget Mac-F1 (↓)        | 0.597  | 0.393 (3%) / 0.328 (6%) / 0.313 (10%)
  Test AUC      (↑)        | 0.674  | 0.625 (3%) / 0.599 (6%) / 0.565 (10%)
  Test Mac-F1   (↑)        | 0.387  | 0.250 (3%) / 0.270 (6%) / 0.252 (10%)
  1 - CosSim    (↓)        | 0.371  |  ~0.4-0.5
────────────────────────────────────────────────────────────
  Time (hours)             | 0.094  | Forget-MI paper: ~5h
  GPU peak (GB)            | 11.76   |
  Trainable params         | 442,368 (0.389%)
────────────────────────────────────────────────────────────
```

### 8.3 Phân tích nhanh

| Khía cạnh | Đánh giá |
|---|---|
| ✅ Tốc độ | **53× nhanh hơn paper** (0.094h vs 5h) |
| ✅ Tham số | **256× ít hơn** (442K vs 113.7M) |
| ✅ MIA | Tương đương paper, gần F_re hơn |
| ✅ Utility (Test AUC/F1) | **Tốt hơn paper rõ rệt** (+0.049 AUC, +0.137 F1) |
| ✅ Distance vs retrained | Trong khoảng tốt |
| ❌ Forget AUC | **Cao hơn paper +0.094** — chưa "quên" đủ mạnh |
| ❌ Forget F1 | **Cao hơn paper +0.204** — vẫn classify đúng forget |

**Diễn giải**: Mô hình LoKU đang ở chế độ **"soft unlearn"** — giữ utility rất tốt nhưng forget yếu hơn. MIA thấp là do cả forget và test đều được classify tốt → loss distribution overlap → attacker không phân biệt được.

---

## 9. Hạn chế và hướng cải thiện

### 9.1 Hạn chế đã xác định

1. **Forget strength yếu**: Forget AUC = 0.829 (paper 0.735) — model vẫn nhận ra forget data
2. **L_MD = 0 trong mọi epoch**: `forget_margin=8` quá nhỏ so với joint distance thực tế → mất 1 thành phần loss
3. **CosSim không cải thiện**: 0.6220 → 0.6217 — anchor về F_re yếu (η=0.5 nhỏ so với L_RE=20.8)
4. **Early stop quá sớm**: epoch 3/8 — chưa khai thác hết potential
5. **LoRA rank=8**: Có thể chật cho task forget mạnh

### 9.2 Phương án cải thiện

**Phương án 1 — Tune hyperparameter** (rẻ, nhanh):
```yaml
forget_margin:    20.0     # kích hoạt L_MD
eta_re_anchor:    0.0      # bỏ kéo về retrained
beta:             2.0      # UU mạnh hơn
theta:            2.0      # MD mạnh hơn
unlearn_epochs:   15
early_stop_patience: 5
```

**Phương án 2 — Thêm NegGrad term**:
```
L_neg = -CE(F_ul(I_f, T_f), y_f)    # gradient ascent trên forget
L_total += ζ · clamp(L_neg, min=-5)
```

**Phương án 3 — Tăng LoRA capacity**:
```yaml
lora_r:           16        # 8 → 16
lora_alpha:       32        # giữ tỉ lệ
lora_target_modules: ["query", "key", "value", "intermediate.dense", "output.dense"]
```

**Phương án 4 — Two-stage training**: Stage 1 forget aggressive, Stage 2 restore utility

### 9.3 Future work (cho luận văn)

- Mở rộng experiment với forget percentage 6% và 10% (paper test cả 3 mức)
- Multi-seed (3-5 seeds) để có mean ± std
- Ablation: LoRA only vs FIM only vs LoRA+FIM
- So sánh với các baseline khác: NegGrad+, SCRUB, CF-k, EU-k, MultiDelete
- Áp dụng lên foundation model (như CLIP-style medical models)

---

## 10. Mô tả luận điểm cho luận văn

> **Đóng góp chính**: Chúng tôi đề xuất **Forget-MI-LoKU**, một biến thể của Forget-MI sử dụng **LoRA adapter** và **Fisher Information Matrix** để giảm chi phí tính toán của machine unlearning trên dữ liệu y tế đa phương thức. Trên MIMIC-CXR với forget 3%, phương pháp đạt **MIA = 0.552** (tương đương baseline 0.571), **bảo tồn utility tốt hơn** (Test AUC +0.049, F1 +0.137) với **53× tốc độ nhanh hơn** và **256× ít tham số trainable**. Trade-off duy nhất là forget strength yếu hơn baseline (Forget AUC +0.094), phản ánh giới hạn capacity của PEFT trong unlearning — một hướng nghiên cứu mở cho future work.

---

## 11. File cấu hình tái lập

### 11.1 Phần mềm
```
python==3.12
torch==2.x (CUDA)
transformers==4.38.0
peft==0.10.0
accelerate==0.27.0
sklearn, scipy, numpy, pandas
wandb (optional)
pydicom, scikit-image
```

### 11.2 Hardware
- GPU: NVIDIA T4 (Colab) — 16 GB VRAM
- Peak GPU usage: 11.76 GB
- Wall-clock: 0.094h ≈ 5.6 phút

### 11.3 Code repository
- Local: `d:/Hoang Nhu/UNIVERSITY/4th YEAR/Khoa luan tot nghiep/Code/Forget-MI-main/Forget-MI-main/`
- GitHub: https://github.com/nhnhu146/Forget-MI-LoKU
- Main entry: `training/forgetmi_loku.py`
- Config: `config.yaml`

---

## 12. Lịch sử các bug đã sửa (cho phụ lục luận văn)

| # | Bug | Triệu chứng | Fix |
|---|---|---|---|
| 1 | Loss divergence | Loss → −65 sau 12 epoch | F.relu hinge thay vì cộng âm |
| 2 | Hinge reversed | L_UKR/MKR mất gradient sau margin | Đảo `min()` thành `relu(L − margin)` |
| 3 | Early stop fail | Loss âm dần → không bao giờ stop | Theo dõi CosSim ↑ thay vì loss ↓ |
| 4 | SVD aggressive init | Subtract 2× SVD ra khỏi base → phá model | Soft init scale=0.05, không subtract base |
| 5 | FIM proxy yếu | Dùng sum(logits) | Dùng cross-entropy với label thật |
| 6 | LoRA fc1 collision | Target "fc1" đụng img_model.fc1 | Đổi thành ["query","key","value"] |
| 7 | dtype mismatch | fp16 model + fp32 input crash conv2d | Helper `safe_forward(model, inputs)` |
| 8 | AUC NaN, F1 zero-biased | Dùng `label_raw [N,1]` thay vì one-hot | Dùng `label_onehot [N,4]` từ batch |
