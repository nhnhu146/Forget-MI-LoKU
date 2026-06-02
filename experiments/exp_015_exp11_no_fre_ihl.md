# Exp 015 — exp11_no_fre_ihl

| Field | Value |
|---|---|
| **Date** | 2026-06-02 |
| **Git commit** | `f27a174 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.179h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

exp11: BO phu thuoc F_re khi train. retain-distill teacher = F_og (model goc, luon co san), TAT distill_forget (can F_re), bat IHL=1.5 cho forget (bounded, khong can teacher). Giu image-FILA scale0.3. F_re chi con dung de DANH GIA (CosSim) - hop le. Muc tieu: chung minh phuong phap van tot ma KHONG hoc tu gold model. Doi: metric gan exp10c (co the kem hon chut), nhung HOP LE hon han.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `distill_retain_weight` | `1.5                                            # KL teacher trên RETAIN - giữ utility` | `1.5                                            # KL(student || F_og) trên RETAIN - giữ utility gốc` |
| `distill_forget_weight` | `1.0                                            # [EXP 10c] NHẸ — ép logits forget khớp teacher retrained` | `0.0                                            # [EXP 11] TẮT - cần F_re; thay bằng IHL` |
| `ihl_forget_weight` | `0.0                                            # [EXP 10] off — isolate image-FILA` | `1.5                                            # [EXP 11] on — forget push không cần F_re` |
<!-- /AUTO -->

### Code changes (điền thủ công nếu có sửa .py)
- _(file:line) Mô tả_

---

## 3. Full configuration snapshot

<details>
<summary>config.yaml</summary>

```yaml
program: training/forgetmi_loku.py
method: random
parameters:
  # ===== WandB =====
  project:
    value: "forget_mi_loku"
  entity:
    value: "unlearning"

  # ===== Data paths =====
  base_model_path:
    value: "./forgetme/training_original_model"
  retrained_model_path:
    value: "./model_retrained_3per/"
  bert_pretrained_dir:
    value: "./forgetme/training_original_model/"
  text_data_dir:
    value: "./data/metadata"
  img_data_dir:
    value: "./data/img_data"
  data_split_path:
    value: "./data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"
  synonyms_dir:
    value: "./data_splits/Synonyms.csv"
  forget_set_path:
    value: "./data_splits/forget_set_3per.csv"          # 3% — start here for fast sanity
  output_dir:
    value: "./unlearning_output/"

  # ===== Dataset prep =====
  max_seq_length:
    value: 320
  text_noise_level:
    value: 0.5
  output_channel_encoding:
    value: "multiclass"
  reprocess_input_data:
    value: false
  validation_ratio:
    value: 0.1
  random_point_ratio:
    value: 0.1
  random_seed:
    value: 42
  id:
    value: "loku_3per"

  # ===== Training =====
  do_train:
    value: true
  do_eval:
    value: false
  overwrite_output_dir:
    value: true
  use_noise:
    value: false                                          # false → use bounded forget-push
  learning_rate:
    value: 5.0e-4                                         # LoRA tolerates higher lr than full FT
  weight_decay:
    value: 0.01
  unlearn_epochs:
    value: 8                                              # [EXP 08] ít hơn — subtraction làm việc tại init
  early_stop_patience:
    value: 4                                              # [EXP 08] CosSim sẽ TĂNG (gần F_re hơn) → stop khi plateau
  grad_clip:
    value: 1.0
  warmup_proportion:
    value: 0.1
  scheduler:
    value: "ReduceLROnPlateau"
  unlearn_batch_size:
    value: 16
  eval_batch_size:
    value: 16
  eval_max_retain:
    value: 512                                            # cap retain forwards in MIA+CosSim (0 = full/old behavior)
  mia_paper_batch_size:
    value: 32                                             # nhóm loss theo batch cho MIA_paper (giống eval_unlearning.py gốc)
  num_cpu_workers:
    value: 2

  # ===== Noise (only used if use_noise=true) =====
  noise_mean:
    value: 0
  noise_std:
    value: 0.1

  # ===== Loss weights (UR=alpha, UU=beta, MD=theta, MR=gamma, RE-anchor=eta) =====
  # "Unimodal" setting (best for 3% in paper)
  # [EXP 008] TRUE LoKU (FILA subtraction) + Retain-only training
  # Sau 7 exps, đọc lại LoKU paper kỹ → tìm ra cái thiếu: subtraction tại init!
  # FILA paper: W* = W - B*A* (TRỪ forget direction), LoRA giữ direction.
  # Training CHỈ giữ retain → khi LoRA giảm về 0, model = W - sub → forget bị xóa thật.
  # Exp 07 phát hiện distill_forget gây giữ forget → BỎ luôn. Chỉ giữ retain signals.
  alpha:
    value: 1.0                                            # UKR mạnh - giữ retain embedding
  beta:
    value: 0.0                                            # UU off - subtraction xử lý forget
  theta:
    value: 0.0                                            # MD off
  gamma:
    value: 1.0                                            # MKR mạnh - giữ joint retain
  eta_re_anchor:
    value: 0.0                                            # tắt re-anchor

  # ===== Classification losses (retain only) =====
  kappa_cls_retain:
    value: 2.0                                            # CE retain MẠNH - anchor classifier
  kappa_cls_forget:
    value: 0.0                                            # KHÔNG đụng vào forget data
  cls_forget_clamp:
    value: 4.0
  uniform_prior_weight:
    value: 0.0
  unfreeze_classifier_heads:
    value: true

  # ===== Distillation =====
  # [EXP 11] BỎ phụ thuộc F_re khi TRAIN: retain-distill dùng teacher = F_og (model gốc,
  # luôn có sẵn). F_re chỉ còn dùng để ĐÁNH GIÁ (CosSim) — hợp lệ. Forget không distill nữa
  # (F_og biết forget) → forget xử lý bằng IHL bên dưới.
  distill_teacher:
    value: "og"                                           # "og" = F_og (exp11) | "re" = F_re (exp10c)
  distill_retain_weight:
    value: 1.5                                            # KL(student || F_og) trên RETAIN - giữ utility gốc
  distill_forget_weight:
    value: 0.0                                            # [EXP 11] TẮT - cần F_re; thay bằng IHL
  distill_temperature:
    value: 2.0

  # ===== Inverted Hinge Loss (IHL) — từ paper LoKU =====
  # L_IHL = 1 + p(true_forget_label) - max_{v ≠ true}(p(v|x))
  # Bounded [0, 2], self-stopping khi unlearning đạt. An toàn hơn NegGrad (unbounded).
  # [EXP 11] BẬT IHL làm tín hiệu forget (bounded, KHÔNG cần teacher F_re) thay cho
  # distill_forget. Đẩy forget loss lên kiểu self-stopping → ghìm MIA mà không lệ thuộc gold.
  ihl_forget_weight:
    value: 1.5                                            # [EXP 11] on — forget push không cần F_re

  # ===== Forget-margin (bounded loss) =====
  forget_margin:
    value: 20.0                                           # [EXP 002] tăng margin để kích hoạt L_MD

  # ===== LoRA =====
  lora_r:
    value: 8
  lora_alpha:
    value: 16
  lora_dropout:
    value: 0.05
  lora_target_modules:
    value: ["query", "key", "value"]                      # BERT (text) attention

  # ===== [EXP 10 NEW] Modality-aware PEFT on the IMAGE pathway =====
  # Chẩn đoán: MIA chỉ đọc img_logits (eval_unlearning.py), mà LoRA/FILA của exp08 CHỈ
  # chạm BERT (text) → image forget behavior không đổi → Forget-AUC kẹt ở 0.833.
  # exp10: thêm FILA subtraction lên Conv2d ở các stage cuối của image encoder.
  lora_image_last_k_blocks:
    value: 1                                              # 1 = chỉ img_model.layer7; 2 = +layer6; 0 = tắt
  lora_image_include_fc1:
    value: false                                          # true → FILA cả image classifier head (rank-clamped)
  loku_image_subtract_scale:
    value: 0.3                                            # [EXP 10b] 0.5 đẩy forget tốt nhưng MIA tăng 0.562→0.612;
                                                          # hạ 0.3 để giữ Forget-AUC thấp mà MIA bớt tăng. Ablate 0.3/0.5/0.7

  # ===== Fisher Information =====
  fisher_max_samples:
    value: 256
  fisher_batch_size:
    value: 16
  loku_init_scale:
    value: 0.05                                           # only used when loku_subtract_scale = 0
  loku_subtract_scale:
    value: 1.0                                            # [EXP 08 KEY] TRUE LoKU FILA: W* = W - B*A* * scale
                                                          # 0 = soft init (legacy), >0 = real subtraction

  # ===== Model architecture flags =====
  bert_pool_last_hidden:
    value: false
  bert_pool_use_img:
    value: false
  bert_pool_img_lowerlevel:
    value: false

  # ===== Misc (kept for compatibility with shared modules) =====
  use_text_data_dir:
    value: false
  use_data_split_path:
    value: false
  training_mode:
    value: "supervised"
  semisupervised_training_data:
    value: "allCXR"
  use_masked_txt:
    value: false
  use_all_data:
    value: false
  use_pretrained_checkpoint:
    value: false
  print_predictions:
    value: false
  print_embeddings:
    value: false
  logging_steps:
    value: 50
  save_epochs:
    value: 1
  ihl_margin:
    value: 0.5

command:
  - ${env:PYTHON}
  - training/forgetmi_loku.py
  - --config=config.yaml

```

</details>

---

## 4. Execution

```bash
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp11_no_fre_ihl
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+8.212  UKR=+0.851  MKR=+2.241  CLS_ret=+1.368  DSL_ret=+0.5495  DSL_frg=+0.0000  IHL_frg=+1.0405  | CosSim(ul,re)=0.6675
[E01] loss=+7.208  UKR=+0.464  MKR=+2.012  CLS_ret=+1.200  DSL_ret=+0.5043  DSL_frg=+0.0000  IHL_frg=+1.0499  | CosSim(ul,re)=0.6226
[E02] loss=+6.296  UKR=+0.397  MKR=+1.846  CLS_ret=+1.042  DSL_ret=+0.3083  DSL_frg=+0.0000  IHL_frg=+1.0045  | CosSim(ul,re)=0.6648
[E03] loss=+5.825  UKR=+0.351  MKR=+1.761  CLS_ret=+0.956  DSL_ret=+0.2445  DSL_frg=+0.0000  IHL_frg=+0.9557  | CosSim(ul,re)=0.6792
[E04] loss=+5.587  UKR=+0.360  MKR=+1.711  CLS_ret=+0.880  DSL_ret=+0.2598  DSL_frg=+0.0000  IHL_frg=+0.9112  | CosSim(ul,re)=0.6777
[E05] loss=+5.514  UKR=+0.339  MKR=+1.700  CLS_ret=+0.851  DSL_ret=+0.2976  DSL_frg=+0.0000  IHL_frg=+0.8849  | CosSim(ul,re)=0.6503
[E06] loss=+5.335  UKR=+0.318  MKR=+1.645  CLS_ret=+0.880  DSL_ret=+0.2347  DSL_frg=+0.0000  IHL_frg=+0.8397  | CosSim(ul,re)=0.6472
[E07] loss=+5.180  UKR=+0.319  MKR=+1.601  CLS_ret=+0.860  DSL_ret=+0.2201  DSL_frg=+0.0000  IHL_frg=+0.8062  | CosSim(ul,re)=0.6568
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA_persample ↓ | **0.333** | (LoKU per-sample SVM) |
| MIA_paper ↓ | **0.143** | -0.428 ✅ |
| Forget AUC ↓ | **0.687** | -0.048 ✅ |
| Forget F1 ↓ | **0.308** | -0.085 ✅ |
| Test AUC ↑ | **0.673** | +0.048 ✅ |
| Test F1 ↑ | **0.380** | +0.130 ✅ |
| 1−CosSim ↓ | **0.341** | -0.109 ✅ |
| Time (h) | **0.179** | -4.821 ✅ |
| GPU peak (GB) | **11.78** | — |
| Trainable params | **0.451%** | (vs 100% paper) |

<!-- /AUTO -->

---

## 6. Observations (✍️ điền thủ công)

_(Quan sát gì bất thường? Khớp/khác predict?)_

-

---

## 7. Conclusion (✍️ điền thủ công)

- **Hypothesis verdict**: ✅ Confirmed / ❌ Rejected / 🤷 Inconclusive
- **Keep changes**: Y / N
- **Why**:

---

## 8. Next steps (✍️ điền thủ công)

-
