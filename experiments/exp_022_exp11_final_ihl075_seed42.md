# Exp 022 — exp11_final_ihl075_seed42

| Field | Value |
|---|---|
| **Date** | 2026-06-03 |
| **Git commit** | `d31313f (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.221h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

exp11 FINAL (honest, no F_re): distill_teacher=og, early_stop=val, distill_forget=0, IHL=0.75 (sweet-spot tu sweep exp11d), image-FILA scale0.3. Multi-seed de bao cao mean+-std. Da kiem chung ket qua bat bien voi tieu chi early-stop -> khong ky xao.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
_(không thay đổi config so với exp trước)_
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
    value: 4                                              # số epoch chờ trước khi dừng
  early_stop_metric:
    value: "val"                                          # [EXP 11] HONEST: dừng theo loss VALIDATION (không đụng F_re).
                                                          # "cossim" = theo độ giống F_re (gold) — CHỈ cho biến thể cho phép
                                                          # F_re (exp10c). "none" = chạy đủ epoch. exp11 phải để "val".
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
    value: 0.75                                           # [EXP 11c] sweet-spot: forget_ce ≈ test_ce (không over-forget),
                                                          # Forget-AUC ≈ paper, MIA lành mạnh. (IHL=1.5 over-forget; 0.5 hơi nhẹ)

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp11_final_ihl075_seed42
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+8.258  UKR=+1.262  MKR=+2.507  CLS_ret=+1.419  DSL_ret=+0.5865  DSL_frg=+0.0000  IHL_frg=+1.0271  | val_CE=1.7318
[E01] loss=+6.917  UKR=+0.420  MKR=+2.149  CLS_ret=+1.319  DSL_ret=+0.6222  DSL_frg=+0.0000  IHL_frg=+1.0362  | val_CE=0.9381
[E02] loss=+5.620  UKR=+0.361  MKR=+1.865  CLS_ret=+1.076  DSL_ret=+0.3147  DSL_frg=+0.0000  IHL_frg=+1.0258  | val_CE=0.9157
[E03] loss=+5.182  UKR=+0.331  MKR=+1.771  CLS_ret=+0.964  DSL_ret=+0.2659  DSL_frg=+0.0000  IHL_frg=+1.0031  | val_CE=1.0286
[E04] loss=+4.961  UKR=+0.342  MKR=+1.718  CLS_ret=+0.896  DSL_ret=+0.2502  DSL_frg=+0.0000  IHL_frg=+0.9779  | val_CE=1.1005
[E05] loss=+4.875  UKR=+0.360  MKR=+1.686  CLS_ret=+0.850  DSL_ret=+0.2736  DSL_frg=+0.0000  IHL_frg=+0.9576  | val_CE=0.9383
[E06] loss=+4.684  UKR=+0.306  MKR=+1.624  CLS_ret=+0.883  DSL_ret=+0.2009  DSL_frg=+0.0000  IHL_frg=+0.9154  | val_CE=0.8951
[E07] loss=+4.602  UKR=+0.299  MKR=+1.585  CLS_ret=+0.862  DSL_ret=+0.2201  DSL_frg=+0.0000  IHL_frg=+0.8848  | val_CE=1.0120
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA_persample ↓ | **0.453** | (LoKU per-sample SVM) |
| MIA_paper ↓ | **0.429** | -0.142 ✅ |
| Forget AUC ↓ | **0.734** | -0.001 ✅ |
| Forget F1 ↓ | **0.379** | -0.015 ✅ |
| Test AUC ↑ | **0.677** | +0.052 ✅ |
| Test F1 ↑ | **0.363** | +0.113 ✅ |
| 1−CosSim ↓ | **0.329** | -0.121 ✅ |
| Time (h) | **0.221** | -4.779 ✅ |
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
