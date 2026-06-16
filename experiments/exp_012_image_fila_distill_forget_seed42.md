# Exp 012 — exp10c_image_fila_distill_forget_seed42

| Field | Value |
|---|---|
| **Date** | 2026-06-02 |
| **Git commit** | `63e218f (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.182h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Multi-seed cho exp10c (image-FILA scale0.3 + distill_forget=1.0) de bao cao mean+-std. Luu y MIA_paper tho (~1/7) nen can nhieu seed.

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

  # ===== Distillation (retain ONLY) =====
  distill_retain_weight:
    value: 1.5                                            # KL teacher trên RETAIN - giữ utility
  distill_forget_weight:
    value: 1.0                                            # [EXP 10c] NHẸ — ép logits forget khớp teacher retrained
                                                          # (MIA=0). exp10/10b mở conv ảnh -> fit retain tốt -> forget
                                                          # loss thấp -> MIA tăng 0.61. distill_forget kéo forget về
                                                          # phân bố "chưa từng thấy" của F_re. exp07 dùng 4.0 (quá mạnh,
                                                          # chưa có image-PEFT) nên hỏng; giờ 1.0 nhẹ + đúng nhánh ảnh.
  distill_temperature:
    value: 2.0

  # ===== Inverted Hinge Loss (IHL) — từ paper LoKU =====
  # L_IHL = 1 + p(true_forget_label) - max_{v ≠ true}(p(v|x))
  # Bounded [0, 2], self-stopping khi unlearning đạt. An toàn hơn NegGrad (unbounded).
  # [EXP 10] TẮT (0.0) để CÔ LẬP hiệu ứng image-FILA: exp10 = exp08 + image pathway,
  # không trộn thêm forget-loss. IHL sẽ bật lại ở exp11 (kèm orthogonal projection).
  ihl_forget_weight:
    value: 0.0                                            # [EXP 10] off — isolate image-FILA

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp10c_image_fila_distill_forget_seed42
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+15.653  UKR=+0.949  MKR=+2.288  CLS_ret=+1.506  DSL_ret=+3.6230  DSL_frg=+3.9691  IHL_frg=+0.0000  | CosSim(ul,re)=0.6749
[E01] loss=+14.284  UKR=+0.468  MKR=+2.015  CLS_ret=+1.325  DSL_ret=+3.5108  DSL_frg=+3.8849  IHL_frg=+0.0000  | CosSim(ul,re)=0.6526
[E02] loss=+12.619  UKR=+0.430  MKR=+1.896  CLS_ret=+1.123  DSL_ret=+3.0671  DSL_frg=+3.4452  IHL_frg=+0.0000  | CosSim(ul,re)=0.7103
[E03] loss=+12.154  UKR=+0.456  MKR=+1.888  CLS_ret=+1.029  DSL_ret=+2.9722  DSL_frg=+3.2944  IHL_frg=+0.0000  | CosSim(ul,re)=0.6977
[E04] loss=+11.886  UKR=+0.529  MKR=+1.843  CLS_ret=+0.982  DSL_ret=+2.8779  DSL_frg=+3.2336  IHL_frg=+0.0000  | CosSim(ul,re)=0.7181
[E05] loss=+11.531  UKR=+0.391  MKR=+1.816  CLS_ret=+0.956  DSL_ret=+2.8291  DSL_frg=+3.1693  IHL_frg=+0.0000  | CosSim(ul,re)=0.7023
[E06] loss=+11.172  UKR=+0.387  MKR=+1.781  CLS_ret=+0.879  DSL_ret=+2.7723  DSL_frg=+3.0863  IHL_frg=+0.0000  | CosSim(ul,re)=0.7135
[E07] loss=+10.865  UKR=+0.403  MKR=+1.760  CLS_ret=+0.820  DSL_ret=+2.7168  DSL_frg=+2.9869  IHL_frg=+0.0000  | CosSim(ul,re)=0.7107
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA_persample ↓ | **0.493** | (LoKU per-sample SVM) |
| MIA_paper ↓ | **0.571** | +0.000 ❌ |
| Forget AUC ↓ | **0.718** | -0.017 ✅ |
| Forget F1 ↓ | **0.271** | -0.122 ✅ |
| Test AUC ↑ | **0.689** | +0.064 ✅ |
| Test F1 ↑ | **0.330** | +0.080 ✅ |
| 1−CosSim ↓ | **0.297** | -0.153 ✅ |
| Time (h) | **0.182** | -4.818 ✅ |
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
