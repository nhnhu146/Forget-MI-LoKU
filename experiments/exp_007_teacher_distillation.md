# Exp 007 — exp07_teacher_distillation

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Git commit** | `5714dd2 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.232h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

FUNDAMENTAL pivot sau 6 exps that bai: thay vi day forget classifier sai (gay MIA tang), DAY student bat chuoc model_re (gold standard chua tung thay forget). KL divergence (student || teacher) tren CA retain (giu utility) VA forget (lay teacher behavior). Vi teacher khong biet forget -> student cung khong biet -> forget loss ~ test loss -> MIA tu nhien thap. Approach co trong SCRUB, BadT papers.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `alpha` | `1.0                                            # UKR mạnh giữ retain embedding` | `0.5                                            # [EXP 007] UKR nhẹ (distillation đã giữ retain)` |
| `gamma` | `1.0                                            # MKR mạnh giữ joint retain` | `0.5                                            # [EXP 007] MKR nhẹ (distillation đã giữ joint)` |
| `eta_re_anchor` | `0.0                                            # tắt re-anchor` | `0.0                                            # tắt re-anchor (distill thay thế)` |
| `kappa_cls_retain` | `3.0                                            # CE retain (mạnh — anchor classifier giữ retain đúng)` | `0.5                                            # CE nhẹ giữ classifier ổn định` |
| `kappa_cls_forget` | `0.0                                            # [EXP 006] TẮT NegGrad (gây MIA tăng)` | `0.0                                            # TẮT NegGrad (đã chứng minh gây MIA tăng)` |
| `cls_forget_clamp` | `4.0                                            # (không dùng khi uniform_prior_weight > 0)` | `4.0` |
| `uniform_prior_weight` | `2.0                                            # [EXP 006] NEW — push forget softmax về uniform` | `0.0                                            # TẮT uniform prior (cũng gây MIA tăng)` |
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
    value: 10                                             # 10 epochs đủ cho uniform prior
  early_stop_patience:
    value: 999                                            # [EXP 006] EFFECTIVELY OFF — CosSim không hợp với NegGrad/Uniform
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
  num_cpu_workers:
    value: 2

  # ===== Noise (only used if use_noise=true) =====
  noise_mean:
    value: 0
  noise_std:
    value: 0.1

  # ===== Loss weights (UR=alpha, UU=beta, MD=theta, MR=gamma, RE-anchor=eta) =====
  # "Unimodal" setting (best for 3% in paper)
  # [EXP 007] Teacher-Student Distillation — hướng đi HOÀN TOÀN MỚI
  # Triết lý: thay vì đẩy forget classifier sai (gây MIA tăng), DẠY student bắt chước
  # model_re (đã retrain không có forget). Vì model_re "không biết" forget → student
  # cũng "không biết" → forget loss distribution giống test → MIA THẤP tự nhiên.
  # Approach này có trong SCRUB, BadT papers — well-established.
  alpha:
    value: 0.5                                            # [EXP 007] UKR nhẹ (distillation đã giữ retain)
  beta:
    value: 0.0                                            # UU off
  theta:
    value: 0.0                                            # MD off
  gamma:
    value: 0.5                                            # [EXP 007] MKR nhẹ (distillation đã giữ joint)
  eta_re_anchor:
    value: 0.0                                            # tắt re-anchor (distill thay thế)

  # ===== Classification losses — TẮT HẾT (Exp 07 dùng distillation thay) =====
  kappa_cls_retain:
    value: 0.5                                            # CE nhẹ giữ classifier ổn định
  kappa_cls_forget:
    value: 0.0                                            # TẮT NegGrad (đã chứng minh gây MIA tăng)
  cls_forget_clamp:
    value: 4.0
  uniform_prior_weight:
    value: 0.0                                            # TẮT uniform prior (cũng gây MIA tăng)
  unfreeze_classifier_heads:
    value: true

  # ===== Teacher-Student Distillation (KEY của Exp 07) =====
  distill_retain_weight:
    value: 2.0                                            # KL(student || teacher) trên retain — keep utility
  distill_forget_weight:
    value: 4.0                                            # [EXP 07 KEY] KL trên forget — student học teacher "không biết forget"
  distill_temperature:
    value: 2.0                                            # softening temperature (chuẩn distillation)

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
    value: ["query", "key", "value"]                      # BERT attention only — avoids img_model.fc1

  # ===== Fisher Information =====
  fisher_max_samples:
    value: 256
  fisher_batch_size:
    value: 16
  loku_init_scale:
    value: 0.05                                           # small → base behavior preserved at init

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp07_teacher_distillation
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+26.053  UKR=+0.296  MKR=+1.862  CLS_ret=+1.281  DSL_ret=+3.6209  DSL_frg=+4.2730  | CosSim(ul,re)=0.6408
[E01] loss=+24.969  UKR=+0.296  MKR=+1.459  CLS_ret=+1.316  DSL_ret=+3.5399  DSL_frg=+4.0884  | CosSim(ul,re)=0.6525
[E02] loss=+24.351  UKR=+0.294  MKR=+1.394  CLS_ret=+1.347  DSL_ret=+3.4890  DSL_frg=+3.9639  | CosSim(ul,re)=0.6591
[E03] loss=+23.881  UKR=+0.291  MKR=+1.350  CLS_ret=+1.361  DSL_ret=+3.4461  DSL_frg=+3.8719  | CosSim(ul,re)=0.6627
[E04] loss=+23.468  UKR=+0.288  MKR=+1.312  CLS_ret=+1.357  DSL_ret=+3.4017  DSL_frg=+3.7964  | CosSim(ul,re)=0.6651
[E05] loss=+23.086  UKR=+0.286  MKR=+1.275  CLS_ret=+1.344  DSL_ret=+3.3569  DSL_frg=+3.7298  | CosSim(ul,re)=0.6672
[E06] loss=+22.730  UKR=+0.287  MKR=+1.238  CLS_ret=+1.328  DSL_ret=+3.3141  DSL_frg=+3.6689  | CosSim(ul,re)=0.6694
[E07] loss=+22.399  UKR=+0.290  MKR=+1.200  CLS_ret=+1.313  DSL_ret=+3.2743  DSL_frg=+3.6121  | CosSim(ul,re)=0.6717
[E08] loss=+22.095  UKR=+0.301  MKR=+1.163  CLS_ret=+1.301  DSL_ret=+3.2370  DSL_frg=+3.5596  | CosSim(ul,re)=0.6738
[E09] loss=+21.816  UKR=+0.315  MKR=+1.125  CLS_ret=+1.291  DSL_ret=+3.2051  DSL_frg=+3.5101  | CosSim(ul,re)=0.6756
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.657** | +0.086 ❌ |
| Forget AUC ↓ | **0.819** | +0.084 ❌ |
| Forget F1 ↓ | **0.497** | +0.104 ❌ |
| Test AUC ↑ | **0.687** | +0.062 ✅ |
| Test F1 ↑ | **0.324** | +0.074 ✅ |
| 1−CosSim ↓ | **0.317** | -0.133 ✅ |
| Time (h) | **0.232** | -4.768 ✅ |
| GPU peak (GB) | **11.76** | — |
| Trainable params | **0.395%** | (vs 100% paper) |

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
