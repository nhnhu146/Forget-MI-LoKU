# Exp 006 — exp06_uniform_prior

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Git commit** | `81bd05c (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.222h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

BREAKTHROUGH approach: thay NegGrad (ep CE forget cao -> MIA tang) bang Uniform Prior (ep softmax forget ve uniform 25%/25%/25%/25%). Ly thuyet: Forget F1 thap vi predictions random, MIA THAP vi forget loss = -log(0.25) ~ 1.39 (gan test loss). Tat early stop tren CosSim (khong hop voi unlearning). Doi: MIA <= 0.55, Forget F1 < 0.45, Test giu nguyen.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `unlearn_epochs` | `10                                             # [EXP 005] was 15 (vừa đủ, tránh over-train)` | `10                                             # 10 epochs đủ cho uniform prior` |
| `early_stop_patience` | `3                                              # [EXP 005] was 6 (stop sớm khi CosSim giảm)` | `999                                            # [EXP 006] EFFECTIVELY OFF — CosSim không hợp với NegGrad/Uniform` |
| `alpha` | `1.0                                            # [EXP 005] UKR mạnh để giữ retain embedding` | `1.0                                            # UKR mạnh giữ retain embedding` |
| `beta` | `0.0                                            # [EXP 005] BỎ HẲN UU (xáo trộn BERT không cần thiết)` | `0.0                                            # UU off` |
| `theta` | `0.0                                            # MD = 0, bỏ` | `0.0                                            # MD off` |
| `gamma` | `1.0                                            # [EXP 005] MKR mạnh để giữ joint retain` | `1.0                                            # MKR mạnh giữ joint retain` |
| `kappa_cls_retain` | `3.0                                            # [EXP 005] was 2.0 (anchor mạnh hơn)` | `3.0                                            # CE retain (mạnh — anchor classifier giữ retain đúng)` |
| `kappa_cls_forget` | `1.0                                            # [EXP 005] was 3.0 (giảm mạnh — tránh over-unlearn)` | `0.0                                            # [EXP 006] TẮT NegGrad (gây MIA tăng)` |
| `cls_forget_clamp` | `4.0                                            # [EXP 005] was 10 (cap thấp hơn → tránh đẩy quá)` | `4.0                                            # (không dùng khi uniform_prior_weight > 0)` |
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
  # [EXP 006] Uniform Prior unlearning — kỹ thuật mới (KL toward uniform thay vì NegGrad)
  # Ý tưởng: thay vì đẩy CE forget lên cao (gây MIA tăng), đẩy softmax forget về uniform 25%/25%/25%/25%
  # → Forget F1 thấp (random predictions) + forget loss ≈ test loss (MIA thấp)
  alpha:
    value: 1.0                                            # UKR mạnh giữ retain embedding
  beta:
    value: 0.0                                            # UU off
  theta:
    value: 0.0                                            # MD off
  gamma:
    value: 1.0                                            # MKR mạnh giữ joint retain
  eta_re_anchor:
    value: 0.0                                            # tắt re-anchor

  # ===== Classification losses — Uniform Prior mode (Exp 06) =====
  kappa_cls_retain:
    value: 3.0                                            # CE retain (mạnh — anchor classifier giữ retain đúng)
  kappa_cls_forget:
    value: 0.0                                            # [EXP 006] TẮT NegGrad (gây MIA tăng)
  cls_forget_clamp:
    value: 4.0                                            # (không dùng khi uniform_prior_weight > 0)
  uniform_prior_weight:
    value: 2.0                                            # [EXP 006] NEW — push forget softmax về uniform
  unfreeze_classifier_heads:
    value: true

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp06_uniform_prior
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+4.158  UU=+17.075  MD=+0.000  UKR=+0.295  MKR=+1.854  RE=+0.000  CLS_ret=+1.231  CLS_frg=-1.684  | CosSim(ul,re)=0.6113
[E01] loss=+3.547  UU=+17.088  MD=+0.000  UKR=+0.293  MKR=+1.452  RE=+0.000  CLS_ret=+1.180  CLS_frg=-1.739  | CosSim(ul,re)=0.5926
[E02] loss=+3.345  UU=+17.087  MD=+0.000  UKR=+0.290  MKR=+1.386  RE=+0.000  CLS_ret=+1.149  CLS_frg=-1.778  | CosSim(ul,re)=0.5737
[E03] loss=+3.204  UU=+17.087  MD=+0.000  UKR=+0.285  MKR=+1.338  RE=+0.000  CLS_ret=+1.131  CLS_frg=-1.811  | CosSim(ul,re)=0.5587
[E04] loss=+3.094  UU=+17.087  MD=+0.000  UKR=+0.284  MKR=+1.298  RE=+0.000  CLS_ret=+1.118  CLS_frg=-1.841  | CosSim(ul,re)=0.5494
[E05] loss=+2.994  UU=+17.091  MD=+0.000  UKR=+0.284  MKR=+1.260  RE=+0.000  CLS_ret=+1.106  CLS_frg=-1.867  | CosSim(ul,re)=0.5451
[E06] loss=+2.896  UU=+17.093  MD=+0.000  UKR=+0.283  MKR=+1.221  RE=+0.000  CLS_ret=+1.094  CLS_frg=-1.892  | CosSim(ul,re)=0.5441
[E07] loss=+2.799  UU=+17.095  MD=+0.000  UKR=+0.283  MKR=+1.182  RE=+0.000  CLS_ret=+1.083  CLS_frg=-1.916  | CosSim(ul,re)=0.5450
[E08] loss=+2.702  UU=+17.093  MD=+0.000  UKR=+0.283  MKR=+1.143  RE=+0.000  CLS_ret=+1.072  CLS_frg=-1.941  | CosSim(ul,re)=0.5473
[E09] loss=+2.606  UU=+17.091  MD=+0.000  UKR=+0.283  MKR=+1.103  RE=+0.000  CLS_ret=+1.062  CLS_frg=-1.967  | CosSim(ul,re)=0.5505
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.672** | +0.101 ❌ |
| Forget AUC ↓ | **0.839** | +0.104 ❌ |
| Forget F1 ↓ | **0.654** | +0.261 ❌ |
| Test AUC ↑ | **0.669** | +0.044 ✅ |
| Test F1 ↑ | **0.387** | +0.137 ✅ |
| 1−CosSim ↓ | **0.463** | +0.013 ❌ |
| Time (h) | **0.222** | -4.778 ✅ |
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
