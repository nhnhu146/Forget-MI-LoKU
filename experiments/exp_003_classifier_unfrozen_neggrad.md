# Exp 003 — exp03_classifier_unfrozen_neggrad

| Field | Value |
|---|---|
| **Date** | 2026-05-30 |
| **Git commit** | `093a3fd` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.094h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

FIX BUG: unfreeze img_model.fc1 + text_model.classifier + them classification loss (CE retain, neg CE forget). Truoc do LoRA chi train BERT, eval dung img_logits nen ket qua bat bien. Doi: MIA giam, Forget AUC/F1 giam manh, Test giam nhe.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `eta_re_anchor` | `0.0                                            # [EXP 002] tắt re-anchor (was 0.5)` | `0.0                                            # [EXP 002] tắt re-anchor` |
| `forget_margin` | `20.0                                           # [EXP 002] tăng margin (was 8.0) — kích hoạt L_MD` | `20.0                                           # [EXP 002] tăng margin để kích hoạt L_MD` |
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
    value: 8                                              # LoRA usually converges <10 epochs
  early_stop_patience:
    value: 3
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
  alpha:
    value: 1.0
  beta:
    value: 1.0
  theta:
    value: 0.5
  gamma:
    value: 0.5
  eta_re_anchor:
    value: 0.0                                            # [EXP 002] tắt re-anchor

  # ===== Classification losses (NEW — for gradient signal to img/txt classifier) =====
  # Without these, LoRA on BERT does NOT affect img_logits (which eval uses)
  # → results would be invariant to all other hyperparameters
  kappa_cls_retain:
    value: 1.0                                            # CE on retain (keep classification correct)
  kappa_cls_forget:
    value: 0.5                                            # neg CE on forget (push classification wrong)
  cls_forget_clamp:
    value: 5.0                                            # cap forget CE to avoid divergence
  unfreeze_classifier_heads:
    value: true                                           # unfreeze img_model.fc1 + text_model.classifier

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp03_classifier_unfrozen_neggrad
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+18.823  UU=+17.056  MD=+0.000  UKR=+0.299  MKR=+1.854  RE=+0.000  CLS_ret=+1.253  CLS_frg=-1.423  | CosSim(ul,re)=0.6255
[E01] loss=+18.437  UU=+16.878  MD=+0.000  UKR=+0.322  MKR=+1.457  RE=+0.000  CLS_ret=+1.222  CLS_frg=-1.426  | CosSim(ul,re)=0.6229
[E02] loss=+18.014  UU=+16.308  MD=+0.000  UKR=+0.515  MKR=+1.398  RE=+0.000  CLS_ret=+1.206  CLS_frg=-1.428  | CosSim(ul,re)=0.6198
[E03] loss=+17.966  UU=+15.542  MD=+0.000  UKR=+1.255  MKR=+1.384  RE=+0.000  CLS_ret=+1.192  CLS_frg=-1.429  | CosSim(ul,re)=0.6162
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.557** | -0.014 ✅ |
| Forget AUC ↓ | **0.830** | +0.095 ❌ |
| Forget F1 ↓ | **0.597** | +0.204 ❌ |
| Test AUC ↑ | **0.669** | +0.044 ✅ |
| Test F1 ↑ | **0.386** | +0.136 ✅ |
| 1−CosSim ↓ | **0.391** | -0.059 ✅ |
| Time (h) | **0.094** | -4.906 ✅ |
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
