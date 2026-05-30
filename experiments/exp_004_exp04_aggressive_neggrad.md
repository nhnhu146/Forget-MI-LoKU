# Exp 004 — exp04_aggressive_neggrad

| Field | Value |
|---|---|
| **Date** | 2026-05-30 |
| **Git commit** | `ce36869 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.271h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Exp 03 cho thay UU=15 de bep CLS_frg (chi -0.7 sau weight). Exp 04: giam beta 1.0->0.1 (UU bot manh), tang kappa_cls_forget 0.5->3.0 (gradient ascent gap 6 lan), tang epoch 8->15. Doi: Forget AUC giam ve ~0.65-0.75, MIA giam ve ~0.45-0.55, Test AUC tuc co the giam ve 0.55-0.65 (trade-off chap nhan).

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `unlearn_epochs` | `8                                              # LoRA usually converges <10 epochs` | `15                                             # [EXP 004] was 8 (NegGrad cần nhiều epoch hơn)` |
| `early_stop_patience` | `3` | `6                                              # [EXP 004] was 3 (đỡ stop quá sớm)` |
| `alpha` | `1.0` | `0.3                                            # was 1.0 (UKR — giảm)` |
| `beta` | `1.0` | `0.1                                            # was 1.0 (UU — giảm mạnh, hết đè bẹp)` |
| `theta` | `0.5` | `0.0                                            # was 0.5 (MD = 0 luôn, bỏ luôn)` |
| `gamma` | `0.5` | `0.5                                            # was 0.5 (MKR giữ nguyên)` |
| `kappa_cls_retain` | `1.0                                            # CE on retain (keep classification correct)` | `2.0                                            # [EXP 004] was 1.0 (anchor retain mạnh hơn)` |
| `kappa_cls_forget` | `0.5                                            # neg CE on forget (push classification wrong)` | `3.0                                            # [EXP 004] was 0.5 (gradient ascent mạnh gấp 6!)` |
| `cls_forget_clamp` | `5.0                                            # cap forget CE to avoid divergence` | `10.0                                           # [EXP 004] was 5.0 (cho CE forget lên cao hơn)` |
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
    value: 15                                             # [EXP 004] was 8 (NegGrad cần nhiều epoch hơn)
  early_stop_patience:
    value: 6                                              # [EXP 004] was 3 (đỡ stop quá sớm)
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
  # [EXP 004] Giảm mạnh embedding losses (UU=15+ trước đây đè bẹp CLS_frg)
  alpha:
    value: 0.3                                            # was 1.0 (UKR — giảm)
  beta:
    value: 0.1                                            # was 1.0 (UU — giảm mạnh, hết đè bẹp)
  theta:
    value: 0.0                                            # was 0.5 (MD = 0 luôn, bỏ luôn)
  gamma:
    value: 0.5                                            # was 0.5 (MKR giữ nguyên)
  eta_re_anchor:
    value: 0.0                                            # [EXP 002] tắt re-anchor

  # ===== Classification losses (KEY của Exp 04) =====
  # Tăng kappa_cls_forget gấp 6 lần để NegGrad thật sự chiếm lĩnh gradient
  kappa_cls_retain:
    value: 2.0                                            # [EXP 004] was 1.0 (anchor retain mạnh hơn)
  kappa_cls_forget:
    value: 3.0                                            # [EXP 004] was 0.5 (gradient ascent mạnh gấp 6!)
  cls_forget_clamp:
    value: 10.0                                           # [EXP 004] was 5.0 (cho CE forget lên cao hơn)
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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp04_aggressive_neggrad
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+0.935  UU=+17.064  MD=+0.000  UKR=+0.297  MKR=+1.868  RE=+0.000  CLS_ret=+1.286  CLS_frg=-1.456  | CosSim(ul,re)=0.6335
[E01] loss=+0.517  UU=+17.029  MD=+0.000  UKR=+0.299  MKR=+1.464  RE=+0.000  CLS_ret=+1.341  CLS_frg=-1.563  | CosSim(ul,re)=0.6390
[E02] loss=+0.170  UU=+16.811  MD=+0.000  UKR=+0.364  MKR=+1.402  RE=+0.000  CLS_ret=+1.449  CLS_frg=-1.740  | CosSim(ul,re)=0.6422
[E03] loss=-0.523  UU=+14.115  MD=+0.000  UKR=+2.824  MKR=+1.371  RE=+0.000  CLS_ret=+1.992  CLS_frg=-2.484  | CosSim(ul,re)=0.6441
[E04] loss=-0.912  UU=+13.375  MD=+0.000  UKR=+3.743  MKR=+1.352  RE=+0.000  CLS_ret=+2.338  CLS_frg=-2.908  | CosSim(ul,re)=0.6449
[E05] loss=-1.609  UU=+13.404  MD=+0.000  UKR=+3.838  MKR=+1.342  RE=+0.000  CLS_ret=+2.732  CLS_frg=-3.412  | CosSim(ul,re)=0.6447
[E06] loss=-2.563  UU=+12.015  MD=+0.000  UKR=+5.261  MKR=+1.335  RE=+0.000  CLS_ret=+3.349  CLS_frg=-4.236  | CosSim(ul,re)=0.6442
[E07] loss=-3.711  UU=+11.179  MD=+0.000  UKR=+6.092  MKR=+1.329  RE=+0.000  CLS_ret=+3.931  CLS_frg=-5.061  | CosSim(ul,re)=0.6445
[E08] loss=-4.666  UU=+11.061  MD=+0.000  UKR=+6.022  MKR=+1.325  RE=+0.000  CLS_ret=+4.265  CLS_frg=-5.591  | CosSim(ul,re)=0.6448
[E09] loss=-5.033  UU=+11.630  MD=+0.000  UKR=+5.330  MKR=+1.321  RE=+0.000  CLS_ret=+4.328  CLS_frg=-5.704  | CosSim(ul,re)=0.6444
[E10] loss=-4.133  UU=+12.068  MD=+0.000  UKR=+5.428  MKR=+1.318  RE=+0.000  CLS_ret=+4.642  CLS_frg=-5.637  | CosSim(ul,re)=0.6441
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.687** | +0.116 ❌ |
| Forget AUC ↓ | **0.809** | +0.074 ❌ |
| Forget F1 ↓ | **0.478** | +0.085 ❌ |
| Test AUC ↑ | **0.672** | +0.047 ✅ |
| Test F1 ↑ | **0.327** | +0.077 ✅ |
| 1−CosSim ↓ | **0.338** | -0.112 ✅ |
| Time (h) | **0.271** | -4.729 ✅ |
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
