# Exp 005 — exp05_calibrated_neggrad

| Field | Value |
|---|---|
| **Date** | 2026-05-30 |
| **Git commit** | `04b4da4 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.090h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Exp 04 over-unlearn: kappa_frg=3 day CE forget len 5.7, retain CE cung tang gap 3.6 lan -> MIA TANG (xau hon). Exp 05 calibrated: bo UU/MD hoan toan, kappa_frg=1.0 (giam 3x), kappa_ret=3.0 (anchor manh), clamp=4. Muc tieu: forget CE chi tang den ~2-3 (gan test CE), retain CE giu thap. Doi: MIA quay ve ~0.5-0.55, Forget F1 giu khoang 0.45-0.55, Test F1 cai thien ve 0.35+.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `unlearn_epochs` | `15                                             # [EXP 004] was 8 (NegGrad cần nhiều epoch hơn)` | `10                                             # [EXP 005] was 15 (vừa đủ, tránh over-train)` |
| `early_stop_patience` | `6                                              # [EXP 004] was 3 (đỡ stop quá sớm)` | `3                                              # [EXP 005] was 6 (stop sớm khi CosSim giảm)` |
| `alpha` | `0.3                                            # was 1.0 (UKR — giảm)` | `1.0                                            # [EXP 005] UKR mạnh để giữ retain embedding` |
| `beta` | `0.1                                            # was 1.0 (UU — giảm mạnh, hết đè bẹp)` | `0.0                                            # [EXP 005] BỎ HẲN UU (xáo trộn BERT không cần thiết)` |
| `theta` | `0.0                                            # was 0.5 (MD = 0 luôn, bỏ luôn)` | `0.0                                            # MD = 0, bỏ` |
| `gamma` | `0.5                                            # was 0.5 (MKR giữ nguyên)` | `1.0                                            # [EXP 005] MKR mạnh để giữ joint retain` |
| `eta_re_anchor` | `0.0                                            # [EXP 002] tắt re-anchor` | `0.0                                            # tắt re-anchor` |
| `kappa_cls_retain` | `2.0                                            # [EXP 004] was 1.0 (anchor retain mạnh hơn)` | `3.0                                            # [EXP 005] was 2.0 (anchor mạnh hơn)` |
| `kappa_cls_forget` | `3.0                                            # [EXP 004] was 0.5 (gradient ascent mạnh gấp 6!)` | `1.0                                            # [EXP 005] was 3.0 (giảm mạnh — tránh over-unlearn)` |
| `cls_forget_clamp` | `10.0                                           # [EXP 004] was 5.0 (cho CE forget lên cao hơn)` | `4.0                                            # [EXP 005] was 10 (cap thấp hơn → tránh đẩy quá)` |
| `unfreeze_classifier_heads` | `true                                           # unfreeze img_model.fc1 + text_model.classifier` | `true` |
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
    value: 10                                             # [EXP 005] was 15 (vừa đủ, tránh over-train)
  early_stop_patience:
    value: 3                                              # [EXP 005] was 6 (stop sớm khi CosSim giảm)
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
  # [EXP 005] Calibrated unlearning: BỎ HẲN UU/MD (gây hại trong Exp 04),
  # tăng retention strong, giảm NegGrad về moderate (mục tiêu: forget CE ≈ test CE)
  alpha:
    value: 1.0                                            # [EXP 005] UKR mạnh để giữ retain embedding
  beta:
    value: 0.0                                            # [EXP 005] BỎ HẲN UU (xáo trộn BERT không cần thiết)
  theta:
    value: 0.0                                            # MD = 0, bỏ
  gamma:
    value: 1.0                                            # [EXP 005] MKR mạnh để giữ joint retain
  eta_re_anchor:
    value: 0.0                                            # tắt re-anchor

  # ===== Classification losses (calibrated) =====
  # Exp 04: kappa_frg=3.0 quá mạnh → CE forget bay lên 5.7 → MIA tăng vọt
  # Exp 05: kappa_frg=1.0 moderate → đẩy forget CE đến ~2-3 (gần test) → MIA giảm
  kappa_cls_retain:
    value: 3.0                                            # [EXP 005] was 2.0 (anchor mạnh hơn)
  kappa_cls_forget:
    value: 1.0                                            # [EXP 005] was 3.0 (giảm mạnh — tránh over-unlearn)
  cls_forget_clamp:
    value: 4.0                                            # [EXP 005] was 10 (cap thấp hơn → tránh đẩy quá)
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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp05_calibrated_neggrad
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+4.465  UU=+17.074  MD=+0.000  UKR=+0.295  MKR=+1.853  RE=+0.000  CLS_ret=+1.243  CLS_frg=-1.414  | CosSim(ul,re)=0.6222
[E01] loss=+3.953  UU=+17.084  MD=+0.000  UKR=+0.293  MKR=+1.453  RE=+0.000  CLS_ret=+1.202  CLS_frg=-1.401  | CosSim(ul,re)=0.6153
[E02] loss=+3.814  UU=+17.084  MD=+0.000  UKR=+0.291  MKR=+1.387  RE=+0.000  CLS_ret=+1.175  CLS_frg=-1.389  | CosSim(ul,re)=0.6086
[E03] loss=+3.716  UU=+17.085  MD=+0.000  UKR=+0.287  MKR=+1.341  RE=+0.000  CLS_ret=+1.156  CLS_frg=-1.379  | CosSim(ul,re)=0.6035
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.607** | +0.036 ❌ |
| Forget AUC ↓ | **0.832** | +0.097 ❌ |
| Forget F1 ↓ | **0.631** | +0.238 ❌ |
| Test AUC ↑ | **0.667** | +0.042 ✅ |
| Test F1 ↑ | **0.399** | +0.149 ✅ |
| 1−CosSim ↓ | **0.408** | -0.042 ✅ |
| Time (h) | **0.090** | -4.910 ✅ |
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
