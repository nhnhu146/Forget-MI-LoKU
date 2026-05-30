# Exp 002 — forget_margin_20_no_anchor

| Field | Value |
|---|---|
| **Date** | 2026-05-30 |
| **Git commit** | `aa16ca3 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.092h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Tang forget_margin tu 8 len 20 (kich hoat L_MD vi joint distance > 8) va tat re-anchor (eta=0) de cho forget manh hon. Doi: Forget AUC giam tu 0.829 xuong ~0.75, MIA tang nhe ~0.58, Test AUC giam nhe ~0.65.

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
    value: 0.0                                            # [EXP 002] tắt re-anchor (was 0.5)

  # ===== Forget-margin (bounded loss) =====
  forget_margin:
    value: 20.0                                           # [EXP 002] tăng margin (was 8.0) — kích hoạt L_MD

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp forget_margin_20_no_anchor
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+18.281  UU=+17.055  MD=+0.000  UKR=+0.299  MKR=+1.855  RE=+0.000  | CosSim(ul,re)=0.6220
[E01] loss=+17.921  UU=+16.871  MD=+0.000  UKR=+0.323  MKR=+1.457  RE=+0.000  | CosSim(ul,re)=0.6217
[E02] loss=+17.518  UU=+16.304  MD=+0.000  UKR=+0.515  MKR=+1.398  RE=+0.000  | CosSim(ul,re)=0.6217
[E03] loss=+17.512  UU=+15.548  MD=+0.000  UKR=+1.272  MKR=+1.384  RE=+0.000  | CosSim(ul,re)=0.6217
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.552** | -0.019 ✅ |
| Forget AUC ↓ | **0.829** | +0.094 ❌ |
| Forget F1 ↓ | **0.597** | +0.204 ❌ |
| Test AUC ↑ | **0.674** | +0.049 ✅ |
| Test F1 ↑ | **0.387** | +0.137 ✅ |
| 1−CosSim ↓ | **0.371** | -0.079 ✅ |
| Time (h) | **0.092** | -4.908 ✅ |
| GPU peak (GB) | **11.75** | — |
| Trainable params | **0.389%** | (vs 100% paper) |

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
