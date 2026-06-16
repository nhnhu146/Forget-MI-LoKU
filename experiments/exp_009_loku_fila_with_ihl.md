# Exp 009 — exp09_loku_fila_with_ihl

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Git commit** | `253a429 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.189h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Exp 08 dat MIA TOT NHAT (0.562 < paper 0.571) nho TRUE LoKU FILA subtraction, nhung retain anchor manh keo LoRA ve khoi phuc forget direction -> Forget metrics van cao (0.833 AUC, 0.589 F1). Exp 09 them Inverted Hinge Loss (IHL) tu paper LoKU goc: L_IHL = 1 + p(true_forget) - max_v!=true(p(v)). Bounded [0,2] va self-stopping (khong can clamp nhu NegGrad). Adapted tu Sec 3.3 cua LoKU paper. Doi: Subtraction (init) + IHL (training mild push) cong huong -> Forget metrics giam ve gan paper, MIA giu thap, Test giu cao -> BALANCED.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `distill_forget_weight` | `0.0                                            # [EXP 08 KEY] BỎ - đây là lỗi Exp 07` | `0.0                                            # BỎ - đã chứng minh có hại` |
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
    value: 0.0                                            # BỎ - đã chứng minh có hại
  distill_temperature:
    value: 2.0

  # ===== [EXP 09 NEW] Inverted Hinge Loss (IHL) — từ paper LoKU =====
  # L_IHL = 1 + p(true_forget_label) - max_{v ≠ true}(p(v|x))
  # Bounded [0, 2], self-stopping khi unlearning đạt. An toàn hơn NegGrad (unbounded).
  # Adapted từ LoKU paper Sec 3.3 (gốc cho LLM next-token, ta adapt cho classification).
  ihl_forget_weight:
    value: 1.5                                            # [EXP 09 KEY] mạnh nhưng IHL self-bounded

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp09_loku_fila_with_ihl
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+12.220  UKR=+0.353  MKR=+2.342  CLS_ret=+1.253  DSL_ret=+3.6264  IHL_frg=+1.0531  | CosSim(ul,re)=0.6264
[E01] loss=+11.399  UKR=+0.335  MKR=+1.639  CLS_ret=+1.259  DSL_ret=+3.5640  IHL_frg=+1.0405  | CosSim(ul,re)=0.6277
[E02] loss=+11.121  UKR=+0.323  MKR=+1.495  CLS_ret=+1.246  DSL_ret=+3.5095  IHL_frg=+1.0307  | CosSim(ul,re)=0.6295
[E03] loss=+10.956  UKR=+0.321  MKR=+1.429  CLS_ret=+1.233  DSL_ret=+3.4684  IHL_frg=+1.0244  | CosSim(ul,re)=0.6255
[E04] loss=+10.811  UKR=+0.319  MKR=+1.386  CLS_ret=+1.213  DSL_ret=+3.4341  IHL_frg=+1.0190  | CosSim(ul,re)=0.6265
[E05] loss=+10.703  UKR=+0.341  MKR=+1.344  CLS_ret=+1.198  DSL_ret=+3.3954  IHL_frg=+1.0193  | CosSim(ul,re)=0.6311
[E06] loss=+10.609  UKR=+0.312  MKR=+1.318  CLS_ret=+1.179  DSL_ret=+3.3852  IHL_frg=+1.0286  | CosSim(ul,re)=0.6336
[E07] loss=+10.812  UKR=+0.625  MKR=+1.294  CLS_ret=+1.178  DSL_ret=+3.3425  IHL_frg=+1.0150  | CosSim(ul,re)=0.6377
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.537** | -0.034 ✅ |
| Forget AUC ↓ | **0.832** | +0.097 ❌ |
| Forget F1 ↓ | **0.593** | +0.200 ❌ |
| Test AUC ↑ | **0.678** | +0.053 ✅ |
| Test F1 ↑ | **0.385** | +0.135 ✅ |
| 1−CosSim ↓ | **0.361** | -0.089 ✅ |
| Time (h) | **0.189** | -4.811 ✅ |
| GPU peak (GB) | **11.75** | — |
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
