# Exp 008 — exp08_true_loku_fila_subtraction

| Field | Value |
|---|---|
| **Date** | 2026-06-01 |
| **Git commit** | `784c836 (dirty)` |
| **Branch** | `master` |
| **Status** | ✅ Auto-tracked (cần điền Observations + Conclusion) |
| **Duration** | <!-- AUTO:duration -->
0.139h
<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Sau 7 exps, doc lai LoKU paper ky -> tim ra cai THIEU: FILA tai init phai TRU forget direction tu base (W*=W-B*A*). Cac exps truoc dung soft init (init_scale=0.05) khong tru gi -> moi cong viec phai do training dam nhan -> cham vao forget data -> MIA tang. Exp 08: dung TRUE LoKU FILA (subtract_scale=1.0) + chi train tren RETAIN (drop NegGrad, drop distill_forget). Doi: subtraction lam phan lon viec unlearn tai init, training chi restore retain -> balanced metrics.

**Predict trước khi chạy**: _(điền số mong đợi TRƯỚC khi xem kết quả)_

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
| Param | Trước | Sau |
|---|---|---|
| `unlearn_epochs` | `10                                             # 10 epochs đủ cho uniform prior` | `8                                              # [EXP 08] ít hơn — subtraction làm việc tại init` |
| `early_stop_patience` | `999                                            # [EXP 006] EFFECTIVELY OFF — CosSim không hợp với NegGrad/Uniform` | `4                                              # [EXP 08] CosSim sẽ TĂNG (gần F_re hơn) → stop khi plateau` |
| `alpha` | `0.5                                            # [EXP 007] UKR nhẹ (distillation đã giữ retain)` | `1.0                                            # UKR mạnh - giữ retain embedding` |
| `beta` | `0.0                                            # UU off` | `0.0                                            # UU off - subtraction xử lý forget` |
| `gamma` | `0.5                                            # [EXP 007] MKR nhẹ (distillation đã giữ joint)` | `1.0                                            # MKR mạnh - giữ joint retain` |
| `eta_re_anchor` | `0.0                                            # tắt re-anchor (distill thay thế)` | `0.0                                            # tắt re-anchor` |
| `kappa_cls_retain` | `0.5                                            # CE nhẹ giữ classifier ổn định` | `2.0                                            # CE retain MẠNH - anchor classifier` |
| `kappa_cls_forget` | `0.0                                            # TẮT NegGrad (đã chứng minh gây MIA tăng)` | `0.0                                            # KHÔNG đụng vào forget data` |
| `uniform_prior_weight` | `0.0                                            # TẮT uniform prior (cũng gây MIA tăng)` | `0.0` |
| `distill_retain_weight` | `2.0                                            # KL(student || teacher) trên retain — keep utility` | `1.5                                            # KL teacher trên RETAIN - giữ utility` |
| `distill_forget_weight` | `4.0                                            # [EXP 07 KEY] KL trên forget — student học teacher "không biết forget"` | `0.0                                            # [EXP 08 KEY] BỎ - đây là lỗi Exp 07` |
| `distill_temperature` | `2.0                                            # softening temperature (chuẩn distillation)` | `2.0` |
| `loku_init_scale` | `0.05                                           # small → base behavior preserved at init` | `0.05                                           # only used when loku_subtract_scale = 0` |
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
    value: 0.0                                            # [EXP 08 KEY] BỎ - đây là lỗi Exp 07
  distill_temperature:
    value: 2.0

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
python training/forgetmi_loku.py --config config.yaml --fresh --exp exp08_true_loku_fila_subtraction
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
[E00] loss=+10.714  UKR=+0.500  MKR=+2.265  CLS_ret=+1.251  DSL_ret=+3.6322  DSL_frg=+0.0000  | CosSim(ul,re)=0.6228
[E01] loss=+9.821  UKR=+0.350  MKR=+1.626  CLS_ret=+1.238  DSL_ret=+3.5795  DSL_frg=+0.0000  | CosSim(ul,re)=0.6254
[E02] loss=+9.730  UKR=+0.345  MKR=+1.618  CLS_ret=+1.234  DSL_ret=+3.5323  DSL_frg=+0.0000  | CosSim(ul,re)=0.6228
[E03] loss=+9.472  UKR=+0.340  MKR=+1.433  CLS_ret=+1.229  DSL_ret=+3.4946  DSL_frg=+0.0000  | CosSim(ul,re)=0.6229
[E04] loss=+9.362  UKR=+0.336  MKR=+1.398  CLS_ret=+1.218  DSL_ret=+3.4610  DSL_frg=+0.0000  | CosSim(ul,re)=0.6208
[E05] loss=+9.282  UKR=+0.341  MKR=+1.371  CLS_ret=+1.210  DSL_ret=+3.4325  DSL_frg=+0.0000  | CosSim(ul,re)=0.6207
```
<!-- /AUTO -->

### 5.2 Final metrics
<!-- AUTO:metrics -->
| Metric | Value | Δ vs Paper (3%) |
|---|---|---|
| MIA ↓ | **0.562** | -0.009 ✅ |
| Forget AUC ↓ | **0.833** | +0.098 ❌ |
| Forget F1 ↓ | **0.589** | +0.196 ❌ |
| Test AUC ↑ | **0.679** | +0.054 ✅ |
| Test F1 ↑ | **0.388** | +0.138 ✅ |
| 1−CosSim ↓ | **0.371** | -0.079 ✅ |
| Time (h) | **0.139** | -4.861 ✅ |
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
