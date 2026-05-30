# Exp 001 — loku_v1_baseline

| Field | Value |
|---|---|
| **Date** | 2025-05-30 |
| **Git commit** | `0c4db06` (after metrics fix) |
| **Branch** | `master` |
| **Status** | ✅ Done — baseline reference (re-created after cleanup) |
| **Duration** | <!-- AUTO:duration -->0.094h<!-- /AUTO --> |

---

## 1. Hypothesis (giả thuyết)

Phương pháp Forget-MI-LoKU (LoRA + FIM-guided soft init + bounded losses + re-anchor) có thể đạt **MIA ≈ Forget-MI baseline (0.571)** và **Test AUC ≈ baseline (0.625)** với **thời gian ≤ 1h** (so với 5h của baseline) và **trainable params < 1%** (so với 100%).

**Predict trước khi chạy**: MIA 0.50-0.65 | Test AUC 0.55-0.65 | Forget AUC 0.65-0.80 | Time ≤ 0.5h | Trainable 0.3-0.5%

---

## 2. Changes from previous experiment

<!-- AUTO:config_diff -->
_(no previous experiment to compare — this is the first exp)_
<!-- /AUTO -->

### Code changes (so với Forget-MI baseline)
- Tạo mới [training/forgetmi_loku.py](../training/forgetmi_loku.py) (rewrite từ `forgetmi_partial.py`)
- Thêm: `compute_fisher_importance`, `apply_loku_soft_init`, `safe_forward`, `run_mia`, `perf_metrics`
- Loss: `F.relu` hinge thay vì `-distance`; đảo chiều hinge L_UKR/L_MKR
- Thêm L_RE anchor term + final eval block

---

## 3. Full configuration snapshot

<details>
<summary>config.yaml (xem rút gọn các tham số chính)</summary>

```yaml
learning_rate: 5.0e-4
unlearn_epochs: 8
early_stop_patience: 3
unlearn_batch_size: 16
use_noise: false
forget_margin: 8.0
alpha: 1.0       # UKR
beta: 1.0        # UU
theta: 0.5       # MD
gamma: 0.5       # MKR
eta_re_anchor: 0.5
lora_r: 8
lora_alpha: 16
lora_target_modules: ["query", "key", "value"]
fisher_max_samples: 256
loku_init_scale: 0.05
random_seed: 42
forget_set_path: "./data_splits/forget_set_3per.csv"
```

</details>

---

## 4. Execution

```bash
python training/forgetmi_loku.py --config config.yaml --fresh --exp loku_v1_baseline
```

---

## 5. Results

### 5.1 Training curves
<!-- AUTO:training_curves -->
```
Epoch 0: 13it [01:05, 5.07s/it]
[E00] loss=+16.684  UU=+5.052  MD=+0.000  UKR=+0.300  MKR=+1.855  RE=+20.810  | CosSim=0.6220
[E01] loss=+16.292  UU=+4.845  MD=+0.000  UKR=+0.325  MKR=+1.458  RE=+20.785  | CosSim=0.6217
[E02] loss=+15.891  UU=+4.268  MD=+0.000  UKR=+0.517  MKR=+1.399  RE=+20.814  | CosSim=0.6217
[E03] loss=+15.786  UU=+3.585  MD=+0.000  UKR=+1.050  MKR=+1.380  RE=+20.923  | CosSim=0.6217
⏹ Early stop at epoch 3 (best CosSim=0.6220)
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
| Time (h) | **0.094** | -4.906 ✅ |
| GPU peak (GB) | **11.76** | — |
| Trainable params | **0.389%** | (vs 100% paper) |
<!-- /AUTO -->

---

## 6. Observations

- **L_MD = 0 mọi epoch**: `forget_margin=8` quá nhỏ — joint distance đã > 8 từ đầu → ReLU hinge satisfied → MD không đóng góp gradient
- **L_RE ≈ 20.8 cứng đơ**: η=0.5 quá nhỏ so với scale của L_RE (20+) → không đủ kéo F_ul gần F_re → CosSim không cải thiện
- **L_UU giảm 5.05 → 3.58**: Forget unimodal đang được đẩy ra đúng hướng (hinge chưa satisfied)
- **L_UKR tăng nhẹ 0.30 → 1.05**: Retain drift đang tăng nhưng vẫn bị bound bởi 0.1·d term
- **Early stop ở epoch 3**: CosSim plateau → patience=3 trigger sớm

---

## 7. Conclusion

- **Hypothesis verdict**: 🤷 Partially confirmed
  - Speed/efficiency: ✅ vượt mong đợi (53× nhanh hơn paper)
  - MIA: ✅ matched (0.552 vs 0.571)
  - Test utility: ✅ tốt hơn paper (+0.049 AUC, +0.137 F1)
  - Forget strength: ❌ yếu hơn paper (Forget AUC +0.094)
- **Keep changes**: Y — pipeline foundation OK, không có bug nghiêm trọng
- **Why**: Trade-off đang nghiêng về "giữ utility quá tốt, quên chưa đủ". Cần tune trong exp tiếp theo để đẩy forget mạnh hơn.

---

## 8. Next steps

- **Exp 002** — Tune `forget_margin: 8→20`, `eta_re_anchor: 0.5→0` → kích hoạt L_MD và bỏ kéo về F_re
- **Exp 003** — Nếu Exp 002 chưa đủ: thêm NegGrad term (gradient ascent trên forget)
- **Exp 004** — Tăng `lora_r=16` để có nhiều capacity hơn
