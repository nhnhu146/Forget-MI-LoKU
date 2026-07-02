# LoKU Kaggle Multi-seed — FORGET 3% — final_ihl125_3per

_Generated: 2026-07-02 07:58:48_

**Config**: `final_ihl125` (honest, no F_re), IHL=1.25, image-FILA scale=0.3

**Hardware**: Kaggle T4 (so sánh trực tiếp được với baseline run_kaggle_baseline.ipynb)

**Seeds**: [42, 123, 7]

**Gold retrained**: ❌ N/A — 1−CosSim KHÔNG hợp lệ

| Metric | seed 42 | seed 123 | seed 7 | **mean ± std** | Paper | Δ vs paper |
|---|---|---|---|---|---|---|
| MIA_persample | 0.239 |  |  | **0.239 ± 0.000** | — | — |
| MIA_paper | 0.000 |  |  | **0.000 ± 0.000** | 0.571 | -0.571 |
| forget_ce | 2.957 |  |  | **2.957 ± 0.000** | — | — |
| test_ce | 1.803 |  |  | **1.803 ± 0.000** | — | — |
| Forget AUC | 0.562 |  |  | **0.562 ± 0.000** | 0.735 | -0.173 |
| Forget Mac-F1 | 0.134 |  |  | **0.134 ± 0.000** | 0.393 | -0.259 |
| Test AUC | 0.662 |  |  | **0.662 ± 0.000** | 0.625 | +0.037 |
| Test Mac-F1 | 0.324 |  |  | **0.324 ± 0.000** | 0.250 | +0.074 |
| 1 − CosSim ⚠️ | 0.341 |  |  | **0.341 ± 0.000** | — | — |
| Time (h) | 0.218 |  |  | **0.218 ± 0.000** | 5.000 | -4.782 |
| GPU peak (GB) | 11.780 |  |  | **11.780 ± 0.000** | — | — |
| Trainable ratio | 0.005 |  |  | **0.005 ± 0.000** | — | — |

**Paper Forget-MI (3%)**: MIA=0.571 | Df_AUC=0.735 | Dt_AUC=0.625 | Time≈5.0h

_Δ vs paper_: âm = LoKU tốt hơn (↓ metrics) hoặc kém hơn (↑ metrics). ✅ = LoKU thắng.