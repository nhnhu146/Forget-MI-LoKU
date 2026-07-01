# Bảng baseline Kaggle — Cross-dataset (MIMIC + IU)

_Auto-generated từ `/kaggle/working/results_summary.csv` — 2026-07-01 17:09_

Bảng chứa BASELINE Forget-MI trên Kaggle cho cả 2 datasets.
Để so với LoKU, merge CSV này với CSV từ Colab.

| Dataset | Forget% | Method | MIA | Df_AUC | Df_F1 | Dt_AUC | Dt_F1 | 1−CosSim | Time(h) |
|---|---|---|---|---|---|---|---|---|---|
| **MIMIC-CXR** | | |  |  |  |  |  |  |  |
| MIMIC-CXR | 3% | Paper | 0.571 | 0.735 | 0.393 | 0.625 | 0.250 | — | 5.000 |
| MIMIC-CXR | 3% | **Baseline (n=1)** | 1.000±0.000 | 0.561±0.000 | 0.156±0.000 | 0.632±0.000 | 0.234±0.000 | 0.458±0.000 | 2.959±0.000 |
| MIMIC-CXR | 3% | Δ (LoKU−paper) | ❌+0.429 | ✅-0.174 | ✅-0.237 | ✅+0.007 | ❌-0.016 | — | ✅-2.041 |

| MIMIC-CXR | 6% | Paper | 0.615 | 0.654 | 0.328 | 0.599 | 0.270 | — | 5.000 |
| MIMIC-CXR | 6% | _(chưa chạy)_ | — | — | — | — | — | — | — |

| MIMIC-CXR | 10% | Paper | 0.810 | 0.656 | 0.313 | 0.565 | 0.252 | — | 5.000 |
| MIMIC-CXR | 10% | _(chưa chạy)_ | — | — | — | — | — | — | — |
