# Bảng baseline Kaggle — Cross-dataset (MIMIC + IU)

_Auto-generated từ `/kaggle/working/results_summary.csv` — 2026-06-24 21:01_

Bảng chứa BASELINE Forget-MI trên Kaggle cho cả 2 datasets.
Để so với LoKU, merge CSV này với CSV từ Colab.

| Dataset | Forget% | Method | MIA | Df_AUC | Df_F1 | Dt_AUC | Dt_F1 | 1−CosSim | Time(h) |
|---|---|---|---|---|---|---|---|---|---|
| **MIMIC-CXR** | | |  |  |  |  |  |  |  |
| MIMIC-CXR | 3% | Paper | 0.571 | 0.735 | 0.393 | 0.625 | 0.250 | — | 5.000 |
| MIMIC-CXR | 3% | **Baseline (n=3)** | 0.857±0.000 | 0.571±0.000 | 0.172±0.000 | 0.651±0.000 | 0.277±0.000 | 0.384±0.007 | 2.853±0.035 |
| MIMIC-CXR | 3% | Δ (LoKU−paper) | ❌+0.286 | ✅-0.164 | ✅-0.221 | ✅+0.026 | ✅+0.027 | — | ✅-2.147 |

| MIMIC-CXR | 6% | Paper | 0.615 | 0.654 | 0.328 | 0.599 | 0.270 | — | 5.000 |
| MIMIC-CXR | 6% | _(chưa chạy)_ | — | — | — | — | — | — | — |

| MIMIC-CXR | 10% | Paper | 0.810 | 0.656 | 0.313 | 0.565 | 0.252 | — | 5.000 |
| MIMIC-CXR | 10% | _(chưa chạy)_ | — | — | — | — | — | — | — |
