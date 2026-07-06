# Bảng baseline Kaggle — Cross-dataset (MIMIC + IU)

_Auto-generated từ `/kaggle/working/results_summary.csv` — 2026-07-06 19:50_

Bảng chứa BASELINE Forget-MI trên Kaggle cho cả 2 datasets.
Để so với LoKU, merge CSV này với CSV từ Colab.

| Dataset | Forget% | Method | MIA | Df_AUC | Df_F1 | Dt_AUC | Dt_F1 | 1−CosSim | Time(h) |
|---|---|---|---|---|---|---|---|---|---|
| **MIMIC-CXR** | | |  |  |  |  |  |  |  |
| MIMIC-CXR | 3% | Paper | 0.571 | 0.735 | 0.393 | 0.625 | 0.250 | — | 5.000 |
| MIMIC-CXR | 3% | **Baseline (n=1)** | 1.000±0.000 | 0.561±0.000 | 0.156±0.000 | 0.632±0.000 | 0.234±0.000 | 0.458±0.000 | 2.959±0.000 |
| MIMIC-CXR | 3% | Δ (LoKU−paper) | ❌+0.429 | ✅-0.174 | ✅-0.237 | ✅+0.007 | ❌-0.016 | — | ✅-2.041 |

| MIMIC-CXR | 6% | Paper | 0.615 | 0.654 | 0.328 | 0.599 | 0.270 | — | 5.000 |
| MIMIC-CXR | 6% | **Baseline (n=1)** | 0.769±0.000 | 0.679±0.000 | 0.411±0.000 | 0.648±0.000 | 0.341±0.000 | 0.356±0.000 | 3.180±0.000 |
| MIMIC-CXR | 6% | Δ (LoKU−paper) | ❌+0.154 | ❌+0.025 | ❌+0.083 | ✅+0.049 | ✅+0.071 | — | ✅-1.820 |

| MIMIC-CXR | 10% | Paper | 0.810 | 0.656 | 0.313 | 0.565 | 0.252 | — | 5.000 |
| MIMIC-CXR | 10% | **Baseline (n=1)** | 0.909±0.000 | 0.757±0.000 | 0.396±0.000 | 0.641±0.000 | 0.295±0.000 | 0.461±0.000 | 3.671±0.000 |
| MIMIC-CXR | 10% | Δ (LoKU−paper) | ❌+0.099 | ❌+0.101 | ❌+0.083 | ✅+0.076 | ✅+0.043 | — | ✅-1.329 |

| **Indiana University CXR** | | |  |  |  |  |  |  |  |
| Indiana University CXR | 3% | **Baseline (n=1)** | 0.000±0.000 | 0.653±0.000 | 0.573±0.000 | 0.636±0.000 | 0.571±0.000 | 0.098±0.000⚠️ | 9.518±0.000 |

| Indiana University CXR | 6% | _(chưa chạy)_ | — | — | — | — | — | — | — |

| Indiana University CXR | 10% | _(chưa chạy)_ | — | — | — | — | — | — | — |
