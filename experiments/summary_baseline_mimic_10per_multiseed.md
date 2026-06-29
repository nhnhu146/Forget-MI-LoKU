# BASELINE Forget-MI — MIMIC-CXR — FORGET 10%

_Generated: 2026-06-29 15:07:16 (recovered từ Kaggle Save Version output)_

**Dataset**: MIMIC-CXR (`mimic`)
**Seeds**: [42, 123]  _(seed 7 chưa xong — timeout ở epoch 28/30)_
**Gold retrain**: ✅ (model_retrained_10per từ forget-mi-models-full)

| Metric | seed 42 | seed 123 | **mean ± std** | Paper | Δ vs paper |
|---|---|---|---|---|---|
| MIA_persample | 0.589 | 0.613 | **0.601 ± 0.012** | — | — |
| MIA_paper | 0.909 | 0.909 | **0.909 ± 0.000** | 0.810 | +0.099 |
| forget_ce | 2.454 | 2.454 | **2.454 ± 0.000** | — | — |
| test_ce | 2.426 | 2.426 | **2.426 ± 0.000** | — | — |
| Forget AUC | 0.764 | 0.764 | **0.764 ± 0.000** | 0.656 | +0.108 |
| Forget Mac-F1 | 0.385 | 0.385 | **0.385 ± 0.000** | 0.313 | +0.072 |
| Test AUC | 0.654 | 0.654 | **0.654 ± 0.000** | 0.565 | +0.089 |
| Test Mac-F1 | 0.322 | 0.322 | **0.322 ± 0.000** | 0.252 | +0.070 |
| 1 - CosSim | 0.429 | 0.414 | **0.421 ± 0.008** | — | — |
| Time (h) | 3.838 | 3.850 | **3.844 ± 0.006** | 5.000 | -1.156 |
| GPU peak (GB) | 8.490 | 8.490 | **8.490 ± 0.000** | — | — |
| Trainable ratio | 1.000 | 1.000 | **1.000 ± 0.000** | — | — |