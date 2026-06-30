# BASELINE Forget-MI — MIMIC-CXR — FORGET 3%

_Generated: 2026-06-30 11:24:22 on Kaggle_

**Dataset**: MIMIC-CXR (`mimic`)
**Seeds**: [42]
**Gold retrain**: ✅

| Metric | seed 42 | **mean ± std** | Paper | Δ vs paper |
|---|---|---|---|---|
| MIA_persample | 0.493 | **0.493 ± 0.000** | — | — |
| MIA_paper | 1.000 | **1.000 ± 0.000** | 0.571 | +0.429 |
| forget_ce | 4.807 | **4.807 ± 0.000** | — | — |
| test_ce | 3.446 | **3.446 ± 0.000** | — | — |
| Forget AUC | 0.561 | **0.561 ± 0.000** | 0.735 | -0.174 |
| Forget Mac-F1 | 0.156 | **0.156 ± 0.000** | 0.393 | -0.237 |
| Test AUC | 0.632 | **0.632 ± 0.000** | 0.625 | +0.007 |
| Test Mac-F1 | 0.234 | **0.234 ± 0.000** | 0.250 | -0.016 |
| 1 − CosSim | 0.458 | **0.458 ± 0.000** | — | — |
| Time (h) | 2.959 | **2.959 ± 0.000** | 5.000 | -2.041 |
| GPU peak (GB) | 8.490 | **8.490 ± 0.000** | — | — |
| Trainable ratio | 1.000 | **1.000 ± 0.000** | — | — |