# BASELINE Forget-MI — MIMIC-CXR — FORGET 3%

_Generated: 2026-06-30 07:54:44 on Kaggle_

**Dataset**: MIMIC-CXR (`mimic`)
**Seeds**: [42]
**Gold retrain**: ✅

| Metric | seed 42 | **mean ± std** | Paper | Δ vs paper |
|---|---|---|---|---|
| MIA_persample | 0.433 | **0.433 ± 0.000** | — | — |
| MIA_paper | 0.857 | **0.857 ± 0.000** | 0.571 | +0.286 |
| forget_ce | 4.738 | **4.738 ± 0.000** | — | — |
| test_ce | 3.049 | **3.049 ± 0.000** | — | — |
| Forget AUC | 0.571 | **0.571 ± 0.000** | 0.735 | -0.164 |
| Forget Mac-F1 | 0.172 | **0.172 ± 0.000** | 0.393 | -0.221 |
| Test AUC | 0.651 | **0.651 ± 0.000** | 0.625 | +0.026 |
| Test Mac-F1 | 0.277 | **0.277 ± 0.000** | 0.250 | +0.027 |
| 1 − CosSim | 0.394 | **0.394 ± 0.000** | — | — |
| Time (h) | 2.804 | **2.804 ± 0.000** | 5.000 | -2.196 |
| GPU peak (GB) | 8.490 | **8.490 ± 0.000** | — | — |
| Trainable ratio | 1.000 | **1.000 ± 0.000** | — | — |