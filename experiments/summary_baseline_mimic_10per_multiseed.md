# BASELINE Forget-MI — MIMIC-CXR — FORGET 10%

_Generated: 2026-07-02 00:25:12 on Kaggle_

**Dataset**: MIMIC-CXR (`mimic`)
**Seeds**: [42]
**Gold retrain**: ✅

| Metric | seed 42 | **mean ± std** | Paper | Δ vs paper |
|---|---|---|---|---|
| MIA_persample | 0.595 | **0.595 ± 0.000** | — | — |
| MIA_paper | 0.909 | **0.909 ± 0.000** | 0.810 | +0.099 |
| forget_ce | 2.264 | **2.264 ± 0.000** | — | — |
| test_ce | 2.378 | **2.378 ± 0.000** | — | — |
| Forget AUC | 0.757 | **0.757 ± 0.000** | 0.656 | +0.101 |
| Forget Mac-F1 | 0.396 | **0.396 ± 0.000** | 0.313 | +0.083 |
| Test AUC | 0.641 | **0.641 ± 0.000** | 0.565 | +0.076 |
| Test Mac-F1 | 0.295 | **0.295 ± 0.000** | 0.252 | +0.043 |
| 1 − CosSim | 0.461 | **0.461 ± 0.000** | — | — |
| Time (h) | 3.671 | **3.671 ± 0.000** | 5.000 | -1.329 |
| GPU peak (GB) | 8.490 | **8.490 ± 0.000** | — | — |
| Trainable ratio | 1.000 | **1.000 ± 0.000** | — | — |