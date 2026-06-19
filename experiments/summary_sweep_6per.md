# Sweep Summary — FORGET 6% (1 seed test)

_Generated: 2026-06-19 12:50:42_

**Seed**: [42, 123, 7]
**Configs tested**: 1

**Paper Forget-MI (6%)**: MIA=0.615, Df_AUC=0.654, Dt_AUC=0.599

## Comparison Table

| Config | IHL | img | ep | κ | MIA↓ | Df_AUC↓ | Df_F1↓ | Dt_AUC↑ | Dt_F1↑ | f_ce | t_ce | Time | PUS↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (n=3) | 0.75 | 0.30 | 8 | 2.0 | 0.589 | 0.729 | 0.359 | 0.689 | 0.347 | 2.02 | 2.08 | 0.28 | — |
| paper | — | — | — | — | 0.615 | 0.654 | 0.328 | 0.599 | 0.270 | — | — | 5.00 | — |
| C_combo_moderate | 1.00 | 0.50 | 8 | 2.0 | — | — | — | — | — | — | — | — | — |

**PUS** = (1 − MIA) × Dt_AUC — higher = better privacy + utility balance