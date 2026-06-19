# Sweep Summary — FORGET 10% (1 seed test)

_Generated: 2026-06-19 12:51:12_

**Seed**: [42, 123, 7]
**Configs tested**: 1

**Paper Forget-MI (10%)**: MIA=0.81, Df_AUC=0.656, Dt_AUC=0.565

## Comparison Table

| Config | IHL | img | ep | κ | MIA↓ | Df_AUC↓ | Df_F1↓ | Dt_AUC↑ | Dt_F1↑ | f_ce | t_ce | Time | PUS↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline (n=3) | 0.75 | 0.30 | 8 | 2.0 | 0.773 | 0.781 | 0.474 | 0.681 | 0.361 | 1.31 | 1.70 | 0.35 | — |
| paper | — | — | — | — | 0.810 | 0.656 | 0.313 | 0.565 | 0.252 | — | — | 5.00 | — |
| D_combo_aggressive | 1.25 | 0.50 | 8 | 2.0 | — | — | — | — | — | — | — | — | — |

**PUS** = (1 − MIA) × Dt_AUC — higher = better privacy + utility balance