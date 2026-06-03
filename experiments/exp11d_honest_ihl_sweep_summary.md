# IHL sweep (HONEST, early-stop=val, no F_re) — exp11d_honest_ihl_sweep

IHL values: [0.5, 0.75, 1.0] (single seed 42)

| IHL | forget_ce | test_ce | MIA_ps | MIA_pp | Df_AUC | Df_F1 | Dt_AUC | Dt_F1 | over-forget? |
|---|---|---|---|---|---|---|---|---|---|
| 0.500 | 1.571 | 1.739 | 0.507 | 0.429 | 0.758 | 0.470 | 0.679 | 0.368 | no |
| 0.750 | 1.800 | 1.821 | 0.453 | 0.429 | 0.734 | 0.379 | 0.677 | 0.363 | no |
| 1.000 | 1.972 | 2.003 | 0.488 | 0.571 | 0.741 | 0.369 | 0.674 | 0.330 | no |

**Paper (3%)**: MIA=0.571 | Df_AUC=0.735 | Df_F1=0.393 | Dt_AUC=0.625 | Dt_F1=0.250
**Sweet-spot**: IHL lớn nhất mà forget_ce ≲ test_ce (Forget-AUC thấp nhất, MIA lành mạnh).
