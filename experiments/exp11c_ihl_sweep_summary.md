# IHL sweep — exp11c_ihl_sweep

IHL values: [0.5, 0.75, 1.0] (single seed 42, no F_re)

| IHL | forget_ce | test_ce | MIA_ps | MIA_pp | Df_AUC | Df_F1 | Dt_AUC | Dt_F1 | over-forget? |
|---|---|---|---|---|---|---|---|---|---|
| 0.500 | 1.571 | 1.739 | 0.507 | 0.429 | 0.758 | 0.470 | 0.679 | 0.368 | no |
| 0.750 | 1.800 | 1.821 | 0.453 | 0.429 | 0.734 | 0.379 | 0.677 | 0.363 | no |
| 1.000 | 1.617 | 1.773 | 0.483 | 0.571 | 0.767 | 0.456 | 0.681 | 0.352 | no |

**Paper (3%)**: MIA=0.571 | Df_AUC=0.735 | Df_F1=0.393 | Dt_AUC=0.625 | Dt_F1=0.250
**Sweet-spot**: chọn IHL lớn nhất mà forget_ce ≲ test_ce (Forget-AUC thấp nhất, MIA lành mạnh).
