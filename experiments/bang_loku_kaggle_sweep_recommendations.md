# LoKU Kaggle Sweep — Recommendations (seed 42, 1 seed)

_Generated: 2026-06-30 09:09:49 — recovered từ Save Version output (session timeout @ 10% config H)_

**PUS** = (1−MIA_paper) × Dt_AUC (cân bằng privacy + utility). Cao = tốt.
Thấp hơn baseline ở MIA & Df_AUC = quên tốt hơn; cao hơn ở Dt_AUC = giữ utility tốt hơn.

## Forget 3%

**Baseline (repro)**: MIA=0.857 Df_AUC=0.571 Dt_AUC=0.651 | PUS=0.093
**Paper Forget-MI**: MIA=0.571 Df_AUC=0.735 Dt_AUC=0.625

| Config | desc | MIA_p ↓ | Df_AUC ↓ | Dt_AUC ↑ | Dt_F1 ↑ | PUS ↑ | beat base? |
|---|---|---|---|---|---|---|---|
| A_ihl100 | IHL1.0 | 0.571 | 0.741 | 0.674 | 0.330 | 0.289 | ✅ |
| B_img050 | img0.5 | 0.286 | 0.709 | 0.681 | 0.368 | 0.486 | ✅ |
| C_combo_moderate | IHL1.0+img0.5 | 0.286 | 0.692 | 0.673 | 0.389 | 0.481 | ✅ |
| D_combo_aggressive | IHL1.25+img0.5 | 0.143 | 0.651 | 0.674 | 0.354 | 0.578 | ✅ |
| E_extreme | IHL1.5+img0.7 | 0.143 | 0.632 | 0.656 | 0.342 | 0.562 | ✅ |
| F_more_epochs | 12ep | 0.286 | 0.689 | 0.661 | 0.352 | 0.472 | ✅ |
| G_less_retain | kappa1.0 | 0.571 | 0.775 | 0.689 | 0.381 | 0.296 | ✅ |
| H_long_aggressive | IHL1.25+img0.5+12ep | 0.143 | 0.637 | 0.672 | 0.356 | 0.576 | ✅ |
| I_very_aggressive | IHL1.75+img0.7 | 0.143 | 0.653 | 0.670 | 0.319 | 0.574 | ✅ |
| J_max_push | IHL2.0+img0.8+12ep | 0.000 | 0.562 | 0.662 | 0.324 | 0.662 | ✅ |

🏆 **Winner 3%: `J_max_push`** — PUS=0.662, MIA=0.000, Df_AUC=0.562, Dt_AUC=0.662

## Forget 6%

**Baseline (repro)**: MIA=0.769 Df_AUC=0.661 Dt_AUC=0.65 | PUS=0.150
**Paper Forget-MI**: MIA=0.615 Df_AUC=0.654 Dt_AUC=0.599

| Config | desc | MIA_p ↓ | Df_AUC ↓ | Dt_AUC ↑ | Dt_F1 ↑ | PUS ↑ | beat base? |
|---|---|---|---|---|---|---|---|
| A_ihl100 | IHL1.0 | 0.538 | 0.740 | 0.681 | 0.354 | 0.315 | ✅ |
| B_img050 | img0.5 | 0.462 | 0.707 | 0.686 | 0.344 | 0.369 | ✅ |
| C_combo_moderate | IHL1.0+img0.5 | 0.538 | 0.728 | 0.687 | 0.341 | 0.317 | ✅ |
| D_combo_aggressive | IHL1.25+img0.5 | 0.385 | 0.658 | 0.682 | 0.361 | 0.419 | ✅ |
| E_extreme | IHL1.5+img0.7 | 0.385 | 0.629 | 0.690 | 0.332 | 0.424 | ✅ |
| F_more_epochs | 12ep | 0.538 | 0.743 | 0.679 | 0.351 | 0.314 | ✅ |
| G_less_retain | kappa1.0 | 0.385 | 0.720 | 0.685 | 0.378 | 0.421 | ✅ |
| H_long_aggressive | IHL1.25+img0.5+12ep | 0.385 | 0.658 | 0.682 | 0.361 | 0.419 | ✅ |
| I_very_aggressive | IHL1.75+img0.7 | 0.538 | 0.601 | 0.689 | 0.352 | 0.318 | ✅ |
| J_max_push | IHL2.0+img0.8+12ep | 0.154 | 0.560 | 0.678 | 0.364 | 0.574 | ✅ |

🏆 **Winner 6%: `J_max_push`** — PUS=0.574, MIA=0.154, Df_AUC=0.560, Dt_AUC=0.678

## Forget 10%

**Baseline (repro)**: MIA=0.909 Df_AUC=0.764 Dt_AUC=0.654 | PUS=0.060
**Paper Forget-MI**: MIA=0.81 Df_AUC=0.656 Dt_AUC=0.565

| Config | desc | MIA_p ↓ | Df_AUC ↓ | Dt_AUC ↑ | Dt_F1 ↑ | PUS ↑ | beat base? |
|---|---|---|---|---|---|---|---|
| A_ihl100 | IHL1.0 | 0.591 | 0.739 | 0.679 | 0.369 | 0.278 | ✅ |
| B_img050 | img0.5 | 0.727 | 0.737 | 0.680 | 0.329 | 0.186 | ✅ |
| C_combo_moderate | IHL1.0+img0.5 | 0.636 | 0.744 | 0.668 | 0.343 | 0.243 | ✅ |
| D_combo_aggressive | IHL1.25+img0.5 | 0.364 | 0.688 | 0.668 | 0.335 | 0.425 | ✅ |
| E_extreme | IHL1.5+img0.7 | 0.545 | 0.716 | 0.677 | 0.337 | 0.308 | ✅ |
| F_more_epochs | 12ep | 0.818 | 0.794 | 0.682 | 0.373 | 0.124 | ✅ |
| G_less_retain | kappa1.0 | 0.773 | 0.754 | 0.687 | 0.316 | 0.156 | ✅ |

🏆 **Winner 10%: `D_combo_aggressive`** — PUS=0.425, MIA=0.364, Df_AUC=0.688, Dt_AUC=0.668
> ⚠️ 10%% mới chạy A–G (H/I/J bị timeout). Winner thật có thể là J (thắng ở 3%/6%) — cần rerun H/I/J@10%.
