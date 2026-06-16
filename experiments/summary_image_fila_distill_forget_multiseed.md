# Multi-seed summary — exp10c_image_fila_distill_forget

Seeds: [42, 123, 7]

| Metric | seed 42 | seed 123 | seed 7 | **mean ± std** |
|---|---|---|---|---|
| MIA_persample ↓ | 0.493 | 0.473 | 0.403 | **0.456 ± 0.039** |
| MIA_paper ↓ | 0.571 | 0.429 | 0.429 | **0.476 ± 0.067** |
| Forget AUC ↓ | 0.718 | 0.700 | 0.686 | **0.701 ± 0.013** |
| Forget F1 ↓ | 0.271 | 0.307 | 0.273 | **0.284 ± 0.017** |
| Test AUC ↑ | 0.689 | 0.673 | 0.675 | **0.679 ± 0.007** |
| Test F1 ↑ | 0.330 | 0.317 | 0.348 | **0.332 ± 0.013** |

**Paper (3%)**: MIA=0.571 | Df_AUC=0.735 | Df_F1=0.393 | Dt_AUC=0.625 | Dt_F1=0.250  
**Gold retrained**: MIA=0.000 | Df_AUC=0.566 | Df_F1=0.310 | Dt_AUC=0.626 | Dt_F1=0.362
