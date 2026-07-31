# BẢNG TỔNG HỢP ĐẦY ĐỦ — MIMIC 3%, seed 42

Thời gian = **train thuần** (train+Fisher, giờ) · Para = tỉ lệ tham số huấn luyện.

| Phương pháp | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | fce/tce | Giờ (train) | Para |
|---|---|---|---|---|---|---|---|---|
| OG (Original) | 0.731 | 0.419 | 0.695 | 0.384 | 0.657 | 1.97/2.11 | – | 100% |
| **GOLD (Retrained) — ĐÍCH** | 0.498 | 0.179 | 0.615 | 0.297 | 0.423 | 4.74/3.05 | – | 100% |
| *Forget-MI công bố (paper)* | 0.735 | 0.393 | 0.625 | 0.250 | 0.571 | --/-- | – | 100%‡ |
| **Forget-MI tái lập (E14)** † | 0.750 | 0.383 | 0.668 | 0.338 | 0.572 | 2.76/2.57 | 0.54h | 100% |
| | | | | | | | | |
| **P3 (tổng quát)** | 0.689 | 0.344 | 0.696 | 0.400 | 0.483 | 1.79/1.69 | 0.26h | 0.45% |
| P4 k=5 | 0.728 | 0.446 | 0.697 | 0.392 | 0.552 | 1.68/1.76 | 0.27h | 0.45% |
| P4 k=10 | 0.716 | 0.411 | 0.698 | 0.389 | 0.562 | 1.64/1.65 | 0.29h | 0.45% |
| P4 k=15 | 0.699 | 0.377 | 0.696 | 0.395 | 0.502 | 1.71/1.67 | 0.28h | 0.45% |
| P4 k=20 | 0.693 | 0.357 | 0.696 | 0.399 | 0.493 | 1.75/1.67 | 0.27h | 0.45% |
| P5 | 0.721 | 0.427 | 0.697 | 0.385 | 0.537 | 1.68/1.74 | 0.27h | 0.45% |
| **P6 (quên mạnh nhất)** | 0.647 | 0.286 | 0.691 | 0.345 | 0.328 | 1.92/1.71 | 0.27h | 0.44% |
| MAIN (S_val) | 0.700 | 0.361 | 0.698 | 0.374 | 0.408 | 1.64/1.66 | 0.28h | 0.44% |
| MAIN (val_ce) | 0.700 | 0.361 | 0.698 | 0.374 | 0.408 | 1.64/1.66 | 0.27h | 0.44% |

**Ghi chú:** dòng P = checkpoint `last` (E30) · Forget-MI = E14 (khớp paper) · ‡ số Bảng 2 bài báo · † eval trên test đầy đủ (split cũ): Df-AUC/Df-F1 so trực tiếp được, Dt/MIA tham khảo · P-methods eval trên D_t_final.

**Thời gian:** Forget-MI train FULL model (113M) → ~0.54h; P-methods chỉ LoRA (0.5M) → ~0.27h. *(Lần chạy baseline có eval-mỗi-epoch nên WALL 6.55h, nhưng train thuần vẫn 0.54h — số trên là train thuần cho công bằng.)* GOLD = retrain toàn bộ (đắt nhất, paper ~14h).
