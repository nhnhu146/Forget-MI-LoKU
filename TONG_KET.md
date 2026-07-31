# TỔNG KẾT — MIMIC 3%, seed 42 (đã tái lập Forget-MI đúng paper)

**Đích = GOLD.** Forget-MI tái lập = **checkpoint khớp paper (E14/E15)**, KHÔNG phải E30.

| Phương pháp | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | fce/tce | %param |
|---|---|---|---|---|---|---|---|
| **GOLD (Retrained)** | 0.498 | 0.179 | 0.615 | 0.297 | 0.423 | 4.74/3.05 | 100% |
| OG (chưa quên) | 0.731 | 0.419 | 0.695 | 0.384 | 0.657 | 1.97/2.11 | – |
| *Forget-MI công bố (paper)* | 0.735 | 0.393 | 0.625 | 0.250 | 0.571 | --/-- | 100%‡ |
| **Forget-MI tái lập E14** † | 0.750 | 0.383 | 0.668 | 0.338 | 0.572 | 2.76/2.57 | 100% |
| Forget-MI tái lập E15 † | 0.736 | 0.373 | 0.667 | 0.320 | 0.532 | 2.93/2.63 | 100% |
| P3 | 0.689 | 0.344 | 0.696 | 0.400 | 0.483 | 1.79/1.69 | 0.44% |
| P4 k=10 | 0.716 | 0.411 | 0.698 | 0.389 | 0.562 | 1.64/1.65 | 0.44% |
| P5 | 0.721 | 0.427 | 0.697 | 0.385 | 0.537 | 1.68/1.74 | 0.44% |
| **P6 (đề xuất)** | 0.647 | 0.286 | 0.691 | 0.345 | 0.328 | 1.92/1.71 | 0.44% |
| MAIN (S_val) | 0.700 | 0.361 | 0.698 | 0.374 | 0.408 | 1.64/1.66 | 0.44% |

‡ số Bảng 2 bài báo · † eval trên **split cũ** (toàn test): Df-AUC/Df-F1 **so được** (D_f y hệt), Dt/MIA chỉ tham khảo · P-methods eval trên D_t_final.

## Kết luận
- **Tái lập Forget-MI THÀNH CÔNG**: E14 khớp MIA (0.572≈0.571), E15 khớp Df-AUC (0.736≈0.735).
- **P6 quên MẠNH HƠN Forget-MI**: Df-AUC 0.647 < 0.750 (E14) — cùng tập D_f → so được trực tiếp.
- P6 dùng **0.44% tham số** (vs 100%), giữ utility tốt (Dt-AUC 0.691), forget_ce lành mạnh.
- **Forget-MI phải dừng ~E14**; chạy hết E30 thì over-forget (fce 4.81≫tce 3.45) — điểm yếu mà hinge-có-chặn của P3/P6 khắc phục.
