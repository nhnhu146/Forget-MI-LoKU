# Experiments Index

Master index của tất cả thí nghiệm. Mỗi dòng ↔ 1 file `exp_NNN_*.md`.

> **Workflow tự động** (xem chi tiết ở phần header của [run.ipynb](../run.ipynb)):
> 1. Sửa `config.yaml` → push code
> 2. Trên Colab: đổi `EXP_NAME` + `HYPOTHESIS` ở đầu Cell 4
> 3. Chạy Cell 4 (auto train + tạo file MD + cập nhật bảng dưới)
> 4. Chạy Cell 5 (auto push lên GitHub)
> 5. Trên local: `git pull`, điền 3 section thủ công vào file MD

---

## Tổng quan kết quả

Sắp xếp theo thứ tự thời gian (mới nhất ở dưới). Auto-tracker sẽ tự append dòng mới.

| # | Tên | Date | Status | MIA ↓ | Df AUC ↓ | Df F1 ↓ | Dt AUC ↑ | Dt F1 ↑ | Time(h) | Trainable% | Kết luận ngắn |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 001 | loku_v1_baseline | 2025-05-30 | ✅ | 0.552 | 0.829 | 0.597 | 0.674 | 0.387 | 0.094 | 0.389% | Speed/utility tốt hơn paper; forget yếu hơn |
| 002 | forget_margin_20_no_anchor | 2026-05-30 | 🔄 | 0.552 | 0.829 | 0.597 | 0.674 | 0.387 | 0.092 | 0.389% | _(auto, cần kết luận)_ |
| 003 | exp03_classifier_unfrozen_neggrad | 2026-05-30 | 🔄 | 0.557 | 0.830 | 0.597 | 0.669 | 0.386 | 0.094 | 0.395% | _(auto, cần kết luận)_ |
| 004 | exp04_aggressive_neggrad | 2026-05-30 | 🔄 | 0.687 | 0.809 | 0.478 | 0.672 | 0.327 | 0.271 | 0.395% | _(auto, cần kết luận)_ |
| 005 | exp05_calibrated_neggrad | 2026-05-30 | 🔄 | 0.607 | 0.832 | 0.631 | 0.667 | 0.399 | 0.090 | 0.395% | _(auto, cần kết luận)_ |
| 006 | exp06_uniform_prior | 2026-06-01 | 🔄 | 0.672 | 0.839 | 0.654 | 0.669 | 0.387 | 0.222 | 0.395% | _(auto, cần kết luận)_ |
| 007 | exp07_teacher_distillation | 2026-06-01 | 🔄 | 0.657 | 0.819 | 0.497 | 0.687 | 0.324 | 0.232 | 0.395% | _(auto, cần kết luận)_ |
| 008 | exp08_true_loku_fila_subtraction | 2026-06-01 | 🔄 | 0.562 | 0.833 | 0.589 | 0.679 | 0.388 | 0.139 | 0.395% | _(auto, cần kết luận)_ |
| 009 | exp09_loku_fila_with_ihl | 2026-06-01 | 🔄 | 0.537 | 0.832 | 0.593 | 0.678 | 0.385 | 0.189 | 0.395% | _(auto, cần kết luận)_ |
| 011 | exp10c_image_fila_distill_forget | 2026-06-02 | 🔄 | 0.493 | 0.718 | 0.271 | 0.689 | 0.330 | 0.177 | 0.451% | _(auto, cần kết luận)_ |
| 012 | exp10c_image_fila_distill_forget_seed42 | 2026-06-02 | 🔄 | 0.493 | 0.718 | 0.271 | 0.689 | 0.330 | 0.182 | 0.451% | _(auto, cần kết luận)_ |
| 013 | exp10c_image_fila_distill_forget_seed123 | 2026-06-02 | 🔄 | 0.473 | 0.700 | 0.307 | 0.673 | 0.317 | 0.184 | 0.451% | _(auto, cần kết luận)_ |
| 014 | exp10c_image_fila_distill_forget_seed7 | 2026-06-02 | 🔄 | 0.403 | 0.686 | 0.273 | 0.675 | 0.348 | 0.185 | 0.451% | _(auto, cần kết luận)_ |
| 015 | exp11_no_fre_ihl | 2026-06-02 | 🔄 | 0.333 | 0.687 | 0.308 | 0.673 | 0.380 | 0.179 | 0.451% | _(auto, cần kết luận)_ |
| 016 | exp11c_ihl_sweep_ihl050 | 2026-06-03 | 🔄 | 0.507 | 0.758 | 0.470 | 0.679 | 0.368 | 0.170 | 0.451% | _(auto, cần kết luận)_ |
| 017 | exp11c_ihl_sweep_ihl075 | 2026-06-03 | 🔄 | 0.453 | 0.734 | 0.379 | 0.677 | 0.363 | 0.173 | 0.451% | _(auto, cần kết luận)_ |
| 018 | exp11c_ihl_sweep_ihl100 | 2026-06-03 | 🔄 | 0.483 | 0.767 | 0.456 | 0.681 | 0.352 | 0.110 | 0.451% | _(auto, cần kết luận)_ |

**Reference (paper Forget-MI, 3%)**: MIA=0.571 | Df_AUC=0.735 | Df_F1=0.393 | Dt_AUC=0.625 | Dt_F1=0.250 | Time=5h | Trainable=100%

**Gold standard (retrained F_re, 3%)**: MIA=0.000 | Df_AUC=0.566 | Df_F1=0.310 | Dt_AUC=0.626 | Dt_F1=0.362

**Original (F_og, before unlearn)**: MIA=1.000 | Df_AUC=0.999 | Df_F1=0.965 | Dt_AUC=0.677 | Dt_F1=0.388

---

## Roadmap (kế hoạch thí nghiệm)

Tick khi xong:

- [ ] **Exp 001** — `loku_v1_baseline` — Run đầu tiên với code đã fix
- [ ] **Exp 002** — `forget_margin_20` — Tăng forget_margin 8→20 để kích hoạt L_MD
- [ ] **Exp 003** — `no_re_anchor` — Tắt eta_re_anchor để forget mạnh hơn
- [ ] **Exp 004** — `neggrad` — Thêm gradient ascent trên forget set
- [ ] **Exp 005** — `lora_r16` — Tăng rank LoRA + mở rộng target_modules
- [ ] **Exp 006** — `two_stage` — Aggressive forget → restore utility
- [ ] **Exp 007** — Lặp exp tốt nhất với seed [0, 123, 7] để có mean±std
- [ ] **Exp 008** — Chạy trên forget 6% với config tốt nhất
- [ ] **Exp 009** — Chạy trên forget 10%
- [ ] **Exp 010** — Ablation: LoRA only (không FIM init)
- [ ] **Exp 011** — Ablation: FIM only (full FT + FIM mask)
- [ ] **Exp 012** — Reproduce Forget-MI baseline trên cùng môi trường

---

## Legend Status

| Icon | Meaning |
|---|---|
| 🔄 | Auto-tracked, chưa có Conclusion |
| ✅ | Done — kết quả OK |
| ⭐ | Done — best so far ở metric nào đó |
| ❌ | Abandoned |
| ⏸ | Paused |
| 🐛 | Có bug — kết quả không tin được |

---

## Quick links

- Template: [_TEMPLATE.md](_TEMPLATE.md)
- **Nhật ký thay đổi code lớn**: [../CHANGELOG.md](../CHANGELOG.md)
- Auto-tracker module: [../scripts/exp_tracker.py](../scripts/exp_tracker.py)
- Báo cáo tổng: [../EXPERIMENT_REPORT.md](../EXPERIMENT_REPORT.md)
- Code chính: [../training/forgetmi_loku.py](../training/forgetmi_loku.py)
- Config: [../config.yaml](../config.yaml)
- Notebook: [../run.ipynb](../run.ipynb)
