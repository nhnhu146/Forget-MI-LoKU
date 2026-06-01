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
- Auto-tracker module: [../scripts/exp_tracker.py](../scripts/exp_tracker.py)
- Báo cáo tổng: [../EXPERIMENT_REPORT.md](../EXPERIMENT_REPORT.md)
- Code chính: [../training/forgetmi_loku.py](../training/forgetmi_loku.py)
- Config: [../config.yaml](../config.yaml)
- Notebook: [../run.ipynb](../run.ipynb)
