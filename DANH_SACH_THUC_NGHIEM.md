# Danh sách thực nghiệm đã chạy

> Mỗi khi user gửi kết quả, thêm TÊN thực nghiệm đó vào đây. Nhóm theo **batch (đợt chạy)**.
> Ký hiệu: ✅ số dùng được · ⚠️ có lưu ý · ❌ bỏ (bug).

---

## Batch 1 — 2026-07-21/22 · code CŨ (dính bug)
**Bug của batch này:** `S_val=nan` từ ~E3–E11 → **`selected` chọn nhầm E1 (chưa train xong) ❌** ·
`UU/MU=0` (quên biểu diễn chưa hoạt động) → **`last(E30)` là số thật nhưng CHƯA phải bản cuối ⚠️** ·
**references (OG/RE) ✅ đúng** (chỉ eval model cố định, không dính bug).

3% · seed 42 · split test_final (75%):

- ⚠️ `p3_3per_s42`
- ⚠️ `p4_k5_3per_s42`
- ⚠️ `p4_k10_3per_s42`
- ⚠️ `p4_k15_3per_s42`
- ⚠️ `p4_k20_3per_s42`
- ⚠️ `p5_3per_s42`
- ⚠️ `p6_3per_s42`
- ⚠️ `main_sval_3per_s42`
- ⚠️ `main_valce_3per_s42`
- ✅ `reference_og_3per` (mốc vàng OG)
- ✅ `reference_re_3per` (mốc vàng GOLD)

---

## Batch 2 — 2026-07-22 · code ĐÃ FIX (nan + UU/MU per-sample + BatchNorm eval)
**Trạng thái:** `S_val` nan = **0/30** ✅ · UU/MU **kích hoạt thật** (act E1≈0.94, `d_u` 2.45→7.99 vượt
trần 5.82 rồi tự dừng) ✅ · BN đóng băng eval ✅ → **số dùng được**.

3% · seed 42 (acc1):
- ✅ `p6_3per_s42` — **tốt nhất hiện tại** (last E30: Forget_AUC 0.647, MIA_p 0.286)
- ✅ `main_sval_3per_s42` (selected E17)
- ✅ `main_valce_3per_s42` (selected E11) — ablation selector
- ❌ `main_sval_noUUMU_3per_s42` — **HỎNG: trùng hệt `main_sval`** (cờ tắt UU/MU không ăn vì
  main lấy trọng số từ P5 preset, bỏ qua `lambda_uu/mu`). Đã sửa bằng `ablate_uu_mu` → **chạy lại**.

3% · seed 42 (acc2):
- ✅ `p5_3per_s42`
- ✅ `p4_k5_3per_s42`
- ✅ `p4_k20_3per_s42`

3% · seed 42 (acc3):
- ✅ `p3_3per_s42`
- ✅ `p4_k10_3per_s42`
- ✅ `p4_k15_3per_s42`

---

## ⚠️ PHÁT HIỆN QUAN TRỌNG — bản "Forget-MI tái lập" cũ KHÔNG phải Forget-MI

Số cũ `baseline_mimic_3per` (`experiments/results_summary_kaggle.csv`, 2026-06-30):
`MIA 1.000 · Df-AUC 0.561 · Df-F1 0.156 · Dt-AUC 0.632 · Dt-F1 0.234`.

Đối chiếu **Bảng 2** bài báo (3%) → khớp nhất với biến thể **"No Noise"**
(`1.000 / 0.534 / 0.116 / 0.508 / 0.154`, sai lệch 0.271) — biến thể mà **bài báo nói cho kết quả
TỆ NHẤT** (*"Noiseless experiments resulted in the poorest outcomes"*).

**Forget-MI thật @3% (Unimodal) = `MIA 0.571 · Df-AUC 0.735 · Df-F1 0.393 · Dt-AUC 0.625 · Dt-F1 0.250`.**

→ Hệ quả: **0.735 quên NHẸ HƠN P6 (0.647) của ta.** Mọi kết luận kiểu "Forget-MI sát gold hơn ta"
dựa trên 0.561 đều **PHẢI LẬT LẠI** sau khi chạy 2 notebook dưới.

### Còn thiếu
- [ ] `run_baseline_forgetmi_kaggle.ipynb` — Forget-MI **theo code gốc** (kỳ vọng lại ra ~0.56)
- [ ] `run_forgetmi_paperfix_kaggle.ipynb` — Forget-MI **sửa theo bài báo** (kỳ vọng ~0.735)
- [ ] `main_sval_noUUMU` chạy lại bằng cờ `ablate_uu_mu`
- [ ] Dựng lại bảng tổng hợp SAU khi có 2 số trên (bảng cũ đã xoá vì dựa trên số sai)
