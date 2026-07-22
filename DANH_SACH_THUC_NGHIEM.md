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

## Batch 2 — (chờ) · code ĐÃ FIX (nan + UU/MU per-sample)
*(chưa có — sẽ thêm khi user gửi)*
