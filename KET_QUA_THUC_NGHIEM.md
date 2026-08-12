# Kết quả thực nghiệm — Forget-MI vs P3-NoKD-More

Tổng hợp toàn bộ run đã hoàn thành. Sinh lại bằng:

```
python tools/build_tables.py "D:\Run_KLTN\2.8" --out-md bang.md
```

**Trạng thái: 20 run** (12 run của danh sách tối thiểu + `p3_iu_lr2e4_s42` +
`p3_m3_lr1e4_s42` + 3 run NegGrad+ + 3 run CF-k). Xem mục Trạng thái ở cuối.

---

## 0. Điều kiện thực nghiệm

| Hạng mục | Giá trị |
|---|---|
| GPU | NVIDIA Tesla T4 (mọi run) |
| CUDA / PyTorch | 12.8 / 2.10.0+cu128 |
| Seed | 42 |
| Số epoch | 30 |
| Số optimizer update | 30 (tích lũy gradient cả epoch, cập nhật 1 lần cuối epoch) |
| Batch size | 16 |
| Learning rate | Forget-MI 1e-5 (full-FT) · P3 2e-4 (LoRA) |
| Dataloader | `num_workers=2`, `pin_memory=True` (giống nhau hai phương pháp) |
| Precision | vòng train FP32, không GradScaler; autocast chỉ ở eval/monitor |

### Ký hiệu

- **S2** = selector *Closest CE* (gold-free): chọn epoch có `|forget_CE − nm_val_CE|` nhỏ nhất.
  Epoch ghi trong bảng đã quy về đếm-từ-1.
- **E30** = checkpoint cuối cùng.
- `—` = chưa chạy. (Bảng từng có `n/a` ở Df-AUC/Dt-AUC của P3 trên IU; nguyên nhân là
  một lỗi trong hàm tính AUC, đã sửa và điền số thật — xem Phụ lục C mục 6.)
- Ở hàng **S2**, cột cuối là **nm_val-CE** (đại lượng selector dùng), không phải Test-CE.
- **Giao thức MIA** (thống nhất cho mọi hàng của cả hai phương pháp): member = `D_r` lấy
  mẫu còn **512** (`eval_max_retain`, seed 42), non-member = `D_t_final`, mục tiêu tấn
  công = `D_f`. `MIA` dùng CE theo từng mẫu và cân bằng 1:1; `MIA_paper` dùng CE trung
  bình theo lô và **không** cân bằng — nên `MIA_paper` nhạy với kích thước tập member.

### Công thức thời gian lõi

```
T_core(Forget-MI) = T_train
T_core(P3)        = T_Fisher + T_FILA + T_train
```

Thời gian chọn checkpoint và đánh giá cuối được đo riêng, **không** tính vào `T_core`
vì chúng thuộc giao thức thực nghiệm chứ không phải bản thân thuật toán gỡ bỏ.
GPU peak của lõi lấy **MAX** giữa các giai đoạn, không cộng dồn.

---

## Bảng 1 — MIMIC-CXR 3%

| Mô hình | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| θ_og (gốc) | – | – | 0.731 | 0.419 | 0.695 | 0.384 | 0.657 | 0.714 | 1.966 | 2.108 |
| θ_re (gold) | – | – | 0.498 | 0.179 | 0.615 | 0.297 | 0.423 | 0.000 | 4.736 | 3.046 |
| Forget-MI | S2 | E7 | 0.734 | 0.345 | 0.668 | 0.319 | 0.657 | 0.429 | 2.979 | 2.975 |
| Forget-MI | E30 | 30 | 0.623 | 0.278 | 0.648 | 0.307 | 0.627 | 0.286 | 3.578 | 2.906 |
| **P3-NoKD-More** | S2 | E30 | 0.682 | 0.365 | **0.704** | **0.394** | **0.552** | **0.286** | 2.067 | 2.073 |
| **P3-NoKD-More** | E30 | 30 | 0.683 | 0.365 | 0.704 | 0.394 | 0.552 | 0.286 | 2.067 | 1.829 |
| † P3 lr 1e-4 | E30 | 30 | 0.707 | 0.405 | 0.703 | 0.388 | 0.562 | 0.429 | 1.887 | 1.812 |

† = run kiểm tra độ nhạy learning rate, **không** phải kết quả chính. Xem mục ngay dưới.

| Mô hình | Tham số cập nhật | Tỉ lệ | T_Fisher | T_FILA | T_train | **T_core** | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forget-MI | 113 238 164 | 100 % | 0 | 0 | 2 172,5 s | **2 172,5 s** | 6,92 GB | 7,25 GB |
| P3-NoKD-More | 1 451 008 | **1,27 %** | 270,9 s | 0,9 s | 1 592,7 s | **1 864,5 s** | 13,92 GB | 14,08 GB |
| † P3 lr 1e-4 | 1 451 008 | 1,27 % | 267,9 s | 0,8 s | 1 560,1 s | **1 828,8 s** | 13,92 GB | – |

**Nhận xét.** P3 tốt hơn ở trục riêng tư theo `MIA` (0.552 vs 0.627 tại E30; 0.552 vs
0.657 tại S2) nhưng **hoà** ở `MIA_paper` (0.286 vs 0.286 tại E30). Tiện ích cao hơn rõ
(Dt-AUC 0.704 vs 0.648, còn cao hơn cả θ_og 0.695) và Test-CE thấp hơn (1.829 vs 2.906).
Đổi lại Df-AUC xa gold hơn (0.683 vs 0.623 — gold 0.498), tức quên ít hơn.

S2 của Forget-MI rơi vào **E7** vì CE cắt nhau rất sớm, lúc đó mô hình gần như chưa quên
gì (Df-AUC 0.734 còn cao hơn θ_og 0.731, MIA 0.657 = θ_og). Từ E8 trở đi Forget-MI
over-forget nhanh.

S2 của P3 rơi vào **E30**, nhưng **không phải vì phát hiện điểm cắt**: `ce_gap` của P3
âm suốt 30 epoch (−0.322 → −0.0055), không bao giờ vượt 0, nên S1/S3 không xác định và
S2 lấy epoch có |gap| nhỏ nhất = epoch cuối. Vì vậy hàng S2 và hàng E30 trùng số nhau.

### Độ nhạy learning rate (run bổ sung `p3_m3_lr1e4_s42`)

Learning rate của hai phương pháp khác nhau **theo thiết kế** — Forget-MI 1e-5 (chỉnh
toàn bộ 113 triệu tham số), P3 2e-4 (chỉnh 1,45 triệu tham số LoRA) — và **không bên nào
được sweep**: mỗi bên lấy giá trị từ nguồn của chính nó. Để trả lời câu "P3 thắng ở 3 %
có phải nhờ lr may mắn không", đã chạy lại P3 ở **1e-4** (một nửa), giữ nguyên mọi thứ
khác.

| tại E30 | P3 lr 2e-4 (chính) | † P3 lr 1e-4 | Forget-MI | gold |
|---|---:|---:|---:|---:|
| Dt-AUC ↑ | 0.704 | **0.703** | 0.648 | 0.615 |
| Dt-F1 ↑ | 0.394 | **0.388** | 0.307 | 0.297 |
| Test-CE | 1.829 | **1.812** | 2.906 | 3.046 |
| MIA ↓ | 0.552 | **0.562** | 0.627 | 0.423 |
| MIA_paper ↓ | 0.286 | **0.429** | 0.286 | 0.000 |
| Df-AUC ↓ | 0.683 | **0.707** | 0.623 | 0.498 |
| Forget-CE | 2.067 | **1.887** | 3.578 | 4.736 |

**Kết luận không nhạy learning rate.** Trục tiện ích gần như trùng khít (Dt-AUC 0.703 vs
0.704, chênh 0.001). `MIA` chênh 0.010 — nhỏ hơn nhiễu của phép đo cả một bậc (±0.07). Xu hướng duy nhất thấy được đúng như trực giác: lr thấp hơn thì quên ít hơn
(forget-CE 1.887 vs 2.067, Df-AUC 0.707 vs 0.683 — xa gold 0.498 hơn), nhưng mức thay đổi
nhỏ. Ưu thế của P3 so với Forget-MI ở mức 3 % **giữ nguyên ở cả hai learning rate**.

⚠️ **Ngoại lệ:** `MIA_paper` nhảy từ 0.286 sang
0.429, tức từ "hòa với Forget-MI" thành "thua Forget-MI". Nhưng ở mức 3 % thì
`|D_f| ≈ 200` chia 32 chỉ được **7 lô**, nên 0.286 = 2/7 và 0.429 = 3/7 — khác biệt đúng
bằng **một lô đổi phía**, tức bước lượng tử nhỏ nhất mà chỉ số này có thể biểu diễn.

⇒ **Không được dùng `MIA_paper` để kết luận ở mức 3 %.** Bằng chứng riêng tư ở 3 % phải
dựa vào `MIA` (0.552 và 0.562, đều thấp hơn Forget-MI 0.627 ở cả hai lr) chứ không phải
`MIA_paper`. Đây cũng là lý do phải công bố cỡ tập cùng với chỉ số.

---

## Bảng 2 — MIMIC-CXR 6%

| Mô hình | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| θ_og (gốc) | – | – | 0.730 | 0.434 | 0.695 | 0.384 | 0.736 | 0.769 | 1.755 | 2.108 |
| θ_re (gold) | – | – | 0.596 | 0.339 | 0.665 | 0.381 | 0.613 | 0.462 | 2.450 | 2.191 |
| Forget-MI | S2 | E14 | 0.702 | 0.330 | 0.658 | 0.291 | **0.357** | 0.385 | 3.019 | 3.022 |
| Forget-MI | E30 | 30 | 0.633 | 0.296 | 0.641 | 0.296 | 0.773 | 0.538 | 3.142 | 3.088 |
| **P3-NoKD-More** | S2 | E30 | 0.670 | 0.353 | **0.689** | **0.370** | 0.706 | **0.231** | 2.087 | 2.087 |
| **P3-NoKD-More** | E30 | 30 | 0.670 | 0.353 | 0.689 | 0.370 | 0.706 | 0.231 | 2.087 | 1.895 |

| Mô hình | Tham số cập nhật | Tỉ lệ | T_Fisher | T_FILA | T_train | **T_core** | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forget-MI | 113 238 164 | 100 % | 0 | 0 | 3 995,5 s | **3 995,5 s** | 6,92 GB | 7,20 GB |
| P3-NoKD-More | 1 451 008 | **1,27 %** | 286,9 s | 0,9 s | 3 102,4 s | **3 390,2 s** | 13,92 GB | 14,20 GB |

**Nhận xét.** Kết luận ở mức 6% **phụ thuộc chốt checkpoint**, cần nói rõ:

- **Tại S2**: MIA của Forget-MI xuống 0.357, **thấp hơn cả gold 0.613**. Đây **không phải
  ưu thế riêng tư** mà là dấu hiệu **quên quá đà**: forget-CE của Forget-MI tại E14 là
  3.019 so với gold 2.450, tức `D_f` bị đẩy ra xa hơn cả một mô hình chưa từng thấy nó,
  nên bộ tấn công đoán "non-member". Thêm nữa 0.357 chỉ cách 0.5 khoảng một độ lệch chuẩn
  của chính phép đo (±0.15). `MIA_paper` lại ngược: P3 0.231 vs 0.385.
- **Tại E30**: P3 tốt hơn ở **cả hai** chỉ số (MIA 0.706 vs 0.773; MIA_paper 0.231 vs
  0.538).

Điểm E14 mà S2 chọn cho Forget-MI rơi đúng vào lúc MIA của nó chạm đáy; đến E30 thì
MIA vọt lên 0.773, cao hơn cả θ_og 0.736. Đây là biểu hiện mất ổn định của Forget-MI
theo epoch chứ không phải một ưu thế bền vững.

P3 giữ tiện ích tốt hơn ở mọi chốt (Dt-AUC 0.689 vs 0.641–0.658) và Test-CE thấp nhất
bảng (1.895).

---

## Bảng 3 — MIMIC-CXR 10%

| Mô hình | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| θ_og (gốc) | – | – | 0.757 | 0.469 | 0.695 | 0.384 | 0.717 | 0.682 | 1.662 | 2.108 |
| θ_re (gold) | – | – | 0.547 | 0.274 | 0.607 | 0.338 | 0.364 | 0.227 | 3.264 | 2.881 |
| Forget-MI | S2 | E30 | **0.704** | **0.349** | 0.653 | 0.308 | **0.616** | **0.455** | 2.549 | 2.898 |
| Forget-MI | E30 | 30 | 0.704 | 0.349 | 0.653 | 0.308 | 0.616 | 0.455 | 2.549 | 2.906 |
| P3-NoKD-More | S2 | E30 | 0.728 | 0.417 | **0.692** | **0.384** | 0.730 | 0.591 | 1.683 | 2.109 |
| P3-NoKD-More | E30 | 30 | 0.728 | 0.417 | 0.692 | 0.384 | 0.731 | 0.591 | 1.683 | 1.864 |

| Mô hình | Tham số cập nhật | Tỉ lệ | T_Fisher | T_FILA | T_train | **T_core** | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forget-MI | 113 238 164 | 100 % | 0 | 0 | 6 403,3 s | **6 403,3 s** | 6,92 GB | 7,17 GB |
| P3-NoKD-More | 1 451 008 | **1,27 %** | 294,2 s | 0,9 s | 5 447,5 s | **5 742,5 s** | 13,92 GB | 14,08 GB |

**Nhận xét — ở mức 10 % P3 gần như KHÔNG quên được.** forget-CE chỉ đi từ 1.662 (θ_og)
lên **1.683**, trong khi khoảng cách tới gold là 3.264 — dịch được 0.021 trên tổng 1.6.
Forget-MI ở cùng mức thì **có** quên (forget-CE 2.549, Df-AUC 0.704).

Forget-MI tốt hơn ở trục quên và riêng tư: Df-AUC 0.704 vs 0.728, Df-F1 0.349 vs 0.417,
MIA_paper 0.455 vs 0.591.

⚠️ Riêng `MIA` (0.616 vs 0.730) thì **không kết luận được**: nhiễu của chính phép đo ở
kích thước tập này là ±0.07. Cũng **không** được viết "MIA của P3
(0.730) còn cao hơn θ_og (0.717)" như một phát hiện — chênh 0.013 nhỏ hơn nhiễu nhiều
lần. Phát biểu đúng là **"không phân biệt được với θ_og"**, và đó là hệ quả trực tiếp của
việc P3 để nguyên mô hình: đầu vào của phép đo (phân bố CE trên `D_f`) gần như không đổi
thì đầu ra cũng không đổi.

P3 chỉ thắng ở trục tiện ích (Dt-AUC 0.692 vs 0.653, Dt-F1 0.384 vs 0.308, Test-CE
1.864 vs 2.906) — nhưng đó là hệ quả trực tiếp của việc nó ít thay đổi mô hình.

Nguyên nhân nhìn thấy rõ ở quỹ đạo CE: forget-CE của P3 gần như **đứng yên** suốt 30
epoch (1.616 → 1.574 → 1.676 → 1.683), `ce_gap` chỉ dao động trong khoảng
[−0.645, −0.427] và không bao giờ cắt. Tập quên 10% lớn hơn nhiều nên 30 lần cập nhật
trên 1,27 % tham số không đủ để dịch chuyển mô hình.

Cả hai phương pháp đều không có điểm cắt CE ở mức 10% (S1/S3 không xác định), S2 lấy E30.

> **Cần ghi vào khóa luận:** khả năng quên của P3 suy giảm theo tỉ lệ quên — mạnh ở 3%,
> hòa/lẫn lộn ở 6%, thua rõ ở 10%. Đây là giới hạn của thiết kế hiệu-quả-tham-số với
> ngân sách cập nhật cố định, không phải lỗi cài đặt.

---

## Bảng 4 — Ablation pipeline, MIMIC-CXR 3%

Mọi biến thể giữ nguyên toàn bộ cấu hình P3-NoKD-More, chỉ bỏ đúng một thành phần.

| Biến thể | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **P3 đầy đủ** | S2 | E30 | 0.682 | 0.365 | 0.704 | 0.394 | 0.552 | 0.286 | 2.067 | 2.073 |
| **P3 đầy đủ** | E30 | 30 | 0.683 | 0.365 | 0.704 | 0.394 | 0.552 | 0.286 | 2.067 | 1.829 |
| w/o Fisher/FILA | S2 | E30 | 0.722 | 0.406 | 0.701 | 0.393 | 0.522 | 0.429 | 1.825 | 2.098 |
| w/o Fisher/FILA | E30 | 30 | 0.722 | 0.406 | 0.701 | 0.393 | 0.522 | 0.429 | 1.825 | 1.851 |
| w/o IHL | S2 | E10 | 0.742 | 0.499 | 0.707 | 0.381 | 0.602 | 0.429 | 1.681 | 1.990 |
| w/o IHL | E30 | 30 | 0.742 | 0.488 | 0.708 | 0.384 | 0.572 | 0.429 | 1.698 | 1.773 |
| w/o MU/MR | S2 | E9 | 0.676 | 0.350 | 0.700 | 0.389 | 0.592 | 0.286 | 1.892 | 1.901 |
| w/o MU/MR | E30 | 30 | 0.638 | 0.262 | 0.700 | 0.409 | 0.418 | 0.143 | 2.495 | 1.850 |

| Biến thể | Tham số | Tỉ lệ | T_Fisher | T_FILA | T_train | **T_core** | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| P3 đầy đủ | 1 451 008 | 1,27 % | 260,7 s | 0,9 s | 1 644,9 s | **1 906,4 s** | 13,92 GB | 14,08 GB |
| w/o Fisher/FILA | 1 451 008 | 1,27 % | **0,0 s** | 0,1 s | 1 660,4 s | **1 660,5 s** | 13,27 GB | 13,32 GB |
| w/o IHL | 1 451 008 | 1,27 % | 252,9 s | 0,9 s | 1 650,5 s | **1 904,3 s** | 13,92 GB | 14,08 GB |
| w/o MU/MR | 1 451 008 | 1,27 % | 257,5 s | 0,8 s | 1 623,8 s | **1 882,2 s** | 13,92 GB | 14,08 GB |

**Nhận xét từng thành phần.**

*Fisher/FILA* — bỏ đi thì Df-AUC tăng từ 0.683 lên 0.722 (xa gold 0.498 hơn) và
MIA_paper xấu đi rõ (0.286 → 0.429), đổi lại tiết kiệm 246 s (12,9 % thời gian lõi) và
0,65 GB GPU. Tức khởi tạo dẫn hướng bằng Fisher **có đóng góp vào chất lượng quên**, với
chi phí tiền xử lý xác định được.

*IHL* — bỏ đi cho kết quả **tệ nhất** trên trục quên: Df-AUC 0.742, Df-F1 0.488 (bản đầy
đủ 0.365), MIA_paper 0.429. Chi phí gần như không đổi. IHL là thành phần đóng góp mạnh
nhất cho khả năng quên.

*MU/MR* — bỏ đi thì tại E30 **quên nhiều hơn** bản đầy đủ (Df-AUC 0.638, MIA 0.418,
MIA_paper 0.143) và giữ nguyên tiện ích (Dt-AUC 0.700). Nhưng forget-CE vọt lên 2.495
trong khi nm_val-CE chỉ 1.850, tức bắt đầu over-forget; S2 vì thế chốt sớm ở E9.

> **Lưu ý bắt buộc ghi khi trích Bảng 4:** biến thể `w/o MU/MR` đặt w_MU = w_MR = 0 và
> **không chuẩn hóa lại** trọng số, nên tổng trọng số khối Forget-MI của riêng biến thể
> này là **2/3** thay vì 1. Chuẩn hóa lại sẽ vừa bỏ MU/MR vừa nhân đôi UU/UR, tức đổi
> hai thứ cùng lúc và không còn là ablation sạch.

---

## Bảng 5 — IU Chest X-rays 3%

> **Cấu hình của hàng P3 đã đổi (2026-08-11).** Run IU đầu tiên nạp
> `config_loku_iu_kaggle.yaml` với **lr 5e-4 và grad_clip 1.0**, trong khi P3 trên MIMIC
> chạy **lr 2e-4 và grad_clip 0.0** — tức câu "dùng nguyên cấu hình khóa từ MIMIC" khi đó
> **không đúng với run thật**. Đã đối chiếu mọi khóa còn lại (`lora_r` 8 · `lora_alpha` 16
> · dropout 0.05 · `weight_decay` 0.01 · warmup 0.1 · batch 16 · Fisher 256/16 · splits
> 4/10): **giống hệt nhau**, chỉ lệch đúng hai khóa đó. Hàng chính nay là run đã khôi phục
> đúng cấu hình khóa (`p3_iu_lr2e4_s42`); run lr 5e-4 giữ lại làm hàng đối chứng độ nhạy.

| Mô hình | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| θ_og (gốc) | – | – | 1.000 | 1.000 | 0.676 | 0.622 | 0.990 | 1.000 | 0.002 | 2.174 |
| θ_re (gold) | – | – | 0.651 | 0.587 | 0.670 | 0.614 | 0.545 | 0.000 | 2.329 | 2.307 |
| Forget-MI | S2 | E30 | 0.854 | **0.616** | 0.635 | **0.533** | 0.581 | 0.167 | 3.019 | 3.565 |
| Forget-MI | E30 | 30 | 0.854 | 0.616 | 0.635 | 0.533 | 0.581 | 0.167 | 3.019 | 3.661 |
| **P3-NoKD-More** | S2 | E23 | 0.860 | 0.465 | **0.664** | 0.414 | 0.597 | 0.167 | 9.574 | 9.576 |
| **P3-NoKD-More** | S_val | E28 | 0.832 | 0.414 | 0.656 | 0.407 | 0.592 | 0.167 | 11.354 | 10.907 |
| **P3-NoKD-More** | E30 | 30 | 0.825 | 0.414 | 0.654 | 0.402 | **0.555** | 0.167 | 12.042 | 11.407 |
| ‡ P3 lr 5e-4 | S2 | E11 | 0.872 | 0.486 | **0.665** | 0.410 | 0.597 | 0.167 | 8.946 | 9.180 |
| ‡ P3 lr 5e-4 | E30 | 30 | **0.710** | 0.334 | 0.590 | 0.351 | 0.560 | **0.000** | 37.771 | 28.553 |

‡ = run cũ, cấu hình lệch MIMIC (lr 5e-4 + grad_clip 1.0). Giữ lại để đối chứng độ nhạy
learning rate, **không** dùng làm kết quả chính.

Hàng `S_val` là checkpoint do selector `S_val` chọn (E28, `S_val = 1.1309`) — **không
phải** S2, ghi thêm để đối chiếu. Rebuild checkpoint đó khớp bit-exact với model lúc chạy
(`selected_rebuild_maxdiff = 0.0`).

**Selector không cứu được gì ở đây.** S2 chốt E23, sớm hơn 7 epoch, nhưng forget-CE tại
đó vẫn là **9.57 so với gold 2.33** — vẫn gấp 4,1 lần. Ba chốt S2 / S_val / E30 chỉ khác
nhau về mức độ sụp đổ (9.57 → 11.35 → 12.04), không có chốt nào đưa mô hình về vùng lành
mạnh. So với run lr 5e-4 thì điểm cắt CE lùi rất xa: S1/S3 từ epoch 11 sang **epoch 23**.

| Mô hình | Tham số cập nhật | Tỉ lệ | T_Fisher | T_FILA | T_train | **T_core** | Peak alloc | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Forget-MI | 113 238 164 | 100 % | 0 | 0 | 1 882,6 s | **1 882,6 s** | 6,92 GB | 7,23 GB |
| P3-NoKD-More (lr 2e-4) | 1 451 008 | **1,27 %** | 250,4 s | 0,8 s | 1 282,3 s | **1 533,5 s** | 13,92 GB | – |
| ‡ P3 lr 5e-4 | 1 451 008 | 1,27 % | 259,3 s | 0,8 s | 1 273,5 s | **1 533,6 s** | 13,92 GB | 14,08 GB |

Đổi learning rate **không** đổi chi phí: `T_core` 1 533,5 s so với 1 533,6 s.

### Hạ learning rate KHÔNG cứu được IU

Đây là kết quả quan trọng nhất của run bổ sung. Với cấu hình đã khôi phục đúng như MIMIC,
forget-CE vẫn là **12.04 so với gold 2.33 — gấp 5,2 lần**. Giảm được so với 37.77 nhưng
vẫn là sụp đổ.

⇒ Nguyên nhân chính là **đặc tính dữ liệu IU, không phải learning rate**. Trước đây hai
nguyên nhân không tách rời được; nay đã tách: θ_og trên IU **thuộc lòng** tập quên
(forget-CE 0.002, Df-AUC 1.000), nên mọi áp lực quên ở ngân sách 30 epoch đều đẩy mô hình
quá đà. Và lần này câu "dùng nguyên cấu hình khóa từ MIMIC" **đúng với run thật**.

**Cạm bẫy phải nói rõ:** test-CE cũng nổ lên **11.41** (gold 2.31), tức mô hình hỏng trên
**cả tập kiểm thử** chứ không riêng tập quên. Tỉ số forget/test = 12.04/11.41 = **1.06**,
gần bằng tỉ số của gold (1.01) — theo tiêu chí "CE cân bằng" thì trông rất *lành mạnh*
trong khi thực chất là **suy giảm toàn cục**, không phải quên có chọn lọc. Chỉ nhìn tỉ số
CE là bị đánh lừa; phải nhìn **mức tuyệt đối** so với gold. Tỉ số của chính gold cũng không phải 1.0
(MIMIC 3 %: 1.55), nên selector S2 — vốn nhắm tỉ số 1.0 — không lấy gold làm mốc.

Cùng một ý ở trục hiệu năng: Dt-AUC 0.654 sát gold 0.670 (thứ hạng còn giữ) nhưng Dt-F1
chỉ 0.402 so với gold 0.614 (ngưỡng quyết định đã hỏng).

**So với Forget-MI tại E30 (cấu hình đúng):** P3 quên nhiều hơn chút (Df-AUC 0.825 vs
0.854, gold 0.651), riêng tư gần gold hơn (MIA 0.555 vs 0.581, gold 0.545), hòa ở
MIA_paper (0.167 — chỉ 6 lô nên bước nhỏ nhất là 0.167). Đổi lại cái giá lớn hơn hẳn:
CE 12.04/11.41 so với 3.02/3.66.

### (đối chứng) Run lr 5e-4 — vì sao vẫn giữ lại

Quỹ đạo CE của run lr 5e-4:

| Epoch | E1 | E6 | E11 (S2) | E12 (cắt) | E13 | E16 | E21 | E30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| forget-CE | 0.04 | 1.80 | 8.95 | 10.59 | 11.99 | 21.39 | 31.28 | **37.77** |
| nm_val-CE | 2.50 | 4.28 | 9.18 | 10.09 | 10.65 | 16.69 | 23.44 | **27.97** |

Gold chỉ ở mức 2.33 — tức forget-CE của run này cao gấp **16 lần** mức mà huấn luyện lại
từ đầu đạt được (bản lr 2e-4 là 5,2 lần).

So sánh hai learning rate ở E30 cho thấy lr **có** tác động, nhưng không phải nguyên nhân:

| | gold | P3 lr 2e-4 | P3 lr 5e-4 |
|---|---:|---:|---:|
| forget-CE | 2.33 | 12.04 (5,2×) | 37.77 (16×) |
| test-CE | 2.31 | 11.41 (4,9×) | 28.55 (12×) |
| Dt-AUC | 0.670 | 0.654 | 0.590 |
| Dt-F1 | 0.614 | 0.402 | 0.351 |
| Df-AUC | 0.651 | 0.825 | 0.710 |

Hạ lr kéo được mức sụp đổ xuống một phần ba và giữ Dt-AUC sát gold, nhưng **không** đưa
được CE về vùng của gold. Đổi lại, quên ít đi (Df-AUC 0.825 xa gold hơn 0.710).

S2 của Forget-MI trên IU chốt đúng **E30** nên hai hàng của nó trùng nhau (khác biệt duy
nhất ở cột cuối là do hàng S2 ghi nm_val-CE 3.565 còn hàng E30 ghi Test-CE 3.661).

**Tóm lại cho IU:** với cấu hình đúng (lr 2e-4), tại E30 P3 quên nhỉnh hơn Forget-MI (Df-AUC 0.825
vs 0.854) và gần gold hơn ở MIA (0.555 vs 0.581, gold 0.545), nhưng phá mô hình nặng hơn
nhiều (CE 12.04/11.41 vs 3.02/3.66; Dt-F1 0.402 vs 0.533). **Đây là hạn chế phải đưa vào
khóa luận**, và nay đã có bằng chứng nguyên nhân là dữ liệu chứ không phải siêu tham số.

---

## Bảng 6 — Các baseline gỡ bỏ khác trên MIMIC-CXR

Hai phương pháp tham chiếu ngoài Forget-MI, chạy trên **cùng pipeline / cùng data split /
cùng giao thức MIA** nên so được trực tiếp với Bảng 1–3.

- **NegGrad+** — tinh chỉnh trên `D_r` đồng thời đảo gradient trên `D_f`
  (`loss = CE(retain) − λ·CE(forget)`, λ = |D_f|/|D_r|). Đại diện nhóm *modality-agnostic*.
- **CF-k** — đóng băng 2 tầng đầu của mỗi encoder, tinh chỉnh phần còn lại trên `D_r`.
  Đại diện nhóm *hạn chế tham số được cập nhật* — cùng họ với phương pháp đề xuất.

| Mức quên | Phương pháp | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE | 1−Sim |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 % | θ_og (gốc) | 0.731 | 0.419 | 0.695 | 0.384 | 0.657 | 0.714 | 1.966 | 2.108 | – |
| 3 % | θ_re (gold) | 0.498 | 0.179 | 0.615 | 0.297 | 0.423 | 0.000 | 4.736 | 3.046 | – |
| 3 % | Forget-MI | 0.623 | 0.278 | 0.648 | 0.307 | 0.627 | 0.286 | 3.578 | 2.906 | – |
| 3 % | P3-NoKD-More | 0.683 | 0.365 | **0.704** | **0.394** | 0.552 | 0.286 | 2.067 | 1.829 | – |
| 3 % | **NegGrad+** | **0.162** | 0.000 | 0.615 | 0.316 | 0.000 | 0.000 | **129.45** | **9.56** | 0.453 |
| 3 % | **CF-k** | **0.912** | 0.701 | 0.669 | 0.380 | 0.637 | 0.143 | 1.488 | 4.165 | 0.453 |
| 6 % | θ_og (gốc) | 0.730 | 0.434 | 0.695 | 0.384 | 0.736 | 0.769 | 1.755 | 2.108 | – |
| 6 % | θ_re (gold) | 0.596 | 0.339 | 0.665 | 0.381 | 0.613 | 0.462 | 2.450 | 2.191 | – |
| 6 % | Forget-MI | 0.633 | 0.296 | 0.641 | 0.296 | 0.773 | 0.538 | 3.142 | 3.088 | – |
| 6 % | P3-NoKD-More | 0.670 | 0.353 | **0.689** | 0.370 | 0.706 | 0.231 | 2.087 | 1.895 | – |
| 6 % | **NegGrad+** | **0.102** | 0.000 | 0.592 | 0.318 | 0.000 | 0.000 | **114.59** | **10.81** | 0.582 |
| 6 % | **CF-k** | **0.927** | 0.771 | 0.676 | **0.403** | 0.663 | 0.615 | 1.130 | 3.923 | 0.534 |
| 10 % | θ_og (gốc) | 0.757 | 0.469 | 0.695 | 0.384 | 0.717 | 0.682 | 1.662 | 2.108 | – |
| 10 % | θ_re (gold) | 0.547 | 0.274 | 0.607 | 0.338 | 0.364 | 0.227 | 3.264 | 2.881 | – |
| 10 % | Forget-MI | 0.704 | 0.349 | 0.653 | 0.308 | 0.616 | 0.455 | 2.549 | 2.906 | – |
| 10 % | P3-NoKD-More | 0.728 | 0.417 | **0.692** | **0.384** | 0.731 | 0.591 | 1.683 | 1.864 | – |
| 10 % | **NegGrad+** | **0.207** | 0.000 | 0.570 | 0.278 | 0.000 | 0.000 | **99.53** | **11.56** | 0.473 |
| 10 % | **CF-k** | **0.925** | 0.775 | 0.667 | 0.383 | 0.706 | 0.545 | 1.175 | 4.114 | 0.427 |

### Hai baseline này là hai THÁI CỰC, và cả hai đều hỏng

Đọc cột Df-AUC theo chiều dọc sẽ thấy toàn cảnh (mốc vàng in đậm):

```
NegGrad+  0.10–0.21   ← phá hủy mô hình
θ_re      0.50–0.60   ← MỐC VÀNG
Forget-MI 0.62–0.70
P3        0.67–0.73
θ_og      0.73–0.76   ← chưa quên gì
CF-k      0.91–0.93   ← còn nhớ HƠN cả mô hình gốc
```

Forget-MI và P3 nằm giữa hai kiểu hỏng: một bên quên đến mức phá mô hình, một bên không
quên gì. Đây là khung đọc kết quả tốt nhất cho Chương 4 — nó cho thấy khoảng giá trị hợp
lệ hẹp đến mức nào.

### CF-k không quên gì — thậm chí còn nhớ HƠN mô hình gốc

Df-AUC 0.912–0.927, **cao hơn cả θ_og** (0.731–0.757), và Df-F1 0.70–0.78 so với θ_og
0.42–0.47. forget-CE 1.13–1.49 cũng **thấp hơn** θ_og (1.66–1.97).

Nghe nghịch lý nhưng cơ chế rất rõ: CF-k chỉ đóng băng 2 tầng đầu rồi **tinh chỉnh tiếp
trên `D_r` suốt 30 epoch**. Trong hàm mất mát của nó **không có bất kỳ thành phần nào
hướng tới việc quên**. Mà `D_f` được rút ra từ chính tập huấn luyện, cùng phân bố và cùng
nhóm bệnh nhân với `D_r`, nên huấn luyện thêm trên `D_r` **khớp tốt hơn** luôn cả `D_f`.

Bằng chứng chốt là hướng đi ngược nhau của hai cột CE: forget-CE **giảm** (1.97 → 1.49 ở
3 %) trong khi test-CE **tăng** (2.11 → 4.17). Đó là chữ ký của quá khớp — mô hình bám
chặt hơn vào phân bố huấn luyện và mất khả năng khái quát.

⇒ **CF-k đóng vai cận trên "không quên"**, đúng như bài báo gốc báo cáo. Nó không phải
phương pháp cạnh tranh mà là mốc đối chứng.

Một điểm đáng nói cho họ phương pháp *hạn chế tham số*: CF-k vẫn cập nhật **72,3 %** tham
số (81,9/113,2 triệu) mà không quên được gì, trong khi P3 chỉ cập nhật **1,27 %** và có
quên. Việc quên hay không **không** do số lượng tham số được mở, mà do hàm mất mát có
thành phần hướng tới quên hay không.

### NegGrad+ phân kỳ ở cả ba mức quên

forget-CE đạt **99–129** trong khi gold chỉ 2.5–4.7, tức gấp **25–35 lần**. Đây là hành vi
đã biết của gradient ascent không có chặn: không có gì giới hạn `−CE(forget)` nên nó chạy
tới vô cùng. Ba dấu hiệu xác nhận:

1. **Df-AUC 0.10–0.21, tức DƯỚI 0.5.** Mô hình không chỉ quên mà còn xếp hạng **ngược**
   trên tập quên. Df-F1 bằng đúng 0.000.
2. **Test-CE cũng nổ lên 9.6–11.6** (gold 2.2–3.0). Mô hình hỏng **toàn cục**, không phải
   quên có chọn lọc — dù nhánh giữ lại `CE(retain)` vẫn có trong hàm mất mát.
3. **1−Sim so với gold 0.45–0.58**, xa hơn mọi phương pháp khác.

> ⚠️ **NegGrad+ đạt MIA = 0.000 và MIA_paper = 0.000 ở cả ba mức — bằng đúng gold. Đây là
> con số VÔ NGHĨA, không phải chiến thắng.** Nó là bẫy "MIA = 0 giả": tập quên bị đẩy ra
> xa đến mức bộ tấn công đoán "non-member" cho toàn bộ, trong khi mô hình đã bị phá hủy
> (forget-CE 129). Đây là lý do **mọi con số MIA phải công bố kèm forget-CE và test-CE**.
> Nếu chỉ nhìn cột MIA thì NegGrad+ là phương pháp tốt nhất trong toàn bộ tài liệu này.

Kết quả trùng với quan sát ở nhánh mã cũ (forget-CE 54 rồi 117), tức **không phải sự cố
của một lần chạy** mà là tính chất của phương pháp trong thiết lập này. Báo cáo trung thực,
không chạy lại để "sửa".

### Không được đưa hai baseline này vào bảng chi phí

Chúng gọi `optimizer.step()` **mỗi batch trên toàn bộ tập giữ lại**:

| Run | Tham số cập nhật | Số lần cập nhật | Thời gian | GPU peak |
|---|---:|---:|---:|---:|
| NegGrad+ 3 / 6 / 10 % | 113 238 164 (100 %) | 10 140 / 9 810 / 9 390 | 5,90 / 5,84 / 5,36 h | 14,76 GB |
| CF-k 3 / 6 / 10 % | 81 874 028 (**72,3 %**) | 10 140 / 9 810 / 9 390 | 2,70 / 2,54 / 2,41 h | 7,09 GB |
| Forget-MI (mọi mức) | 113 238 164 (100 %) | **30** | 0,52 – 1,78 h | 6,92 GB |
| P3-NoKD-More (mọi mức) | 1 451 008 (**1,27 %**) | **30** | 0,43 – 1,60 h | 13,92 GB |

Tức hai baseline này chạy gấp **313 – 338 lần** số bước cập nhật. Đây là cách chúng **được
thiết kế** để chạy (và là cách bài báo gốc chạy), nên so sánh thời gian là vô nghĩa.

Ngược lại, điều đó làm kết luận về **chất lượng** mạnh hơn hẳn: kể cả với hơn 300 lần số
bước cập nhật, NegGrad+ vẫn phá hủy mô hình còn CF-k vẫn không quên được gì.

---

## Phụ lục A — Checkpoint theo cả 4 selector

Chỉ **S2** được dùng trong các bảng chính; S1/S3/S4 ghi lại để đối chiếu.
`—` nghĩa là CE không cắt nhau nên selector đó không xác định. Epoch đếm-từ-0.

| Run | S1 (first crossing) | S2 (closest CE) | S3 (first stable) | S4 (CE + tiện ích) |
|---|---|---|---|---|
| Forget-MI MIMIC 3% | 6 | 6 | 6 | 9 |
| P3-NoKD-More MIMIC 3% | — | 29 | — | 12 |
| Forget-MI MIMIC 6% | 14 | 13 | 14 | 5 |
| P3-NoKD-More MIMIC 6% | — | 29 | — | 11 |
| Forget-MI MIMIC 10% | — | 29 | — | 29 |
| P3-NoKD-More MIMIC 10% | — | 29 | — | 29 |
| Forget-MI IU 3% | — | 29 | — | 29 |
| ‡ P3-NoKD-More IU 3% (lr 5e-4) | 11 | 10 | 11 | 10 |
| P3-NoKD-More IU 3% (lr 2e-4) | 23 | 22 | 23 | 22 |
| Ablation: P3 đầy đủ | — | 29 | — | 12 |
| Ablation: w/o Fisher/FILA | — | 29 | — | 29 |
| Ablation: w/o IHL | — | 9 | — | 9 |
| Ablation: w/o MU/MR | 9 | 8 | 9 | 8 |

Quan sát: CE của P3 trên MIMIC **không bao giờ cắt** ở mọi mức quên và mọi biến thể
ablation trừ `w/o MU/MR` — tức P3 giữ forget-CE luôn dưới nm_val-CE, hành vi quên không
vượt ngưỡng. Trên IU thì cắt sớm (epoch index 11, tức E12) rồi vọt lên rất nhanh.

---

## Phụ lục B — Bóc tách thời gian đầy đủ

Các cột dưới đây **không** đưa vào bảng kết quả chính (chúng thuộc giao thức, không phải
thuật toán), chỉ dùng để giải thích wall-clock.

| Run | T_core | T_selection | T_eval | T_pipeline | Chẩn đoán | Thời gian 1 epoch train |
|---|---:|---:|---:|---:|---:|---:|
| Forget-MI MIMIC 3% | 2 172,5 | 2 848,9 | 1 239,0 | 6 260,3 | 1 046,1 | 72,42 ± 0,35 s |
| P3-NoKD-More MIMIC 3% | 1 864,5 | 7 333,3 | 408,5 | 9 606,3 | 0,0 | 53,09 ± 0,82 s |
| Forget-MI MIMIC 6% | 3 995,5 | 3 516,9 | 1 328,6 | 8 840,9 | 1 024,6 | 133,18 ± 0,44 s |
| P3-NoKD-More MIMIC 6% | 3 390,2 | 9 578,0 | 474,1 | 13 442,3 | 0,0 | 103,41 ± 0,63 s |
| Forget-MI MIMIC 10% | 6 403,3 | 3 822,5 | 1 342,5 | 11 568,4 | 1 061,7 | 213,44 ± 0,50 s |
| P3-NoKD-More MIMIC 10% | 5 742,5 | 13 163,6 | 617,7 | 19 523,8 | 0,0 | 181,58 ± 1,37 s |
| Forget-MI IU 3% | 1 882,6 | 3 084,8 | 1 284,8 | 6 252,2 | 8 367,8 | 62,75 ± 0,49 s |
| P3-NoKD-More IU 3% (lr 2e-4) | 1 533,5 | 8 991,7 | 532,8 | 11 058,0 | 0,0 | 42,75 ± 1,08 s |
| ‡ P3-NoKD-More IU 3% (lr 5e-4) | 1 533,6 | 8 690,3 | 536,4 | 10 760,3 | 0,0 | 42,45 ± 0,52 s |
| Ablation: P3 đầy đủ | 1 906,4 | 7 494,0 | 409,0 | 9 809,4 | 0,0 | 54,83 ± 0,72 s |
| Ablation: w/o Fisher/FILA | 1 660,5 | 7 421,5 | 412,0 | 9 494,1 | 0,0 | 55,35 ± 1,97 s |
| Ablation: w/o IHL | 1 904,3 | 7 401,6 | 412,1 | 9 718,0 | 0,0 | 55,02 ± 0,51 s |
| Ablation: w/o MU/MR | 1 882,2 | 7 698,5 | 416,4 | 9 997,1 | 0,0 | 54,13 ± 1,62 s |

Đơn vị: giây. `single-run timing` — mỗi cấu hình chạy một lần, không có mean ± std giữa
các lần chạy.

---

## Phụ lục C — Các điểm phải công bố kèm số liệu

**0. MIA của Forget-MI ở hàng E30 ĐÃ ĐƯỢC TÍNH LẠI.** Bản `_final_evaluation` của
Forget-MI bỏ qua `eval_max_retain=512` và dùng toàn bộ retain (~5410) làm tập member,
trong khi selector và P3 đều dùng bản lấy mẫu 512 — khiến `MIA_paper` bão hoà ở 1.000
và không so được giữa hai phương pháp. Đã đánh giá lại 4 checkpoint `last.pt` bằng
`forgetmi_eval_only.py` (dùng `final_evaluation` của `adv_common`, có lấy mẫu 512).

Kiểm chứng: mọi Df-AUC/Df-F1/Dt-AUC/Dt-F1/CE **trùng khít từng chữ số** với lần trước
(cùng checkpoint, cùng `D_t_final`), chỉ MIA/MIA_paper đổi. Và ở MIMIC 10% cùng IU 3% —
hai bộ mà S2 chốt đúng E30 — MIA tính lại **trùng khít giá trị S2** (0.616/0.455 và
0.581/0.167), trong khi trước đó lệch (0.559 và 0.602).

| Bộ | MIA cũ → mới | MIA_paper cũ → mới |
|---|---|---|
| MIMIC 3% | 0.567 → **0.627** | 1.000 → **0.286** |
| MIMIC 6% | 0.544 → **0.773** | 1.000 → **0.538** |
| MIMIC 10% | 0.559 → **0.616** | 1.000 → **0.455** |
| IU 3% | 0.602 → **0.581** | 0.667 → **0.167** |

Các hàng **S2 không đổi** (selector vốn đã dùng member 512). Hàng E30 của P3 cũng không
đổi (đã dùng 512 sẵn).

**1. Không được so `T_pipeline` giữa hai phương pháp.** P3 chọn checkpoint theo `S_val`,
Forget-MI theo `val_ce` — khác selector nên `T_selection` không so được, kéo theo
`T_pipeline` cũng vậy. Chỉ dùng cột **`T_core`**.

**2. Precision của mô hình tham chiếu lệch nhau.** P3 chạy `model_og` ở **fp16**,
Forget-MI ở **fp32**. Một phần lợi thế `T_train` của P3 đến từ đây chứ không phải từ
LoRA. Chênh lệch `T_core` thực tế do thuật toán còn nhỏ hơn con số trong bảng.

**3. P3 KHÔNG tiết kiệm bộ nhớ.** GPU peak 13,92 GB so với 6,92 GB — gấp **2,01×**. Chỉ
được kết luận hiệu quả **tham số** (1,27 % vs 100 %, tức ít hơn 78×), không được kết
luận hiệu quả bộ nhớ.

**4. Mức tăng tốc khiêm tốn.** `T_core` nhanh hơn 1,17× (3 %), 1,18× (6 %), 1,12×
(10 %), 1,23× (IU). Không phải "hàng chục lần" như số liệu của code cũ trong bản thảo
Chương 4 — con số `≈1/17` cũ phải bỏ.

**5. Khả năng quên của P3 suy giảm theo tỉ lệ quên.** Thắng ở 3 % (hơn ở `MIA`, hoà ở
`MIA_paper`), phụ thuộc chốt ở 6 % (S2 nghiêng Forget-MI, E30 nghiêng P3), **ở 10 % thì
gần như không quên được** — forget-CE 1.662 → 1.683 trong khi gold ở 3.264. Không được
phát biểu "P3 tốt hơn Forget-MI" một cách tổng quát — phải nói theo từng mức quên **và**
theo chốt checkpoint. Bằng chứng ở 10 % là `MIA_paper` (0.455 vs 0.591) và forget-CE,
**không** phải `MIA` (0.616 vs 0.730 nằm trong nhiễu ±0.07).

**6. Df-AUC/Dt-AUC của P3 trên IU tại E30 từng bị mất do LỖI HÀM METRIC, nay đã có số.**
`compute_auc` tính Pairwise-AUC bằng `p[c1] / (p[c0] + p[c1])`. Logit ảnh của P3 tại E30
trải `[−65.3, +68.6]`; softmax chạy trên FP32 nên `exp(−134)` underflow về **đúng 0.0** ở
cả hai kênh → `0/0`. Vì là `float` của Python nên nó **ném `ZeroDivisionError`** thay vì
trả `nan`, và lỗi rơi vào trước cả bước kiểm tra "cặp thiếu lớp". Tệ hơn, `perf_metrics`
bọc AUC-theo-kênh và Pairwise-AUC trong **một** `try/except`, nên phần AUC-theo-kênh
*đã tính xong* cũng mất theo.

Đã sửa: chặn mẫu số bằng 0 (gán 0.5 — mẫu đó không phân biệt được hai lớp) và tách
`try/except` cho hai đại lượng. Với biên độ logit thông thường, kết quả **trùng khít**
bản cũ nên **không số nào trong Bảng 1–4 bị ảnh hưởng**.

Xác nhận mô hình dựng lại đúng là mô hình lúc chạy thật: `Macro_F1` đo lại
(0.3335 / 0.3507) **trùng khít** giá trị đã ghi trong bảng (0.334 / 0.351), trong khi
AUC chuyển từ `NaN` sang 0.710 / 0.590. Số dùng trong Bảng 5 lấy ở chế độ autocast FP16
— đúng chế độ mọi hàng khác đã chạy; bản FP32 chênh không đáng kể (0.7098 / 0.58965).

**7. Kiểm tra tái lập bit-exact.** Run `p3_m3` (chạy lại P3-NoKD-More 3%) cho kết quả
**trùng khít từng chữ số** với run gốc: forget-CE tại E1/E11/E30 = 1.9197 / 1.9334 /
2.0671 ở cả hai, S2 giống nhau đến 4 chữ số thập phân.

**8. HẠN CHẾ — thứ tự dữ liệu và dãy Gate bị cố định giữa các epoch.**
`AlignedSampler.__iter__` gọi `torch.manual_seed(42)` mỗi lần được duyệt, tức **đầu mỗi
epoch**. Hệ quả đo được:

- thứ tự batch của `forget`, `random` **và** `retain` lặp y hệt qua cả 30 epoch;
- Gate được tạo mới mỗi batch, nhưng vì RNG bị ghim nên đó là **cùng một dãy Gate phát
  lại 30 lần** chứ không phải 30 dãy khác nhau;
- `--seed 7` **không** đổi thứ tự mẫu lẫn dãy Gate (giá trị 42 bị viết cứng ở nơi tạo
  sampler); seed chỉ đổi phân hoạch dữ liệu, khởi tạo và MIA.

Đây là **hành vi kế thừa nguyên vẹn từ code Forget-MI gốc** (`torch.manual_seed(self.seed)`
và `AlignedSampler(..., shuffle=True, seed=42)` đều có trong bản gốc), không phải do bản
tái lập thêm vào. Bản tái lập chỉ sửa đúng một lỗi hiển nhiên: bản gốc gọi
`torch.randperm(...)` mà **không gán kết quả** nên xáo trộn vô hiệu — sửa rồi thì thứ tự
là một hoán vị cố định thay vì tuần tự, tính chất "lặp lại mỗi epoch" thì không đổi.

Ảnh hưởng: **cả hai phương pháp chịu như nhau** nên mọi so sánh trong tài liệu này vẫn
công bằng, và đây cũng là lý do kết quả tái lập bit-exact ở mục 7. Nhưng nếu về sau chạy
đa-seed để báo mean ± std thì **bắt buộc phải sửa**, nếu không độ biến thiên đo được sẽ
hẹp giả tạo.

---

## Tổng kết theo câu hỏi nghiên cứu

**RQ1 — khả năng quên và rò rỉ thành viên.** Phụ thuộc mạnh vào tỉ lệ quên.
Mọi số dưới đây đều lấy ở checkpoint **S2** cho cả hai phương pháp (các giá trị S2 không
bị ảnh hưởng bởi lần tính lại MIA ở Phụ lục C, vì selector vốn đã dùng member 512):

| Mức quên | MIA (FMI → P3) | MIA_paper (FMI → P3) | Kết luận |
|---|---|---|---|
| MIMIC 3 % | 0.657 → **0.552** | 0.429 → **0.286** | P3 tốt hơn |
| MIMIC 6 % | **0.357** → 0.706 | 0.385 → **0.231** | hai chỉ số ngược nhau |
| MIMIC 10 % | 0.616 → 0.730 | **0.455** → 0.591 | xem ghi chú (a) |
| IU 3 % | **0.581** → 0.597 | 0.167 = 0.167 | xấp xỉ (nhưng P3 sụp — xem Bảng 5) |
| ‡ IU 3 % (lr 5e-4) | **0.581** → 0.597 | 0.167 = 0.167 | – |

(a) Chênh lệch `MIA` 0.616 → 0.730 **không kết luận được**: độ lệch chuẩn của chính phép
đo ở kích thước tập này là ±0.07. Chỉ `MIA_paper` mới đủ cách biệt.
Xem thêm ghi chú ở Bảng 3 về việc P3 gần như không quên ở mức 10 %.

⚠️ Hàng IU chưa có số S2 cho cấu hình đúng (lr 2e-4) — xem Bảng 5.

Cùng các mức quên đó **tại E30** — bức tranh khác hẳn, nên phải công bố cả hai chốt:

| Mức quên | MIA (FMI → P3) | MIA_paper (FMI → P3) | Kết luận |
|---|---|---|---|
| MIMIC 3 % | 0.627 → **0.552** | 0.286 = 0.286 | P3 hơn ở `MIA`, **hoà** ở `MIA_paper` |
| MIMIC 6 % | 0.773 → **0.706** | 0.538 → **0.231** | P3 tốt hơn ở **cả hai** |
| MIMIC 10 % | 0.616 → 0.731 | **0.455** → 0.591 | xem ghi chú (a) |
| IU 3 % | 0.581 → **0.555** | 0.167 = 0.167 | P3 gần gold hơn (gold 0.545), nhưng xem RQ2 |
| ‡ IU 3 % (lr 5e-4) | 0.581 → 0.560 | 0.167 → 0.000 | – |

Chỗ lật rõ nhất là **MIMIC 6 %**: tại S2 Forget-MI thắng `MIA` rất đậm (0.357), tại E30
lại thua cả hai chỉ số (0.773 vs 0.706). Nguyên nhân là MIA của Forget-MI biến động mạnh
theo epoch, còn P3 gần như đứng yên. Không được chọn chốt có lợi rồi kết luận.

**RQ2 — bảo toàn hiệu năng.** Dt-AUC tại **S2** — P3 thắng ở mọi cấu hình:

| | MIMIC 3 % | MIMIC 6 % | MIMIC 10 % | IU 3 % | ‡ IU (lr 5e-4) |
|---|---:|---:|---:|---:|---:|
| Forget-MI | 0.668 | 0.658 | 0.653 | 0.635 | 0.635 |
| **P3-NoKD-More** | **0.704** | **0.689** | **0.692** | **0.664** | **0.665** |

Dt-AUC tại **E30** — P3 thắng **cả 4**:

| | MIMIC 3 % | MIMIC 6 % | MIMIC 10 % | IU 3 % | ‡ IU (lr 5e-4) |
|---|---:|---:|---:|---:|---:|
| Forget-MI | 0.648 | 0.641 | 0.653 | 0.635 | 0.635 |
| **P3-NoKD-More** | **0.704** | **0.689** | **0.692** | **0.654** | 0.590 |

Chỗ thua ở IU tại E30 **đã biến mất** sau khi chạy đúng cấu hình khóa (0.590 → 0.654).
Test-CE của P3 thấp hơn ở mọi cấu hình MIMIC.

⚠️ Nhưng **không được dừng ở Dt-AUC với IU**: Dt-F1 của P3 chỉ 0.402 so với Forget-MI
0.533 và gold 0.614, còn test-CE 11.41 so với gold 2.31. Thứ hạng dự đoán giữ được nhưng
ngưỡng quyết định đã hỏng — thắng ở AUC mà thua ở F1 và CE thì **không** được phát biểu
là "bảo toàn hiệu năng tốt hơn" trên IU.

**RQ3 — hiệu quả tài nguyên.**

| Trục | Kết luận |
|---|---|
| Tham số cập nhật | **1,27 % vs 100 %** — ít hơn 78×, kết luận vững, không phụ thuộc phần cứng |
| Thời gian lõi | nhanh hơn **1,12–1,23×** — khiêm tốn, và còn được lợi từ chênh lệch precision |
| GPU peak | **13,92 vs 6,92 GB — P3 tốn gấp 2,01×**, không được nói là tiết kiệm bộ nhớ |

---

## Phụ lục D — Chuẩn bị phản biện

Tổng hợp những chỗ dễ bị hỏi, kèm câu trả lời dựa trên số liệu đã có. Mọi thay đổi trong
quá trình làm đều có lịch sử git, có thể trình ra khi cần.

**"Sao trong bảng từng có ô `n/a` rồi lại có số?"**
Do một lỗi trong hàm tính AUC, không phải do chạy lại thí nghiệm. Cùng một checkpoint:
trước khi vá cho `NaN`, sau khi vá cho 0.710 / 0.590, trong khi `Macro_F1` **không đổi**
(0.334 / 0.351) — chứng minh vẫn đúng mô hình đó. Chi tiết ở Phụ lục C mục 6.

**"Bản vá đó có làm sai lệch các số khác không?"**
Không. Bản vá chỉ kích hoạt khi mẫu số **đúng bằng 0** — tình huống trước đây làm chương
trình văng lỗi, tức không có số nào để mà thay đổi. Đã quét toàn bộ kết quả: trong 12 run
chỉ **đúng một ô** từng bị lỗi này (P3 / IU / E30), và nó đã được điền.

**"Vì sao MIA của Forget-MI bị tính lại?"**
Bản đánh giá cuối của Forget-MI dùng toàn bộ tập giữ lại (~5410 mẫu) làm member, trong
khi selector và P3 đều dùng bản lấy mẫu 512. Cùng một checkpoint mà ra hai giá trị MIA
khác nhau. Đã đánh giá lại cả 4 checkpoint bằng đúng đường tính của P3. Bằng chứng đúng:
mọi Df-AUC/Df-F1/Dt-AUC/Dt-F1/CE **trùng khít từng chữ số** lần trước, chỉ MIA đổi; và ở
MIMIC 10% cùng IU — hai bộ mà S2 chốt đúng E30 — MIA tính lại **trùng khít giá trị S2**.

**"P3 nhanh hơn bao nhiêu?"**
`T_core` nhanh hơn **1,12–1,23×**, không hơn. Và phải nói kèm: mô hình tham chiếu của P3
chạy fp16 còn Forget-MI chạy fp32, nên phần chênh do thuật toán còn nhỏ hơn con số này.
Số `≈1/17` trong bản thảo cũ là của code cũ, phải bỏ.

**"P3 có tiết kiệm tài nguyên không?"**
Chỉ **tham số**: 1,27 % so với 100 %, tức ít hơn 78 lần — kết luận này vững vì không phụ
thuộc phần cứng. **Bộ nhớ thì ngược lại**: GPU đỉnh 13,92 GB so với 6,92 GB, tức P3 tốn
gấp 2,01 lần. Không được phát biểu P3 tiết kiệm bộ nhớ.

**"P3 có tốt hơn Forget-MI không?"**
Không thể trả lời chung. Phụ thuộc **tỉ lệ quên** và **chốt checkpoint**: thắng ở MIMIC
3%, lẫn lộn ở 6% (S2 nghiêng Forget-MI, E30 nghiêng P3), thua rõ ở 10%. Trên IU thì P3
quên nhiều hơn nhưng mất tiện ích. Xem hai bảng ở phần RQ1 và RQ2.

**"Vì sao chỉ chạy một lần, không có mean ± std?"**
Ngân sách GPU: 12 run × ~3h trên tài khoản Kaggle miễn phí. Mọi bảng đều ghi rõ
`single-run timing`. Lưu ý thêm: theo Phụ lục C mục 8, muốn chạy đa-seed cho có ý nghĩa
thì phải sửa `AlignedSampler` trước, nếu không độ biến thiên sẽ hẹp giả tạo.

**"Vì sao không so tổng thời gian pipeline?"**
Hai phương pháp dùng selector khác nhau (`S_val` và `val_ce`) nên thời gian chọn
checkpoint không so được, kéo theo pipeline cũng vậy. Chỉ dùng `T_core`. Công cụ dựng
bảng tự phát cảnh báo này.

**"Ablation `w/o MU/MR` có tổng trọng số 2/3, có phải lỗi không?"**
Cố ý. Chuẩn hoá lại sẽ vừa bỏ MU/MR vừa nhân đôi UU/UR — đổi hai thứ cùng lúc, không còn
là ablation sạch. Cả hai phương án đều có nhược điểm; đã chọn phương án giữ nguyên trọng
số các thành phần còn lại và ghi rõ trong Bảng 4. Vì optimizer là AdamW (bất biến theo tỉ
lệ loss) nên việc tổng nhỏ hơn 1 gần như không đổi độ lớn bước cập nhật.

**"P3 sụp đổ trên IU, có phải cài đặt sai không?"**
Không. Đó là hệ quả đúng của giao thức đã chốt trước khi chạy: dùng nguyên cấu hình khoá
từ MIMIC, không tuning lại theo IU. IU nhỏ hơn nhiều và mô hình gốc đã khớp tập quên gần
như hoàn hảo (Df-AUC 1.000, forget-CE 0.002), nên cùng learning rate với 30 epoch là quá
mạnh. Đây là một hạn chế đáng ghi, không phải lỗi.

---

## Trạng thái

**14 run.** 12 run của danh sách tối thiểu, cộng hai run bổ sung chạy 2026-08-11:

| Run | Mục đích | Kết luận |
|---|---|---|
| `p3_iu_lr2e4_s42` | khôi phục đúng cấu hình khóa cho IU | hạ lr **không** cứu được IU ⇒ nguyên nhân là dữ liệu |
| `p3_m3_lr1e4_s42` | kiểm tra độ nhạy learning rate ở MIMIC 3 % | kết luận **không** nhạy lr; riêng `MIA_paper` lật vì lượng tử hóa |

Cộng **6 run baseline** ở Bảng 6 — NegGrad+ và CF-k, mỗi bên 3/6/10 % → tổng **20 run**.
Bảng 6 nay **đầy đủ**, không còn ô trống ở bất kỳ bảng nào.

Còn thiếu:
- Đánh giá bổ sung trên cả hai nhánh ảnh + văn bản (`run_eval_multimodal_kaggle.ipynb`) —
  hiện mọi độ đo chỉ dùng nhánh ảnh.
