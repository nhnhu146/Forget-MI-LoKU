# Hướng dẫn end-to-end: Forget-MI baseline + LoKU trên Indiana University CXR (Kaggle)

> Mục tiêu: chạy **baseline (Forget-MI)** và **LoKU** trên IU-CXR, **chỉ forget 3% × multi-seed**, để có
> kết quả cross-dataset cho Chương 4 luận văn. Dùng dataset gốc `raddar/chest-xrays-indiana-university` trên Kaggle.

IU-CXR **không có** model pretrained như MIMIC (paper chỉ release cho MIMIC), nên phải **tự train**
`model_og_IU` + `model_retrained_iu_3per` trước khi unlearning. Toàn bộ đã được wire sẵn.

---

## 0. Chuẩn bị 1 lần

| Cần | Ghi chú |
|---|---|
| Kaggle account + GPU T4/P100, Internet ON | |
| Kaggle Dataset gốc: `raddar/chest-xrays-indiana-university` | Add vào notebook khi cần |
| Kaggle Dataset `forget-mi-models` (hoặc `forget-mi-models-full`) | MIMIC models — dùng để **warm-start** train IU |
| Kaggle Secrets: `GITHUB_TOKEN`, `GIT_EMAIL`, `GIT_NAME` | Để push kết quả |

---

## 1. Thứ tự thực hiện (BẮT BUỘC theo đúng thứ tự — không đảo được)

```
Bước 1  PREPROCESS   preprocess_iu_kaggle.ipynb        ~5 phút  CPU   → Dataset forget-mi-data-iu
Bước 2  TRAIN og+re  run_kaggle_train_iu.ipynb         ~16–28h  GPU   → Dataset forget-mi-models-iu
Bước 3  BASELINE     run_kaggle_baseline.ipynb Cell 4d ~15h     GPU
Bước 4  LoKU         run_kaggle_loku.ipynb Cell 4d     ~40 phút GPU
Bước 5  PUSH         Cell 6 mỗi notebook                              → GitHub
```

**Phụ thuộc cứng**: Bước 3 và 4 đều load `model_og_IU` → không có Bước 2 thì không chạy được gì.
Bước 2 chỉ làm **1 lần**, dùng lại cho mọi seed/mọi lần chạy sau.

**Quota free 30h/tuần** (chỉ làm 3%): phần nặng là Bước 2 (train 2 model ≈ 16–28h).
Có thể tách Cell 3 (train og) và Cell 4 (train re) của Bước 2 ra **2 session/2 tuần** — cả hai đều warm-start độc lập từ MIMIC, có cơ chế skip nếu đã train xong.

---

## 2. Chi tiết từng bước

### Bước 1 — Preprocess → `forget-mi-data-iu`
1. New Notebook → **+ Add data**: `raddar/chest-xrays-indiana-university`
2. Upload & **Run All** [`preprocess_iu_kaggle.ipynb`](preprocess_iu_kaggle.ipynb)
3. **Save Version → Save & Run All** → khi Successful → **"Create Dataset from Output"** → đặt tên `forget-mi-data-iu`

Sinh ra: `data/metadata/all_data.tsv`, `data/img_data/*.png`, `data_splits/iu-split.csv`,
`data_splits/forget_set_{3,6,10}per_iu.csv`.

### Bước 2 — Train models → `forget-mi-models-iu`
1. New Notebook → **+ Add data**: `forget-mi-data-iu` **và** `forget-mi-models` (cho warm-start)
2. Upload [`run_kaggle_train_iu.ipynb`](run_kaggle_train_iu.ipynb) → GPU ON → chạy:
   - Cell 1 (setup) → Cell 2 (detect, phải thấy ✅ hết) → **Cell 3** (train `model_og_IU`, ~8–14h) → **Cell 4** (train `model_retrained_iu_3per`, ~8–14h) → Cell 5 (verify)
3. **"Create Dataset from Output"** → đặt tên `forget-mi-models-iu`

> ⚙️ Chỉnh `EPOCHS` trong Cell 3/4 (mặc định 20) nếu sợ vượt 12h. Warm-start nên 15–20 epoch là đủ.

> ⚠️ **Ghi vào luận văn**: `model_og_IU` được **warm-start từ model MIMIC** rồi fine-tune trên IU
> (transfer learning chest-xray→chest-xray) — cần thiết vì train from scratch BERT+ResNet trên ~3K mẫu
> không khả thi trong giới hạn Kaggle. Đây là lựa chọn chính đáng, nêu rõ như một thiết lập thực nghiệm.

### Bước 3 — Baseline Forget-MI trên IU (3% multi-seed)
1. Mở [`run_kaggle_baseline.ipynb`](run_kaggle_baseline.ipynb) → **+ Add data**: `forget-mi-data-iu`, `forget-mi-models-iu` (và `forget-mi-models` nếu vẫn chạy MIMIC)
2. Cell 1 → Cell 2 → Cell 3 (phải thấy `iu ✅ ENABLED`) → **Cell 4d**: đặt `RUN_IU_3PER = True`, `SEEDS_THIS_RUN = (42, 123, 7)` → chạy (~5h × 3 ≈ 15h)
3. Cell 5 (bảng) → Cell 6 (push)

### Bước 4 — LoKU trên IU (3% multi-seed)
1. Mở [`run_kaggle_loku.ipynb`](run_kaggle_loku.ipynb) → **+ Add data**: `forget-mi-data-iu`, `forget-mi-models-iu`
2. Cell 1 → Cell 2 → Cell 3 (helpers) → **Cell 4d (IU)**: `RUN_IU_LOKU = True`, `SEEDS_IU = (42, 123, 7)` → chạy (~12 phút × 3)
3. Cell 5 → Cell 6 (push). IU ghi CSV riêng `unlearning_iu_output/` — không đụng MIMIC.

### Bước 5 — Lấy kết quả
Local: `git pull` → đọc trong `experiments/`:
- `summary_baseline_iu_3per_multiseed.md`
- `summary_loku_iu_3per_kaggle_multiseed.md`
- `bang_baseline_kaggle_cross_dataset.md` (MIMIC vs IU)

---

## 3. Thông số mặc định

| | Baseline IU | LoKU IU |
|---|---|---|
| Script / Config | `forgetmi_partial.py` / `config_baseline_iu_kaggle.yaml` | `forgetmi_loku.py` / `config_loku_iu_kaggle.yaml` |
| Forget % | 3 | 3 |
| Seeds | 42, 123, 7 | 42, 123, 7 |
| `output_channel_encoding` | `multiclass` | `multiclass` |
| `max_seq_length` | 320 | 320 |
| Trainer | `scripts/train_iu_model.py` (og + re) | dùng chung models từ Bước 2 |

> IU là task **binary** (normal/abnormal, label 0/1) nhưng đi qua **head 4-class** (`multiclass` + one-hot).
> Lý do: data loader không hỗ trợ encoding `binary` (sẽ crash). Label 2,3 không bao giờ xuất hiện → tương đương binary.

---

## 4. Các fix đã thực hiện để IU chạy được (tham khảo)

IU pipeline trước đó **chưa từng chạy thông**. Đã sửa:

1. `joint_img_txt/model_utils.py` — `report_id = int("CXR349")` crash → strip ký tự không phải số;
   đuôi ảnh hardcode `.jpg` → fallback `.png`; thêm nhánh label cho `binary`/`multiclass` (trước đó `binary` để `txt_label` undefined → crash).
2. `config_baseline_iu_kaggle.yaml`, `config_loku_iu_kaggle.yaml` — `binary`→`multiclass`, thống nhất `max_seq_length=320`.
3. `scripts/train_iu_model.py` (mới) — train `model_og_IU` (full train) + `model_retrained_iu_Nper` (train − forget), tái dùng `build_dataset` nên pipeline data y hệt lúc unlearning.
4. `run_kaggle_train_iu.ipynb` (mới) — notebook chạy trainer → `forget-mi-models-iu`.
5. `run_kaggle_baseline.ipynb` Cell 3 — sửa path split + `multiclass`, auto-detect đệ quy (robust nesting).
6. `run_kaggle_loku.ipynb` — thêm Cell 4d (IU) `run_loku_iu()`.

---

## 5. Troubleshooting

| Triệu chứng | Xử lý |
|---|---|
| Cell 2 (train) thiếu `MIMIC init model` | Add `forget-mi-models` vào Input (cần để warm-start) |
| `iu ❌ DISABLED` ở baseline Cell 3 | Chưa add `forget-mi-data-iu`/`forget-mi-models-iu`, hoặc Bước 2 chưa xong |
| Out of memory khi train | Giảm `--batch_size` 16→8 trong Cell 3/4 train notebook |
| Vượt 12h session khi train | Giảm `EPOCHS`; train og và re ở 2 session khác nhau (có skip tự động) |
| `1−CosSim` báo ⚠️ | Thiếu `model_retrained_iu_3per` (gold) — chạy Cell 4 của Bước 2 |
