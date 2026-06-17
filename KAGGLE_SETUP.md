# Kaggle Setup — Forget-MI Baseline

> Hướng dẫn 1 lần để chạy `run_kaggle_baseline.ipynb` trên Kaggle.

## TL;DR

```
1. Upload data + models lên Kaggle Datasets — 1 LẦN
2. Setup Kaggle Secrets cho GitHub push — 1 LẦN
3. Tạo Kaggle Notebook → Add data → Add secrets → Settings: GPU T4
4. Upload run_kaggle_baseline.ipynb → Run cells
```

> ⚠️ **Quan trọng**: Data CONTENT giống hệt với Colab — bạn **KHÔNG cần chuẩn bị zip mới**.
> Chỉ cần upload 3 zip đã có sẵn trên Google Drive lên Kaggle.

---

## 1. Chuẩn bị Kaggle Datasets

### 🎯 Data cần upload = chính xác 3 zip bạn đang dùng trên Colab

Theo `setup_data.py`, project dùng 3 zip:

| Zip file | Nội dung | Sau extract (Colab) |
|---|---|---|
| `data.zip` | `text_data/`, `img_data/`, `metadata/` | `./data/` |
| `base_model.zip` | Pretrained model files | `./forgetme/training_original_model/` |
| `retrained_model.zip` | Retrained 3% model files | `./model_retrained_3per/` |

### Bước 1: Tải 3 zip từ Drive về máy local
```
Google Drive → Forget-MI-Project/data.zip → Tải xuống
Google Drive → Forget-MI-Project/base_model.zip → Tải xuống
Google Drive → Forget-MI-Project/retrained_model.zip → Tải xuống
```

### Bước 2: Upload lên Kaggle — chọn 1 trong 2 option

#### 🟢 Option A: 3 Datasets riêng (1:1 với Drive — DỄ NHẤT)

| Kaggle Dataset | Upload zip | Path sau extract |
|---|---|---|
| `forget-mi-data` | `data.zip` | `/kaggle/input/forget-mi-data/...` |
| `forget-mi-base-model` | `base_model.zip` | `/kaggle/input/forget-mi-base-model/...` |
| `forget-mi-retrained` | `retrained_model.zip` | `/kaggle/input/forget-mi-retrained/...` |

→ Tốn 3 lần "Add data" trong notebook nhưng KHÔNG cần repackage.

#### 🟡 Option B: 2 Datasets (gộp model — match KAGGLE_SETUP cũ)

| Kaggle Dataset | Upload zip(s) | Path sau extract |
|---|---|---|
| `forget-mi-data` | `data.zip` | `/kaggle/input/forget-mi-data/...` |
| `forget-mi-models` | **CẢ** `base_model.zip` + `retrained_model.zip` | `/kaggle/input/forget-mi-models/...` |

→ Kaggle Dataset cho phép multi-file: chọn cả 2 zip cùng lúc khi tạo dataset.

### Bước 3: ⚠️ XÁC NHẬN paths sau khi Kaggle auto-extract

Kaggle tự giải nén zip khi tạo Dataset. **Cấu trúc cuối cùng phụ thuộc nested level trong zip gốc**:

| Zip nội dung | Path Kaggle | Khớp config mặc định? |
|---|---|---|
| **Nested**: `data.zip` chứa `data/text_data/...` | `/kaggle/input/forget-mi-data/data/text_data/...` | ❌ — cần sửa config |
| **Flat**: `data.zip` chứa thẳng `text_data/, img_data/, metadata/` | `/kaggle/input/forget-mi-data/text_data/...` | ❌ — cần sửa config |
| **Nested 1 lớp** (như mặc định KAGGLE_SETUP): `data.zip` chứa `metadata/, img_data/, text_data/` (không có wrapper `data/`) | `/kaggle/input/forget-mi-data/metadata/...` | ✅ |

→ **Kiểm tra sau upload**: Vào Kaggle Dataset → "Data" tab → "File Browser" → xem cấu trúc thực tế.

→ Nếu khác mặc định, **sửa paths** trong `config_baseline_kaggle.yaml` (mục `base_model_path`, `text_data_dir`, `img_data_dir`, `bert_pretrained_dir`, `retrained_model_path`).

### Bước 4: Cell 2 trong notebook sẽ verify paths

Sau khi tạo Kaggle Notebook + Add Data, chạy **Cell 2** sẽ check:
```
✅ Base model      /kaggle/input/forget-mi-models/training_original_model/pytorch_model.bin
✅ Text metadata   /kaggle/input/forget-mi-data/metadata
...
```
Nếu thấy ❌ → chỉnh paths trong config_baseline_kaggle.yaml cho đúng cấu trúc thực tế.

---

## 2. Setup Kaggle Secrets (GitHub push)

Để `Cell 6` push được kết quả về GitHub:

1. Tạo GitHub Personal Access Token: https://github.com/settings/tokens
   - Scope: ✓ `repo` (đủ rồi)
   - Copy token bắt đầu `ghp_...`

2. Trong Kaggle Notebook:
   - Sidebar phải → **Add-ons** → **Secrets**
   - Add 3 secrets:

| Label | Value |
|---|---|
| `GITHUB_TOKEN` | `ghp_xxxxxxxxxx` |
| `GIT_EMAIL` | `your@email.com` |
| `GIT_NAME` | `Your Name` |

3. Trong notebook, các secrets được load qua `kaggle_secrets.UserSecretsClient()` (Cell 6 đã handle).

---

## 3. Tạo Kaggle Notebook & chạy

1. **Tạo notebook mới**: kaggle.com → Code → New Notebook
2. **Upload notebook**: File → Upload Notebook → chọn `run_kaggle_baseline.ipynb`
3. **Settings** (sidebar phải):
   - Accelerator: **GPU T4 x2** (free) hoặc **GPU P100** (Pro)
   - Internet: **ON** (để git clone)
   - Persistence: Files only
4. **Add data** (sidebar phải):
   - + Add Data → search `forget-mi-data` (private) → Add
   - + Add Data → search `forget-mi-models` (private) → Add
5. **Run cells theo thứ tự**:
   - Cell 1 (setup) → Cell 2 (verify) → Cell 3 (helpers) → **Cell 4a/4b/4c** (theo nhu cầu)
   - Cell 5 (summary) → Cell 6 (push)

---

## 4. Quota GPU Kaggle

| Plan | GPU quota |
|---|---|
| **Free** | 30h/tuần (reset Sat 00:00 UTC) |
| **Kaggle Pro** | Unlimited (paid) |

**Mỗi run baseline ~5h** (paper). Multi-seed 3% × 3 seeds = **~15h** → vừa khít với free 30h/tuần (1 forget% mỗi tuần).

**Strategy cho free user**:
- Tuần 1: Cell 4a (3%) — ~15h
- Tuần 2: Cell 4b + Cell 4c (6% + 10%) — ~30h

**Cảnh báo**: Mỗi session Kaggle giới hạn **12h liên tục**. Multi-seed 3 seeds ≈ 15h → cần break giữa các seeds. Helper `_seed_done()` tự skip seed đã xong khi rerun → an toàn nếu disconnect.

---

## 5. Setup IU-CXR sau này (placeholder)

Khi đã preprocess IU-CXR (theo Section 13 `THESIS_ROADMAP.md`):

1. Upload thêm 2 Kaggle Datasets:
   - `forget-mi-data-iu` — IU text + image data
   - `forget-mi-models-iu` — pretrained `model_og_IU` + retrained `model_re_IU_*per`

2. Tạo `config_baseline_iu_kaggle.yaml` (copy `config_baseline_kaggle.yaml`, đổi paths):
   ```yaml
   base_model_path: "/kaggle/input/forget-mi-models-iu/model_og_IU"
   text_data_dir:   "/kaggle/input/forget-mi-data-iu/metadata"
   img_data_dir:    "/kaggle/input/forget-mi-data-iu/img_data"
   output_channel_encoding: "binary"     # nếu task là normal/abnormal
   ```

3. Tạo `run_kaggle_baseline_iu.ipynb` (copy `run_kaggle_baseline.ipynb`, đổi `--config` thành `config_baseline_iu_kaggle.yaml`).

---

## 6. Troubleshooting

### Q: `MODULE NOT FOUND: peft`
A: Cell 1 sẽ tự `pip install`. Restart kernel nếu install giữa session.

### Q: Out of memory
A: Giảm `unlearn_batch_size` từ 16 → 8 trong `config_baseline_kaggle.yaml`.

### Q: GPU không có
A: Settings → Accelerator → GPU T4 x2 → Save & Restart.

### Q: Push GitHub fail "permission denied"
A: Token sai scope. Tạo lại token với scope `repo` (full access).

### Q: Kaggle session disconnect giữa cell 4a
A: Run lại cell 4a — `_seed_done()` auto-skip seed đã xong.

### Q: Path `/kaggle/input/forget-mi-data/metadata` không tồn tại
A: Tên dataset upload khác. Vào Cell 2 xem path thật, update `config_baseline_kaggle.yaml`.

### Q: Cache features bị regenerate (~5 phút)
A: Đã cache trong dataset upload chưa? Verify `cachedfeatures_train_seqlen-320_multiclass` có trong `/kaggle/input/forget-mi-data/metadata/`.

---

## 7. Workflow hoàn chỉnh (recap)

```
[LẦN ĐẦU — 1 giờ]
1. Upload forget-mi-data + forget-mi-models (15 phút mỗi cái)
2. Setup Kaggle Secrets (5 phút)
3. Tạo notebook, add data, add secrets (5 phút)
4. Verify Cell 1-3 chạy OK (5 phút)

[MỖI TUẦN — chạy thí nghiệm]
1. Open notebook, Run Cell 1 (pull latest code), Cell 2 (verify), Cell 3 (helpers)
2. Run Cell 4a/b/c tùy quota
3. Run Cell 5 (summary)
4. Run Cell 6 (push) → đẩy kết quả về GitHub
5. Local: git pull → đọc experiments/summary_baseline_*.md
```

---

## 8. Output files (sau khi chạy)

Trên `/kaggle/working/`:
- `results_summary.csv` — 9 rows (3 forget% × 3 seeds)
- `baseline_output/<run_name>/` — checkpoints (nếu enable)

Trên `experiments/` (sẽ push lên GitHub qua Cell 6):
- `exp_NNN_baseline_3per_seed42.md` ... `exp_NNN_baseline_10per_seed7.md` (9 files)
- `summary_baseline_3per_multiseed.md`
- `summary_baseline_6per_multiseed.md`
- `summary_baseline_10per_multiseed.md`
- `bang_baseline_kaggle.md` ← **dùng trực tiếp cho Chương 4 luận văn**

Local sau `git pull`:
- Tất cả file trên + `results_summary.csv` (nếu có push)
