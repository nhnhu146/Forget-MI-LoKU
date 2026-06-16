# Kaggle Setup — Forget-MI Baseline

> Hướng dẫn 1 lần để chạy `run_kaggle_baseline.ipynb` trên Kaggle.

## TL;DR

```
1. Upload 2 Kaggle Datasets (data + models) — 1 LẦN
2. Setup Kaggle Secrets cho GitHub push — 1 LẦN
3. Tạo Kaggle Notebook → Add data → Add secrets → Settings: GPU T4
4. Upload run_kaggle_baseline.ipynb → Run cells
```

---

## 1. Chuẩn bị 2 Kaggle Datasets

### Dataset 1: `forget-mi-data`
**Nội dung**: Toàn bộ thư mục `data/` của project (text + image).

**Cấu trúc upload**:
```
forget-mi-data/
├── metadata/
│   ├── all_data.tsv
│   ├── cachedfeatures_train_seqlen-320_multiclass     (sau khi đã generate)
│   └── cachednoisyfeatures_train_seqlen-320_multiclass
├── img_data/
│   ├── <patient_id_1>/
│   │   └── *.png
│   └── ...
└── text_data/                                         (raw reports, có thể skip nếu đã có cache)
    └── ...
```

**Cách upload**:

**Option A — Local zip + upload** (đơn giản, nhưng phải đợi upload):
```bash
# Trên máy bạn:
cd "d:/Hoang Nhu/UNIVERSITY/4th YEAR/Khoa luan tot nghiep/Code/Forget-MI-main/Forget-MI-main"
zip -r forget-mi-data.zip data/metadata data/img_data data/text_data
# Upload zip lên Kaggle Datasets → Kaggle tự giải nén
```

**Option B — Copy từ Drive xuống Kaggle qua API** (nhanh nếu Drive đã có):
```bash
# Sau khi đã setup Kaggle CLI:
kaggle datasets init -p ./forget_mi_data_kaggle/
cd forget_mi_data_kaggle/
# ... copy files ...
kaggle datasets create -p ./
```

Sau khi upload, dataset path sẽ là: `/kaggle/input/forget-mi-data/`

### Dataset 2: `forget-mi-models`
**Nội dung**: Pretrained model + retrained model.

**Cấu trúc**:
```
forget-mi-models/
├── training_original_model/
│   ├── pytorch_model.bin
│   ├── config.json
│   ├── vocab.txt
│   └── ...
└── model_retrained_3per/
    ├── pytorch_model.bin
    ├── config.json
    └── ...
```

**Cách upload**: Tương tự dataset 1 — zip + upload qua web UI hoặc CLI.

→ Path: `/kaggle/input/forget-mi-models/`

> ⚠️ Nếu đặt tên dataset khác (vd `mimic-data`), nhớ **đổi paths** trong `config_baseline_kaggle.yaml`.

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
