---
name: luu-y-dac-biet
description: ĐỌC FILE NÀY TRƯỚC MỖI LẦN LÀM VIỆC — tất cả lưu ý quan trọng về dự án Forget-MI-LoKU và cách làm việc với user
metadata:
  type: project
---

# ⚠️ LƯU Ý ĐẶC BIỆT — đọc trước khi làm bất cứ việc gì

> File tổng hợp mọi lưu ý. Cập nhật khi có thay đổi (đặc biệt mục 2 — trạng thái).

---

## 0. QUY TẮC LÀM VIỆC VỚI USER
- **Commit theo pathspec**: `git commit <đường/dẫn/file> -m "..."`. KHÔNG `git add .` / `git add -A`. Lý do: user chạy **nhiều tab Claude** trên cùng repo → index dùng chung, dễ cuốn nhầm file tab khác. Sau commit, `git show --stat <sha>` kiểm tra không lẫn file lạ. Chỉ 1 tab nên sửa 1 file tại 1 thời điểm.
- **Sau mỗi lần sửa cho 1 experiment**, chủ động đưa **commit message sẵn để dán** (style: `fix(...)/feat(...)`, kèm `Co-Authored-By` trailer). User tự commit/push từ local.
- **Hiểu đúng vấn đề trước khi sửa.** User đã phải sửa lưng tôi nhiều lần — đừng "sửa linh tinh" hay tự chế logic; hỏi lại nếu chưa chắc.

## 1. BASELINE PHẢI KHỚP PAPER (quan trọng nhất)
- Baseline tái lập (`training/forgetmi_partial.py`) là **THƯỚC ĐO** để chấm LoKU — **KHÔNG** dùng số paper công bố (so với paper bị lệch pipeline/confounded). Cả LoKU và baseline phải chạy **cùng pipeline/máy/eval/seed** → đối chứng có kiểm soát.
- NHƯNG baseline phải **tái lập ĐÚNG số paper** (Df_AUC~0.735, MIA~0.571 ở 3%) để đáng tin — **bằng paper, không hơn không kém**.
- **Dùng NGUYÊN code + config gốc**, KHÔNG sửa logic, KHÔNG "cải tiến", KHÔNG tự chế chọn-epoch. Code gốc ở `C:\Users\admin\Downloads\Forget-MI-main\Forget-MI-main`. Logic train trong repo đã giống hệt gốc (data_split, vòng unlearn, loss, hinge, `optimizer.step()` 1 lần/epoch).
- **Bug đã sửa**: weights bị đổi sang `4/4/1/1` (Unimodal) → over-forget (Df_AUC 0.571). Paper `config.yaml` dùng `alpha=1,beta=1,theta=2,gamma=2` (0.17/0.17/0.33/0.33). Đã sửa trong `config_baseline_kaggle.yaml`. lr: paper sweep [1e-4,1e-5], đang để 1e-5; thử 1e-4 nếu Df_AUC còn thấp.

## 2. TRẠNG THÁI HIỆN TẠI + ĐANG CHỜ
- **Deadline ~1 tuần (từ 2026-06-30)**. Bắt buộc có **IU-CXR 3% rút gọn** (1–2 seed); IU đầy đủ đã cắt.
- **⏳ ĐANG CHỜ (quan trọng)**: kết quả **baseline 3% MIMIC (1 seed)** chạy lại với config đúng paper (weights 1/1/2/2). Repo CSV `results_summary_kaggle.csv` đã reset header-only + xóa summary cũ → user phải chạy trong **session Kaggle MỚI** để Cell 1 khôi phục CSV sạch. Kiểm tra **Df_AUC≈0.735, MIA≈0.571**; chưa khớp → đổi lr 1e-5→1e-4. Cả `config_baseline_kaggle.yaml` và `config_baseline_iu_kaggle.yaml` đều đã sửa weights về 1/1/2/2.
- **⏳ ĐANG CHỜ (IU)**: user tạo dataset **`forget-mi-data-iu`** từ preprocess. Preprocess OK (3826 report, 7426 ảnh) + fix: header all_data.tsv, dò ảnh đệ quy. **QUYẾT ĐỊNH ẢNH: `--link_mode skip`** (KHÔNG đóng gói ảnh) — vì copy 7426 file ~2GB làm Kaggle treo lúc lưu output. → `forget-mi-data-iu` chỉ chứa metadata (~3MB). Lúc train/baseline/loku IU phải **attach kèm dataset raddar** và trỏ `img_data_dir` sang `images/images_normalized` của raddar (tên file khớp `<dicom_id>.png`). Đã sửa `preprocess` + `run_kaggle_train_iu` (fallback raddar); **CÒN PHẢI sửa baseline IU cell + loku IU cell** trước khi chạy 2 bước đó. Sau đó mới train og+re → baseline IU → LoKU IU. Xem `HUONG_DAN_IU.md`.
- **Config LoKU đã chốt**: `D_combo_aggressive` (IHL=1.25, img=0.5, 8ep, kappa=2.0) — quên lành mạnh (forget_ce≈test_ce), thắng số paper. Mới seed 42, cần multi-seed (123, 7). Sweep: `run_kaggle_loku_sweep.ipynb` chưa có push-mỗi-config (timeout mất việc); 10% mới A–G (thiếu H/I/J).
- **Đã xong & trên GitHub**: code LoKU chạy được Kaggle; sweep 27 config (seed 42). **Chưa**: baseline đúng-paper (đang chạy lại), LoKU-D multi-seed, toàn bộ IU 3%.

## 3. CÁCH TÍNH MIA
- Đặc trưng = cross-entropy **nhánh ảnh (img_logits)** thôi. `SVC(C=3,rbf)` tách retain(member=1)/test(non-member=0) → **MIA = tỉ lệ forget bị đoán "member"** ∈ [0,1].
- **MIA_persample** (per-sample, balanced) | **MIA_paper** (per-batch-mean, replicate `eval_unlearning.py` gốc, thô ~7 điểm → cần multi-seed).
- **Thang đo: THẤP = tốt. Hoàn hảo ≈ 0** (= retrained gold), Original ≈ 1. **KHÔNG phải 0.5.**
- Chỉ đo ảnh vì report y khoa khuôn mẫu; đó là lý do có **Image-FILA**.
- **Bẫy MIA=0 giả**: do over-forget (`forget_ce >> test_ce`). Luôn báo cáo kèm forget_ce/test_ce (lành mạnh khi ≈ nhau, code cảnh báo nếu >1.3×).

## 4. KAGGLE GOTCHAS
- User chạy **nhiều account** (hoangnhu03, hoangnhu2…). **Mỗi account phải cài Secrets** `GITHUB_TOKEN`/`GIT_EMAIL`/`GIT_NAME`, nếu không auto-push **skip im lặng** → kết quả kẹt trong `/kaggle/working`.
- **Giới hạn 12h/session** → run dài timeout. Đã có push-mỗi-seed (baseline). Chạy ≤2 seed/ít config mỗi session; resume tự skip qua CSV.
- **Cứu khi timeout**: bảo user tải `/kaggle/working/results_summary.csv` (+exp_*.md) từ Output của Save Version → tôi merge vào repo CSV + regenerate summary + push.
- **Sửa notebook phải re-import**: Cell 1 `git reset` chỉ cập nhật **code repo**, KHÔNG cập nhật **cell notebook** (nằm trong editor Kaggle). Sửa cell xong, user phải re-import notebook từ GitHub.
- **Dataset slugs**: `forget-mi-models-full` (base `original_model/forgetme/training_original_model`, gold `model_retrained_<N>per/model_retrained_<N>per`) + `forget-mi-data`. IU: `forget-mi-data-iu` / `forget-mi-models-iu`.
