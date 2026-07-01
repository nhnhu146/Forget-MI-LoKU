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
- **Bug đã sửa**: weights bị đổi sang `4/4/1/1` (Unimodal) → over-forget. Paper `config.yaml` dùng `alpha=1,beta=1,theta=2,gamma=2`. Đã sửa.
- **LỜI THẦY (kim chỉ nam)**: "chạy code gốc mà số ra kì thì cứ kệ nó" → **KHÔNG đẽo gọt để ra số paper** (loại phương án curve-fit về 0.735).
- **KẾT QUẢ baseline 3% MIMIC (seed 42, 30 epoch, config đúng)**: Df_AUC=**0.561**, Df_F1=0.156, Dt_AUC=0.632, MIA_persample=0.493, MIA_paper=1.000 (per-batch nhiễu), forget_ce=4.807 ≫ test_ce=3.446 → **OVER-FORGET**. Weights đã đúng mà vẫn over-forget → **nguyên nhân KHÔNG phải weights** mà là **eval ở epoch CUỐI (E29)**: code gốc lưu MỌI epoch + `eval_unlearning.py` chọn 1 checkpoint; pipeline mình xóa hết chỉ eval E29 (chỗ over-forget nhất).
- **DIỄN GIẢI ĐÚNG (quan trọng)**: thước đo thật = **model gold retrained**, KHÔNG phải "Df_AUC thấp nhất". Df_AUC 0.561 = quên QUÁ ĐÀ (vượt gold) + phá model (CosSim 0.66→0.565). Over-forget là **ĐIỂM YẾU của Forget-MI**, không phải mạnh. LoKU thắng bằng **quên lành mạnh + giữ retain/MIA + hiệu quả**, KHÔNG phải "quên mạnh hơn".
- **GIẢI PHÁP đã code (commit 186b87f)**: thêm eval per-epoch READ-ONLY (`--override eval_every_epoch=1`, toggle `EVAL_EVERY_EPOCH` notebook baseline, mặc định bật). Mỗi epoch ghi `/kaggle/working/perepoch_<id>.csv` (Df_AUC/Dt_AUC/MIA/forget_ce/test_ce...). **1 run ra CẢ HAI**: E29 trung thực (RNG được khôi phục → y hệt bản gốc) + quỹ đạo để dò epoch khớp paper. Re-run baseline 3% để lấy quỹ đạo.

## 2. TRẠNG THÁI HIỆN TẠI + ĐANG CHỜ
- **Deadline ~1 tuần (từ 2026-06-30)**. Bắt buộc có **IU-CXR 3% rút gọn** (1–2 seed); IU đầy đủ đã cắt.
### ⏳ ĐANG CHỜ (cập nhật khi xong thì XÓA dòng đó)
1. **baseline 3% MIMIC re-run có per-epoch eval** (account hoangnhu2) → lấy `perepoch_*.csv` (quỹ đạo Df_AUC/MIA theo epoch). E29 đã có (0.561); re-run để có TRAJECTORY dò epoch khớp paper. RNG được khôi phục nên E29 không đổi.
2. **baseline 6% + 10% MIMIC** — chạy SONG SONG account khác, **mỗi account 1 %** (per-epoch eval làm mỗi run ~4.5h → cả 3 không vừa 12h). Cờ: account B `RUN_6PER=True` (tắt 3/10), account C `RUN_10PER=True` (tắt 3/6). Paper ref Df_AUC 0.654(6%)/0.656(10%). Để ý log gold `model_retrained_6per/10per` (nếu fallback→3per thì dist_vs_re vô nghĩa).
3. **IU baseline 3% (kết quả) + LoKU IU 3%**. baseline IU đang chạy trên hoangnhu03 (đã qua detection/gold OK, đang eval). Cảnh báo `Skipping AUC pair X vs Y / No positive samples` là **BÌNH THƯỜNG** — IU nhị phân (0/1) chạy trên head 4-class → lớp 2,3 rỗng; cặp 0v1 vẫn tính (AUC nhị phân hợp lệ). CHỜ: metrics IU cuối + `perepoch_baseline_iu_3per*.csv`. LoKU IU 3% **chưa chạy**. Attach 4 dataset: forget-mi-data-iu + raddar + forget-mi-models-iu + forget-mi-models-iu-re. ⚠️ hoangnhu03 nếu THIẾU Secret GITHUB_TOKEN → tải kết quả thủ công.
4. **Unlearning baselines NegGrad+/CF-k/EU-k** (Forget-MI Table 1) — CODE XONG (`scripts/unlearn_baselines.py` + cell "UNLEARNING BASELINES", commit 56b1c12). CHỜ user CHẠY: cell đã set sẵn `UB_FORGET=3, UB_METHODS=('neggrad','cfk','euk'), UB_DATASET='mimic', seed 42`. Attach forget-mi-data + forget-mi-models-full. Mỗi method **tự push sau khi xong** (`_push_progress` → resume-able, `_method_done` skip cái đã có). ~10-20p/method, ~1h tổng. Paper @3%: cả 3 **MIA≈1.0, Df_AUC≈1.0** (quên KÉM). CSV method=neggrad/cfk/euk (dedup keyed cả 'method'). Rồi mở 6/10% + IU (đổi UB_FORGET / UB_DATASET).
5. **LoKU-D multi-seed** (123, 7) @ 3/6/10% — HOÃN (chưa gấp).
6. **Bảng so sánh cuối** gồm mọi method (Retrain|NegGrad+|CF-k|EU-k|Forget-MI|LoKU) — dựng sau khi có đủ số trong CSV.

### ✅ ĐÃ XONG (khỏi chờ)
- **baseline 3% MIMIC** seed42 first result: Df_AUC 0.561 (over-forget E29) — xem mục 1.
- **IU preprocess** → `forget-mi-data-iu` (metadata, `--link_mode skip`, ~3MB). Ảnh dùng raddar trực tiếp.
- **IU og + re model**: og (`model_og_IU`) 15 epoch 110min val_acc~1.0; re (`model_retrained_iu_3per`) 15 epoch 108.7min val_acc~1.0 (train trên train−forget3%: 6130 vs og 6321). Cùng 15 epoch/setup → mốc vàng công bằng. Cả 2 đã train xong (453MB mỗi cái).
- **IU model datasets ĐÃ TẠO**: og→`forget-mi-models-iu` (lồng 2 lớp `forget-mi-models-iu/forget-mi-models-iu/base_model/...`, glob `**` tự xử lý), re→`forget-mi-models-iu-re`. Baseline sửa để nhận 2 dataset riêng (e902dbb) + fix bug `models_root=''` 3 chỗ (fb9b728); loku sẵn ổn (glob đệ quy).
- code LoKU chạy Kaggle + sweep 27 config (seed 42), config D chốt: `D_combo_aggressive` (IHL=1.25, img=0.5, 8ep, kappa=2.0), quên lành mạnh.

### Workflow IU (tham chiếu): preprocess(skip) → train og+re (warm-start MIMIC, attach raddar) → tạo 2 dataset model → baseline IU + LoKU IU. Mỗi run IU attach: **forget-mi-data-iu + raddar/chest-xrays-indiana-university** (+ model IU). Xem `HUONG_DAN_IU.md`.

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
