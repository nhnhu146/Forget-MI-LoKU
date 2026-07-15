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
2. ~~LoKU IU sweep~~ **✅ XONG** — xem ĐÃ XONG.
3. **Unlearning baselines cho IU 3%** (neggrad/cfk/euk) — cell "UNLEARNING BASELINES" với `UB_DATASET='iu'`. *Nice-to-have* (bảng cross-dataset đầy đủ); bỏ được nếu hết quota.
4. **LoKU-D multi-seed** (123, 7) @ 3/6/10% — HOÃN.
5. **Bảng so sánh cuối** gồm mọi method (Retrain|NegGrad+|CF-k|EU-k|Forget-MI|LoKU) × (MIMIC 3/6/10% + IU 3%) — dựng sau khi đủ số. **user sẽ gửi file thật** exp_028(6%)/exp_058(10%) + CSV hoangnhu4 để verify (tôi đã merge tạm từ số paste).

### ✅ ĐÃ XONG (khỏi chờ)
- **baseline 3% MIMIC** seed42 first result: Df_AUC 0.561 (over-forget E29) — xem mục 1.
- **IU preprocess** → `forget-mi-data-iu` (metadata, `--link_mode skip`, ~3MB). Ảnh dùng raddar trực tiếp.
- **IU og + re model**: og (`model_og_IU`) 15 epoch 110min val_acc~1.0; re (`model_retrained_iu_3per`) 15 epoch 108.7min val_acc~1.0 (train trên train−forget3%: 6130 vs og 6321). Cùng 15 epoch/setup → mốc vàng công bằng. Cả 2 đã train xong (453MB mỗi cái).
- **IU model datasets ĐÃ TẠO**: og→`forget-mi-models-iu` (lồng 2 lớp `forget-mi-models-iu/forget-mi-models-iu/base_model/...`, glob `**` tự xử lý), re→`forget-mi-models-iu-re`. Baseline sửa để nhận 2 dataset riêng (e902dbb) + fix bug `models_root=''` 3 chỗ (fb9b728); loku sẵn ổn (glob đệ quy).
- **baseline IU 3% seed42 XONG** (recover từ Save Version, commit 50fdfc2): Df_AUC 0.653, Dt_AUC 0.636, MIA_persample 0.565, forget_ce 4.718>test_ce 3.355 (over-forget E29 giống MIMIC). `perepoch_baseline_iu_3per.csv` đủ 30 epoch → điểm quên-lành-mạnh (forget_ce≈test_ce) ~E21, Df_AUC~0.77.
- **unlearn baselines MIMIC 3% (full retain)**: CF-k Df_AUC **0.927** ✅ (không unlearn, ~paper). **EU-k 30ep = 0.555** ✅ (≈ Retrain gold 0.566 — hợp lý, EU-k là retrain-like; test AUC 0.590 hơi tụt do overfit; khác paper 0.996 nhưng reimplement hợp lệ). NegGrad+ 0.049 forget_ce=54 → **PHÂN KỲ** (unbounded GA); user chọn **(A) báo cáo trung thực**, KHÔNG chạy lại.
- **baseline MIMIC 6% + 10% XONG** (E29, no per-epoch): **6% Df_AUC 0.679** (≈paper 0.654, forget_ce 2.476≈test_ce 2.250 = LÀNH MẠNH ✅), **10% Df_AUC 0.757** (>paper 0.656, forget_ce<test_ce, quên nhẹ hơn). → Pattern: 3% over-forget (0.561) nhưng **6/10% lành mạnh sát paper** (forget lớn → unlearning ổn định hơn). Merge tạm từ paste (commit 6e8e792); chờ file thật + exp_028/exp_058 MD.
- code LoKU chạy Kaggle + sweep 27 config (seed 42), config D chốt: `D_combo_aggressive` (IHL=1.25, img=0.5, 8ep, kappa=2.0), quên lành mạnh.
- **LoKU IU sweep 5 config (exp063-067) XONG**. **CHỌN config C (exp065)** làm kết quả chính IU: honest (teacher=og), **ihl=3.0**, early_stop=none 15ep. MIA_ps **0.550**, Df_AUC 0.714, Df_F1 0.462, Dt_AUC 0.620, Dt_F1 0.554, forget_ce/test 2.90/2.09 (over nhẹ 1.39×), 0.13h, 0.45%. **Fix cho IU = tăng IHL (1.25→3.0)+bỏ early-stop, KHÔNG phải nhiều epoch** (IHL_frg bão hòa, thêm epoch vô ích — trả lời câu hỏi epoch của user). A/B/E dùng `teacher=re` = **ORACLE mớm θ_re** (số đẹp hơn: E MIA 0.497/Dt 0.667) → KHÔNG hợp lệ làm main, chỉ là cận trên; luận văn ghi rõ minh bạch. D=NegGrad phá model (Df 0.403, forget_ce 6.3). **ĐÃ ĐIỀN tab:main-iu (dòng LoKU=C) + tab:efficiency IU + mục 4.9 + thảo luận 4.12 + tổng kết 4.13.**

### 📝 BÁO CÁO LaTeX (D:\LatexProjects\KLTN_NHNHU\Chapter4\chapter4.tex — KHÔNG phải repo git)
- **VIẾT LẠI TOÀN BỘ Chương 4 theo barem 13 mục user đưa** (2026-07-02): 4.1 Mục tiêu(RQ1/2/3) → 4.2 Dữ liệu → 4.3 Kịch bản → 4.4 Phương pháp so sánh → 4.5 Độ đo → 4.6 Môi trường → **4.7 Tái lập baseline MIMIC** (tab:repro: tái lập vs công bố 3/6/10%) → 4.8 Đề xuất MIMIC (tab:main-mimic+ratios) → 4.9 IU → 4.10 Ablation → 4.11 Chi phí → 4.12 Thảo luận → 4.13 Tổng kết. main.tex chỉ \include, KHÔNG chứa số (user tưởng chưa điền vì mở main.tex).
- **LỖI ĐÃ SỬA khi viết lại**: (1) "MIA gần 0.5 càng tốt" → SỬA thành "MIA càng THẤP (gần 0=gold) càng tốt" (2 chỗ). (2) tab:config ghi ihl=0.75/γ_img=0.3 → SỬA thành ihl=1.25/γ_img=0.5 (khớp config D thật). (3) abl-ihl "0.75 chọn" SAI → sửa "1.25 chọn".
- **ĐÃ ĐIỀN (số thật)**: tab:dataset; tab:repro(3/6/10%: tái lập 0.561/0.679/0.757 vs công bố 0.735/0.654/0.656); tab:main-mimic 3%(FMI 0.561 + NegGrad0.049/CFk0.927/EUk0.555 + LoKU 0.651); tab:ratios; tab:efficiency(FMI 2.96h vs LoKU 0.17h/0.45%); **tab:main-iu Forget-MI tái lập IU 0.653** (MIA0.565/DfF1.573/Dt.636/DtF1.571 — recover results(5)); **tab:abl-ihl 2 điểm SẠCH** (λ=1.0 config C: MIA.358/Df.692/Dt.673/CE1.76-1.64; λ=1.25 config D chọn: .373/.651/.674/2.42-1.92 — cùng img0.5/8ep/κ2.0); abl-fisher/modality/rank điền dòng "đề xuất"(=LoKU D), dòng biến thể "--"; **sec:discussion + 4.13 tổng kết VIẾT XONG**.
- **θ_og/θ_re MIMIC ĐÃ ĐIỀN từ số công bố bài báo** (Bảng 1a docx dịch, C:\Users\admin\Downloads\Forget-MI_ban_dich_tieng_Viet.docx): Original MIA1.000/Df0.999/DfF1.965/Dt0.677/DtF1.388; Retrain MIA0.000/Df0.566/DfF1.310/Dt0.626/DtF1.362. Điền cột MIA_paper (per-batch), MIA_ps để "--". Dấu ‡ + footnote caption ghi rõ "số công bố". **Số công bố đầy đủ Bảng 1 (3/6/10%) đã có trong scratchpad/fmi_text.txt** nếu cần: 6% Retrain Df0.675/MIA0.769; 10% Retrain Df0.588/MIA0.190; Original ~ giống nhau mọi tỉ lệ (Df0.999).
- **θ_og/θ_re IU ĐÃ EVAL XONG (2026-07-03)** bằng mẹo `unlearn_epochs=0` (eval-only, đã fix bug retain_dataloader unbound, commit cc6c1f2). Cell tự-dò-path (glob) + trỏ ảnh raddar, config_baseline_iu_kaggle.yaml. **θ_og: MIA 0.990/Df-AUC 1.000/DfF1 1.000/Dt 0.678/DtF1 0.621, forget_ce 0.002** (nhớ hết). **θ_re GOLD: MIA 0.529/Df-AUC 0.651/DfF1 0.587/Dt 0.673/DtF1 0.617, forget_ce 2.329≈test 2.293** (lành mạnh chuẩn). → **tab:main-iu ĐẦY ĐỦ**. Mốc vàng Df-AUC IU = **0.651**: Forget-MI 0.653 trùng gold NHƯNG qua over-forget (CE 4.72≫3.36); LoKU-C 0.714 hơi cao hơn gold nhưng CE 2.90/2.09 gần gold 2.33/2.29 → lành mạnh hơn. LoKU MIA 0.550 gần gold 0.529 hơn FMI 0.565.
- **baseline 6% MIMIC**: run mới của user (kernel 125577582) KHÔNG push (thiếu secret) nhưng số TRÙNG KHÍT repo (đã có từ df58567). neggrad_mimic_6per MỚI trong CSV = PHÂN KỲ (forget_ce 117, Df nan).
- **unlearn baselines 6% MIMIC XONG (2026-07-03)**: cfk_mimic_6per Df 0.926 (không quên, ≈og), euk_mimic_6per Df 0.540 (dưới gold 0.675, over nhẹ, Dt 0.58 tụt). → 6% đủ cả 3 baselines.
- **10% baselines ĐỦ TRỌN (2026-07-06)**: NegGrad ✅(nan phân kỳ, 6.7h), CF-k ✅(0.92, 3.0h), **EU-k ✅ (Df 0.560, MIA 0.226, retain-CE 0.003 overfit, Dt 0.602, forget/test 3.99/3.59, 2.73h)**. → **TẤT CẢ MIMIC baselines 3/6/10% XONG** (9 ô). EU-k chạy riêng có GPU (7.2GB, 5.5min/epoch × 30 = 2.73h). NegGrad chậm do lặp FULL retain ×2 forward ×30ep.
- **NOTEBOOKS ĐÃ PRESET cho reimport (commit 9ad593f, 18e10ca)**: (1) run_kaggle_loku.ipynb +CELL 4g ablation (5 biến thể: lora_random/text_only/ihl_zero(λ=0)/rank4/rank16, tự-dò-path, cần cờ loku_random_init commit 7665ea2); (2) baseline UB cell → euk 10%; (3) loku 4b/4c RUN_6PER/10PER=True (multi-seed config D seeds 42,123,7 id loku_Nper). **Multi-seed config D CHƯA có** (final_ihl125 cũ dùng img=0.3 + seed123/7 kẹt Running → BỎ). 3 account song song: P1 ablation / P2 euk10% / P3 multiseed.
- **IU baselines cell ĐÃ THÊM** (commit bf75b57): baseline nb cell self-contained "UNLEARNING BASELINES cho IU", tự-dò-path + trỏ raddar (vá bug _iu_img: Cell 3 chỉ tìm ảnh dưới forget-mi-data-iu, KHÔNG thấy raddar dataset riêng → _check_dataset_ready fail). Mặc định IU_METHODS=('cfk','euk') ~6h (NegGrad ~6.7h+phân kỳ → bỏ, cả 3 tràn 12h). id=<m>_iu_3per.
- **3 ACCOUNT SONG SONG**: acc1=P1 ablation(loku CELL 4g) · acc2=EU-k 10% MIMIC(baseline UB cell preset) · acc3=IU baselines cfk+euk(baseline cell IU mới). Reimport cả 2 notebook.
- **CÒN THIẾU (cập nhật)**: 5 ablation (P1) + euk_10per (P2) + multiseed 123/7 (deferred) hoặc IU baselines (acc3 chọn cái này). Xong P1+P2+acc3 là Chương 4 trọn.
- **CÒN "--"**: tab:main-iu og/re (chờ user gửi); 10% forget/test-CE tab:repro; abl-fisher LoRA-random, abl-modality text-only, abl-rank r=4/16 (cần run riêng, KHÔNG bịa).

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

## 5. KHUNG IN + THỜI GIAN (đổi 2026-07-15)
- **Khung in kết quả (`_final_evaluation` baseline + final eval LoKU) ĐÃ BỎ cột "PAPER TARGET"**. Giờ in gọn: Forget/Test (AUC/F1/CE), MIA (persample/paper), 1−CosSim, khối Time + Compute. User KHÔNG muốn cột so paper nữa.
- **Bóc tách thời gian (cả baseline + LoKU)**: `unlearn()` trả **dict timing** `{train_h, monitor_h, ckpt_h, wall_h}`; main thêm `fisher_h` (LoKU=Fisher+FILA init; baseline=0) + `load_h` + `final_eval_h`. **Số THỜI GIAN CÔNG BẰNG dùng cho luận văn = `unlearn_core_hours` = train_h + fisher_h** (KHÔNG gồm monitor CosSim per-epoch của baseline — chỉ để log, không cần cho unlearning). CSV cột cũ `unlearn_time_hours` = **wall** (giữ nguyên để notebook aggregate không vỡ).
- **CSV**: dùng helper chung `_append_row_csv` (trong `forgetmi_partial.py`, LoKU import lại) — tự migrate header khi thêm cột, **ghi theo thứ tự header thực → không lệch cột** (thay khối schema-migration cũ). Đã test độc lập OK.
- **LƯU Ý báo cáo**: tab:efficiency cũ (FMI 2.96h vs LoKU 0.17h) là **wall**; baseline wall bị "thổi" bởi monitor CosSim per-epoch. Re-run để lấy `unlearn_core_hours` (train thuần) cho so sánh thời gian chuẩn hơn. Notebook Cell 5 vẫn hiện `unlearn_time_hours` (wall) — đổi sang `unlearn_core_hours` nếu muốn.
