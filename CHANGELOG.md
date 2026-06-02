# CHANGELOG — Nhật ký thay đổi code lớn

> Nơi ghi lại **các thay đổi code/kiến trúc đáng kể** (thêm cơ chế mới, đổi pipeline,
> thêm loss/PEFT mới...). Khác với `experiments/exp_NNN_*.md` (chỉ track **config + kết quả**
> của từng lần chạy), file này tập trung vào **"đã sửa gì trong code và vì sao"** để sau này
> đọc lại không phải lần mò qua git diff.
>
> **Quy ước ghi** (mới nhất ở trên cùng):
> - Mỗi mục: `## [tag/exp] — YYYY-MM-DD — tiêu đề ngắn`
> - **Động cơ**: vì sao cần sửa (gắn với kết quả/bottleneck nào).
> - **Thay đổi**: liệt kê file + hàm + ý chính (không cần dán code).
> - **Config liên quan**: param mới/đổi trong `config.yaml`.
> - **Kiểm chứng**: đã test/verify ra sao.
> - **Rủi ro / cần theo dõi**: điều có thể hỏng, knob để revert.

---

## [exp11] — 2026-06-02 — Bỏ phụ thuộc F_re khi train (retain-distill ← F_og + IHL)

**Động cơ.** exp10c đạt số đẹp (thắng paper cả 5 metric qua 3 seed) NHƯNG distill dùng
**F_re (model retrained)** làm teacher → (1) cần retrain mới có F_re ⇒ phá tiền đề "unlearn
thay vì retrain"; (2) gần như "học tủ" cho chính metric đánh giá (CosSim đo độ giống F_re;
Forget-AUC mục tiêu về phía F_re). Phản biện sẽ bắt lỗi. Cần chứng minh phương pháp vẫn tốt
mà KHÔNG dùng gold model khi train.

**Thay đổi.** (`training/forgetmi_loku.py`) thêm config `distill_teacher`:
- `"og"` (exp11): retain-distill teacher = **F_og** (model gốc, luôn có sẵn). Tái dùng forward
  `model_og` đã tính cho UKR/MKR (bắt thêm logits `og_ret_il/og_ret_tl`) → KHÔNG tốn forward
  thêm, lại bỏ hẳn forward `model_re` khi train. Forget-distill tự tắt (F_og biết forget) →
  forget xử lý bằng **IHL**. `F_re` chỉ còn dùng để ĐÁNH GIÁ (CosSim) — hợp lệ.
- `"re"` (mặc định, exp10c): giữ nguyên hành vi cũ (teacher = F_re).

**Config (exp11).** `distill_teacher: "og"`, `distill_forget_weight: 0.0`, `ihl_forget_weight: 1.5`,
giữ image-FILA (`lora_image_last_k_blocks=1`, `loku_image_subtract_scale=0.3`).

**Notebook.** Cell 4 multi-seed đổi sang `EXP_NAME="exp11_no_fre_ihl"`; Cell 3.5 verify thêm
`distill_teacher` + check `use_og_teacher`.

**Kiểm chứng.** `py_compile` OK. Khi `teacher='og'`: bỏ qua forward F_re lúc train, dùng
logits F_og đã có; forget-distill bị vô hiệu an toàn (teach_frg=None, distill_frg_w→0).

---

## [multi-seed] — 2026-06-02 — Hỗ trợ chạy nhiều seed + tổng hợp mean±std

**Động cơ.** Một lần chạy không đủ tin: MIA_persample dao động ~0.02 giữa các lần, MIA_paper
lượng tử hóa ~1/7. Cần chạy nhiều seed và báo cáo mean±std cho luận văn.

**Thay đổi.**
- (`training/forgetmi_loku.py`) thêm cờ `--seed` override `config.random_seed` (đặt trước
  `wandb.init`). Data split vẫn cố định (random.seed(0)/random_state=42) nên forget/retain/test
  KHÔNG đổi giữa các seed — chỉ phần ngẫu nhiên huấn luyện/đánh giá đổi (đúng tinh thần).
- (`run.ipynb` Cell 4) chuyển sang vòng lặp multi-seed [42,123,7]: mỗi seed train+eval đầy đủ
  (`--fresh --seed`), tự đọc `results_summary.csv` tính mean±std, in bảng và ghi
  `experiments/<exp>_multiseed_summary.md` (Cell 5 push). Mỗi seed vẫn auto-track riêng.

**Kiểm chứng.** `py_compile` OK.

**Động cơ.** MIA của LoKU dùng loss **per-sample** (`per_sample_ce`), còn MIA gốc của
Forget-MI (`evaluation/eval_unlearning.py`) dùng loss **trung bình mỗi BATCH** rồi mới đưa
vào SVM. Hai cách khác nhau → so "MIA vs paper" không apples-to-apples. Cần BỔ SUNG (không
thay thế) một MIA tính đúng kiểu paper để so sánh chuẩn.

**Thay đổi.** (`training/forgetmi_loku.py`) `run_mia` giờ trả về dict 2 giá trị, tính từ
CÙNG một lượt forward (không tốn thêm GPU):
- `persample` — giữ nguyên metric cũ (SVM trên per-sample, balanced retain/test).
- `paper` — gom per-sample loss theo chunk `mia_paper_batch_size` (mặc định 32) qua helper
  `_batch_means`, rồi chạy SVC(C=3, rbf, gamma=auto), retain+test KHÔNG cân bằng — y hệt
  `eval_unlearning.py`. In ra bảng kết quả 2 dòng: `MIA_persample` và `MIA_paper`.
- Log `final/MIA` (giữ key cũ) + `final/MIA_paper`; thêm vào CSV + bảng MD (exp_tracker.py).

**Config.** thêm `mia_paper_batch_size: 32` (khớp default của eval gốc).

**Kiểm chứng.** `py_compile` OK (cả exp_tracker.py); `_batch_means(201, 32)` → 7 batch-mean
đúng, chunk-mean ≡ per-batch CE mean. MIA_persample giữ nguyên công thức cũ.

**Sửa lỗi trung thực (cùng ngày).** Bản đầu subsample retain xuống 512 *trước* khi tính MIA
→ MIA_paper chỉ train SVM trên ~16 retain batch-means (paper dùng full 5409 → ~169) nên KHÔNG
trung thực, cho ra 0.143 không so sánh được. Sửa: `run_mia` dùng **FULL retain** (bỏ `max_retain`),
cả 2 MIA lấy từ 1 lượt forward full retain; CosSim vẫn subsample (`eval_max_retain`) vì nó mới là
phần tốn 2/3 thời gian (forward 2 model). Eval vẫn ~5× nhanh hơn bản gốc. Lưu ý: MIA_paper vốn THÔ
(forget 201 mẫu / bs32 = 7 điểm → giá trị bội 1/7, phương sai cao) → bắt buộc multi-seed khi báo cáo.

---

## [eval-speedup] — 2026-06-02 — Tăng tốc bước evaluation ~6–10×

**Động cơ.** Train rất nhanh (vòng lặp `zip(forget, rand, retain)` dừng ở forget ~201 mẫu →
chỉ ~13 batch/epoch) nhưng eval lại chậm vì duyệt TOÀN BỘ retain (5409) tới 3 lần, ở fp32.
Hai chỗ lãng phí thật sự: (1) MIA forward cả 5409 retain rồi chỉ giữ ~531 (cân bằng theo
len(test)); (2) CosSim chạy full retain × 2 model.

**Thay đổi.** (`training/forgetmi_loku.py`)
- Thêm `_eval_autocast()` (fp16) và bọc quanh forward trong `per_sample_ce`,
  `cosine_sim_models`, `perf_metrics` → mỗi forward nhanh ~2×.
- Thêm `_subsample_dataset(dataset, max_n, seed)`; `run_mia` nhận `max_retain` và subsample
  retain TRƯỚC khi forward; CosSim cũng dùng retain đã subsample trong `run()`.

**Config.** (`config.yaml`) thêm `eval_max_retain: 512` (0 = full/giữ hành vi cũ để lặp lại
kết quả cũ khi cần).

**Kiểm chứng.** `py_compile` OK. Metric gần như không đổi: MIA vốn chỉ dùng ~531 mẫu;
CosSim/AUC trên ~512 mẫu ≈ trên 5409 (sai số rất nhỏ). Thuần tối ưu tốc độ, không đụng logic unlearning.

---

## [exp10] — 2026-06-01 — Modality-aware PEFT: FILA subtraction lên nhánh ẢNH

**Động cơ.** Chẩn đoán từ kết quả exp01–09: MIA (`evaluation/eval_unlearning.py`,
`forgetmi_loku.py::per_sample_ce`) **chỉ dùng cross-entropy của `img_logits`** làm feature.
Nhưng LoRA + FILA subtraction của exp08 **chỉ chạm BERT (text) `query/key/value`** — image
encoder (`ImageResNet`) và `fc1` không bị đụng tới, các loss khi train lại chỉ chạy trên retain
→ behavior trên forget của nhánh ảnh gần như không đổi → **Forget AUC kẹt ở 0.833** (gold
retrained = 0.566). Nói cách khác: ta đang "quên" sai chỗ so với cái MIA đo.

**Thay đổi.** (`training/forgetmi_loku.py`)
- Thêm `resolve_image_targets(model, last_k_blocks, include_fc1)`: liệt kê **full-name** các
  `nn.Conv2d` ở các stage cuối của image encoder (`img_model.layer{N}.*`), tùy chọn thêm
  `img_model.fc1`. Trả full-name để dùng được cho cả PEFT (exact-match) lẫn Fisher (substring).
- Tách `_fila_decompose()` và tổng quát hóa `apply_loku_soft_init()`:
  - Hỗ trợ **Conv2d (weight 4D)**: reshape `[out,in,kh,kw]→[out,in·kh·kw]` để SVD, rồi reshape
    factor về `lora_A=[r,in,kh,kw]`, `lora_B=[out,r,1,1]` (đúng API conv-LoRA của PEFT).
  - **Clamp rank + zero-pad** để chạy được trên ma trận nhỏ (vd `fc1` 768→4).
  - Thêm **scale riêng cho ảnh** (`image_subtract_scale`) vì conv + BatchNorm nhạy hơn BERT.
- `run()`: mở rộng `target_modules` bằng image targets **trước** khi tính Fisher & wrap PEFT;
  guard không unfreeze base của `fc1` khi `fc1` đã được PEFT-wrap.

**Config liên quan.** (`config.yaml`) thêm:
- `lora_image_last_k_blocks: 1` — 1 = chỉ `layer7`; 2 = +`layer6`; **0 = tắt (revert exp08)**.
- `lora_image_include_fc1: false` — bật để FILA cả image classifier head.
- `loku_image_subtract_scale: 0.5` — scale FILA cho ảnh (nhẹ hơn text=1.0).
- `ihl_forget_weight: 0.0` — **tắt IHL** để cô lập hiệu ứng image-FILA (exp10 = exp08 + nhánh ảnh).

**Kiểm chứng.**
- `python -m py_compile training/forgetmi_loku.py` → OK.
- Unit test với PEFT 0.19.1: `apply_loku_soft_init` giữ đúng **identity-at-init** (diff ~5e-7)
  cho Conv2d / Linear / Linear rank-hạn-chế; shape conv-LoRA đúng.
- `resolve_image_targets`: last_k=1 → 5 conv của layer7; last_k=2 → 10; last_k=0 → rỗng.

**Rủi ro / cần theo dõi.**
- FILA trên conv làm running-stats của BatchNorm layer7 lệch; `model.train()` mỗi epoch giúp BN
  thích nghi lại trên retain, nhưng nếu **Test AUC tụt** → hạ `loku_image_subtract_scale` về 0.3.
- Nếu forget vẫn mạnh → tăng `lora_image_last_k_blocks: 2` / scale `1.0` / bật `include_fc1`.
- Revert nhanh: đặt `lora_image_last_k_blocks: 0`.

---

## [exp09] — 2026-06-01 — Inverted Hinge Loss (IHL)

**Động cơ.** exp08 đạt MIA tốt nhất (0.562) nhưng Forget AUC/F1 vẫn cao → cần một forget-push
"hiền" hơn NegGrad (vốn gây spike MIA ở exp04–06).

**Thay đổi.** (`training/forgetmi_loku.py`, vùng loss ~L675) thêm `L_IHL = 1 + p(true) − max_{v≠true} p(v)`
trên `img/txt logits` của forget, bounded [0,2], self-stopping. Thêm `ihl_forget_weight` vào tổng loss.

**Config.** `ihl_forget_weight` (exp09 = 1.5). *(exp10 đặt lại 0.0 để cô lập.)*

---

## [exp08] — 2026-06-01 — TRUE LoKU FILA subtraction (mốc breakthrough)

**Động cơ.** exp01–07 dùng soft-init (`init_scale=0.05`, không trừ gì) → mọi việc unlearn dồn
vào training → chạm forget data → MIA tăng.

**Thay đổi.** (`apply_loku_soft_init`) thêm chế độ **subtraction thật**: `W* = W − B·A·scaling·scale`,
LoRA giữ forget direction; train retain-only → khi LoRA đi khỏi init thì lộ phần đã trừ = unlearn thật.

**Config.** `loku_subtract_scale: 1.0` (0 = soft-init legacy). Tắt NegGrad/uniform/distill_forget.

**Kết quả.** MIA 0.562 (✅ < paper 0.571), Test AUC 0.679, Time 0.139h. Forget AUC còn 0.833 → động cơ của exp09/exp10.
