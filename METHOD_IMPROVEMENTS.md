# Forget-MI gốc → Forget-MI-LoKU: So sánh code ban đầu ↔ phương pháp cuối

> Tài liệu này so sánh **trực tiếp** hai bản code:
> - **BAN ĐẦU** = [`training/forgetmi_partial.py`](training/forgetmi_partial.py) — bản tái hiện
>   Forget-MI gốc (full fine-tuning).
> - **CUỐI (exp11 final)** = [`training/forgetmi_loku.py`](training/forgetmi_loku.py) +
>   [`config.yaml`](config.yaml) — phương pháp đề xuất Forget-MI-LoKU.
>
> Không ghi theo từng exp; chỉ ghi **đã đổi gì giữa điểm đầu và điểm cuối**, kèm *vị trí code,
> công thức, lý do, ảnh hưởng*. Dùng để viết Chương 3 + Chương 4 khóa luận.

---

## 0. Kiến trúc nền — GIỮ NGUYÊN (không đổi)

Cả hai bản dùng chung [`joint_img_txt/model.py`](joint_img_txt/model.py):
- Nhánh ảnh `ImageResNet`: CNN residual 7 stage (`layer1..layer7`) → `z_img`(768) → `fc1` →
  `img_logits` (4 lớp pulmonary edema).
- Nhánh văn bản `TextBertForSequenceClassification` (SciBERT) → `[CLS]` → `z_txt`(768) →
  `classifier` → `txt_logits`.
- Hợp nhất `Gate` (cơ chế cổng). Bài toán: phân loại 4 mức edema, MIMIC-CXR.

→ **Đóng góp của khóa luận KHÔNG đụng kiến trúc**, chỉ thay đổi *cách gỡ học* (cơ chế + tham số
được cập nhật + hàm mất mát + quy trình đánh giá).

---

## 1. Bảng so sánh tổng quan (đầu ↔ cuối)

| Khía cạnh | BAN ĐẦU (`forgetmi_partial.py`) | CUỐI (`forgetmi_loku.py`, exp11) |
|---|---|---|
| Tham số cập nhật | **Toàn bộ 100%** (full-FT) | **LoRA + heads + gates = 0.451%** |
| Khởi tạo | Không có (train thẳng từ F_og) | **Fisher → SVD → FILA subtraction** `W*=W−BA` |
| Phạm vi can thiệp | Mọi trọng số | Q/K/V của BERT + Conv2d `layer7` ảnh + `fc1`/`classifier` |
| Loss forget | `L_uu=−d`, `L_md=−d` (**âm, KHÔNG chặn**) | **bounded hinge** `relu(margin−d)` + **IHL** `1+p_true−max p_other` |
| Loss retain | `min(d, margin)` (UKR, MKR) | `relu(d−margin)+0.1d` + **CE retain** + **KL-distill F_og** |
| Gate fusion | **Tạo mới ngẫu nhiên MỖI batch, không train** | 4 Gate **cố định, được train** |
| Cập nhật optimizer | `step()` **1 lần/epoch**, **bỏ epoch 0** | `step()` **mỗi batch** + grad-clip |
| Teacher | F_og (chỉ để tính UU/MD/UKR/MKR) | F_og (distill retain) — **F_re KHÔNG dùng khi train** |
| Dừng sớm | Không (chạy đủ epoch) | **validation CE** (held-out, không đụng F_re) |
| Đánh giá | Chỉ log CosSim(F_re,ul); MIA/AUC ngoài file | **Inline đầy đủ**: 2×MIA, CosSim, AUC, F1, time/GPU/params |
| Tốc độ eval | fp32, full retain | **fp16 + subsample CosSim** (~5–10×) |
| Hạ tầng | 1 lần chạy thủ công | multi-seed, `--override`, auto-tracker, GPU-guard |
| Thời gian / lần | ~5h (full-FT) | **~0.1–0.15h** |

---

## 2. Thay đổi #1 — Full fine-tuning → LoRA PEFT

**Ban đầu** ([`forgetmi_partial.py:606-608`](training/forgetmi_partial.py#L606-L608)):
```python
for param in model_unlearn.parameters():
    param.requires_grad = True        # TẤT CẢ tham số trainable (100%)
```
Optimizer nhận toàn bộ named_parameters ([dòng 612-619](training/forgetmi_partial.py#L612-L619)).

**Cuối** ([`forgetmi_loku.py:1062-1100`](training/forgetmi_loku.py#L1062-L1100)):
- Bọc model bằng PEFT LoRA: `W = W₀ + B·A`, đóng băng `W₀`, chỉ train `A,B` (`r=8`, α=16).
- Chỉ mở thêm 2 classifier head (`fc1`, `classifier`) + 4 Gate.

**Lý do.** Hạn chế #1 của Forget-MI: full-FT đắt. Giả thiết LoRA: cập nhật khi thích ứng có hạng
nội tại thấp.

**Ảnh hưởng.** Trainable 100% → **0.451%**; time ~5h → **~0.1h** (≈30–50×); VRAM giảm mạnh.

---

## 3. Thay đổi #2 — Khởi tạo: train-từ-gốc → Fisher + FILA subtraction

**Ban đầu.** Không có bước khởi tạo định hướng. `model_unlearn = copy.deepcopy(model_og)`
([dòng 594](training/forgetmi_partial.py#L594)) rồi train thẳng → mọi việc quên dồn vào gradient,
buộc chạm forget data.

**Cuối — thêm 2 bước mới TRƯỚC khi train:**

**(a) Fisher importance** ([`compute_fisher_importance`](training/forgetmi_loku.py#L293-L323)):
```
F̂(θ) ≈ E[(∂ CE(img,txt | y_true) / ∂θ)²]
```
Tính riêng `F̂^f` (forget) và `F̂^r` (retain) → tỷ số tương đối `R = F̂^f/(F̂^r+ε)`.

**(b) FILA subtraction — lõi cơ chế quên mới** ([`apply_loku_soft_init`](training/forgetmi_loku.py#L350-L427),
[`_fila_decompose`](training/forgetmi_loku.py#L326-L347)):
1. Nhân hàng theo `√R` → SVD hạng thấp → `(A,B)`, `sub = B·A` (hướng forget).
2. **Trừ thẳng khỏi base**: `W' = W − sub·peft_scaling·s`.
3. Khởi tạo LoRA `A√s, B√s` để forward tại init = `W` (**identity-at-init**).
4. Khi train kéo LoRA rời init → phần `−sub` "lộ ra" → **quên thật** dù chỉ train trên retain.

**Cấu hình.** `loku_subtract_scale=1.0` (text), `loku_image_subtract_scale=0.3` (ảnh),
`fisher_max_samples=256`.

**Lý do.** Đưa việc quên về **init** (đại số) thay vì ép gradient chạm forget data → giảm rủi ro
MIA tăng. Đây là điểm "Fisher-guided" + "LoKU/FILA" của khóa luận.

---

## 4. Thay đổi #3 — Phạm vi can thiệp: mọi nơi → text + image có chọn lọc

**Ban đầu.** Full-FT → đụng mọi trọng số (kể cả những phần không liên quan forget).

**Cuối.** Chỉ can thiệp đúng chỗ:
- BERT Q/K/V (`lora_target_modules=[query,key,value]`).
- **Conv2d ở `img_model.layer7`** (modality-aware) — [`resolve_image_targets`](training/forgetmi_loku.py#L90-L111),
  FILA tổng quát cho Conv2d 4D ([`forgetmi_loku.py:393-420`](training/forgetmi_loku.py#L393-L420)).
- Head `fc1` (ảnh) + `classifier` (text) mở băng.

**Lý do (chẩn đoán quan trọng).** MIA + Df/Dt-AUC/F1 đều đọc **`img_logits`**
([`model.py:321`](joint_img_txt/model.py#L321)). Nếu chỉ chạm text → hành vi ảnh trên forget không
đổi → **Forget-AUC kẹt 0.833**. Thêm FILA nhánh ảnh `layer7` → Df-AUC về ~0.736 (mốc paper).

> ⚠️ Tài liệu (abstract/§1.3) đang viết "chỉ LoRA bộ mã hóa **văn bản**". Code thực tế **text +
> image**. Cần sửa báo cáo cho khớp (image-FILA chính là cải tiến tạo kết quả).

---

## 5. Thay đổi #4 — Loss forget: âm-không-chặn → bounded hinge + IHL

**Ban đầu** ([`forgetmi_partial.py:422-431`](training/forgetmi_partial.py#L422-L431)) khi
`use_noise=False`:
```python
L_uu = -euclidean(ul_frgt_concat, og_rand_concat).mean()   # ÂM, không chặn
L_md = -euclidean(ul_frgt_joint,  og_rand_joint ).mean()   # ÂM, không chặn
```
→ tối đa hóa khoảng cách vô hạn → **phân kỳ** (hạn chế #2 của Forget-MI).

**Cuối — 2 thay đổi:**
- Hinge có chặn ([`forgetmi_loku.py:755-758`](training/forgetmi_loku.py#L755-L758)):
  `L_uu = relu(forget_margin − d_uu)` → chỉ đẩy tới `forget_margin` rồi dừng.
- **IHL** ([`forgetmi_loku.py:826-839`](training/forgetmi_loku.py#L826-L839)):
  `L_IHL = 1 + p(nhãn_thật) − max_{v≠thật} p(v) ∈ [0,2]`, bounded, self-stopping.

**Cấu hình cuối.** `forget_margin=20`, `beta(UU)=0`, `theta(MD)=0` (hai loss forget gốc **tắt** —
quên do **FILA + IHL** đảm nhận), `ihl_forget_weight=0.75` (sweet-spot: `forget_ce ≈ test_ce`,
không over-forget).

> ⚠️ Tài liệu (đề cương §2.4) viết "IHL + **4** loss Forget-MI". Thực tế: IHL + **2 loss retain**
> (UR, MR) + FILA; 2 loss forget (UU, MU) **tắt**. Cần sửa câu cho trung thực.

---

## 6. Thay đổi #5 — Loss retain: hinge đơn → hinge + CE + distill F_og

**Ban đầu** ([`forgetmi_partial.py:433-449`](training/forgetmi_partial.py#L433-L449)):
```python
L_ukr = euclidean(ul_ret_concat, og_ret_concat).mean()
L_mkr = euclidean(ul_ret_joint,  og_ret_joint ).mean()
if epoch != 0: L_ukr = min(L_ukr, margin_ukr); L_mkr = min(L_mkr, margin_mkr)
```
→ chỉ giữ retain qua khoảng cách embedding so với F_og.

**Cuối — 3 tín hiệu giữ retain mạnh hơn:**
- Hinge đúng chiều + lực kéo nhỏ ([`forgetmi_loku.py:765-770`](training/forgetmi_loku.py#L765-L770)):
  `L_ukr = relu(d_ukr − margin) + 0.1·d_ukr`.
- **CE phân loại retain** ([dòng 782](training/forgetmi_loku.py#L782)): neo classifier head
  (`kappa_cls_retain=2.0`).
- **KL-distillation từ F_og trên retain** ([dòng 804-821](training/forgetmi_loku.py#L804-L821)):
  `KL(softmax(student/T) ‖ softmax(F_og/T))·T²`, `distill_retain_weight=1.5`, `T=2.0`. Tái dùng
  forward F_og đã có → không tốn thêm.

**Lý do.** FILA + IHL đẩy forget mạnh hơn → cần neo retain chắc hơn để Dt-AUC/F1 không tụt.

---

## 7. Thay đổi #6 — Gate fusion: ngẫu nhiên mỗi batch → cố định & được train

**Ban đầu (đáng lưu ý)** ([`forgetmi_partial.py:396-409`](training/forgetmi_partial.py#L396-L409)):
```python
gate_ul_ret = Gate(...).to(device)   # TẠO MỚI ngẫu nhiên TRONG vòng lặp batch
ul_ret_joint_emb = gate_ul_ret(...)
...
og_frgt_joint_emb = gate_ul_frgt(og_frgt_img_emb, og_frgt_img_emb)  # truyền img 2 lần
```
→ Gate khởi tạo lại **ngẫu nhiên mỗi batch**, **không nằm trong optimizer**, **không train** →
các loss joint (MD, MKR) ban đầu dựa trên gate nhiễu, kém ổn định.

**Cuối** ([`forgetmi_loku.py:1102-1108`](training/forgetmi_loku.py#L1102-L1108)): 4 Gate cố định
(`ul_ret, ul_frg, og_rnd, og_ret`) tạo **một lần**, **đưa vào optimizer**, **được train** cùng
LoRA ([dòng 1111-1113](training/forgetmi_loku.py#L1111-L1113)).

**Lý do.** Để biểu diễn joint nhất quán giữa các batch và học được trọng số cổng hợp lý.

---

## 8. Thay đổi #7 — Lịch cập nhật optimizer

**Ban đầu** ([`forgetmi_partial.py:461-472`](training/forgetmi_partial.py#L461-L472)):
- `loss.backward()` cộng dồn gradient cả epoch; `optimizer.step()` **1 lần/epoch** (ngoài vòng
  batch); **epoch 0 không train** (chỉ tính margin).

**Cuối** ([`forgetmi_loku.py:848-855`](training/forgetmi_loku.py#L848-L855)): `zero_grad →
backward → clip_grad_norm → step` **mỗi batch**, grad-clip=1.0.

**Lý do.** Cập nhật mỗi batch hội tụ tốt hơn cho LoRA + lr cao (5e-4); grad-clip ổn định.

---

## 9. Thay đổi #8 — Vai trò F_re: tham chiếu → CHỈ ở eval (honest)

**Ban đầu** ([`forgetmi_partial.py:483-489`](training/forgetmi_partial.py#L483-L489)): mỗi epoch
gọi `get_probability_measure(model_re, model_ul, retain)` để log CosSim — **forward F_re trong
vòng train** (dù chỉ để log).

**Cuối.** F_re **không tham gia train**: `distill_teacher="og"`, `eta_re_anchor=0`,
`early_stop_metric="val"`. F_re chỉ dùng ở **eval** (CosSim cuối) — hợp lệ vì là tham chiếu đánh
giá.

**Lý do (then chốt phản biện).** Dùng F_re khi train phá tiền đề "unlearn thay vì retrain" và dễ
"học tủ" metric. Phương pháp cuối chứng minh tốt **mà không cần gold model** lúc train.

---

## 10. Thay đổi #9 — Dừng sớm: không có → validation CE (không đụng F_re)

**Ban đầu.** Không dừng sớm; chạy đủ `unlearn_epochs`, lưu mọi epoch.

**Cuối** ([`forgetmi_loku.py:870-903`](training/forgetmi_loku.py#L870-L903)): `early_stop_metric`:
- `"val"` (mặc định, honest): dừng theo mean img-CE trên **validation** held-out, `patience=4`.
- `"cossim"`/`"none"`: chỉ cho biến thể cho phép F_re / chạy cố định.

**Lý do.** Chọn checkpoint khách quan, **không** dùng F_re (tránh kỹ xảo early-stop theo metric
báo cáo).

---

## 11. Thay đổi #10 — Đánh giá: rời rạc/ngoài → inline đầy đủ

**Ban đầu.** Trong file chỉ log CosSim; phần MIA/AUC/F1 bị comment, chạy thủ công ở
[`evaluation/eval_unlearning.py`](evaluation/eval_unlearning.py).

**Cuối** ([`forgetmi_loku.py:1132-1236`](training/forgetmi_loku.py#L1132-L1236)): merge LoRA rồi
đánh giá **trong cùng run**:
- **Dual MIA** ([`run_mia`](training/forgetmi_loku.py#L511-L562)): `persample` (SVM per-sample,
  balanced) + `paper` (per-batch-mean, giống eval gốc, `mia_paper_batch_size=32`).
- CosSim vs F_re, AUC/F1 trên test & forget, **chẩn đoán over-forgetting** (in mean-CE retain/
  test/forget + cảnh báo `forget>1.3×test`).
- Hiệu quả: time, GPU peak, trainable %; ghi CSV + auto-track MD.
- **Tốc độ**: fp16 autocast + subsample CosSim (`eval_max_retain=512`) → ~5–10× nhanh, metric ~không đổi.

---

## 12. Thay đổi #11 — Hạ tầng thực nghiệm (mới hoàn toàn)

Không có ở bản đầu; thêm ở bản cuối ([`main()`](training/forgetmi_loku.py#L932)):
- `--seed` override (data split vẫn cố định → forget/retain/test không đổi giữa seed) → **multi-seed
  mean±std**.
- `--override "k=v,..."` patch config → sweep không cần sửa file.
- **GPU guard**: `SystemExit` nếu chạy CPU (tránh treo).
- Auto-tracker ([`scripts/exp_tracker.py`](scripts/exp_tracker.py)) tự điền `experiments/*.md` +
  INDEX + CSV.

---

## 13. Tổng hợp config cuối (đóng băng — exp11 honest)

```yaml
lora_r: 8 ; lora_alpha: 16 ; lora_dropout: 0.05
lora_target_modules: [query, key, value]
lora_image_last_k_blocks: 1 ; lora_image_include_fc1: false
loku_subtract_scale: 1.0 ; loku_image_subtract_scale: 0.3
fisher_max_samples: 256 ; fisher_batch_size: 16
alpha: 1.0 (UR) ; gamma: 1.0 (MR) ; beta: 0.0 (UU) ; theta: 0.0 (MD)
kappa_cls_retain: 2.0 ; kappa_cls_forget: 0.0 ; ihl_forget_weight: 0.75
distill_teacher: "og" ; distill_retain_weight: 1.5 ; distill_forget_weight: 0.0
eta_re_anchor: 0.0 ; early_stop_metric: "val" ; early_stop_patience: 4
learning_rate: 5e-4 ; unlearn_epochs: 8 ; unlearn_batch_size: 16
eval_max_retain: 512 ; mia_paper_batch_size: 32
```

- **Cơ chế quên** = FILA subtraction (init, text+image) + IHL (train).
- **Cơ chế giữ** = hinge UR/MR + CE retain + KL-distill từ F_og.
- **F_re** chỉ ở EVAL.

---

## 14. Kết quả (cuối ↔ paper, forget 3%, multi-seed 42/123/7)

| Metric | BAN ĐẦU/paper Forget-MI | CUỐI Forget-MI-LoKU | Nhận xét |
|---|---|---|---|
| MIA_persample ↓ | 0.571 | **0.455 ± 0.027** | thắng |
| MIA_paper ↓ | 0.571 | 0.429 ± 0.116 | thắng (thô, multi-seed) |
| Df-AUC ↓ | 0.735 | **0.736 ± 0.004** | hòa |
| Df-F1 ↓ | 0.393 | 0.379 ± 0.013 | hòa/nhỉnh |
| Dt-AUC ↑ | 0.625 | **0.677 ± 0.002** | thắng |
| Dt-F1 ↑ | 0.250 | 0.364 ± 0.020 | thắng |
| Trainable | 100% | **0.451%** | ≈220× ít hơn |
| Time | ~5h | **~0.1–0.15h** | ≈30–50× nhanh |

→ **Hòa khả năng quên, thắng utility + MIA + hiệu quả tài nguyên**, không kỹ xảo.

---

## 15. Hai điểm LỆCH code ↔ tài liệu cần đồng bộ khi viết báo cáo

1. **PEFT đa-phương-thức**: code là **text + image**, không phải "chỉ text" (abstract/§1.3).
2. **Loss forget**: code = **IHL + 2 loss retain (UR/MR) + FILA**, không phải "IHL + cả 4 loss
   Forget-MI" (UU/MU đang tắt).

Ngoài ra: abstract nêu **MIMIC-CXR + Indiana**; code hiện chỉ chạy MIMIC-CXR.
