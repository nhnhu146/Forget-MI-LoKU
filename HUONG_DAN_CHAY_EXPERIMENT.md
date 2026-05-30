# Hướng Dẫn Chạy 1 Experiment — Từng Bước

> File này = checklist cầm tay. Mở mỗi lần làm exp mới, làm theo từng bước.

---

## ⚙️ SETUP MỘT LẦN DUY NHẤT (chỉ làm 1 lần ở exp đầu tiên)

### Bước A: Tạo Personal Access Token trên GitHub

1. Vào https://github.com/settings/tokens
2. Click **"Generate new token (classic)"**
3. Đặt tên (Note): `colab-forget-mi`
4. Expiration: chọn `90 days` hoặc lâu hơn
5. Tick scope: **`repo`** (Full control of private repositories)
6. Click **"Generate token"** ở dưới cùng
7. **Copy chuỗi bắt đầu bằng `ghp_xxx...`** (chỉ hiện 1 lần, mất là phải tạo lại)

### Bước B: Lưu token theo môi trường bạn dùng

Chọn 1 trong 3 cách (theo độ tiện):

#### 🌟 CÁCH 1 — Lưu file vào Google Drive (KHUYẾN NGHỊ, dùng được mọi nơi)

Tốt nhất nếu bạn dùng **VS Code + Colab** hoặc **web Colab** đều được.

1. Mở Google Drive bằng browser
2. Vào thư mục `Forget-MI-Project` (cùng chỗ có data.zip)
3. Tạo file mới tên `.git-secrets.json` với nội dung:
   ```json
   {
     "GITHUB_TOKEN": "ghp_xxxxxxxxxxxx",
     "GIT_EMAIL": "ban@gmail.com",
     "GIT_NAME": "Nguyen Hoang Nhu"
   }
   ```
4. Lưu file. Xong.

> **Tip nếu Drive không cho tạo file bắt đầu bằng dấu chấm**:
> - Tạo file trên máy local trước (vd: Notepad → save as `.git-secrets.json`), rồi upload lên Drive
> - Hoặc đặt tên `git-secrets.json` (không có chấm) → mở Cell 5 sửa đường dẫn trong code bỏ dấu chấm đi (dòng `drive_path = Path(".../.git-secrets.json")`)

> File này nằm trên Drive, không bị commit vào git. Mỗi lần Colab chạy Cell 5, nó tự đọc.

#### CÁCH 2 — Colab Secrets (chỉ dùng được trên web colab.research.google.com)

1. Mở Colab trên trình duyệt
2. Click icon **🔑** ở thanh bên trái
3. Add 3 secrets: `GITHUB_TOKEN`, `GIT_EMAIL`, `GIT_NAME` (bật Notebook access)

> Cách này KHÔNG dùng được với VS Code Colab extension.

#### CÁCH 3 — Nhập tay mỗi session (lười, không cần setup)

Bỏ qua Bước B luôn. Cell 5 sẽ tự hỏi token mỗi lần chạy.

✅ Setup xong. Không phải làm lại lần nào nữa.

---

## 🔄 QUY TRÌNH CHO MỖI EXPERIMENT MỚI

Có **3 phần**, tổng cộng **~25 phút**:

| Phần | Việc làm | Ở đâu | Thời gian |
|---|---|---|---|
| 1 | Sửa config + push lên GitHub | Máy bạn | 5 phút |
| 2 | Chạy 5 cells trên Colab | Colab | 15 phút |
| 3 | Điền observations + conclusion | Máy bạn | 5 phút |

---

## 📦 PHẦN 1: Trên máy bạn (5 phút)

### Bước 1: Mở file `config.yaml`

Mở bằng VS Code / Notepad / editor gì cũng được.

### Bước 2: Sửa các tham số bạn muốn test

Ví dụ — Exp "tăng forget_margin từ 8 lên 20":

Tìm dòng (khoảng dòng 84):
```yaml
forget_margin:
  value: 8.0
```

Đổi thành:
```yaml
forget_margin:
  value: 20.0
```

> **Mỗi exp chỉ nên đổi 1-2 tham số** để dễ biết cái nào gây ra thay đổi.

### Bước 3: Lưu file (Ctrl+S)

### Bước 4: Mở terminal trong thư mục project, gõ lần lượt 3 lệnh:

```bash
git add config.yaml
```

```bash
git commit -m "exp NNN: <mô tả ngắn thay đổi>"
```
> _Đổi `NNN` thành số exp (002, 003, ...) và mô tả ngắn._

```bash
git push
```

✅ **Xong phần 1.** Code đã ở trên GitHub.

---

## ☁️ PHẦN 2: Trên Colab (15 phút)

### Bước 5: Mở Colab notebook `run.ipynb`

### Bước 6: Chạy **Cell 1** (click ▶ bên trái cell)

Đợi đến khi thấy: `✅ Môi trường và mã nguồn đã sẵn sàng!`

### Bước 7: Chạy **Cell 2**

Đợi đến khi thấy: `🚀 Tất cả Dữ liệu & Model đã sẵn sàng!`

### Bước 8: Chạy **Cell 3**

Đợi đến khi thấy: `✅ Output sẽ được lưu tại: ...`

### Bước 9: **SỬA Cell 4** TRƯỚC khi chạy

Trong Cell 4, sửa **2 dòng đầu**:

```python
EXP_NAME   = "ten_exp_cua_ban"        # ← đổi tên ngắn (chỉ chữ thường + gạch dưới)
HYPOTHESIS = "Mo ta gia thuyet"       # ← viết 1 câu lý do test
```

**Quy tắc đặt tên `EXP_NAME`:**
- Chỉ dùng chữ thường, số, gạch dưới
- KHÔNG có dấu cách, dấu tiếng Việt
- Ngắn gọn (≤ 30 ký tự)
- Mô tả thay đổi chính

Ví dụ tên tốt:
- ✅ `forget_margin_20`
- ✅ `lora_r16`
- ✅ `neggrad_term`
- ❌ `Exp 2 thử tăng margin` (có dấu cách + tiếng Việt)

### Bước 10: Chạy **Cell 4** (đợi 10-15 phút)

Trong khi đợi, bạn có thể xem training progress. Cuối cùng sẽ thấy bảng:

```
────────────────────────────────────────────────────────────
  MIA           (↓)        | 0.xxx  | ...
  Forget AUC    (↓)        | 0.xxx  | ...
  Forget Mac-F1 (↓)        | 0.xxx  | ...
  Test AUC      (↑)        | 0.xxx  | ...
  Test Mac-F1   (↑)        | 0.xxx  | ...
  ...
────────────────────────────────────────────────────────────

✅ Auto-saved: experiments/exp_NNN_<EXP_NAME>.md
✅ Index updated: experiments/INDEX.md
```

### Bước 11: Chạy **Cell 5** (đợi ~5 giây)

Sẽ thấy:
```
📦 Files sẽ commit:
   - experiments/exp_NNN_<EXP_NAME>.md
   - experiments/INDEX.md

✅ Đã push lên GitHub
🔗 Xem online: https://github.com/...
```

✅ **Xong phần 2.** Kết quả đã được auto-ghi vào GitHub.

---

## ✍️ PHẦN 3: Quay lại máy bạn (5 phút)

### Bước 12: Mở terminal, gõ:

```bash
git pull
```

Bạn sẽ thấy 2 file thay đổi:
- `experiments/exp_NNN_<EXP_NAME>.md` (mới)
- `experiments/INDEX.md` (thêm 1 dòng)

### Bước 13: Mở file `experiments/exp_NNN_<EXP_NAME>.md`

Cuộn xuống cuối, bạn sẽ thấy **3 section trống** cần điền:

```markdown
## 6. Observations (✍️ điền thủ công)
_(Quan sát gì bất thường? Khớp/khác predict?)_
-

## 7. Conclusion (✍️ điền thủ công)
- **Hypothesis verdict**: ✅ Confirmed / ❌ Rejected / 🤷 Inconclusive
- **Keep changes**: Y / N
- **Why**:

## 8. Next steps (✍️ điền thủ công)
-
```

### Bước 14: Điền 3 section theo MẪU

> Tip: Mở [INDEX.md](experiments/INDEX.md) để so sánh với các exp trước.

**Mẫu Section 6 — Observations** (2-4 câu, ghi điều bất thường):
```markdown
## 6. Observations
- L_MD đã > 0 (trước đó = 0 toàn epoch) — margin 20 đã đủ kích hoạt
- Forget AUC giảm từ 0.829 xuống 0.7XX — đúng dự đoán
- Test AUC giảm nhẹ 0.05 — trade-off chấp nhận được
- CosSim vs F_re không cải thiện do đã tắt re-anchor
```

**Mẫu Section 7 — Conclusion** (3 dòng):
```markdown
## 7. Conclusion
- **Hypothesis verdict**: ✅ Confirmed
- **Keep changes**: Y
- **Why**: Forget mạnh hơn rõ rệt, các metric khác chấp nhận được
```

**Mẫu Section 8 — Next steps** (1-2 ý cho exp tiếp):
```markdown
## 8. Next steps
- Exp NNN+1: thử thêm NegGrad term để forget mạnh hơn nữa
- Hoặc: tăng lora_r=16 để có nhiều capacity
```

### Bước 15: Lưu file (Ctrl+S)

### Bước 16: Mở terminal, gõ 3 lệnh:

```bash
git add experiments/
```

```bash
git commit -m "exp NNN: them observations va conclusion"
```

```bash
git push
```

✅✅ **HOÀN TẤT 1 EXPERIMENT.**

---

## 🐛 GẶP LỖI? — Câu hỏi thường gặp

### Q1: Cell 5 báo "Thiếu credentials"

→ Quay lại phần SETUP ở đầu file. Nếu dùng **VS Code + Colab** → dùng CÁCH 1 (file Drive).

### Q1b: Tôi dùng VS Code Colab, không thấy icon 🔑 Secrets

→ VS Code Colab KHÔNG có Secrets panel. Dùng **CÁCH 1** (file Drive) hoặc **CÁCH 3** (nhập tay).

### Q2: `git push` báo "Authentication failed" hoặc "403"

→ Token có thể đã hết hạn / sai. Hướng giải quyết:
1. Tạo token mới ở https://github.com/settings/tokens (tick scope `repo`)
2. Update lại token vào chỗ bạn đã setup:
   - **CÁCH 1**: sửa file `.git-secrets.json` trên Drive
   - **CÁCH 2**: update lại trong Colab Secrets
   - **CÁCH 3**: nhập lại token mới khi cell hỏi

### Q3: Cell 5 báo "Merge conflict"

→ Trên máy bạn, gõ:
```bash
git pull --rebase
git push
```

### Q4: Cell 4 chạy bị OOM (out of memory)

→ Trong `config.yaml`, giảm:
```yaml
unlearn_batch_size:
  value: 8                # từ 16 xuống 8
eval_batch_size:
  value: 8
fisher_batch_size:
  value: 8
```

### Q5: Cell 4 chạy mãi không xong

→ Mặc định tối đa 8 epoch, mỗi epoch ~1 phút → tối đa 10 phút. Nếu lâu hơn, có thể đang train Fisher (đợi thêm 2-3 phút).

### Q9: Cell 5 hỏi token rồi gõ vào nhưng không thấy chữ hiện ra

→ Đây là **bình thường, không phải lỗi**. `getpass` ẩn token khi bạn gõ (để bảo mật, tránh người khác nhìn thấy). Cứ paste token + Enter là OK.

### Q10: Tôi muốn check file `.git-secrets.json` có đọc được không

→ Trong 1 cell mới của Colab, chạy:
```python
import json
from pathlib import Path
p = Path("/content/drive/MyDrive/Forget-MI-Project/.git-secrets.json")
print("File exists:", p.exists())
if p.exists():
    s = json.loads(p.read_text())
    print("Keys:", list(s.keys()))
    print("Token length:", len(s.get('GITHUB_TOKEN', '')))
```
Phải in: `File exists: True`, `Keys: ['GITHUB_TOKEN', 'GIT_EMAIL', 'GIT_NAME']`, `Token length: 40` (hoặc gần đó).

### Q6: Tôi đặt tên `EXP_NAME` trùng với exp cũ

→ Không sao, script sẽ **overwrite** Results section của file cũ (giữ Observations cũ). Nếu muốn tạo file mới riêng, đổi tên khác.

### Q7: Tôi quên đổi `EXP_NAME` ở Bước 9

→ Cell 4 sẽ ghi đè lên file exp gần nhất có cùng tên đó. Nếu lỡ, đổi tên file trong `experiments/` rồi đổi luôn 1 dòng trong INDEX.md.

### Q8: Tôi muốn rerun 1 exp đã chạy

→ Chỉ cần đặt `EXP_NAME` giống lúc trước. Script tự update file cũ.

---

## 📋 Checklist nhanh — copy ra giấy nếu cần

```
PHẦN 1 — Local:
[ ] Sửa config.yaml
[ ] git add config.yaml
[ ] git commit -m "exp NNN: <mô tả>"
[ ] git push

PHẦN 2 — Colab:
[ ] Run Cell 1
[ ] Run Cell 2
[ ] Run Cell 3
[ ] Sửa EXP_NAME + HYPOTHESIS ở Cell 4
[ ] Run Cell 4 (đợi 10-15 phút)
[ ] Run Cell 5

PHẦN 3 — Local:
[ ] git pull
[ ] Mở file experiments/exp_NNN_*.md
[ ] Điền Section 6 (Observations)
[ ] Điền Section 7 (Conclusion)
[ ] Điền Section 8 (Next steps)
[ ] git add experiments/
[ ] git commit -m "exp NNN: observations"
[ ] git push
```

---

## 💡 Ý tưởng cho các Exp tiếp theo

Xem roadmap trong [experiments/INDEX.md](experiments/INDEX.md). Đề xuất theo thứ tự:

| Exp | Thay đổi gì | Mục đích |
|---|---|---|
| 001 | Không đổi gì, chỉ baseline | Có số tham chiếu |
| 002 | `forget_margin: 8→20`, `eta_re_anchor: 0.5→0` | Forget mạnh hơn |
| 003 | Thêm NegGrad term | Forget cực mạnh |
| 004 | `lora_r: 8→16`, mở rộng target_modules | Tăng capacity |
| 005 | Two-stage training | Trade-off tốt nhất |
| 006 | Best config + 3 seeds khác nhau | Mean±std cho luận văn |
| 007 | Forget 6% với best config | Test ở mức cao hơn |
| 008 | Forget 10% với best config | Test ở mức cao nhất |
| 009 | Ablation: chỉ LoRA (không FIM) | Tách đóng góp |
| 010 | Ablation: chỉ FIM (full FT + FIM mask) | Tách đóng góp |
| 011 | Reproduce Forget-MI gốc | Có số baseline để so |
