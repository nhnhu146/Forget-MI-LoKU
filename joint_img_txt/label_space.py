"""
Không gian nhãn (label space) — NGUỒN SỰ THẬT DUY NHẤT cho số lớp đầu ra.

Bối cảnh
--------
Kiến trúc kế thừa từ MIMIC-CXR edema (4 mức độ 0..3): `EdemaClassificationProcessor`
hardcode 4 nhãn, `convert_to_onehot` sinh vector 4 chiều, head phân loại rộng 4.
Khi chạy trên IU Chest X-rays — bài toán NHỊ PHÂN (0=normal, 1=abnormal) — hai kênh
2 và 3 trở thành LỚP CHẾT: không bao giờ xuất hiện trong nhãn, nhưng vẫn có logit,
vẫn nằm trong mẫu số softmax, vẫn là đích hợp lệ của argmax. Hệ quả:

  1. IHL (`L = 1 + p(true) - max_{v != true} p(v)`) được thoả mãn bằng cách dồn xác
     suất sang lớp chết, trong khi quyết định nhị phân normal/abnormal KHÔNG đổi
     → "quên" giả.
  2. Macro-F1 lấy trung bình trên hợp của nhãn thật và nhãn dự đoán; model nào lỡ
     dự đoán lớp 2 thì mẫu số thành 3 thay vì 2 → F1 bị dìm một lượng phụ thuộc
     phương pháp → Df_F1/Dt_F1 không so sánh được giữa các phương pháp.
  3. AUC theo kênh: kênh 2/3 không có mẫu dương → NaN → bị lọc, nhưng AUC của kênh
     0 và kênh 1 chỉ trùng nhau khi p2+p3 ~ 0.
  4. Thang CE: softmax 4 chiều → CE ngẫu nhiên là ln(4) chứ không phải ln(2).

Hai khái niệm PHẢI tách rời
---------------------------
* `num_labels`         — ĐỘ RỘNG HEAD. Thuộc về kiến trúc, gắn chặt với checkpoint
                         (`ImageTextModel.from_pretrained` đọc `num_labels` từ
                         config.json của checkpoint, KHÔNG đọc từ yaml). Đổi nó
                         bắt buộc phải train lại θ_og/θ_re.
* `num_active_classes` — KHÔNG GIAN QUYẾT ĐỊNH. Số lớp thật sự có trong nhãn của
                         dataset. Mọi hàm mất mát quên và mọi độ đo chỉ làm việc
                         trên các kênh `[0, num_active_classes)`.

MIMIC: 4/4. IU (giữ nguyên checkpoint head 4 chiều): 4/2.

ĐƯỜNG LUI
---------
Không khai báo `num_active_classes` trong config → `n_active = n_labels` → mọi hàm
ở đây là NO-OP tuyệt đối, code chạy y hệt trước khi có file này. Vì vậy MIMIC
(4 == 4) bất biến từng bit và KHÔNG cần chạy lại. Để huỷ thực nghiệm, chỉ cần xoá
một dòng `num_active_classes` khỏi config IU.
"""

import numpy as np

DEFAULT_NUM_LABELS = 4


class LabelSpace:
    """Bộ mô tả không gian nhãn đã được phân giải (resolved) cho một lần chạy."""

    __slots__ = ('n_labels', 'n_active')

    def __init__(self, n_labels, n_active):
        self.n_labels = int(n_labels)
        self.n_active = int(n_active)
        if self.n_labels < 2:
            raise ValueError(f"num_labels phải >= 2, nhận {self.n_labels}")
        if not (2 <= self.n_active <= self.n_labels):
            raise ValueError(
                f"num_active_classes={self.n_active} phải nằm trong [2, num_labels={self.n_labels}]. "
                f"Không gian quyết định không thể rộng hơn head.")

    @property
    def is_noop(self):
        """True khi không có kênh chết nào bị loại → mọi thao tác cắt là no-op."""
        return self.n_active == self.n_labels

    def __repr__(self):
        tag = 'no-op (khớp head)' if self.is_noop else f'loại {self.n_labels - self.n_active} kênh chết'
        return f"LabelSpace(head={self.n_labels}, active={self.n_active}) [{tag}]"


def resolve(args, default_num_labels=DEFAULT_NUM_LABELS):
    """Phân giải không gian nhãn từ config/args.

    `num_labels` mặc định 4 (hành vi cũ). `num_active_classes` vắng mặt, None, hoặc
    <= 0 → lấy bằng `num_labels` → no-op.
    """
    n_labels = getattr(args, 'num_labels', None)
    if n_labels in (None, '', 0):
        n_labels = default_num_labels
    n_labels = int(n_labels)

    n_active = getattr(args, 'num_active_classes', None)
    if n_active in (None, '') or (isinstance(n_active, (int, float)) and int(n_active) <= 0):
        n_active = n_labels
    return LabelSpace(n_labels, int(n_active))


def describe(args, default_num_labels=DEFAULT_NUM_LABELS):
    """Chuỗi một dòng để in vào log đầu run — làm mỗi run tự ghi lại không gian nhãn."""
    ls = resolve(args, default_num_labels)
    if ls.is_noop:
        return f"🏷️  Label space: {ls.n_labels} lớp (không có kênh chết)"
    dead = ', '.join(str(c) for c in range(ls.n_active, ls.n_labels))
    return (f"🏷️  Label space: head {ls.n_labels} kênh, ACTIVE {ls.n_active} kênh "
            f"→ loại kênh chết [{dead}] khỏi loss quên + mọi độ đo")


def assert_head_width(model, ls, where=''):
    """Chốt an toàn: độ rộng head của checkpoint phải khớp `num_labels` của config.

    `ImageTextModel.from_pretrained` lấy `num_labels` từ config.json của CHECKPOINT,
    nên nếu config yaml khai báo khác thì nhãn và logits sẽ lệch chiều mà không ai
    báo lỗi. Kiểm tra sớm, hỏng thì dừng ngay.
    """
    head = getattr(getattr(model, 'config', None), 'num_labels', None)
    if head is None:
        return  # không suy ra được → không chặn
    if int(head) != ls.n_labels:
        raise ValueError(
            f"Lệch độ rộng head{' tại ' + where if where else ''}: checkpoint có "
            f"num_labels={int(head)} nhưng config khai báo num_labels={ls.n_labels}. "
            f"Sửa `num_labels` trong config cho khớp checkpoint, HOẶC train lại "
            f"checkpoint với head {ls.n_labels} kênh. Nếu muốn thu hẹp KHÔNG GIAN "
            f"QUYẾT ĐỊNH mà giữ nguyên checkpoint, dùng `num_active_classes`.")


# ---------------------------------------------------------------------------
# Cắt kênh chết
# ---------------------------------------------------------------------------

def slice_logits(logits, ls):
    """Cắt logits (torch.Tensor hoặc np.ndarray) về các kênh active. No-op khi n_active
    == n_labels, hoặc khi chiều cuối đã nhỏ hơn/bằng n_active."""
    if ls is None or ls.is_noop:
        return logits
    if logits.shape[-1] <= ls.n_active:
        return logits
    return logits[..., :ls.n_active]


def slice_onehot(onehot, ls):
    """Cắt nhãn one-hot về các kênh active.

    An toàn vì các kênh bị bỏ theo định nghĩa luôn bằng 0 (nhãn của dataset nằm trong
    [0, n_active)). Nếu phát hiện mẫu có nhãn thuộc kênh chết → dữ liệu mâu thuẫn với
    khai báo config, raise ngay thay vì âm thầm biến nó thành vector toàn 0.
    """
    if ls is None or ls.is_noop:
        return onehot
    if onehot.shape[-1] <= ls.n_active:
        return onehot
    arr = np.asarray(onehot)
    dead_mass = arr[..., ls.n_active:]
    # Nhãn sentinel -1 (mẫu thiếu nhãn) không tính là vi phạm.
    if np.any(dead_mass > 0):
        n_bad = int(np.sum(np.any(dead_mass > 0, axis=-1)))
        raise ValueError(
            f"{n_bad} mẫu có nhãn thuộc kênh chết >= {ls.n_active}, mâu thuẫn với "
            f"num_active_classes={ls.n_active}. Kiểm tra lại nhãn của dataset.")
    return arr[..., :ls.n_active]


def assert_labels_in_range(labels, ls, where=''):
    """Chốt an toàn: chỉ số nhãn thô phải nằm trong [0, n_active)."""
    if ls is None or ls.is_noop:
        return
    arr = np.asarray(labels.detach().cpu() if hasattr(labels, 'detach') else labels)
    arr = arr[arr >= 0]  # bỏ sentinel -1
    if arr.size and int(arr.max()) >= ls.n_active:
        raise ValueError(
            f"Nhãn{' tại ' + where if where else ''} có giá trị {int(arr.max())} >= "
            f"num_active_classes={ls.n_active}. Dataset không phải {ls.n_active} lớp "
            f"như config khai báo.")
