"""
Kiểm chứng refactor không gian nhãn (joint_img_txt/label_space.py).

Chạy: python tools/test_label_space.py     (không cần GPU, không cần dataset)

Hai câu hỏi phải trả lời được BẰNG SỐ trước khi tốn quota Kaggle:
  A. MIMIC (4 kênh, không khai báo num_active_classes) có BẤT BIẾN không?
  B. IU (head 4, active 2) có thật sự bịt được cửa thoát lớp chết không?
"""

import os
import sys
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Các module chỉ cần cho I/O ảnh / logging khi chạy thật; test này thuần số học nên
# thay bằng stub để chạy được trên máy không cài môi trường đầy đủ (không cần GPU).
import types  # noqa: E402


def _stub(name, **attrs):
    if name in sys.modules:
        return
    try:
        __import__(name)
        return
    except ImportError:
        pass
    mod = types.ModuleType(name)
    # accelerate dò gói bằng importlib.util.find_spec → module không có __spec__ làm nó
    # ném ValueError. Gắn spec rỗng để stub trông như một module nhập bình thường.
    import importlib.machinery
    mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    if '.' in name:                       # đăng ký cả package cha
        parent, child = name.rsplit('.', 1)
        _stub(parent)
        setattr(sys.modules[parent], child, mod)


_stub('skimage')
_stub('skimage.io', imread=lambda *a, **k: None)
_stub('pydicom', dcmread=lambda *a, **k: None)
_stub('wandb', init=lambda *a, **k: None, log=lambda *a, **k: None, finish=lambda *a, **k: None)
_stub('peft')

from joint_img_txt import label_space
from joint_img_txt.metrics import compute_auc, get_acc_f1
from joint_img_txt.model_utils import convert_to_onehot, EdemaClassificationProcessor

FAILS = []


def check(name, cond, detail=''):
    print(f"  {'✅' if cond else '❌'} {name}" + (f"  — {detail}" if detail else ''))
    if not cond:
        FAILS.append(name)


# ---------------------------------------------------------------------------
print("\n[A] MIMIC BẤT BIẾN — config không có num_active_classes")
# ---------------------------------------------------------------------------

mimic = SimpleNamespace(output_channel_encoding='multiclass')
ls_mimic = label_space.resolve(mimic)
check("resolve() → 4/4, is_noop", ls_mimic.n_labels == 4 and ls_mimic.n_active == 4
      and ls_mimic.is_noop, repr(ls_mimic))

# convert_to_onehot phải trả ĐÚNG các vector hardcode cũ
OLD_ONEHOT = {0: [1, 0, 0, 0], 1: [0, 1, 0, 0], 2: [0, 0, 1, 0], 3: [0, 0, 0, 1],
              -1: [-1, -1, -1, -1]}
check("convert_to_onehot khớp bản hardcode cũ cho 0/1/2/3/-1",
      all(convert_to_onehot(k) == v for k, v in OLD_ONEHOT.items()))

check("get_labels() vẫn là ['0','1','2','3']",
      EdemaClassificationProcessor().get_labels() == ["0", "1", "2", "3"])

# slice_* là no-op
rng = np.random.default_rng(0)
logits4 = rng.normal(size=(64, 4))
oh4 = np.eye(4)[rng.integers(0, 4, size=64)].astype(int)
check("slice_logits là no-op", np.array_equal(label_space.slice_logits(logits4, ls_mimic), logits4))
check("slice_onehot là no-op", np.array_equal(label_space.slice_onehot(oh4, ls_mimic), oh4))

# pairwise-AUC: sinh động phải cho ĐÚNG 6 khoá cũ, đúng thứ tự, đúng giá trị
probs4 = torch.softmax(torch.tensor(logits4), dim=1).numpy()
aucs, pw = compute_auc(oh4.tolist(), probs4.tolist(), output_channel_encoding='multiclass')
check("pairwise-AUC vẫn đúng 6 khoá, đúng thứ tự",
      list(pw.keys()) == ['0v1', '0v2', '0v3', '1v2', '1v3', '2v3'], str(list(pw.keys())))
check("AUC theo kênh vẫn ra 4 giá trị", len(aucs) == 4)

# ---------------------------------------------------------------------------
print("\n[B] IU — head 4 kênh, không gian quyết định 2 lớp")
# ---------------------------------------------------------------------------

iu = SimpleNamespace(output_channel_encoding='multiclass', num_labels=4, num_active_classes=2)
ls_iu = label_space.resolve(iu)
check("resolve() → 4/2, KHÔNG no-op", ls_iu.n_labels == 4 and ls_iu.n_active == 2
      and not ls_iu.is_noop, repr(ls_iu))

# Kịch bản: model đã "quên" bằng cách dồn xác suất sang LỚP CHẾT số 2.
# Nhãn thật toàn nhị phân 0/1; logit lớp 2 rất cao.
n = 200
y_iu = rng.integers(0, 2, size=n)
oh_iu = np.eye(4)[y_iu].astype(int)
logits_escape = np.zeros((n, 4))
logits_escape[np.arange(n), y_iu] = 2.0        # vẫn phân biệt đúng normal/abnormal
logits_escape[:, 2] = 6.0                       # nhưng dồn phần lớn xác suất sang lớp chết

probs_all4 = torch.softmax(torch.tensor(logits_escape), dim=1).numpy()
f1_all4, _, preds_all4 = get_acc_f1(oh_iu, probs_all4, 'multiclass')

sliced_logits = label_space.slice_logits(logits_escape, ls_iu)
sliced_oh = label_space.slice_onehot(oh_iu, ls_iu)
probs_active = torch.softmax(torch.tensor(sliced_logits), dim=1).numpy()
f1_active, _, preds_active = get_acc_f1(sliced_oh, probs_active, 'multiclass')

check("TRƯỚC khi cắt: argmax rơi vào lớp chết 2", set(np.unique(preds_all4)) == {2},
      f"preds={np.unique(preds_all4)}, macro_f1={f1_all4['macro_f1']}")
check("SAU khi cắt: argmax chỉ còn {0,1} và phân loại đúng",
      set(np.unique(preds_active)) <= {0, 1} and f1_active['accuracy'] == 1.0,
      f"acc={f1_active['accuracy']}, macro_f1={f1_active['macro_f1']}")
check("Macro-F1 hết bị dìm bởi lớp chết",
      f1_active['macro_f1'] > f1_all4['macro_f1'],
      f"{f1_all4['macro_f1']} → {f1_active['macro_f1']}")

# pairwise-AUC cho 2 kênh: đúng 1 cặp, KHÔNG NaN, và không IndexError
aucs_iu, pw_iu = compute_auc(sliced_oh.tolist(), probs_active.tolist(),
                             output_channel_encoding='multiclass')
check("pairwise-AUC chỉ còn cặp 0v1", list(pw_iu.keys()) == ['0v1'], str(pw_iu))
check("pairwise-AUC 0v1 không NaN", not np.isnan(pw_iu['0v1']), str(pw_iu['0v1']))

# Cặp cũ trên 4 kênh: 5/6 cặp là NaN — chính là thứ ta loại bỏ
_, pw_old = compute_auc(oh_iu.tolist(), probs_all4.tolist(), output_channel_encoding='multiclass')
n_nan = sum(1 for v in pw_old.values() if isinstance(v, float) and np.isnan(v))
check("(đối chứng) head 4 kênh cho 5/6 cặp NaN", n_nan == 5, f"{n_nan}/6 NaN")

# ---------------------------------------------------------------------------
print("\n[C] IHL — cửa thoát lớp chết")
# ---------------------------------------------------------------------------

from training.adv_common import inverted_hinge_loss  # noqa: E402

y_t = torch.tensor(y_iu, dtype=torch.long)
lg = torch.tensor(logits_escape, dtype=torch.float32)

ihl_open = float(inverted_hinge_loss(lg, lg, y_t))            # cũ: 4 kênh
ihl_closed = float(inverted_hinge_loss(lg, lg, y_t, ls=ls_iu))  # mới: 2 kênh

check("IHL trên 4 kênh THẤP (tưởng đã quên, nhờ lớp chết)", ihl_open < 0.5, f"{ihl_open:.4f}")
check("IHL trên 2 kênh CAO (chưa quên thật — quyết định vẫn đúng)",
      ihl_closed > 1.0, f"{ihl_closed:.4f}")
check("Cắt kênh làm lộ ra khoảng quên giả", ihl_closed - ihl_open > 0.5,
      f"Δ={ihl_closed - ihl_open:.4f}")

# Trên MIMIC, truyền ls=4/4 phải cho KẾT QUẢ Y HỆT ls=None
lg4 = torch.tensor(logits4, dtype=torch.float32)
y4 = torch.tensor(rng.integers(0, 4, size=64), dtype=torch.long)
check("IHL: ls no-op cho kết quả y hệt ls=None",
      abs(float(inverted_hinge_loss(lg4, lg4, y4))
          - float(inverted_hinge_loss(lg4, lg4, y4, ls=ls_mimic))) < 1e-12)

# ---------------------------------------------------------------------------
print("\n[D] Chốt an toàn")
# ---------------------------------------------------------------------------

try:
    label_space.resolve(SimpleNamespace(num_labels=2, num_active_classes=4))
    check("active > labels phải raise", False)
except ValueError:
    check("active > labels raise ValueError", True)


class _FakeModel:
    def __init__(self, w):
        self.config = SimpleNamespace(num_labels=w)


try:
    label_space.assert_head_width(_FakeModel(4), ls_iu, 'test')
    check("head 4 khớp num_labels 4 → không raise", True)
except ValueError:
    check("head 4 khớp num_labels 4 → không raise", False)

try:
    label_space.assert_head_width(_FakeModel(2), ls_iu, 'test')
    check("head 2 lệch num_labels 4 → phải raise", False)
except ValueError:
    check("head 2 lệch num_labels 4 → raise ValueError", True)

try:
    label_space.slice_onehot(np.eye(4)[[3, 0, 1]].astype(int), ls_iu)
    check("nhãn thuộc kênh chết → phải raise", False)
except ValueError:
    check("nhãn thuộc kênh chết → raise ValueError", True)

# ---------------------------------------------------------------------------
print()
if FAILS:
    print(f"❌ {len(FAILS)} kiểm tra THẤT BẠI: {FAILS}")
    sys.exit(1)
print("✅ Tất cả kiểm tra PASS.")
