#!/usr/bin/env python3
"""Tự kiểm `training/eval_multimodal.py` — chạy được ở LOCAL, không cần GPU/model.

Kiểm đúng một điều quan trọng nhất: các hàm thuần-mảng của file mới phải cho ra
CÙNG con số với đường tính cũ trong `adv_common.py` khi nhận cùng dữ liệu vào.
Nếu pass thì cột view='img' của bản eval mới sẽ tái lập số đã báo cáo.

    python tools/test_eval_multimodal.py
"""
import io
import os
import sys

import numpy as np

# Console Windows mặc định cp1252 → in tiếng Việt sẽ ném UnicodeEncodeError.
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from training.eval_multimodal import (          # noqa: E402
    per_sample_ce_np, batch_means, mia_scores, softmax_np, view_features)

FAIL = []


def check(name, ok, detail=''):
    print(f"  [{'OK ' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ''))
    if not ok:
        FAIL.append(name)


rng = np.random.default_rng(0)
logits = rng.normal(0, 3, size=(200, 4))
labels = rng.integers(0, 4, size=200)

print('1) per_sample_ce_np == torch.nn.functional.cross_entropy(reduction="none")')
try:
    import torch
    import torch.nn.functional as F
    ref = F.cross_entropy(torch.tensor(logits, dtype=torch.float64),
                          torch.tensor(labels), reduction='none').numpy()
    got = per_sample_ce_np(logits, labels)
    check('CE trùng torch', np.allclose(ref, got, atol=1e-10),
          f'sai số lớn nhất {np.abs(ref - got).max():.2e}')
except ImportError:
    print('  (bỏ qua — máy này không có torch)')

print('\n2) per_sample_ce_np ổn định với logit rất lớn (ca P3/IU tại E30, |logit| ~ 68)')
big = np.array([[-65.3, 68.6, -60.0, -70.0], [700.0, -700.0, 0.0, 0.0]])
ce_big = per_sample_ce_np(big, np.array([0, 1]))
check('không NaN/Inf khi logit cực lớn', np.all(np.isfinite(ce_big)), f'CE = {ce_big}')

print('\n3) batch_means == adv_common._batch_means')
x = rng.normal(2, 1, size=205)
ref = np.array([x[i:i + 32].mean() for i in range(0, len(x), 32)])
check('trùng khít', np.allclose(ref, batch_means(x, 32)),
      f'{len(ref)} lô cho {len(x)} mẫu')

print('\n4) mia_scores(1 chiều) == đúng công thức adv_common.run_mia:936-951')
ml = rng.lognormal(0.4, 0.6, 512)
nl = rng.lognormal(0.7, 0.6, 900)
fl = rng.lognormal(0.5, 0.6, 210)


def run_mia_reference(ml, nl, fl, seed=42):
    """Chép NGUYÊN các dòng tính toán của adv_common.run_mia."""
    from sklearn.svm import SVC
    ml2, nl2, fl2 = ml.reshape(-1, 1), nl.reshape(-1, 1), fl.reshape(-1, 1)
    n = min(len(ml2), len(nl2))
    r = np.random.default_rng(seed)
    m_sub = ml2[r.choice(len(ml2), n, replace=False)]
    n_sub = nl2[r.choice(len(nl2), n, replace=False)]
    clf = SVC(C=3, kernel='rbf', gamma='auto', random_state=seed)
    clf.fit(np.vstack([m_sub, n_sub]), np.concatenate([np.ones(n), np.zeros(n)]))
    ps = float(clf.predict(fl2).mean())
    rb = np.array([ml[i:i + 32].mean() for i in range(0, len(ml), 32)]).reshape(-1, 1)
    tb = np.array([nl[i:i + 32].mean() for i in range(0, len(nl), 32)]).reshape(-1, 1)
    fb = np.array([fl[i:i + 32].mean() for i in range(0, len(fl), 32)]).reshape(-1, 1)
    clf2 = SVC(C=3, kernel='rbf', gamma='auto')
    clf2.fit(np.vstack([rb, tb]), np.concatenate([np.ones(len(rb)), np.zeros(len(tb))]))
    return ps, float(clf2.predict(fb).mean())


ref_ps, ref_pp = run_mia_reference(ml, nl, fl)
got = mia_scores(ml, nl, fl, seed=42, scale=False)
check('MIA (per-sample) trùng', abs(ref_ps - got['MIA']) < 1e-12, f"{ref_ps:.6f} vs {got['MIA']:.6f}")
check('MIA_paper trùng', abs(ref_pp - got['MIA_paper']) < 1e-12,
      f"{ref_pp:.6f} vs {got['MIA_paper']:.6f}")
check('forget_ce trùng', abs(fl.mean() - got['forget_ce']) < 1e-12)
check('member_ce ĐƯỢC GHI RA (adv_common tính rồi vứt)', 'member_ce' in got,
      f"retain-CE = {got['member_ce']:.4f}")

print('\n5) StandardScaler CÓ đổi kết quả với SVM-RBF (nên chỉ bật cho nhánh gộp)')
no_sc = mia_scores(ml, nl, fl, seed=42, scale=False)['MIA']
with_sc = mia_scores(ml, nl, fl, seed=42, scale=True)['MIA']
check('scale=False là mặc định của khung nhìn 1 chiều', True,
      f'không scale {no_sc:.3f} · có scale {with_sc:.3f} → khác nhau, '
      f'nên nhánh img/txt PHẢI để scale=False mới tái lập số cũ')

print('\n6) view_features — hình dạng và định nghĩa gộp')
A = {'logits_img': logits, 'logits_txt': rng.normal(0, 3, size=(200, 4)),
     'ce_img': per_sample_ce_np(logits, labels), 'ce_txt': rng.lognormal(0.5, 0.5, 200)}
fi, pi = view_features(A, 'img')
ft, pt = view_features(A, 'txt')
ff, pf = view_features(A, 'fuse')
check('img/txt là 1 chiều, fuse là 2 chiều',
      fi.shape == (200, 1) and ft.shape == (200, 1) and ff.shape == (200, 2))
check('prob_fuse = trung bình hai softmax', np.allclose(pf, 0.5 * (pi + pt)))
check('prob_fuse là phân bố hợp lệ (tổng = 1)', np.allclose(pf.sum(axis=1), 1.0))
check('CE_fuse (tổng hai nhánh) = CE_img + CE_txt',
      np.allclose(ff.sum(axis=1), A['ce_img'] + A['ce_txt']))

print('\n7) perf_from_probs == adv_common.perf_metrics (phần mảng)')
try:
    from training.eval_multimodal import perf_from_probs
    oh = np.eye(4)[labels].astype(int)
    m = perf_from_probs(softmax_np(logits), oh, 'multiclass')
    check('trả đủ khóa', all(k in m for k in ('AUC', 'Macro_F1', 'F1', 'Accuracy')),
          f"AUC={m['AUC']:.4f} MacroF1={m['Macro_F1']:.4f}")
    check('AUC nằm trong [0,1]', 0.0 <= m['AUC'] <= 1.0)
except Exception as e:
    check('perf_from_probs chạy được', False, str(e))

print('\n' + '=' * 70)
print('THẤT BẠI: ' + (', '.join(FAIL) if FAIL else 'không có — file mới khớp đường tính cũ'))
sys.exit(1 if FAIL else 0)
