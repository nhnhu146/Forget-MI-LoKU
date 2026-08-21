#!/usr/bin/env python3
"""
forgetmi_partial_e30.py — Baseline Forget-MI trên PHÂN HOẠCH E30 (giống code gốc)
==================================================================================
Bản mới, KHÔNG sửa ``forgetmi_partial.py``. File cũ giữ nguyên để mọi kết quả baseline
đã chạy vẫn tái lập được.

VÌ SAO CẦN FILE NÀY: ``forgetmi_e30.py`` (P3/P6) chạy trên retain = TOÀN BỘ D_r và
test = TOÀN BỘ D_t. Nếu baseline vẫn chạy trên phép chia cũ (retain 5.410, test 398)
thì hai phía đo trên hai tập dữ liệu khác nhau → bảng so sánh vô nghĩa. File này đưa
baseline về đúng phép chia đó.

CÁCH LÀM: hoán đổi ``forgetmi_partial.data_split`` bằng phiên bản bám code gốc rồi gọi
lại ``main()`` của nó. Toàn bộ vòng unlearn, hàm mất mát, hinge, cadence
``optimizer.step()`` 1 lần/epoch — giữ NGUYÊN, không đụng một dòng nào.

Về ``validation`` và ``sel``: code gốc của tác giả CÓ dựng một tập validation nhưng
nhập ngược lại vào train rồi tạo ``val_dataloader`` mà không bao giờ dùng; tập ``sel``
là của selector, không tồn tại trong bản gốc. Ở đây cả hai trả về RỖNG — hợp lệ với
chốt chặn của ``build_dataset`` (chỉ báo lỗi khi ``n_pre > 0`` mà dict rỗng).

Chạy (thay cho forgetmi_partial.py, cùng bộ tham số CLI):
  WANDB_MODE=disabled python training/forgetmi_partial_e30.py \\
      --config config_baseline_kaggle.yaml --seed 42 \\
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=baseline_e30_3per"
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import training.forgetmi_partial as FP
from training.adv_e30 import data_split_original


def data_split_e30(split_list_path, forget_ids_path, rand_ratio=None,
                   validation_ratio=None, seed=42, test_sel_splits=4,
                   retain_heldout_splits=10):
    """Thay ``forgetmi_partial.data_split`` — giữ NGUYÊN chữ ký và thứ tự 17 giá trị
    trả về, nhưng phép chia là của code gốc: KHÔNG cắt r_heldout, KHÔNG cắt sel.

      train/retain = TOÀN BỘ D_r      test = TOÀN BỘ D_t
      validation   = RỖNG             sel  = RỖNG

    Các tham số ``validation_ratio``/``test_sel_splits``/``retain_heldout_splits`` được
    giữ trong chữ ký cho tương thích nhưng KHÔNG có tác dụng.
    """
    splits = data_split_original(split_list_path, forget_ids_path)

    train_img_txt_ids,  train_labels  = splits['retain']
    test_img_txt_ids,   test_labels   = splits['test_final']
    forget_img_txt_ids, forget_labels = splits['forget']
    rand_img_txt_ids,   rand_labels   = splits['random']

    sel_img_txt_ids,  sel_labels  = {}, {}      # không tồn tại trong code gốc

    # ---- validation: TÁI LẬP ĐÚNG CODE GỐC ----
    # Bản gốc tách 10% bằng train_test_split(random_state=42, stratify) rồi NHẬP NGƯỢC
    # lại vào train (dòng 270-278), nên retain KHÔNG mất mẫu nào và val ⊂ retain.
    # Ta làm y hệt. Hai lý do phải giữ, không được trả về rỗng:
    #   * ``evaluate_last_and_best=1`` cần val để chọn 'val_best'; val rỗng →
    #     RuntimeError("No eligible val-best checkpoint was produced");
    #   * ``evaluate_last_and_best=0`` thì KHÔNG lưu checkpoint nào cả → không eval lại được.
    # ⚠️ val ⊂ retain là LỖI CỦA CODE GỐC (chọn checkpoint trên chính dữ liệu huấn
    #    luyện). Vì vậy CHỈ báo cáo hàng 'last' (E30); KHÔNG dùng hàng 'val_best'.
    from sklearn.model_selection import train_test_split
    ratio = float(validation_ratio) if validation_ratio else 0.1
    _ids = list(train_img_txt_ids.keys())
    _labs = [train_labels[k][0] for k in _ids]
    _, _val_ids = train_test_split(_ids, test_size=ratio, random_state=42, stratify=_labs)
    val_img_txt_ids = {k: train_img_txt_ids[k] for k in _val_ids}
    val_labels = {k: train_labels[k] for k in _val_ids}

    n_train, n_val, n_rand, n_test, n_forget = (
        len(train_img_txt_ids), len(val_img_txt_ids), len(rand_img_txt_ids),
        len(test_img_txt_ids), len(forget_img_txt_ids))

    print(f"Split E30 (bám code gốc): train/retain={n_train} test={n_test} "
          f"forget={n_forget} random={n_rand} | validation={n_val} (⊂ retain, "
          f"CHỈ để chọn val_best — đừng báo cáo) | sel=0 (không tồn tại)")

    return (train_labels, train_img_txt_ids, val_labels, val_img_txt_ids,
            test_labels, test_img_txt_ids, rand_labels, rand_img_txt_ids,
            forget_labels, forget_img_txt_ids, n_train, n_val, n_test, n_rand,
            n_forget, sel_labels, sel_img_txt_ids)


def main():
    """Hai chế độ:

      (a) không tham số vị trí → chạy baseline Forget-MI (``FP.main()``);
      (b) tham số vị trí đầu là một file .py → chạy file đó sau khi đã hoán phép chia.
          Dùng cho ``scripts/unlearn_baselines.py`` (NegGrad+/CF-k/EU-k) vì nó import
          ``build_dataset`` của ``forgetmi_partial``, không đi qua ``adv_common``.

    Ví dụ (b):
      python training/forgetmi_partial_e30.py scripts/unlearn_baselines.py \\
          --config config_baseline_kaggle.yaml --method neggrad --seed 42 --override "..."
    """
    print('=' * 78)
    print('BẢN E30 — phép chia của code gốc')
    print('  retain = TOÀN BỘ D_r, test = TOÀN BỘ D_t')
    print('  không validation, không sel, không selector')
    print('  CHỈ so sánh được với run chạy bằng training/forgetmi_e30.py')
    print('=' * 78)
    FP.data_split = data_split_e30

    if len(sys.argv) > 1 and sys.argv[1].endswith('.py'):
        import runpy
        target = sys.argv[1]
        if not os.path.isabs(target):
            target = os.path.join(project_root, target)
        if not os.path.exists(target):
            raise SystemExit(f"Không thấy script '{sys.argv[1]}'.")
        print(f'→ chạy {os.path.relpath(target, project_root)} trên phép chia E30')
        sys.argv = [target] + sys.argv[2:]
        runpy.run_path(target, run_name='__main__')
        return

    FP.main()


if __name__ == '__main__':
    main()
