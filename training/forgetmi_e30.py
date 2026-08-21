#!/usr/bin/env python3
"""
forgetmi_e30.py — LAUNCHER chạy driver P3/P6/P3-cand trên phân hoạch E30 (code gốc)
====================================================================================
KHÔNG sửa và KHÔNG chép lại bất kỳ driver nào. Nó chỉ hoán ``adv_common.setup_experiment``
thành ``adv_e30.setup_experiment_e30`` rồi chạy driver bạn chỉ định. Nhờ vậy:

  * thêm biến thể mới (p3_cand, p4, p5, main...) KHÔNG phải sửa file này;
  * mọi tham số CLI của driver gốc (--scheme, --ablate, --override...) dùng y nguyên;
  * hàm mất mát / Fisher / FILA / vòng train / ghi CSV giữ NGUYÊN, chỉ dữ liệu đổi.

Cụ thể ``setup_experiment_e30`` làm ba việc:
  1. retain = TOÀN BỘ D_r, test_final = TOÀN BỘ D_t (không cắt r_heldout 10%, sel 25%);
  2. ép ``skip_selection=1`` → không S_val, ReduceLROnPlateau không bao giờ step nên LR
     giữ hằng số sau warmup, CSV chỉ ra hàng 'last' (đúng trọng số sau epoch cuối);
  3. xoá hẳn khoá 'sel'/'r_heldout' khỏi ctx → chạm vào là KeyError, không thể lặng lẽ
     dùng nhầm.

CÁCH CHẠY — đặt tên driver gốc ở vị trí đầu, phần còn lại giữ nguyên như cũ:

  python training/forgetmi_e30.py training/forgetmi_p3_cand.py \\
      --config config_advanced_kaggle.yaml --scheme multi --seed 42 \\
      --override "forget_set_path=./data_splits/forget_set_3per.csv,id=p3cand_e30_3per"

  python training/forgetmi_e30.py training/forgetmi_p3.py --config ... --seed 42 ...
  python training/forgetmi_e30.py training/forgetmi_p6.py --config ... --seed 42 ...

Baseline Forget-MI KHÔNG chạy qua đây (nó không dùng ``setup_experiment``) — dùng
``training/forgetmi_partial_e30.py``.

LƯU Ý SO SÁNH: số ra từ đây KHÔNG so trực tiếp được với run cũ (retain và tập test đã
khác kích thước). Bảng đối chứng phải để baseline chạy bằng ``forgetmi_partial_e30.py``
trên cùng phép chia.

Về ``eval_max_retain``: launcher KHÔNG đụng tới, giữ nguyên giá trị trong config (512).
Muốn MIA cuối lấy member là retain đầy đủ đúng như ``evaluation/eval_unlearning.py``
của tác giả thì tự thêm ``eval_max_retain=0`` vào --override.
"""
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import runpy

import training.adv_common as C
import training.adv_e30 as E


USAGE = (
    "Dùng: python training/forgetmi_e30.py <driver.py> [tham số của driver đó]\n"
    "  ví dụ: python training/forgetmi_e30.py training/forgetmi_p3_cand.py "
    "--config config_advanced_kaggle.yaml --scheme multi --seed 42"
)


def main():
    if len(sys.argv) < 2 or sys.argv[1].startswith('-'):
        raise SystemExit(USAGE)

    target = sys.argv[1]
    if not os.path.isabs(target):
        target = os.path.join(project_root, target)
    if not os.path.exists(target):
        raise SystemExit(f"Không thấy driver '{sys.argv[1]}'.\n{USAGE}")

    print('=' * 78)
    print(f'forgetmi_e30 LAUNCHER → {os.path.relpath(target, project_root)}')
    print('  phân hoạch E30: retain = TOÀN BỘ D_r, test_final = TOÀN BỘ D_t')
    print('  selector: TẮT HOÀN TOÀN (skip_selection ép =1, không r_heldout, không sel)')
    print('=' * 78)

    # Hoán ở CẤP MODULE: driver gọi ``C.setup_experiment(...)`` nên tra tên lúc chạy và
    # sẽ nhận bản E30. ``adv_e30`` đã bind tham chiếu gốc lúc import nên không đệ quy.
    C.setup_experiment = E.setup_experiment_e30
    # ``eval_multimodal.py`` KHÔNG đi qua setup_experiment mà gọi thẳng
    # ``C.build_dataset``. Không hoán luôn thì nó sẽ đánh giá trên test_final=398 của
    # phép chia cũ trong khi model được huấn luyện trên phép chia mới → lệch tập.
    C.build_dataset = E.build_dataset_e30

    # Driver được chạy như chương trình chính, thấy argv y như khi gọi trực tiếp.
    sys.argv = [target] + sys.argv[2:]
    runpy.run_path(target, run_name='__main__')


if __name__ == '__main__':
    main()
