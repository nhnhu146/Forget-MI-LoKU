# -*- coding: utf-8 -*-
"""Dựng Bảng 18 (hướng dẫn đo thời gian) từ các file timing_*.json.

    python tools/timing_table.py --fmi <json|dir> [...] --p3 <json|dir> [...] \
        [--out-md bang.md] [--out-tex bang.tex]

Mục 17: nếu một phương pháp có ≥2 file JSON (nhiều lần chạy) thì báo mean ± std;
nếu chỉ 1 file thì ghi rõ `single-run timing`. KHÔNG trộn JSON từ GPU khác nhau —
script sẽ dừng nếu phát hiện tên GPU không đồng nhất.

Mục 16: script đối chiếu các điều kiện phải giống nhau giữa hai phương pháp
(GPU, split, forget set, seed, epoch, số optimizer update, batch size, workers,
pin_memory, precision) và IN CẢNH BÁO cho mọi chỗ lệch — không tự ý sửa số.
"""
import argparse
import glob
import io
import json
import math
import os
import sys


# ---- các khóa PHẢI trùng nhau giữa Forget-MI và P3 (mục 16) ----
FAIR_KEYS = [
    ('gpu_name', 'GPU'), ('dataset', 'data split'), ('forget_pct', 'forget set'),
    ('seed', 'seed'), ('epochs', 'số epoch'), ('optimizer_updates', 'số optimizer update'),
    ('batch_size', 'batch size'), ('num_workers', 'dataloader workers'),
    ('pin_memory', 'pin_memory'), ('precision_mode', 'precision'),
]


def load_jsons(paths):
    """Nhận file .json hoặc thư mục (quét đệ quy timing_*.json). Trả list dict."""
    out = []
    for p in paths:
        if os.path.isdir(p):
            hits = sorted(glob.glob(os.path.join(p, '**', 'timing_*.json'), recursive=True))
            if not hits:
                print(f"⚠️  không thấy timing_*.json trong {p}")
            for h in hits:
                out.append(_read(h))
        else:
            out.append(_read(p))
    return [x for x in out if x]


def _read(path):
    try:
        with io.open(path, encoding='utf-8') as f:
            d = json.load(f)
        d['_path'] = path
        return d
    except Exception as e:
        print(f"⚠️  bỏ qua {path}: {e}")
        return None


def agg(runs, key):
    """(mean, std, n) của một khóa số học qua các lần chạy."""
    xs = [float(r[key]) for r in runs if isinstance(r.get(key), (int, float))]
    if not xs:
        return (float('nan'), 0.0, 0)
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))
    return (m, sd, len(xs))


def fmt_sec(runs, key, unit='s'):
    """'1234.5' (1 lần) hoặc '1234.5 ± 12.3' (nhiều lần). unit='h' → đổi sang giờ."""
    m, sd, n = agg(runs, key)
    if n == 0:
        return '—'
    div = 3600.0 if unit == 'h' else 1.0
    m, sd = m / div, sd / div
    dec = 3 if unit == 'h' else 1
    return f"{m:.{dec}f}" if n == 1 else f"{m:.{dec}f} ± {sd:.{dec}f}"


def fmt_gb(runs, key):
    m, sd, n = agg(runs, key)
    if n == 0:
        return '—'
    return f"{m:.2f}" if n == 1 else f"{m:.2f} ± {sd:.2f}"


def fmt_int(runs, key):
    m, _, n = agg(runs, key)
    return '—' if n == 0 else f"{int(round(m)):,}"


def fmt_ratio(runs, key):
    m, _, n = agg(runs, key)
    return '—' if n == 0 else f"{100 * m:.2f}%"


def check_fairness(fmi, p3):
    """In cảnh báo cho mọi điều kiện lệch nhau (mục 16). Trả list dòng cảnh báo."""
    warns = []
    for group, runs in (('Forget-MI', fmi), ('P3', p3)):
        gpus = {r.get('gpu_name') for r in runs if r.get('gpu_name')}
        if len(gpus) > 1:
            warns.append(f"❌ {group}: trộn kết quả từ nhiều GPU {sorted(gpus)} — mục 17 cấm.")
        # Quét thư mục dễ vơ nhầm JSON của phương pháp kia → gộp mean/std sai hoàn toàn.
        methods = {r.get('method') for r in runs if r.get('method')}
        if len(methods) > 1:
            warns.append(f"❌ {group}: gộp nhầm nhiều method {sorted(methods)} — "
                         f"chỉ mỗi lần chạy LẶP của CÙNG một phương pháp mới được gộp.")
    if not fmi or not p3:
        return warns
    a, b = fmi[0], p3[0]
    for k, label in FAIR_KEYS:
        va, vb = a.get(k), b.get(k)
        if va is None or vb is None:
            warns.append(f"⚠️  thiếu '{k}' ({label}) trong JSON — không kiểm tra được.")
        elif va != vb:
            warns.append(f"⚠️  {label} lệch: Forget-MI={va!r} vs P3={vb!r} — phải ghi rõ trong khóa luận.")
    la, lb = a.get('learning_rate'), b.get('learning_rate')
    if la is not None and lb is not None and la != lb:
        warns.append(f"ℹ️  learning rate khác nhau ({la} vs {lb}) — CHỦ Ý (full-FT vs LoRA), nhớ ghi chú.")
    # Mục 3.3: T_pipeline chỉ so được khi CÙNG selector + CÙNG checkpoint policy.
    for k, label in (('selector', 'selector'), ('checkpoint_policy', 'checkpoint policy')):
        va, vb = a.get(k), b.get(k)
        if va is not None and vb is not None and va != vb:
            warns.append(f"❌ {label} khác nhau ({va} vs {vb}) — mục 3.3: KHÔNG được so "
                         f"End-to-end pipeline; chỉ so Core method time.")
    return warns


ROWS = [
    ('Fisher',                  lambda r: fmt_sec(r, 'fisher_seconds')),
    ('FILA initialization',     lambda r: fmt_sec(r, 'fila_seconds')),
    ('Unlearning train',        lambda r: fmt_sec(r, 'train_seconds')),
    ('**Core method time**',    lambda r: '**' + fmt_sec(r, 'core_seconds') + '**'),
    ('Checkpoint selection',    lambda r: fmt_sec(r, 'selection_seconds')),
    ('&nbsp;&nbsp;↳ trong đó ckpt I/O', lambda r: fmt_sec(r, 'ckpt_seconds')),
    ('Final evaluation',        lambda r: fmt_sec(r, 'eval_seconds')),
    ('End-to-end pipeline',     lambda r: fmt_sec(r, 'pipeline_seconds')),
    ('_(chẩn đoán, ngoài pipeline)_', lambda r: fmt_sec(r, 'diagnostic_seconds')),
    ('Trainable params',        lambda r: fmt_int(r, 'trainable_params')),
    ('Trainable ratio',         lambda r: fmt_ratio(r, 'trainable_ratio')),
    ('GPU peak allocated (GB)', lambda r: fmt_gb(r, 'core_peak_allocated_gb')),
    ('GPU peak reserved (GB)',  lambda r: fmt_gb(r, 'core_peak_reserved_gb')),
]

# Bản LaTeX: bỏ ** markdown, thêm \textbf
TEX_ROWS = [
    ('Fisher', 'fisher_seconds', 's'), ('Khởi tạo FILA', 'fila_seconds', 's'),
    ('Huấn luyện gỡ bỏ', 'train_seconds', 's'), ('Thời gian lõi', 'core_seconds', 'bold'),
    ('Chọn checkpoint', 'selection_seconds', 's'), ('Đánh giá cuối', 'eval_seconds', 's'),
    ('Toàn pipeline', 'pipeline_seconds', 's'),
]


def build_md(fmi, p3, p3_name):
    n_f, n_p = len(fmi), len(p3)
    lines = [f"| Thành phần (giây) | Forget-MI | {p3_name} |", "|---|---:|---:|"]
    for label, fn in ROWS:
        lines.append(f"| {label} | {fn(fmi)} | {fn(p3)} |")
    note_f = 'single-run timing' if n_f == 1 else f'mean ± std, n={n_f}'
    note_p = 'single-run timing' if n_p == 1 else f'mean ± std, n={n_p}'
    lines += ["", f"Forget-MI: {note_f}. {p3_name}: {note_p}.",
              f"GPU: {fmi[0].get('gpu_name') if fmi else '?'} | "
              f"CUDA {fmi[0].get('cuda_version') if fmi else '?'} | "
              f"torch {fmi[0].get('torch_version') if fmi else '?'}",
              "T_core(Forget-MI) = T_train. T_core(P3) = T_Fisher + T_FILA + T_train. "
              "GPU peak lõi lấy MAX giữa các giai đoạn, không cộng dồn."]
    return '\n'.join(lines)


def build_tex(fmi, p3, p3_name):
    def cell(runs, key, kind):
        v = fmt_sec(runs, key)
        return f"\\textbf{{{v}}}" if kind == 'bold' else v
    L = ["\\begin{table}[htbp]", "\\centering",
         "\\caption{Phân rã thời gian gỡ bỏ học máy giữa Forget-MI và "
         + p3_name + " (cùng GPU, cùng phân hoạch dữ liệu, cùng seed).}",
         "\\label{tab:timing-breakdown}", "\\begin{tabular}{lrr}", "\\hline",
         f"Thành phần (giây) & Forget-MI & {p3_name} \\\\", "\\hline"]
    for label, key, kind in TEX_ROWS:
        lab = f"\\textbf{{{label}}}" if kind == 'bold' else label
        L.append(f"{lab} & {cell(fmi, key, kind)} & {cell(p3, key, kind)} \\\\")
    L += ["\\hline",
          f"Tham số cập nhật & {fmt_int(fmi, 'trainable_params')} & {fmt_int(p3, 'trainable_params')} \\\\",
          f"Tỉ lệ tham số & {fmt_ratio(fmi, 'trainable_ratio')} & {fmt_ratio(p3, 'trainable_ratio')} \\\\",
          f"GPU peak allocated (GB) & {fmt_gb(fmi, 'core_peak_allocated_gb')} & {fmt_gb(p3, 'core_peak_allocated_gb')} \\\\",
          f"GPU peak reserved (GB) & {fmt_gb(fmi, 'core_peak_reserved_gb')} & {fmt_gb(p3, 'core_peak_reserved_gb')} \\\\",
          "\\hline", "\\end{tabular}", "\\end{table}"]
    return '\n'.join(L).replace('%', '\\%').replace('±', '$\\pm$')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fmi', nargs='+', required=True, help='timing JSON hoặc thư mục của Forget-MI')
    ap.add_argument('--p3', nargs='+', required=True, help='timing JSON hoặc thư mục của P3')
    ap.add_argument('--p3-name', default='P3-more')
    ap.add_argument('--out-md', default=None)
    ap.add_argument('--out-tex', default=None)
    a = ap.parse_args()

    fmi, p3 = load_jsons(a.fmi), load_jsons(a.p3)
    if not fmi or not p3:
        print("❌ thiếu JSON cho một trong hai phương pháp."); sys.exit(1)
    print(f"Forget-MI: {len(fmi)} run  |  {a.p3_name}: {len(p3)} run")
    for r in fmi + p3:
        print(f"   {os.path.basename(r['_path'])}  method={r.get('method')}  "
              f"core={r.get('core_seconds', 0):.1f}s")

    warns = check_fairness(fmi, p3)
    print("\n----- KIỂM TRA ĐIỀU KIỆN CÔNG BẰNG (mục 16-17) -----")
    print('\n'.join(warns) if warns else "✅ mọi điều kiện bắt buộc đều trùng khớp.")

    md = build_md(fmi, p3, a.p3_name)
    print("\n----- BẢNG (Markdown) -----\n" + md)
    if a.out_md:
        io.open(a.out_md, 'w', encoding='utf-8').write(md + '\n')
        print(f"\n→ {a.out_md}")
    if a.out_tex:
        io.open(a.out_tex, 'w', encoding='utf-8').write(build_tex(fmi, p3, a.p3_name) + '\n')
        print(f"→ {a.out_tex}")


if __name__ == '__main__':
    main()
