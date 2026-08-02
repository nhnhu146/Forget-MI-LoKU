# -*- coding: utf-8 -*-
"""Gom toan bo ket qua run thanh cac bang cua Chuong 4.

    python tools/build_tables.py <thu_muc_goc> [--out-md bang.md]

Quet de quy tim bo ba file cua moi run:
    timing_*.json|txt        -> T_fisher/T_fila/T_train/T_core, GPU peak, tham so
    selected_checkpoints.*   -> checkpoint S2 (Closest CE)
    results_*.csv|txt        -> checkpoint E30 (last) + moc tham chieu OG/GOLD

Chuan hoa ten cot: Forget-MI ghi Df_AUC/Df_F1/Dt_AUC/Dt_F1, P3 ghi
Forget_AUC/Forget_Macro_F1/Test_AUC/Test_Macro_F1 — cung mot ham perf_metrics.

Run nao thieu file thi bao ro "chua chay", KHONG suy dien va KHONG bo qua im lang.
"""
import argparse
import glob
import io
import json
import os
import sys

import pandas as pd

# run_id -> (nhan hien thi, nhom bang, la_P3)
RUNS = [
    ('fmi_mimic3per_s42', 'Forget-MI',                 'MIMIC 3%',      False),
    ('p3more_mimic3per_s42', 'P3-NoKD-More',           'MIMIC 3%',      True),
    ('fmi_m6_s42',       'Forget-MI',                  'MIMIC 6%',      False),
    ('p3_m6_s42',        'P3-NoKD-More',               'MIMIC 6%',      True),
    ('fmi_m10_s42',      'Forget-MI',                  'MIMIC 10%',     False),
    ('p3_m10_s42',       'P3-NoKD-More',               'MIMIC 10%',     True),
    ('fmi_iu_s42',       'Forget-MI',                  'IU 3%',         False),
    ('p3_iu_s42',        'P3-NoKD-More',               'IU 3%',         True),
    ('p3_m3_s42',        'P3-NoKD-More (day du)',      'Ablation 3%',   True),
    ('abl_fila_s42',     'w/o Fisher/FILA',            'Ablation 3%',   True),
    ('abl_ihl_s42',      'w/o IHL',                    'Ablation 3%',   True),
    ('abl_mumr_s42',     'w/o MU/MR',                  'Ablation 3%',   True),
]
GROUPS = ['MIMIC 3%', 'MIMIC 6%', 'MIMIC 10%', 'IU 3%', 'Ablation 3%']
# nhom -> hau to nhan cua hang tham chieu trong CSV (og_<tag> / re_<tag>)
REF_TAG = {'MIMIC 3%': 'mimic3per', 'MIMIC 6%': 'mimic6per',
           'MIMIC 10%': 'mimic10per', 'IU 3%': 'iu3per', 'Ablation 3%': 'mimic3per'}

M = ['Df_AUC', 'Df_F1', 'Dt_AUC', 'Dt_F1', 'MIA', 'MIA_paper', 'forget_ce', 'test_ce']
ALIAS = {'Df_AUC': ['Df_AUC', 'Forget_AUC'], 'Df_F1': ['Df_F1', 'Forget_Macro_F1'],
         'Dt_AUC': ['Dt_AUC', 'Test_AUC'], 'Dt_F1': ['Dt_F1', 'Test_Macro_F1'],
         'MIA': ['MIA'], 'MIA_paper': ['MIA_paper'],
         'forget_ce': ['forget_ce'], 'test_ce': ['test_ce']}


def pick(row, key):
    """Lay metric theo ten chuan, tu do alias giua hai bo ghi CSV."""
    for c in ALIAS[key]:
        if c in row and pd.notna(row[c]):
            return float(row[c])
    return None


def scan(root):
    """Tra (timings, s2, csvs): map run_id -> du lieu."""
    timings, s2, csv_rows = {}, {}, []
    for p in glob.glob(os.path.join(root, '**', '*'), recursive=True):
        b = os.path.basename(p)
        if not os.path.isfile(p):
            continue
        if b.startswith('timing_') and b.endswith(('.json', '.txt')):
            try:
                d = json.load(io.open(p, encoding='utf-8'))
                timings[str(d.get('run_id'))] = d
            except Exception as e:
                print(f"  bo qua {b}: {e}", file=sys.stderr)
        elif b.startswith('selected_checkpoints') and b.endswith(('.json', '.txt')):
            try:
                d = json.load(io.open(p, encoding='utf-8'))
                # selected_checkpoints.json khong chua run_id -> lay tu timing cung thu muc
                sib = glob.glob(os.path.join(os.path.dirname(p), '..', 'timing_*'))
                sib += glob.glob(os.path.join(os.path.dirname(p), 'timing_*'))
                sib += glob.glob(os.path.join(os.path.dirname(os.path.dirname(p)), 'timing_*'))
                rid = None
                for t in sib:
                    try:
                        rid = json.load(io.open(t, encoding='utf-8')).get('run_id'); break
                    except Exception:
                        pass
                if rid:
                    s2[str(rid)] = d.get('results', {}).get('S2_closest_ce', {})
            except Exception as e:
                print(f"  bo qua {b}: {e}", file=sys.stderr)
        elif b.startswith('results_') and b.endswith(('.csv', '.txt')):
            try:
                csv_rows.append(pd.read_csv(p))
            except Exception as e:
                print(f"  bo qua {b}: {e}", file=sys.stderr)
    allrows = pd.concat(csv_rows, ignore_index=True) if csv_rows else pd.DataFrame()
    return timings, s2, allrows


def last_row(df, rid, want_epoch=30):
    """Hang checkpoint cuoi (E30). P3 dung cot checkpoint_kind, Forget-MI dung checkpoint.

    CAN THAN: mot run_id co the xuat hien o NHIEU file — vi du p3more_mimic3per_s42 co
    ca o run 30 epoch lan run smoke-test 15 epoch cua dot candidate. Lay .iloc[-1] la
    boc nham. Uu tien hang co selected_epoch = want_epoch, va bao ro khi co trung."""
    if df.empty or 'id' not in df.columns:
        return None
    d = df[df['id'].astype(str) == rid]
    for col in ('checkpoint_kind', 'checkpoint'):
        if col in d.columns:
            hit = d[d[col].astype(str) == 'last']
            if not len(hit):
                continue
            if 'selected_epoch' in hit.columns:
                exact = hit[hit['selected_epoch'] == want_epoch]
                if len(hit) > 1:
                    eps = sorted(set(hit['selected_epoch'].dropna().astype(int)))
                    print(f"  ! {rid}: {len(hit)} hang 'last' (epoch {eps}) -> lay E{want_epoch}",
                          file=sys.stderr)
                if len(exact):
                    return exact.iloc[-1]
                return hit.sort_values('selected_epoch').iloc[-1]
            return hit.iloc[-1]
    return None


def ref_row(df, kind, tag):
    if df.empty or 'checkpoint_kind' not in df.columns:
        return None
    hit = df[df['checkpoint_kind'].astype(str) == f'{kind}_{tag}']
    return hit.iloc[-1] if len(hit) else None


def fmt(v, n=3, miss='—'):
    """miss='—' : khong co du lieu.  miss='n/a': co hang nhung metric khong tinh duoc
    (vi du AUC = NaN khi mo hinh sup do, logit tran) — hai truong hop nay KHAC nhau,
    khong duoc hien thi giong nhau."""
    return miss if v is None else f'{v:.{n}f}'


def quality_table(group, timings, s2, df):
    tag = REF_TAG[group]
    head = ('| Mô hình | Chốt | Epoch | Df-AUC ↓ | Df-F1 ↓ | Dt-AUC ↑ | Dt-F1 ↑ | '
            'MIA ↓ | MIA_paper ↓ | Forget-CE | Test-CE / nm_val-CE |')
    L = [head, '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    if group != 'Ablation 3%':
        for kind, name in (('og', 'θ_og (gốc)'), ('re', 'θ_re (gold)')):
            r = ref_row(df, kind, tag)
            if r is not None:
                L.append(f"| {name} | – | – | " + ' | '.join(
                    fmt(pick(r, k)) for k in M) + ' |')
    for rid, label, grp, _is_p3 in RUNS:
        if grp != group:
            continue
        s = s2.get(rid)
        if s and s.get('epoch') is not None:
            # S2 khong co test_ce (selector dung nm_val_ce) -> dien nm_val_ce vao cot cuoi
            vals = [fmt(s.get(k)) for k in M[:-1]] + [fmt(s.get('nm_val_ce'))]
            L.append(f"| {label} | **S2** | E{int(s['epoch']) + 1} | " + ' | '.join(vals) + ' |')
        elif s is not None:
            L.append(f"| {label} | **S2** | *{s.get('note', 'n/a')}* | " +
                     ' | '.join('—' for _ in M) + ' |')
        r = last_row(df, rid)
        if r is not None:
            L.append(f"| {label} | E30 | 30 | " +
                     ' | '.join(fmt(pick(r, k), miss='n/a') for k in M) + ' |')
        if s is None and r is None:
            L.append(f"| {label} | – | **chưa chạy** | " + ' | '.join('—' for _ in M) + ' |')
    return '\n'.join(L)


def cost_table(group, timings):
    L = ['| Mô hình | Tham số cập nhật | Tỉ lệ | T_Fisher (s) | T_FILA (s) | T_train (s) | '
         '**T_core (s)** | GPU peak alloc (GB) | GPU peak reserved (GB) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for rid, label, grp, _ in RUNS:
        if grp != group:
            continue
        d = timings.get(rid)
        if not d:
            L.append(f'| {label} | **chưa chạy** | — | — | — | — | — | — | — |')
            continue
        L.append(f"| {label} | {int(d.get('trainable_params', 0)):,} | "
                 f"{100 * d.get('trainable_ratio', 0):.2f}% | "
                 f"{d.get('fisher_seconds', 0):.1f} | {d.get('fila_seconds', 0):.1f} | "
                 f"{d.get('train_seconds', 0):.1f} | **{d.get('core_seconds', 0):.1f}** | "
                 f"{d.get('core_peak_allocated_gb', 0):.2f} | "
                 f"{d.get('core_peak_reserved_gb', 0):.2f} |")
    return '\n'.join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--out-md', default=None)
    a = ap.parse_args()

    timings, s2, df = scan(a.root)
    have = [r for r, *_ in RUNS if r in timings]
    miss = [r for r, *_ in RUNS if r not in timings]
    print(f'Tim thay {len(have)}/{len(RUNS)} run.')
    if miss:
        print('CHUA CO:', ', '.join(miss))

    out = []
    for g in GROUPS:
        out.append(f'\n## {g} — chất lượng\n')
        out.append(quality_table(g, timings, s2, df))
        out.append(f'\n## {g} — tài nguyên\n')
        out.append(cost_table(g, timings))
    # kiem tra dong nhat phan cung (khong duoc tron GPU)
    gpus = {d.get('gpu_name') for d in timings.values() if d.get('gpu_name')}
    out.append('\n## Điều kiện\n')
    out.append(f"GPU: {', '.join(sorted(gpus))}"
               + ('   ⚠️ TRỘN NHIỀU GPU — thời gian không so được với nhau' if len(gpus) > 1 else ''))
    for rid, label, grp, _ in RUNS:
        d = timings.get(rid)
        if d:
            out.append(f"- `{rid}` ({grp} / {label}): epochs {d.get('epochs')}, "
                       f"updates {d.get('optimizer_updates')}, bs {d.get('batch_size')}, "
                       f"lr {d.get('learning_rate')}, seed {d.get('seed')}, "
                       f"selector `{d.get('selector', 'n/a')}`")
    txt = '\n'.join(out)
    print(txt)
    if a.out_md:
        io.open(a.out_md, 'w', encoding='utf-8').write(txt + '\n')
        print(f'\n-> {a.out_md}')


if __name__ == '__main__':
    main()
