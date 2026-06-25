#!/usr/bin/env python3
"""
Generate Indiana University CXR dataset files in Forget-MI's expected format.

Input:  data_iu/parsed/reports.json  (from parse_iu_reports.py)

Output: data_iu/data/metadata/all_data.tsv      ← consumed by EdemaClassificationProcessor
        data_iu/data/img_data/<image_id>.png    ← symlink/copy from raw
        data_iu/data_splits/iu-split.csv        ← columns: subject_id, study_id, dicom_id, label, fold
        data_iu/data_splits/forget_set_3per_iu.csv
        data_iu/data_splits/forget_set_6per_iu.csv
        data_iu/data_splits/forget_set_10per_iu.csv

Schema notes (to match MIMIC pipeline shared in joint_img_txt/model_utils.py):
- all_data.tsv: tab-separated, 4 cols, NO header
    column[0] = guid/index ignored by processor   (we use report_id)
    column[1] = label string                       (we use "0" or "1")
    column[2] = report_id                          (used as text_key in features)
    column[-1] = text (last col)
- split CSV: ',' separated, header row
    columns: subject_id, study_id, dicom_id, edeme_severity, fold
    edeme_severity = 0/1 (binary) — to be parsed as float
    fold = "TRAIN" or "TEST"

Usage:
    python scripts/make_iu_dataset.py \
        --reports data_iu/parsed/reports.json \
        --img_src data_iu/raw/NLMCXR_png \
        --out_dir data_iu \
        --link_mode copy        # or 'symlink' on Linux

Notes:
- IU does NOT have study_id like MIMIC. We use report_id as both subject_id and study_id.
  This matches Forget-MI's pipeline (which keys on study_id ↔ report_id in cached features).
- Forget sets: chosen by SUBJECT_ID to avoid leak (multi-image-per-subject groups together).
"""
import argparse
import csv
import json
import os
import random
import shutil
import sys
from collections import defaultdict

from sklearn.model_selection import train_test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", default="data_iu/parsed/reports.json")
    ap.add_argument("--img_src", default="data_iu/raw/NLMCXR_png")
    ap.add_argument("--out_dir", default="data_iu")
    ap.add_argument("--test_size", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--link_mode", choices=["copy", "symlink", "skip"], default="copy",
                    help="copy=duplicate PNG; symlink=ln -s (Linux only); skip=assume images already in place")
    args = ap.parse_args()

    if not os.path.exists(args.reports):
        print(f"❌ {args.reports} not found. Run parse_iu_reports.py first.")
        sys.exit(1)

    print(f"📂 Loading {args.reports}")
    with open(args.reports, encoding="utf-8") as f:
        reports = json.load(f)
    print(f"   {len(reports)} reports")

    # ============== Build per-image rows (1 row per (report, image) pair) ==============
    rows = []  # list of dicts: subject_id, study_id, dicom_id, label, text
    img_to_report = {}
    for rid, rec in reports.items():
        label = rec["label_binary"]
        for img_id in rec["image_ids"]:
            rows.append({
                "subject_id": rid,      # IU: subject = report (no patient-level grouping)
                "study_id":   rid,      # used as text_key in Forget-MI's processor
                "dicom_id":   img_id,
                "label":      label,
                "text":       rec["text"],
            })
            img_to_report[img_id] = rid
    print(f"📊 Total (report, image) pairs: {len(rows)}")

    # Drop pairs whose image file doesn't actually exist on disk
    if args.img_src and os.path.isdir(args.img_src):
        kept = []
        dropped = 0
        for r in rows:
            src_png = os.path.join(args.img_src, f"{r['dicom_id']}.png")
            if os.path.exists(src_png):
                kept.append(r)
            else:
                dropped += 1
        if dropped > 0:
            print(f"   Dropped {dropped} rows (PNG missing in {args.img_src})")
        rows = kept

    # ============== Train/test split (by SUBJECT_ID, stratified by label) ==============
    # Group: subject_id → list of indices in rows
    subj_to_indices = defaultdict(list)
    for i, r in enumerate(rows):
        subj_to_indices[r["subject_id"]].append(i)

    subjects = list(subj_to_indices.keys())
    # Subject-level label = majority label across its images
    subj_label = {}
    for s, idxs in subj_to_indices.items():
        labels = [rows[i]["label"] for i in idxs]
        subj_label[s] = max(set(labels), key=labels.count)

    random.seed(args.seed)
    subjects_sorted = sorted(subjects)
    random.shuffle(subjects_sorted)
    subj_labels = [subj_label[s] for s in subjects_sorted]

    train_subj, test_subj = train_test_split(
        subjects_sorted,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=subj_labels,
    )
    print(f"📊 Split (subject-level): train={len(train_subj)}, test={len(test_subj)}")

    # Assign fold to each row
    train_set = set(train_subj)
    for r in rows:
        r["fold"] = "TRAIN" if r["subject_id"] in train_set else "TEST"

    # ============== Forget sets (subject-level, from TRAIN only) ==============
    forget_pcts = [3, 6, 10]
    forget_files = {}
    for pct in forget_pcts:
        n_forget = max(1, int(round(len(train_subj) * pct / 100)))
        # Stratified random sample to keep label distribution
        rng = random.Random(args.seed + pct)
        # Group train subjects by label
        train_by_label = defaultdict(list)
        for s in train_subj:
            train_by_label[subj_label[s]].append(s)
        # Sample proportional from each label
        forget_subjects = []
        for lbl, subs in train_by_label.items():
            k = max(1, int(round(len(subs) * pct / 100)))
            forget_subjects.extend(rng.sample(subs, min(k, len(subs))))
        # Trim if oversampled
        rng.shuffle(forget_subjects)
        forget_subjects = forget_subjects[:n_forget]
        forget_files[pct] = forget_subjects
        print(f"   forget_{pct}%: {len(forget_subjects)} subjects")

    # ============== Write outputs ==============

    # all_data.tsv (no header, 4 cols: guid_idx, label, report_id, text)
    metadata_dir = os.path.join(args.out_dir, "data", "metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    tsv_path = os.path.join(metadata_dir, "all_data.tsv")
    # Deduplicate by report_id (1 row per report — text repeats across images would inflate)
    seen_rid = set()
    tsv_rows = []
    for r in rows:
        rid = r["subject_id"]
        if rid in seen_rid:
            continue
        seen_rid.add(rid)
        tsv_rows.append((str(len(tsv_rows)), str(r["label"]), rid, r["text"]))
    with open(tsv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        for row in tsv_rows:
            w.writerow(row)
    print(f"💾 all_data.tsv: {len(tsv_rows)} reports → {tsv_path}")

    # split CSV: 1 row per (image, report) pair
    splits_dir = os.path.join(args.out_dir, "data_splits")
    os.makedirs(splits_dir, exist_ok=True)
    split_csv = os.path.join(splits_dir, "iu-split.csv")
    with open(split_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "study_id", "dicom_id", "edeme_severity", "fold"])
        for r in rows:
            w.writerow([r["subject_id"], r["study_id"], r["dicom_id"], r["label"], r["fold"]])
    print(f"💾 split CSV: {len(rows)} rows → {split_csv}")

    # Forget set CSVs (subject_id only, header-less per MIMIC convention)
    for pct, subjects in forget_files.items():
        path = os.path.join(splits_dir, f"forget_set_{pct}per_iu.csv")
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["subject_id"])  # MIMIC's forget set has header with subject_id col
            for s in subjects:
                w.writerow([s])
        print(f"💾 forget {pct}%: {len(subjects)} subjects → {path}")

    # ============== Image data ==============
    img_dst_dir = os.path.join(args.out_dir, "data", "img_data")
    os.makedirs(img_dst_dir, exist_ok=True)
    if args.link_mode == "skip":
        print(f"⏭  Image link/copy SKIPPED (--link_mode=skip)")
    else:
        n_done = 0
        n_skip = 0
        for r in rows:
            src = os.path.join(args.img_src, f"{r['dicom_id']}.png")
            dst = os.path.join(img_dst_dir, f"{r['dicom_id']}.png")
            if os.path.exists(dst):
                n_skip += 1
                continue
            if not os.path.exists(src):
                continue
            if args.link_mode == "copy":
                shutil.copy(src, dst)
            else:
                try:
                    os.symlink(os.path.abspath(src), dst)
                except OSError as e:
                    # Fallback to copy on Windows / non-priv
                    shutil.copy(src, dst)
            n_done += 1
        print(f"🖼  Images: {n_done} {args.link_mode}-ed, {n_skip} already in place")

    # ============== Summary ==============
    print(f"\n{'='*60}")
    print(f"✅ IU dataset ready in {args.out_dir}/")
    print(f"{'='*60}")
    print(f"   data/metadata/all_data.tsv     ({len(tsv_rows)} reports)")
    print(f"   data/img_data/*.png            ({len(rows)} images)")
    print(f"   data_splits/iu-split.csv       ({len(rows)} rows)")
    for pct in forget_pcts:
        print(f"   data_splits/forget_set_{pct}per_iu.csv  ({len(forget_files[pct])} subjects)")
    print(f"\nNext: upload data_iu/data/ to Kaggle Dataset 'forget-mi-data-iu'")
    print(f"      (data_splits/*.csv already committed in repo)")


if __name__ == "__main__":
    main()
