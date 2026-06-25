#!/usr/bin/env python3
"""
Parse Indiana University CXR dataset from Kaggle (raddar's chest-xrays-indiana-university).

Different from parse_iu_reports.py (XML-based) — this one uses the preprocessed CSV
format that's common on Kaggle:
  - indiana_reports.csv:    one row per report (uid, MeSH, Problems, image, findings, impression, ...)
  - indiana_projections.csv: one row per image (uid, filename, projection)

Output: JSON identical to parse_iu_reports.py so downstream make_iu_dataset.py works unchanged.

Usage (on Kaggle notebook):
    !python scripts/parse_iu_kaggle.py \
        --reports_csv /kaggle/input/chest-xrays-indiana-university/indiana_reports.csv \
        --projections_csv /kaggle/input/chest-xrays-indiana-university/indiana_projections.csv \
        --output data_iu/parsed/reports.json
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict


NORMAL_KEYWORDS = [
    "no acute", "no evidence of", "no evidence for", "negative for",
    "unremarkable", "normal heart", "normal cardiac", "normal lung",
    "normal chest", "clear lungs", "lungs are clear",
    "no pneumonia", "no effusion", "no pneumothorax", "no infiltrate",
]


def _clean(s):
    if s is None or (isinstance(s, float) and s != s):  # NaN
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s


def label_binary(text, mesh_list):
    """0=normal, 1=abnormal. Same heuristic as parse_iu_reports.py."""
    text_lower = text.lower()
    has_normal_kw = any(kw in text_lower for kw in NORMAL_KEYWORDS)
    mesh_clean = [m for m in mesh_list if m.lower() not in ("normal", "no indexing", "")]
    return 0 if (has_normal_kw and len(mesh_clean) == 0) else 1


def main():
    import pandas as pd

    ap = argparse.ArgumentParser()
    ap.add_argument("--reports_csv", required=True,
                    help="indiana_reports.csv path")
    ap.add_argument("--projections_csv", required=True,
                    help="indiana_projections.csv path")
    ap.add_argument("--img_dir", default=None,
                    help="If set, drop image_ids whose PNG doesn't exist here")
    ap.add_argument("--output", default="data_iu/parsed/reports.json")
    args = ap.parse_args()

    if not os.path.exists(args.reports_csv):
        print(f"❌ {args.reports_csv} not found"); sys.exit(1)
    if not os.path.exists(args.projections_csv):
        print(f"❌ {args.projections_csv} not found"); sys.exit(1)

    print(f"📂 Loading {args.reports_csv}")
    reports_df = pd.read_csv(args.reports_csv)
    print(f"   Reports CSV columns: {list(reports_df.columns)}")
    print(f"   Reports CSV rows   : {len(reports_df)}")

    print(f"\n📂 Loading {args.projections_csv}")
    proj_df = pd.read_csv(args.projections_csv)
    print(f"   Projections CSV columns: {list(proj_df.columns)}")
    print(f"   Projections CSV rows   : {len(proj_df)}")

    # ----- Detect column names (raddar uses 'uid', some variants 'id') -----
    rid_col_r = next((c for c in ('uid', 'id', 'report_id') if c in reports_df.columns), None)
    rid_col_p = next((c for c in ('uid', 'id', 'report_id') if c in proj_df.columns), None)
    if not rid_col_r or not rid_col_p:
        print(f"❌ Cannot find report-id column. Expected one of: uid/id/report_id")
        sys.exit(1)
    fname_col = next((c for c in ('filename', 'image_name', 'file', 'image') if c in proj_df.columns), None)
    if not fname_col:
        print(f"❌ Cannot find image filename column in projections CSV")
        sys.exit(1)
    print(f"\n🔑 Report-id col (reports)     : {rid_col_r}")
    print(f"🔑 Report-id col (projections) : {rid_col_p}")
    print(f"🔑 Image filename col          : {fname_col}")

    # ----- Group images per report -----
    img_per_report = defaultdict(list)
    for _, row in proj_df.iterrows():
        rid = str(row[rid_col_p])
        fname = str(row[fname_col])
        # Strip extension. raddar files: '349_IM-1697-2001.dcm.png' → '349_IM-1697-2001.dcm'
        # (keep .dcm part because actual filename uses it; downstream make script
        # appends .png when copying/symlinking)
        img_id = re.sub(r'\.png$', '', fname, flags=re.IGNORECASE)
        img_id = re.sub(r'\.(jpg|jpeg)$', '', img_id, flags=re.IGNORECASE)
        img_per_report[rid].append(img_id)
    print(f"\n📊 Images grouped per report: {len(img_per_report)} reports have ≥1 image")

    # ----- Detect text/mesh columns in reports CSV -----
    findings_col = next((c for c in reports_df.columns if 'findings' in c.lower()), None)
    impression_col = next((c for c in reports_df.columns if 'impression' in c.lower()), None)
    mesh_col = next((c for c in reports_df.columns if c.lower() in ('mesh', 'problems', 'tags')), None)
    print(f"🔑 Findings col   : {findings_col}")
    print(f"🔑 Impression col : {impression_col}")
    print(f"🔑 MeSH col       : {mesh_col}")

    # ----- Build reports JSON -----
    reports_json = {}
    skipped_no_text = 0
    skipped_no_image = 0
    for _, row in reports_df.iterrows():
        rid = str(row[rid_col_r])

        findings = _clean(row.get(findings_col, "")) if findings_col else ""
        impression = _clean(row.get(impression_col, "")) if impression_col else ""
        text = (findings + " " + impression).strip()

        mesh_str = _clean(row.get(mesh_col, "")) if mesh_col else ""
        # MeSH commonly delimited by ';' or '/'
        mesh = [m.strip() for m in re.split(r'[;/]', mesh_str) if m.strip()]
        mesh = [m for m in mesh if m.lower() not in ("normal", "no indexing")]

        if not text:
            skipped_no_text += 1
            continue

        img_ids = img_per_report.get(rid, [])
        if args.img_dir and os.path.isdir(args.img_dir):
            img_ids = [i for i in img_ids if os.path.exists(os.path.join(args.img_dir, f"{i}.png"))]
        if not img_ids:
            skipped_no_image += 1
            continue

        # Use CXR<uid> as report_id to match Open-i naming convention
        report_id = f"CXR{rid}"
        reports_json[report_id] = {
            "text": text,
            "mesh": mesh,
            "image_ids": img_ids,
            "label_binary": label_binary(text, mesh),
        }

    print(f"\n📊 Parsed {len(reports_json)} valid reports")
    print(f"   Skipped: no_text={skipped_no_text}, no_image={skipped_no_image}")

    n_normal = sum(1 for r in reports_json.values() if r["label_binary"] == 0)
    n_abnormal = len(reports_json) - n_normal
    if reports_json:
        print(f"\n🏷  Label distribution:")
        print(f"   Normal   : {n_normal} ({100*n_normal/len(reports_json):.1f}%)")
        print(f"   Abnormal : {n_abnormal} ({100*n_abnormal/len(reports_json):.1f}%)")

    total_imgs = sum(len(r["image_ids"]) for r in reports_json.values())
    print(f"\n🖼  Total linked images: {total_imgs} ({total_imgs/max(1,len(reports_json)):.2f} per report)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reports_json, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {args.output}")


if __name__ == "__main__":
    main()
