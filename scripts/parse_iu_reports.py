#!/usr/bin/env python3
"""
Parse Indiana University CXR (Open-i) XML reports → per-report text + labels.

Input:  data_iu/raw/ecgen-radiology/*.xml  (NLMCXR_reports.tgz extracted)
        data_iu/raw/NLMCXR_png/*.png        (NLMCXR_png.tgz extracted)

Output: data_iu/parsed/reports.json
        {
          "CXR1": {
            "text": "...",
            "mesh": ["Atelectasis", ...],
            "image_ids": ["CXR1_1_IM-0001-3001", ...],
            "label_binary": 0 | 1,        # 0=normal, 1=abnormal
            "label_multiclass": 0..K-1     # optional, top-K MeSH
          },
          ...
        }

Usage:
    python scripts/parse_iu_reports.py \
        --xml_dir data_iu/raw/ecgen-radiology \
        --img_dir data_iu/raw/NLMCXR_png \
        --output  data_iu/parsed/reports.json

Strategy (per THESIS_ROADMAP Section 13):
- Binary label: 0 (normal) if report contains normality keywords AND has empty/null MeSH;
                1 (abnormal) otherwise
- Discard reports with no text, no images, or unparseable XML
"""
import argparse
import glob
import json
import os
import re
import sys
import xml.etree.ElementTree as ET


NORMAL_KEYWORDS = [
    "no acute",
    "no evidence of",
    "no evidence for",
    "negative for",
    "unremarkable",
    "normal heart",
    "normal cardiac",
    "normal lung",
    "normal chest",
    "clear lungs",
    "lungs are clear",
    "no pneumonia",
    "no effusion",
    "no pneumothorax",
    "no infiltrate",
]


def _clean_text(s):
    if s is None:
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_xml(path):
    """Return dict for one XML, or None if unparseable."""
    try:
        tree = ET.parse(path)
        root = tree.getroot()
    except Exception as e:
        return None

    # Report id from filename
    report_id = os.path.splitext(os.path.basename(path))[0]

    # FINDINGS + IMPRESSION
    findings = _clean_text(
        next((node.text for node in root.findall(".//AbstractText[@Label='FINDINGS']") if node.text), "")
    )
    impression = _clean_text(
        next((node.text for node in root.findall(".//AbstractText[@Label='IMPRESSION']") if node.text), "")
    )
    text = (findings + " " + impression).strip()
    if not text:
        return None

    # MeSH major terms
    mesh = []
    for m in root.findall(".//MeSH/major"):
        if m.text:
            mesh.append(_clean_text(m.text))
    # Drop "normal" MeSH (it's just a placeholder)
    mesh = [m for m in mesh if m.lower() not in ("normal", "no indexing")]

    # Linked images (parentImage id attr)
    image_ids = []
    for p in root.findall(".//parentImage"):
        img_id = p.get("id")
        if img_id:
            image_ids.append(img_id)

    return {
        "text": text,
        "mesh": mesh,
        "image_ids": image_ids,
    }


def label_binary(record):
    """0=normal, 1=abnormal.

    Normal iff:
      - has at least one normality keyword in text, AND
      - MeSH is empty (no flagged abnormality)
    """
    text_lower = record["text"].lower()
    has_normal_kw = any(kw in text_lower for kw in NORMAL_KEYWORDS)
    mesh_empty = len(record["mesh"]) == 0
    return 0 if (has_normal_kw and mesh_empty) else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml_dir", default="data_iu/raw/ecgen-radiology",
                    help="Directory of NLMCXR_reports XML files")
    ap.add_argument("--img_dir", default="data_iu/raw/NLMCXR_png",
                    help="Directory of NLMCXR PNG images (used to filter reports without images)")
    ap.add_argument("--output", default="data_iu/parsed/reports.json")
    ap.add_argument("--verify_images", action="store_true",
                    help="Drop image_ids whose PNG file doesn't exist (slower)")
    args = ap.parse_args()

    xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
    if not xml_files:
        print(f"❌ No XML files in {args.xml_dir}")
        sys.exit(1)
    print(f"📂 Found {len(xml_files)} XML reports in {args.xml_dir}")

    reports = {}
    skipped_no_text = 0
    skipped_no_image = 0
    skipped_parse = 0
    for xml_path in xml_files:
        rec = parse_xml(xml_path)
        if rec is None:
            skipped_parse += 1
            continue
        if not rec["image_ids"]:
            skipped_no_image += 1
            continue
        if not rec["text"]:
            skipped_no_text += 1
            continue

        # Optional: verify images on disk
        if args.verify_images:
            valid = [
                img_id for img_id in rec["image_ids"]
                if os.path.exists(os.path.join(args.img_dir, f"{img_id}.png"))
            ]
            if not valid:
                skipped_no_image += 1
                continue
            rec["image_ids"] = valid

        report_id = os.path.splitext(os.path.basename(xml_path))[0]
        rec["label_binary"] = label_binary(rec)
        reports[report_id] = rec

    print(f"\n📊 Parsed {len(reports)} valid reports")
    print(f"   Skipped: parse_err={skipped_parse}, no_text={skipped_no_text}, no_image={skipped_no_image}")

    # Label distribution
    n_normal = sum(1 for r in reports.values() if r["label_binary"] == 0)
    n_abnormal = len(reports) - n_normal
    print(f"\n🏷  Label distribution (binary):")
    print(f"   Normal   : {n_normal} ({100*n_normal/len(reports):.1f}%)")
    print(f"   Abnormal : {n_abnormal} ({100*n_abnormal/len(reports):.1f}%)")

    # Top MeSH (informational)
    from collections import Counter
    mesh_counter = Counter()
    for r in reports.values():
        mesh_counter.update(r["mesh"])
    print(f"\n🏷  Top-10 MeSH labels:")
    for term, count in mesh_counter.most_common(10):
        print(f"   {term:<35} {count}")

    # Image stats
    total_images = sum(len(r["image_ids"]) for r in reports.values())
    print(f"\n🖼  Total linked images: {total_images} ({total_images/len(reports):.2f} per report)")

    # Save JSON
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(reports, f, indent=2, ensure_ascii=False)
    print(f"\n💾 Saved: {args.output}")


if __name__ == "__main__":
    main()
