"""
Tiền xử lý dữ liệu: Gộp các file báo cáo lâm sàng (.txt) thành file all_data.tsv.
Cách dùng: python make_tsv.py
"""
import os
import csv
import time
import pandas as pd

# --- CONFIG ---
OUTPUT_DIR = "./data/metadata"
SPLIT_CSV = "./data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"

# Auto-detect text_data directory (xử lý cả trường hợp thư mục lồng)
CANDIDATE_DIRS = ["./data/text_data", "./data/data/text_data"]
TEXT_DATA_DIR = next((d for d in CANDIDATE_DIRS if os.path.isdir(d)), None)

if TEXT_DATA_DIR is None:
    print("❌ Không tìm thấy thư mục text_data! Hãy chạy setup_data.py trước.")
    exit(1)
print(f"📂 Sử dụng text_data tại: {TEXT_DATA_DIR}")


# --- 1. ĐỌC MAPPING STUDY_ID -> SEVERITY ---
study_to_label = {}
print("Đang đọc file split để lấy mapping study_id -> severity...")
try:
    df = pd.read_csv(SPLIT_CSV)
    for _, row in df.iterrows():
        sid = str(int(row['study_id']))
        label = str(int(row['edeme_severity']))
        study_to_label[sid] = label
        study_to_label[f"s{sid}"] = label
    print(f"Đã nạp {len(study_to_label)} mapping study_id vào bộ nhớ.")
except Exception as e:
    print(f"❌ Lỗi khi đọc file split: {e}")
    exit(1)


# --- 2. GỘP FILE TEXT THÀNH TSV ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
output_file = os.path.join(OUTPUT_DIR, "all_data.tsv")

print("Đang gộp file văn bản thành all_data.tsv...")
start_time = time.time()

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["index", "label", "report_id", "text"])
    
    count = 0
    scanned = 0
    for root, dirs, files in os.walk(TEXT_DATA_DIR):
        for file in files:
            if not file.endswith(".txt"):
                continue
            scanned += 1
            # Chuẩn hóa ID: bỏ đuôi .txt và bỏ chữ 's' để khớp với CSV
            report_id = file.replace(".txt", "").replace("s", "")
            
            if report_id in study_to_label:
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as tf:
                        text = tf.read().replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                    writer.writerow([count, study_to_label[report_id], report_id, text])
                    count += 1
                except Exception:
                    pass
            
            if scanned % 20000 == 0:
                print(f"  Đã quét {scanned} file... (Khớp {count} báo cáo)")

elapsed = round(time.time() - start_time, 2)
print(f"✅ HOÀN THÀNH! Đã trích xuất {count} báo cáo trong {elapsed} giây.")
print(f"   File TSV: {output_file}")
