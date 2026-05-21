import os
import csv
import time
import re
import pandas as pd

text_data_dir = "./data/text_data" 
output_dir = "./data/metadata"
split_csv_path = "./data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"

# 1. ĐỌC FILE SPLIT ĐỂ LẤY MAPPING STUDY_ID -> SEVERITY
study_to_label = {}
print("Đang đọc file split để lấy mapping study_id -> severity...")
try:
    df = pd.read_csv(split_csv_path)
    # study_id, edeme_severity
    for _, row in df.iterrows():
        sid = str(int(row['study_id']))
        label = str(int(row['edeme_severity']))
        study_to_label[sid] = label
        study_to_label[f"s{sid}"] = label
    print(f"Đã nạp {len(study_to_label)} mapping study_id vào bộ nhớ.")
except Exception as e:
    print(f"Lỗi khi đọc file split: {e}")
    exit()

# 2. BẮT ĐẦU TÌM VÀ GỘP
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "all_data.tsv")

print("Đang gộp file văn bản thành all_data.tsv...")
start_time = time.time()

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    # Tiêu đề cột
    writer.writerow(["index", "label", "report_id", "text"])
    
    count = 0
    scanned = 0
    for root, dirs, files in os.walk(text_data_dir):
        for file in files:
            if file.endswith(".txt"):
                scanned += 1
                report_id = file.replace(".txt", "") # vd: s50414267 hoặc 50414267
                
                if report_id in study_to_label:
                    label = study_to_label[report_id]
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as tf:
                            text = tf.read().replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()
                        # Format: [index, label, report_id, text]
                        writer.writerow([count, label, report_id, text])
                        count += 1
                    except Exception as e:
                        pass
                
                if scanned % 20000 == 0:
                    print(f"Đã quét {scanned} file... (Khớp {count} báo cáo)")

end_time = time.time()
print(f"HOÀN THÀNH! Đã trích xuất {count} báo cáo trong {round(end_time - start_time, 2)} giây.")
print(f"File TSV đã được lưu tại: {output_file}")
