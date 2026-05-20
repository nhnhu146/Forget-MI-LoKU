import os
import csv
import time
import re

text_data_dir = "./data/text_data" 
output_dir = "./data/metadata"
split_csv_path = "./data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"

# 1. ĐỌC BẤT CHẤP MỌI ĐỊNH DẠNG CSV ĐỂ LỌC ID
needed_reports = set()
print("Đang quét file danh sách để lấy mã bệnh nhân...")
try:
    with open(split_csv_path, 'r', encoding='utf-8') as f:
        content = f.read()
        # Tìm tất cả các dãy 8 chữ số (đặc trưng của ID báo cáo)
        matches = re.findall(r'\d{8}', content)
        for m in matches:
            needed_reports.add(m)          # vd: 50414267
            needed_reports.add(f"s{m}")    # vd: s50414267
    print(f"Đã nạp {len(needed_reports)} mã ID vào bộ nhớ bảo vệ.")
except Exception as e:
    print(f"Lỗi khi đọc file split: {e}")
    exit()

# 2. BẮT ĐẦU TÌM VÀ GỘP
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "all_data.tsv")

print("Đang tìm file... Lần này chắc chắn sẽ bắt được!")
start_time = time.time()

with open(output_file, "w", encoding="utf-8", newline="") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow([report_id, "0", report_id, text])
    
    count = 0
    scanned = 0
    for root, dirs, files in os.walk(text_data_dir):
        for file in files:
            if file.endswith(".txt"):
                scanned += 1
                report_id = file.replace(".txt", "") 
                
                # Bắt ID chính xác
                if report_id in needed_reports:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as tf:
                            text = tf.read().replace("\n", " ").replace("\r", " ").strip()
                        writer.writerow([report_id, text])
                        count += 1
                    except Exception as e:
                        pass
                
                if scanned % 50000 == 0:
                    print(f"Đã lướt qua {scanned} file... (Tìm thấy {count} file khớp)")

end_time = time.time()
print(f"HOÀN THÀNH XUẤT SẮC! Đã trích xuất ĐÚNG {count} báo cáo trong {round(end_time - start_time, 2)} giây.")
print(f"File TSV đã được lưu tại: {output_file}")