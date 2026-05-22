"""
Setup script: Giải nén data & models từ Google Drive.
Thay thế cell giải nén trong notebook để xử lý đúng thư mục lồng nhau.

Cách dùng trong Colab:
    !python setup_data.py
"""
import os
import shutil
import glob
import subprocess

# --- CONFIG ---
DRIVE_ROOT_DEFAULT = "/content/drive/MyDrive/Forget-MI-Project"

ZIP_TARGETS = [
    # (zip_name, target_dir)
    ("data.zip",             "./data"),
    ("base_model.zip",       "./forgetme/training_original_model"),
    ("retrained_model.zip",  "./model_retrained_3per"),
]

VERIFY_PATHS = [
    ("./data/text_data",                                    "Text data"),
    ("./data/img_data",                                     "Image data"),
    ("./forgetme/training_original_model/pytorch_model.bin", "Base Model"),
    ("./model_retrained_3per/pytorch_model.bin",             "Retrained Model"),
]

# --- FUNCTIONS ---

def auto_find_drive_path():
    """Tìm thư mục chứa data.zip trên Google Drive."""
    print("🔍 Đang quét Drive để tìm file zip...")
    matches = glob.glob("/content/drive/MyDrive/**/data.zip", recursive=True)
    if matches:
        drive_root = os.path.dirname(matches[0])
        print(f"✅ Đã tìm thấy thư mục dự án tại: {drive_root}")
        return drive_root
    print(f"⚠️ Không tìm thấy data.zip, sử dụng mặc định: {DRIVE_ROOT_DEFAULT}")
    return DRIVE_ROOT_DEFAULT


def flatten_nested_dir(target_dir):
    """Gỡ thư mục lồng nhau sau khi giải nén.
    
    Xử lý 2 trường hợp:
    1. Thư mục con trùng tên: data/data/ -> data/
    2. Chỉ có 1 thư mục con duy nhất: model/xyz/ -> model/
    """
    target_name = os.path.basename(target_dir.rstrip('/'))
    nested_path = os.path.join(target_dir, target_name)
    
    # Trường hợp 1: Thư mục con trùng tên (ưu tiên kiểm tra trước)
    if os.path.isdir(nested_path):
        print(f"  🚚 Phát hiện lồng: {target_name}/{target_name}/ -> Đang gỡ...")
        for item in os.listdir(nested_path):
            src = os.path.join(nested_path, item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        if os.path.exists(nested_path):
            shutil.rmtree(nested_path)
        print(f"  ✅ Gỡ lồng {target_name}/ thành công!")
        return

    # Trường hợp 2: Chỉ có 1 thư mục con duy nhất (tên khác)
    subdirs = os.listdir(target_dir)
    if len(subdirs) == 1 and os.path.isdir(os.path.join(target_dir, subdirs[0])):
        inner_dir = os.path.join(target_dir, subdirs[0])
        print(f"  🚚 Phát hiện lồng đơn: {subdirs[0]}/ -> Đang gỡ...")
        for item in os.listdir(inner_dir):
            src = os.path.join(inner_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.exists(dst):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                else:
                    os.remove(dst)
            shutil.move(src, dst)
        if os.path.exists(inner_dir):
            shutil.rmtree(inner_dir)
        print(f"  ✅ Gỡ lồng {subdirs[0]}/ thành công!")


def robust_extract(zip_name, target_dir, drive_path):
    """Giải nén file zip và tự động gỡ thư mục lồng."""
    zip_full_path = os.path.join(drive_path, zip_name)
    if not os.path.exists(zip_full_path):
        search = glob.glob(f"/content/drive/MyDrive/**/{zip_name}", recursive=True)
        if search:
            zip_full_path = search[0]
        else:
            print(f"❌ KHÔNG THẤY file {zip_name} trên toàn bộ Drive!")
            return

    print(f"📦 Đang giải nén {zip_name} -> {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    subprocess.run(["unzip", "-q", "-o", zip_full_path, "-d", target_dir], check=True)
    
    # Gỡ lồng tự động
    flatten_nested_dir(target_dir)


def verify_paths():
    """Kiểm tra tất cả đường dẫn quan trọng."""
    print("\n--- Kiểm tra kết quả ---")
    all_ok = True
    for check_path, desc in VERIFY_PATHS:
        exists = os.path.exists(check_path)
        status = '✅' if exists else '❌'
        print(f"{status} {desc}: {check_path}")
        if not exists:
            all_ok = False
    return all_ok


# --- MAIN ---
if __name__ == "__main__":
    drive_path = auto_find_drive_path()
    
    print("\n--- Bắt đầu giải nén ---")
    for zip_name, target_dir in ZIP_TARGETS:
        robust_extract(zip_name, target_dir, drive_path)
    
    if verify_paths():
        print("\n🚀 Tất cả Dữ liệu & Model đã sẵn sàng!")
    else:
        print("\n⚠️ Một số file chưa đúng vị trí. Hãy kiểm tra lại cấu trúc file zip!")
