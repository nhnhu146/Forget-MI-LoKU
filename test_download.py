import requests
import pandas as pd

USERNAME = "hoangnhu1406"
PASSWORD = "Nhu@1406"
CSV_FILE_PATH = "data_splits/mimic-cxr-sub-img-edema-split-manualtest.csv"

def authenticate_session(username, password):
    session = requests.Session()
    login_url = "https://physionet.org/login/"
    
    try:
        print("1. Đang lấy token bảo mật (CSRF) từ trang chủ...")
        response = session.get(login_url)
        response.raise_for_status()
        
        csrftoken = session.cookies.get('csrftoken')
        
        login_data = {
            'username': username,
            'password': password,
            'csrfmiddlewaretoken': csrftoken,
            'next': '/files/mimic-cxr-jpg/2.0.0/'
        }
        
        headers = {
            'Referer': login_url,
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        print("2. Đang gửi thông tin đăng nhập...")
        post_response = session.post(login_url, data=login_data, headers=headers)
        post_response.raise_for_status()
        
        if "Sign in to your account" in post_response.text:
             print("❌ Đăng nhập thất bại. Vui lòng kiểm tra lại Username/Password.")
             return None
        
        print("✅ Đăng nhập (giả lập trình duyệt) thành công!")
        return session
        
    except Exception as e:
        print(f"❌ Lỗi khi đăng nhập: {e}")
        return None

def test_single_download():
    # Đọc dòng đầu tiên của file CSV
    df = pd.read_csv(CSV_FILE_PATH)
    row = df.iloc[0]
    
    subject_id_str = str(row['subject_id'])
    study_id_str = str(row['study_id'])
    dicom_id = row['dicom_id']
    
    folder_p1 = f"p{subject_id_str[:2]}"
    folder_p2 = f"p{subject_id_str}"
    folder_s = f"s{study_id_str}"
    filename = f"{dicom_id}.jpg"
    
    url = f"https://physionet.org/files/mimic-cxr-jpg/2.0.0/files/{folder_p1}/{folder_p2}/{folder_s}/{filename}"
    print(f"\nĐang thử tải URL: {url}")
    
    # 1. Gọi hàm xác thực
    session = authenticate_session(USERNAME, PASSWORD)
    if not session:
        return
        
    # 2. Dùng session đã xác thực để tải file
    print("3. Đang gửi yêu cầu tải ảnh...")
    response = session.get(url, stream=True)
    
    print(f"\n--- KẾT QUẢ ---")
    print(f"MÃ TRẠNG THÁI (Status Code): {response.status_code}")
    
    if response.status_code == 200:
        print("🎉 THÀNH CÔNG RỰC RỠ! Cơ chế giả lập trình duyệt đã đánh lừa được tường lửa PhysioNet.")
        print("👉 Bây giờ bạn có thể tự tin copy đoạn code download_images.py đầy đủ ở tin nhắn trước để chạy tải hàng loạt.")
    elif response.status_code == 403:
        print("❌ Vẫn bị lỗi 403 (Forbidden).")
        print("Lý do: PhysioNet đang chặn quyền tải qua API của tài khoản này, dù trên web vẫn cho tải. Bạn có thể sẽ phải dùng công cụ wget truyền thống theo hướng dẫn trên web.")
    elif response.status_code == 404:
         print("❌ Lỗi 404: Không tìm thấy file (Đường dẫn bị sai).")
    else:
        print(f"❌ Lỗi khác: {response.reason}")

if __name__ == "__main__":
    test_single_download()