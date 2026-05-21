# Forget-MI Improvement: Low-Rank Knowledge Unlearning (LoKU)

## 1. Giới thiệu (Overview)
Dự án này là một cải tiến cho framework **Forget-MI**, tập trung vào việc thực hiện **Machine Unlearning** (Máy quên) cho dữ liệu y tế đa phương thức ( Multimodal: Ảnh X-quang và Báo cáo lâm sàng). Mục tiêu chính là xóa bỏ thông tin của một nhóm bệnh nhân cụ thể khỏi mô hình đã huấn luyện mà không cần phải thực hiện quá trình Re-training (huấn luyện lại từ đầu) tốn kém, đồng thời vẫn giữ vững hiệu suất trên phần dữ liệu còn lại.

Phương pháp được đề xuất và tích hợp là **LoKU (Low-Rank Knowledge Unlearning)**.

---

## 2. Phương pháp LoKU (Low-Rank Knowledge Unlearning)

### Cách thức hoạt động:
LoKU thay thế việc tinh chỉnh toàn bộ mô hình (Full Fine-tuning) bằng cách tác động vào các cấu trúc hạng thấp (Low-rank) của trọng số mô hình. Quy trình gồm 3 bước chính:

1.  **Tính toán tầm quan trọng (Importance Scoring):**
    *   Sử dụng dữ liệu cần quên (Forget set) và dữ liệu cần giữ lại (Retain set) để tính toán ma trận thông tin Fisher hoặc độ lớn gradient.
    *   Mục tiêu: Xác định những tham số nào của mô hình chứa nhiều tri thức về bệnh nhân cần xóa nhất.

2.  **Phân tách giá trị suy biến (SVD):**
    *   Áp dụng thuật toán SVD lên các ma trận trọng số quan trọng (thường là Query, Value trong BERT hoặc các lớp FC).
    *   Việc này giúp cô lập "tri thức nhạy cảm" vào các thành phần toán học cụ thể.

3.  **Khấu trừ tri thức (Knowledge Subtraction):**
    *   Sử dụng kỹ thuật LoRA (Low-Rank Adaptation) để trừ đi các thành phần chứa thông tin cần quên khỏi trọng số gốc của mô hình.
    *   Kết quả là mô hình "quên" đi các mẫu dữ liệu cụ thể nhưng không làm hỏng cấu trúc tổng quát của mạng thần kinh.

---

## 3. Cấu trúc Hệ thống & Workflow Hybrid

Dự án sử dụng mô hình **Hybrid Workflow (Local - Cloud - Storage)** để tối ưu hóa tài nguyên:

*   **Local (VS Code):** Nơi soạn thảo mã nguồn, quản lý logic thuật toán và cấu hình (`config.yaml`).
*   **GitHub:** Trung tâm đồng bộ mã nguồn. Giúp đẩy code từ máy cá nhân lên Cloud nhanh chóng.
*   **Google Drive:** Kho lưu trữ "hạng nặng" (Big Data). Chứa các file ảnh y tế (MIMIC-CXR), dữ liệu thô và các trọng số mô hình pre-trained (Checkpoint).
*   **Google Colab:** Cung cấp tài nguyên tính toán (GPU T4/A100). Colab sẽ kéo code từ GitHub, giải nén dữ liệu từ Drive để thực hiện huấn luyện.

---

## 4. Các cải tiến kỹ thuật đã thực hiện

Để LoKU chạy mượt mà trên môi trường này, chúng tôi đã thực hiện các nâng cấp sau:

1.  **Chuẩn hóa định dạng ID bệnh nhân:** Sửa lỗi lệch pha giữa mã số trong file CSV và mã hiệu có tiền tố 's' trong file văn bản, giúp hệ thống không bị crash khi nạp dữ liệu.
2.  **Script tiền xử lý dữ liệu tự động (`make_tsv.py`):** Viết lại script để tự động gộp hàng nghìn file báo cáo lâm sàng thành một file duy nhất (`all_data.tsv`), đồng thời gán nhãn bệnh lý chính xác.
3.  **Cơ chế giải nén Robust:** Tích hợp hàm giải nén thông minh trong Notebook, có khả năng tự tìm file `pytorch_model.bin` ở bất cứ đâu trong file nén để đưa về đúng thư mục yêu cầu.
4.  **Hệ thống Symlink (Liên kết bền vững):** Kết nối thư mục kết quả của Colab trực tiếp với Google Drive. Khi mô hình đang chạy, các checkpoint sẽ được ghi thẳng vào Drive, giúp tránh mất dữ liệu khi mất kết nối mạng.

---

## 5. Hướng dẫn chạy thực nghiệm (Experimental Procedure)

Dành cho người mới, bạn chỉ cần thực hiện theo các bước sau:

1.  **Chuẩn bị Dữ liệu:** 
    *   Nén thư mục ảnh và metadata thành `data.zip`.
    *   Nén mô hình gốc thành `base_model.zip`.
    *   Upload vào thư mục `Forget-MI-Project` trên Google Drive.
2.  **Đồng bộ Code:** Đẩy toàn bộ thư mục dự án lên GitHub của bạn.
3.  **Thực thi trên Colab:**
    *   Mở file `run.ipynb` trên Colab.
    *   Kết nối với Google Drive.
    *   Chạy các Cell tuần tự để tự động Clone code, Giải nén và Tiền xử lý.
4.  **Huấn luyện:** Chạy lệnh cuối cùng để kích hoạt thuật toán LoKU. Kết quả sẽ được log qua **WandB** (Weights & Biases) để theo dõi biểu đồ hội tụ.

---
*Tài liệu này được soạn thảo để tóm tắt các đóng góp và quy trình vận hành dự án Forget-MI LoKU.*
