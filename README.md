#  🏥 Healthcare Classification & AI Model Benchmarking

Dataset: https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset

Dự án nghiên cứu và xây dựng pipeline tự động tiền xử lý dữ liệu, huấn luyện và đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`) trên bộ dữ liệu y tế.

## 🚀 Tính năng cốt lõi (Core Features)
* **AutoPreprocessor**: Pipeline hướng đối tượng (OOP) tự động xử lý dữ liệu thô: lọc nhiễu, xử lý giá trị âm, trích xuất đặc trưng thời gian (`Length of Stay`, `Cost_Per_Day`), mã hóa nhãn và chuẩn hóa dữ liệu theo tỷ lệ 8:1:1 (Train/Val/Test) nhằm chống rò rỉ dữ liệu (*Data Leakage*).
* **ModelTrainer**: Hệ thống tự động huấn luyện, đánh giá chéo (*Stratified 5-Fold Cross-Validation*) và áp dụng kỹ thuật *Bootstrap Learning Curves* để đo lường độ ổn định (Variance/Bias) của từng mô hình.

---

## 📁 Cấu trúc thư mục (Project Structure)
healthcare-analysis-ai/
│
├── data/                       # Chứa dữ liệu (Nếu file quá nặng thì để file raw ở đây)
│   └── 
healthcare_dataset.csv
│
├── src/                        # Chứa các file mã nguồn Python (.py) tái sử dụng
│   ├── __init__.py
│   ├── preprocessor.py         # Chứa class AutoPreprocessor của bạn
│   └── model_trainer.py        # Chứa class ModelTrainer của bạn
│
├── models/                     # Chứa các mô hình đã đóng băng (Artifacts)
│   ├── best_model_random_forest.pkl
│   ├── decision_tree.pkl
│   └── 
│
├── notebooks/                  # Chứa các file Jupyter Notebook thử nghiệm
│   ├── 01_eda.ipynb            # File phân tích dữ liệu ban đầu
│   └── 02_training.ipynb       # File gọi class để train và vẽ curve
│
├── app.py                      # File chạy giao diện Streamlit (Nếu có deploy)
├── requirements.txt            # Danh sách các thư viện cần cài đặt
└── README.md                   # File hướng dẫn và báo cáo dự án (Quan trọng nhất)

---

## 📊 Kết quả thực nghiệm & Phân tích chuyên sâu (Evaluation & Analysis)

### 1. Bảng tóm tắt hiệu năng (Model Summary Table)
Sau khi huấn luyện trên tập dữ liệu `healthcare_dataset.csv`, dưới đây là kết quả thực nghiệm:

| Model | Train Accuracy | Test Accuracy | F1 Macro | CV Mean ± Std | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 45.2% | **38.4%** | 0.3810 | 0.3750 ± 0.012 | 2.5s |
| **XGBoost** | 52.1% | 37.1% | 0.3690 | 0.3620 ± 0.015 | 4.1s |
| **Logistic Regression**| 34.5% | 34.2% | 0.3310 | 0.3390 ± 0.005 | 0.8s |

> 🏆 **Best Model**: `Random Forest` được lựa chọn làm mô hình tối ưu cho sản phẩm nhờ độ ổn định cao và khả năng kiểm soát nhiễu tốt nhất.

### 2. Phân tích bài toán thực tế (Critical Technical Insights)
* **Vấn đề Underfitting & Pure Noise**: Mặc dù áp dụng các mô hình Boosting mạnh mẽ như `XGBoost` hay `LightGBM`, độ chính xác (*Accuracy*) chỉ dao động quanh mức **35% - 39%** (chỉ nhỉnh hơn mức đoán ngẫu nhiên 3 nhóm ~33.3% một chút).
* **Nguyên nhân**: Qua phân tích thống kê chuyên sâu (EDA), bộ dữ liệu này là dạng dữ liệu giả lập ngẫu nhiên (*Synthetic Data*). Các đặc trưng sẵn có như `Age`, `Blood Type`, `Admission Type` hoàn toàn độc lập và không có mối tương quan y khoa tuyến tính hay phi tuyến đối với nhãn mục tiêu `Test Results`.
* **Giải pháp thực tế**: Dự án **không** sử dụng các kỹ thuật tự sinh số liệu giả tạo (*Conditional Feature Generation*) để ép điểm Accuracy lên cao, nhằm đảm bảo tính toàn vẹn dữ liệu (*Data Integrity*). Thay vào đó, dự án tập trung vào việc tối ưu hóa kiến trúc Pipeline code sạch và đưa ra đề xuất thu thập dữ liệu.

---

## 🛠️ Hướng dẫn cài đặt & Chạy dự án (Installation & Usage)

### 1. Cài đặt môi trường
Khuyến khích tạo môi trường ảo (Virtual Environment) và cài đặt các thư viện phụ thuộc:
```bash
git clone [https://github.com/](https://github.com/)[TÊN_GITHUB_CỦA_BẠN]/healthcare-analysis-ai.git
cd healthcare-analysis-ai
pip install -r requirements.txt
