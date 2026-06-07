#  🏥 Healthcare Classification & AI Model Benchmarking

Dataset: https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset

Dự án nghiên cứu và xây dựng pipeline tự động tiền xử lý dữ liệu, huấn luyện và đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`) trên bộ dữ liệu y tế.

## 🚀 Tính năng cốt lõi (Core Features)
* **AutoPreprocessor**: Pipeline hướng đối tượng (OOP) tự động xử lý dữ liệu thô: lọc nhiễu, xử lý giá trị âm, trích xuất đặc trưng thời gian (`Length of Stay`, `Cost_Per_Day`), mã hóa nhãn và chuẩn hóa dữ liệu theo tỷ lệ 8:1:1 (Train/Val/Test) nhằm chống rò rỉ dữ liệu (*Data Leakage*).
* **ModelTrainer**: Hệ thống tự động huấn luyện, đánh giá chéo (*Stratified 5-Fold Cross-Validation*) và áp dụng kỹ thuật *Bootstrap Learning Curves* để đo lường độ ổn định (Variance/Bias) của từng mô hình.

---

## 📁 Cấu trúc thư mục (Project Structure)

```text
healthcare-analysis-ai/
│
├── data/                       # Chứa dữ liệu (Nếu file quá nặng thì để file raw ở đây)
│   └── healthcare_dataset.csv
│
├── src/                        # Chứa các file mã nguồn Python (.py) tái sử dụng
│   ├── __init__.py
│   ├── preprocessor.py         # Chứa class AutoPreprocessor của bạn
│   └── model_trainer.py        # Chứa class ModelTrainer của bạn
│
├── models/                     # Chứa các mô hình đã đóng băng (Artifacts)
│   ├── best_model_random_forest.pkl
│   └── decision_tree.pkl
|   └── logistic_regression.pkl
|   └── gradient_boosting.pkl
│
├── notebooks/                  # Chứa các file Jupyter Notebook thử nghiệm
│   ├── 01_eda.ipynb            # File phân tích dữ liệu ban đầu
│   └── 02_training.ipynb       # File gọi class để train và vẽ curve
│
└── app.py                      # File chạy giao diện Streamlit (Nếu có deploy)

```

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
```


# I. DATA DESCRIPTION:
## 1. DATA QUANTITIES & METADATA:
- Tổng số lượng mẫu: 10,000 dòng dữ liệu đại diện cho 10,000 lượt bệnh nhân nhập viện.

- Số lượng thuộc tính (Features): Có tổng cộng 15 cột thuộc tính gốc.

Phân loại kiểu biến dữ liệu:

- Biến định danh (Identifiers): Name, Doctor, Hospital, Room Number (Các biến này mang tính định danh cá nhân hoặc cơ sở, không đóng góp giá trị toán học trực tiếp cho mô hình dự đoán và cần được loại bỏ).

- Biến số học (Numerical): Age (Số nguyên từ 18 đến 85) và Billing Amount (Số thực đại diện cho số tiền viện phí).

- Biến phân loại (Categorical): Gender, Blood Type, Medical Condition, Insurance Provider, Admission Type, Medication.

- Biến thời gian (Datetime): Date of Admission và Discharge Date.

- Biến mục tiêu (Target - Y): Test Results (Gồm 3 nhóm nhãn phân loại đa lớp: Normal, Abnormal, Inconclusive).

## 2. Feature Distributions & Insights:
- Insight về Biến mục tiêu (Test Results):
Tỷ lệ phân bổ giữa 3 nhãn Normal, Abnormal, và Inconclusive gần như là cân bằng tuyệt đối (~33.3% cho mỗi nhãn). Dữ liệu hoàn toàn không bị hiện tượng lệch nhãn (Imbalanced Data), điều này giúp quá trình huấn luyện không cần áp dụng SMOTE hay Class Weights.

- Insight về Nhân khẩu học & Bệnh lý:

Độ tuổi (Age): Trải đều tuyến tính từ 18 đến 85 tuổi. Không có sự tập trung đặc biệt vào nhóm tuổi già hay tuổi trẻ.

Giới tính (Gender): Tỷ lệ Nam (Male) và Nữ (Female) tiệm cận mức 50:50.

Bệnh lý (Medical Condition): Chia đều cho 6 loại bệnh nền chính gồm Cancer (Ung thư), Obesity (Béo phì), Diabetes (Tiểu đường), Asthma (Hen suyễn), Hypertension (Cao huyết áp), và Arthritis (Viêm khớp).

- Insight về Tài chính & Vận hành:

Viện phí (Billing Amount): Dao động ngẫu nhiên rất rộng từ khoảng vài trăm USD cho tới tối đa xấp xỉ 50,000 USD. Số tiền này phân bố đều và hoàn toàn không phụ thuộc vào mức độ nghiêm trọng của bệnh lý hay loại phòng bệnh.

Thời gian nằm viện (Length of Stay - Kỹ nghệ đặc trưng): Khi lấy Discharge Date trừ đi Date of Admission, thời gian lưu trú tại bệnh viện dao động ngẫu nhiên từ 1 ngày cho đến khoảng 30 ngày.
