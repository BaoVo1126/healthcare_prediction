#  🏥 Healthcare Classification & AI Model Benchmarking

Dataset: https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset

---
# 📋 Table of Contents
---
1. Introduction
2. Project of Structure
3. Data Description, Metadata & Data quality check
4. Evaluate Metrics & Analyze 
5. Snippets



--- 
# 1.🤖 Introduction
Dự án nghiên cứu và xây dựng pipeline tự động tiền xử lý dữ liệu, huấn luyện và đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`) trên bộ dữ liệu y tế.

## 🚀 Tính năng cốt lõi (Core Features)
- **Automated Data Engineering**: Một pipeline `AutoPreprocessor` thông minh tự động hóa toàn bộ quy trình từ làm sạch dữ liệu, xử lý giá trị thiếu đến chuẩn hóa (Scaling) cho 10,000 hồ sơ.
- **Predictive Intelligence**: Triển khai kiến trúc Ensemble với **Random Forest** và **Decision Tree**, cho phép phân tích so sánh để đạt được độ tin cậy tối ưu trong chẩn đoán.
- **Interactive AI Dashboard**: Giao diện người dùng được xây dựng trên **Streamlit**, cung cấp khả năng dự đoán rủi ro theo thời gian thực dựa trên các chỉ số sinh học.
- **Advanced Visual Analytics**: Hệ thống hóa các báo cáo EDA chuyên sâu, trực quan hóa mối tương quan giữa 15+ thuộc tính y tế để tìm ra các biến quan trọng nhất.
- **Scalable Model Persistence**: Quy trình đóng gói mô hình (Serialization) chuyên nghiệp, đảm bảo hệ thống luôn sẵn sàng để triển khai và inference nhanh chóng.

---

## 📁 2. Cấu trúc thư mục (Project Structure)

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
## 📊 3. DATA DESCRIPTION, METADATA & DATA QUALITY CHECK

Dự án này sử dụng bộ dữ liệu gồm **10,000 hồ sơ bệnh án** với **15 thuộc tính gốc**. Qua phân tích, dữ liệu mang bản chất giả lập (Synthetic Data) với các chỉ số phân phối đồng đều.

### 📌 3.1. Từ điển dữ liệu & Định hướng xử lý (Data Metadata)

| Nhóm Dữ Liệu | Tên Biến (Features) | Kiểu Dữ Liệu | Hướng Xử Lý & Khai Thác |
| :--- | :--- | :--- | :--- |
| **Định danh** *(Loại bỏ)* | `Name`, `Doctor`, `Hospital`, `Room Number` | Object / Int | Mang tính hành chính ➔ **Loại bỏ khi huấn luyện** để bảo mật danh tính và tránh nhiễu. |
| **Phân loại** *(Categorical)* | `Gender`, `Blood Type`, `Insurance Provider`, `Admission Type`, `Medication` | Object (String) | Đặc trưng về thực thể ➔ **Mã hóa (Encoding)** về dạng số. |
| **Y khoa Cốt lõi** | `Medical Condition` | Object (String) | Chẩn đoán bệnh nền (6 loại bệnh) ➔ **Biến độc lập chính**. |
| **Thời gian** *(Datetime)* | `Date of Admission`, `Discharge Date` | Datetime | ➔ Biến đổi thành biến quan trọng: **`Length of Stay` (Số ngày nằm viện)**. |
| **Số học** *(Numerical)* | `Age`, `Billing Amount` | Int / Float | Các chỉ số định lượng ➔ **Chuẩn hóa scale** bằng `StandardScaler`. |
| **Mục tiêu** *(Target - Y)* | `Test Results` | Object (String) | Kết quả xét nghiệm (3 nhóm nhãn) ➔ **Nhãn đa lớp cần dự đoán**. |

---

### 🔍 3.2. Số liệu Phân phối & Insights Chi tiết từ EDA

Kết quả phân tích thống kê khám phá (EDA) chứng minh dữ liệu phân bổ theo cấu trúc đồng đều (Uniform Distribution):

* **Bằng chứng về Biến mục tiêu (`Test Results`):** Tỷ lệ phân bổ giữa 3 nhãn *Normal, Abnormal, và Inconclusive* cân bằng gần như tuyệt đối (**~33.3% cho mỗi nhãn**). Dữ liệu hoàn toàn không bị lệch nhãn (Imbalanced Data), không cần áp dụng kỹ thuật SMOTE khi train.
* **Bằng chứng về Nhân khẩu học & Bệnh lý:**
  * **Tuổi (`Age`):** Trải đều tuyến tính từ **18 đến 85 tuổi**, không tập trung vào riêng nhóm tuổi già hay trẻ.
  * **Giới tính (`Gender`):** Tỷ lệ Nam (Male) và Nữ (Female) tiệm cận mức **50:50**.
  * **Bệnh lý (`Medical Condition`):** Chia đều cơ hội xuất hiện cho **6 loại bệnh nền chính** bao gồm: *Cancer (Ung thư), Obesity (Béo phì), Diabetes (Tiểu đường), Asthma (Hen suyễn), Hypertension (Cao huyết áp), và Arthritis (Viêm khớp)*.
* **Bằng chứng về Tài chính & Vận hành:**
  * **Viện phí (`Billing Amount`):** Dao động ngẫu nhiên rất rộng từ **vài trăm USD cho tới tối đa xấp xỉ 50,000 USD**. Số tiền này phân bố đều, hoàn toàn độc lập và không phụ thuộc vào mức độ nặng nhẹ của bệnh nền.
  * **Thời gian nằm viện (`Length of Stay`):** Sau khi trích xuất (`Discharge Date` - `Date of Admission`), thời gian lưu trú tại bệnh viện biến thiên ngẫu nhiên từ **1 ngày đến 30 ngày**.

---

### 🛠️ 3.3. Kiểm tra "Sức khỏe" Dữ liệu & Biến động Ngoại lai (Outliers Check)

* **Dữ liệu thiếu (Missing Values):** **0%**. Toàn bộ 10,000 dòng đều đầy đủ ở tất cả các cột, không có giá trị `NaN` hoặc `Null`. Không cần dùng `SimpleImputer`.
* **Dữ liệu trùng lặp (Duplicate Rows):** **0%**. Không có dòng nào bị lặp lại hoàn toàn.
* **Bằng chứng toán học về Outliers:**
  * Xét trên cột số thực duy nhất là `Billing Amount`, vì dữ liệu phân bố đều (Uniform) từ mức thấp đến mức kịch trần ~50,000 USD nên khi vẽ biểu đồ Boxplot, chúng ta sẽ **không thấy xuất hiện các chấm điểm Outliers** nằm cô lập hẳn ra ngoài hàng ranh định biên toán học.
  * Tuy nhiên, vì khoảng giá trị của `Billing Amount` biến thiên quá lớn (hàng chục nghìn) so với cột `Age` (vài chục), nó tạo ra độ lệch phân phối rất lớn. Bước xử lý `StandardScaler` là **bắt buộc** để ép dữ liệu về phân phối chuẩn (Mean = 0, Std = 1) trước khi đưa vào mô hình tuyến tính.

---

### 🧪 3.4. Nghiên cứu Thực nghiệm Đối chứng: CASE 1 vs CASE 2

Để làm rõ tác động của nhiễu và các giá trị biên, dự án triển khai thử nghiệm so sánh trên 2 tập dữ liệu:
* **CASE 1:** Huấn luyện trên Dữ liệu gốc (Giữ nguyên toàn bộ Outliers toán học và độ nhiễu).
* **CASE 2:** Huấn luyện trên Dữ liệu sạch (Đã dùng phương pháp IQR / Capping để làm mịn cột `Billing Amount` và triệt tiêu các dòng nhiễu nặng phi logic).

#### Data Quality Benchmarking

| Thuật toán (Model) | CASE 1: Dữ liệu gốc (Test Accuracy) | CASE 2: Đã lọc Outliers (Test Accuracy) | Độ chênh lệch ($\Delta$ Accuracy) | Nhận định bản chất toán học từ Kỹ sư |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 34.2% | 34.8% | **+ 0.6%** | Tăng nhẹ vì đường biên tuyến tính không còn bị kéo lệch bởi các hóa đơn kịch trần (~50k USD). |
| **Decision Tree** | 35.1% | 35.0% | **- 0.1%** | Hầu như không đổi vì thuật toán Cây đơn lẻ vốn có tính bền vững (Robust) với Outliers rất tốt. |
| **Random Forest** | **38.4%** | **39.1%** | **+ 0.7%** | **Hiệu năng đạt đỉnh**. Việc làm mịn dữ liệu giúp tập hợp 100 cây quyết định bỏ phiếu đồng thuận và giảm nhiễu tốt hơn ở các nút lá. |
| **XGBoost** | 37.1% | 37.5% | **+ 0.4%** | Thuật toán Boosting sửa sai hiệu quả hơn khi không phải cố gắng tối ưu cho các điểm quá dị biệt. |

#### 💡 Kết luận rút ra từ bằng chứng thực nghiệm:
Số liệu đối chứng giữa hai trường hợp chỉ ra rằng **Accuracy chỉ tăng nhẹ (< 1%)** sau khi đã xử lý Outliers kỹ lưỡng. Điều này chứng minh một định luật tối cao của Machine Learning: **"Garbage in, Garbage out"**. Khi bản chất dữ liệu thô gốc đã bị gán nhãn ngẫu nhiên và thiếu các đặc trưng y khoa lâm sàng cốt lõi (như BMI, huyết áp, đường huyết), thuật toán toán học không thể cứu vãn được mô hình vượt qua ngưỡng trần **~39%**. Để nâng hiệu năng lên >80%, giải pháp duy nhất là thay đổi quy trình thu thập dữ liệu thô ở đầu vào.


---
## 📊 4. Kết quả thực nghiệm & Phân tích chuyên sâu (Evaluation & Analysis)

### 4.1 Final Model Selection Benchmarking

| Model | Train Accuracy | Test Accuracy | F1 Macro | CV Mean ± Std | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 45.2% | **38.4%** | 0.3810 | 0.3750 ± 0.012 | 2.5s |
| **XGBoost** | 52.1% | 37.1% | 0.3690 | 0.3620 ± 0.015 | 4.1s |
| **Logistic Regression**| 34.5% | 34.2% | 0.3310 | 0.3390 ± 0.005 | 0.8s |

> 🏆 **Best Model**: `Random Forest` được lựa chọn làm mô hình tối ưu cho sản phẩm nhờ độ ổn định cao và khả năng kiểm soát nhiễu tốt nhất.

---
### 4.2 Phân tích bài toán thực tế (Critical Technical Insights)
* **Vấn đề Underfitting & Pure Noise**: Mặc dù áp dụng các mô hình Boosting mạnh mẽ như `XGBoost` hay `LightGBM`, độ chính xác (*Accuracy*) chỉ dao động quanh mức **35% - 39%** (chỉ nhỉnh hơn mức đoán ngẫu nhiên 3 nhóm ~33.3% một chút).
* **Nguyên nhân**: Qua phân tích thống kê chuyên sâu (EDA), bộ dữ liệu này là dạng dữ liệu giả lập ngẫu nhiên (*Synthetic Data*). Các đặc trưng sẵn có như `Age`, `Blood Type`, `Admission Type` hoàn toàn độc lập và không có mối tương quan y khoa tuyến tính hay phi tuyến đối với nhãn mục tiêu `Test Results`.
* **Giải pháp thực tế**: Dự án **không** sử dụng các kỹ thuật tự sinh số liệu giả tạo (*Conditional Feature Generation*) để ép điểm Accuracy lên cao, nhằm đảm bảo tính toàn vẹn dữ liệu (*Data Integrity*). Thay vào đó, dự án tập trung vào việc tối ưu hóa kiến trúc Pipeline code sạch và đưa ra đề xuất thu thập dữ liệu.

---

# 💻 5. CODE SNIPPETS & USAGE GUIDE

Dự án được thiết kế theo kiến trúc hướng đối tượng (OOP), cho phép dễ dàng tái sử dụng toàn bộ pipeline tiền xử lý và huấn luyện chỉ với vài dòng code ngắn.

### 🛠️ 5.1. Sử dụng Pipeline Tiền xử lý dữ liệu tự động

Đoạn mã này minh họa cách gọi class `AutoPreprocessor` để tự động làm sạch dữ liệu thô, loại bỏ cột nhiễu hành chính, xử lý viện phí âm và chuẩn hóa dữ liệu theo tỷ lệ 8:1:1:

```python
from src.preprocessor import AutoPreprocessor
import pandas as pd

# 1. Nạp bộ dữ liệu thô
df = pd.read_csv('data/healthcare_dataset.csv')

# 2. Khởi tạo Pipeline (Tự động loại bỏ các biến định danh hành chính)
prep = AutoPreprocessor(
    target_col='Test Results', 
    drop_cols=['Name', 'Doctor', 'Hospital', 'Room Number']
)

# 3. Thực thi tiền xử lý và chia tách tập dữ liệu (Train/Val/Test)
X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)

print(f"Kích thước tập huấn luyện (Train Set): {X_train.shape}")
```
---
## 🛠️ 6. Hướng dẫn cài đặt & Chạy dự án (Installation & Usage)

### 1. Cài đặt môi trường
Khuyến khích tạo môi trường ảo (Virtual Environment) và cài đặt các thư viện phụ thuộc:
```bash
git clone [https://github.com/](https://github.com/)[TÊN_GITHUB_CỦA_BẠN]/healthcare-analysis-ai.git
cd healthcare-analysis-ai
pip install -r requirements.txt
```
---
