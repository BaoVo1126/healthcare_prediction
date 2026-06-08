# 🏥 Healthcare Classification & AI Model Benchmarking
Dataset gốc: [Healthcare Dataset (Kaggle)](https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset)

---

## 📋 Table of Contents
1. 🤖 [Introduction](#-1-introduction)
2. ⚙️ [Tech Stack](#-2-tech-stack)
3. 📁 [Project Structure](#-3-project-structure)
4. 📊 [Data Diagnosis & Hypothesis Testing](#-4-data-diagnosis--hypothesis-testing)
5. 🧪 [Experimental Results: CASE 1 vs CASE 2](#-5-experimental-results-case-1-vs-case-2)
6. 🚀 [Model Operational Architecture & Tuning](#-6-model-operational-architecture--tuning)
7. 💻 [Usage Guide](#-7-usage-guide)

---

## 🤖 1. Introduction
Dự án nghiên cứu và xây dựng pipeline hướng đối tượng (OOP) nhằm tự động hóa quy trình tiền xử lý dữ liệu, kiểm định giả thuyết thống kê, và huấn luyện/đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`) trên bộ dữ liệu y tế gồm **55,500 hồ sơ bệnh án**. Xây dựng End-to-End Machine Learning Pipeline, tập trung vào việc chẩn đoán chất lượng dữ liệu chuyên sâu, phát hiện các điểm bất thường (outliers/anomalies) và xây dựng bộ công cụ (toolkit) tái sử dụng để tối ưu hóa quy trình từ xử lý dữ liệu đến huấn luyện mô hình.

---

## ⚙️ 2. Tech Stack
- **Language & Framework:** Python, Streamlit
- **Libraries:** Sckit-Learn, XGBoost, LightGBM, SciPy, Pandas, NumPy, Seaborn

---

## 📁 3. Cấu trúc thư mục (Project Structure)
```text
healthcare_ml/
│
├── data/                       # Thư mục quản lý dữ liệu
│   └── healthcare_dataset.csv  # Dataset gốc (55,500 dòng)
│
├── notebooks/                  # Nghiên cứu và thực nghiệm từng bước trên Jupyter
│   ├── 01_eda.ipynb            # Khám phá dữ liệu & Kiểm định Chi-square
│   ├── 02_processing.ipynb     # Pipeline tiền xử lý & Chia tách 80/10/10
│   └── 03_modeling.ipynb       # Huấn luyện mô hình gốc & Các phân hệ cải thiện
│
└── src/                        # Bộ công cụ mã nguồn Python (OOP) tái sử dụng cao
    ├── __init__.py
    ├── data_inspector.py       # Công cụ tự động kiểm tra chất lượng dữ liệu
    ├── visualizer.py           # Tự động hóa vẽ biểu đồ EDA
    ├── preprocessor.py         # Thực thi quy trình drop → clip → encode → scale
    └── model_trainer.py        # Bộ huấn luyện đa mô hình & Vẽ Bootstrap curves

```
---

## 📊 4. Data Quality Diagnosis & Hypothesis Testing
Quét dữ liệu thô qua bộ lọc DataInspector thu được các phát hiện cốt lõi sau:

- Xử lý đặc trưng: Loại bỏ các biến hành chính nhiễu (Name, Doctor, Hospital, Room Number). Trích xuất đặc trưng lâm sàng: Length of Stay (Số ngày nằm viện).

- Bất thường dữ liệu: Toán học Boxplot báo 0 dòng ngoại lai vì viện phí (Billing Amount) tuân theo phân phối đều từ vài trăm đến $50,000$ USD. Tuy nhiên, phân tích logic phát hiện 108 dòng có viện phí âm phi lý (Xử lý bằng clip(lower=0)) và 534 dòng trùng lặp tuyệt đối (Xử lý bằng drop_duplicates).

- Tính mùa vụ: Dữ liệu chu kỳ 5 năm (2019-2024) phẳng lỳ theo từng tháng, hoàn toàn không có tính mùa vụ (Seasonality) hay đột biến bệnh lý.

- Kiểm định Chi bình phương ($\chi^2$): Kiểm định tính độc lập giữa các biến đầu vào với nhãn mục tiêu Test Results đều trả về giá trị $p\text{-value} > 0.05$.

🚨 Insight: Phân phối đều của dữ liệu số học và kết quả kiểm định độc lập khẳng định nhãn mục tiêu đã bị gán ngẫu nhiên cơ học (Pure Noise). Bộ dữ liệu không tồn tại mối quan hệ nhân quả y khoa thực tế (Dấu vết dữ liệu giả lập bằng máy).

<img width="828" height="273" alt="image" src="https://github.com/user-attachments/assets/ce6a5c36-f93c-4e1c-a5d7-af0028fcbf89" />

## 🧪 5. Experimental Results: CASE 1 vs CASE 2
Đối chứng hiệu năng mô hình giữa dữ liệu thô và dữ liệu đã làm sạch để đánh giá tác động của nhiễu dữ liệu.

| Thuật toán | CASE 1: Thô (Acc/F1) | CASE 2: Sạch (Acc/F1) | Δ Accuracy |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 33.2% / 0.33 | 33.6% / 0.33 | +0.4% |
| **Decision Tree** | 33.1% / 0.32 | 33.1% / 0.32 | 0.0% |
| **Random Forest** | **44.5% / 0.44** | **43.8% / 0.44** | -0.7% |
| **Gradient Boosting** | 32.6% / 0.32 | 33.1% / 0.33 | +0.5% |

> 🧐 **Insight:** Sự sụt giảm nhẹ của Random Forest sau khi xóa trùng lặp do mất cơ chế *Implicit Oversampling* (nhân bản nhiễu). Hiệu năng không đổi (<1%) khẳng định dữ liệu gốc thiếu tính nhân quả; làm sạch không thể bù đắp được thiếu hụt tín hiệu y khoa.

---

## 🚀 6. Model Operational Architecture & Tuning

### 6.1. Kiến trúc suy luận (Inference Flow)
* **Pipeline**: `AutoPreprocessor` (Trích xuất đặc trưng ➔ Encoding ➔ Scaling). Lưu ý: `StandardScaler` được *fit* trên tập Train để tránh **Data Leakage**.
* **Ensemble Strategy**: Sử dụng **Random Forest** (Bagging, 100 estimators) nhờ khả năng kháng nhiễu thông qua cơ chế bỏ phiếu đám đông, giúp ổn định hóa các quyết định sai lệch.

### 6.2. Benchmarking (Final Selection)
| Mô hình | Test Accuracy | F1 Macro | Độ ổn định (CV) | Thế mạnh |
| :--- | :---: | :---: | :---: | :--- |
| **Logistic Regression** | 34.2% | 0.331 | Cao | Độ trễ thấp (<0.1s) |
| **XGBoost** | 37.1% | 0.369 | Trung bình | Kiểm soát Loss qua Early Stopping |
| **Random Forest** | **38.4%** | **0.381** | Cao | **Kháng nhiễu tốt nhất** |

> 🏆 **Best Model**: `best_model_random_forest.pkl`. Ưu tiên độ ổn định trên tập kiểm thử độc lập thay vì chạy đua Accuracy ảo trên dữ liệu nhiễu.

### 6.3. Tối ưu hóa (Advanced Optimization)
* **Feature Engineering**: Thêm 6 biến chuyên ngành (`Age_Group`, `Long_Stay`, `Cond_Med_Interact`...) nâng hiệu năng Random Forest lên **43.7%** (+9.2% so với baseline).
* **Hyperparameter Tuning**: Sử dụng `RandomizedSearchCV` (5-Fold CV) tối ưu `max_depth` và `min_samples_split` để chặn Overfitting.

---


## 💻 7. Code Snippets & Usage Guide

Dự án được thiết kế hoàn toàn theo kiến trúc hướng đối tượng (OOP), đóng gói thành các Class độc lập giúp bạn có thể dễ dàng tái sử dụng toàn bộ pipeline tiền xử lý và huấn luyện chỉ với vài dòng code ngắn.

### 🛠️ 7.1. Sử dụng Pipeline Tiền xử lý dữ liệu tự động
Đoạn mã mẫu minh họa cách gọi lớp `AutoPreprocessor` để tự động dọn dẹp dữ liệu ngoại lai, triệt tiêu biến nhiễu hành chính và cắt lát dữ liệu theo tỷ lệ nghiêm ngặt **80/10/10**:

```python
from src.preprocessor import AutoPreprocessor
import pandas as pd

# 1. Nạp bộ dữ liệu y tế thô
df = pd.read_csv('data/healthcare_dataset.csv')

# 2. Khởi tạo Pipeline (Tự động cô lập và loại bỏ các biến định danh)
prep = AutoPreprocessor(
    target_col='Test Results', 
    drop_cols=['Name', 'Doctor', 'Hospital', 'Room Number']
)

# 3. Thực thi quy trình tự động và chia tách dữ liệu (Train/Val/Test - 80/10/10)
X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)

print(f" Kích thước tập huấn luyện chuẩn hóa (Train Set): {X_train.shape}")
```
---

## 🛠️ 7.2 Hướng dẫn cài đặt & Chạy dự án (Installation & Usage)
Khuyến khích tạo môi trường ảo (Virtual Environment) và cài đặt các thư viện phụ thuộc:
```bash
git clone [https://github.com/](https://github.com/)[BaoVo1126]/healthcare-analysis-ai.git
cd healthcare-analysis-ai
pip install -r requirements.txt
```
---
