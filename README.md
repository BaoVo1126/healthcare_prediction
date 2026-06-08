# 🏥 Healthcare Classification & AI Model Benchmarking
Dataset gốc: [Healthcare Dataset (Kaggle)](https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset)

---

## 📋 Table of Contents
1. 🤖 [Introduction](#-1-introduction)
2. ⚙️ [Tech Stack](#-2-tech-stack)
3. 🔋 [Features](#-3-tính-năng-cốt-lõi-core-features)
4. 📁 [Project Structure](#-4-cấu-trúc-thư-mục-project-structure)
5. 📊 [Data Quality Diagnosis & Hypothesis Testing](#-5-data-quality-diagnosis--hypothesis-testing)
6. 🧪 [Experimental Results: CASE 1 vs CASE 2](#-6-experimental-results-case-1-vs-case-2)
7. 🚀 [Advanced Improvements & Model Tuning](#-7-advanced-improvements--model-tuning)
8. 💻 [Snippets (Quick Start)](#-8-code-snippets--usage-guide)
9. 🚀 [Installation & Usage](#-9-hướng-dẫn-cài-đặt--chạy-dự-án-installation--usage)

---

## 🤖 1. Introduction
Dự án nghiên cứu và xây dựng pipeline hướng đối tượng (OOP) nhằm tự động hóa quy trình tiền xử lý dữ liệu, kiểm định giả thuyết thống kê, và huấn luyện/đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression`, `Decision Tree`, `Random Forest`, `Gradient Boosting`, `XGBoost`, `LightGBM`) trên bộ dữ liệu y tế gồm **55,500 hồ sơ bệnh án**. Dự án tập trung vào việc phản biện tư duy giữa dữ liệu thô và dữ liệu sạch thông qua các kiểm định toán học nghiêm ngặt.

---

## ⚙️ 2. Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, SciPy, Scikit-Learn, XGBoost, LightGBM
- **Visuals:** Matplotlib, Seaborn
- **Deployment:** Streamlit

---

## 🔋 3. Tính năng cốt lõi (Core Features)
- 💪 **Automated Data Engineering**: Một pipeline `AutoPreprocessor` thông minh tự động hóa toàn bộ quy trình từ làm sạch dữ liệu, xử lý giá trị thiếu đến chuẩn hóa (Scaling) theo tỷ lệ nghiêm ngặt **80/10/10** giúp triệt tiêu hoàn toàn lỗi rò rỉ dữ liệu (*Data Leakage*).
- 💪 **Predictive Intelligence**: Triển khai kiến trúc Ensemble với **Random Forest** và **Decision Tree**, cho phép phân tích so sánh để đạt được độ tin cậy tối ưu trong chẩn đoán.
- 💪 **Interactive AI Dashboard**: Giao diện người dùng được xây dựng trên **Streamlit**, cung cấp khả năng dự đoán rủi ro theo thời gian thực dựa trên các chỉ số sinh học.
- 💪 **Advanced Visual Analytics**: Hệ thống hóa các báo cáo EDA chuyên sâu từ module `Visualizer`, trực quan hóa mối tương quan giữa 15+ thuộc tính y tế để tìm ra các biến quan trọng nhất.
- 💪 **Scalable Model Persistence**: Quy trình đóng gói mô hình (Serialization) chuyên nghiệp sang định dạng `.pkl`, đảm bảo hệ thống luôn sẵn sàng để triển khai và inference nhanh chóng.

---

## 📁 4. Cấu trúc thư mục (Project Structure)
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

## 📊 5. Data Quality Diagnosis & Hypothesis Testing
Trước khi huấn luyện, dữ liệu được đưa qua bộ lọc DataInspector để chẩn đoán "sức khỏe" hệ thống dữ liệu thô và cho ra các bằng chứng toán học cực kỳ quan trọng:

### 📌 5.1. Từ điển dữ liệu & Định hướng xử lý (Data Metadata)

- **Biến hành chính (Drop)**: `Name, Doctor, Hospital, Room Number` mang tính hành chính, có độ tuần hoàn (cardinality) quá cao ➔ Loại bỏ hoàn toàn để tránh mô hình bị học vẹt và bảo mật thông tin.

- **Biến phân loại (Categorical)**: `Gender, Blood Type, Admission Type, Medication` ➔ Mã hóa về dạng số bằng LabelEncoder.

- **Biến thời gian (Datetime)**: `Date of Admission` và `Discharge Date` ➔ Trích xuất thành biến phái sinh: Length of Stay (Số ngày nằm viện).

- **Biến số học (Numerical)**: `Age, Billing Amount` ➔ Chuẩn hóa bằng StandardScaler.

- **Biến mục tiêu (Target)**: `Test Results (Normal / Abnormal / Inconclusive)` ➔ Nhãn đa lớp cần dự đoán. Baseline đoán mò ngẫu nhiên đạt 33.33%.

## 🔍 5.2. Chẩn đoán bất thường & Ngoại lai (Anomalies Check)

- **Dữ liệu trùng lặp**: Phát hiện 534 dòng trùng lặp tuyệt đối (0.96%). Đây là dấu hiệu của lỗi sao chép dữ liệu (Artefacts) trong quá trình giả lập. Hệ thống tự động dùng df.drop_duplicates() để loại bỏ.

- **Viện phí (Billing Amount)**: Biên độ dao động từ $−2,008.49$ đến $52,764.28$. Xuất hiện 108 giá trị âm phi logic (Lỗi nhập liệu). Hệ thống xử lý bằng kỹ thuật clip(lower=0) để làm mịn thay vì xóa bỏ nhằm bảo toàn số dòng cho tập Train.

- **Bản chất phân phối**: Cả `Age, Billing Amount`, và `Length of Stay` đều có dạng phân phối đều (Uniform Distribution) với độ lệch (Skewness) tiệm cận mức 0. Đây là bằng chứng đanh thép khẳng định bộ dữ liệu mang tính chất giả lập (Synthetic Data), không phải dữ liệu lâm sàng thực tế (vốn luôn bị lệch phải).

### 🧪 5.3. Kết quả kiểm định chi bình phương (Chi-Square Test)
Để trả lời câu hỏi: *"Các thuộc tính đầu vào có thực sự liên quan đến kết quả xét nghiệm (Test Results) hay không?"*, ta thực hiện kiểm định với giả thuyết $H_0$ (Thuộc tính và kết quả xét nghiệm độc lập, không có tính nhân quả).

| Thuộc tính (Feature) | Độ tự do (DoF) | Giá trị thống kê $\chi^2$ | Giá trị $p\text{-value}$ | Kết luận toán học |
| :--- | :---: | :---: | :---: | :--- |
| **Medical Condition** | 10 | 13.26 | **0.2098** | ❌ **Fail to Reject $H_0$** (Độc lập) |
| **Admission Type** | 4 | 1.32 | **0.8580** | ❌ **Fail to Reject $H_0$** (Độc lập) |
| **Medication** | 8 | 3.73 | **0.8805** | ❌ **Fail to Reject $H_0$** (Độc lập) |
| **Gender** | 2 | 2.02 | **0.3645** | ❌ **Fail to Reject $H_0$** (Độc lập) |
| **Blood Type** | 14 | 7.63 | **0.9076** | ❌ **Fail to Reject $H_0$** (Độc lập) |

> 🚨 **Insight Y khoa thực tế**: Trong thực tế lâm sàng, một bệnh nhân ung thư nặng phải có phân phối kết quả xét nghiệm khác hoàn toàn với một người khám sức khỏe định kỳ. Việc tất cả các giá trị $p\text{-value}$ đều lớn hơn rất nhiều so với ngưỡng ý nghĩa ($> 0.05$) và hệ số tương quan tuyến tính kịch trần chỉ đạt $0.0065$ là bằng chứng đanh thép chứng minh: **Nhãn mục tiêu đã bị gán ngẫu nhiên cơ học (Pure Noise)** từ quá trình giả lập dữ liệu.

---


## 🧪 6. Experimental Results: CASE 1 vs CASE 2

Để minh chứng tư duy phản biện khoa học trước hội đồng, hệ thống tiến hành thử nghiệm đối chứng nghiêm ngặt giữa hai trường hợp:
- **CASE 1 (Raw Data)**: Huấn luyện trên dữ liệu thô gốc (Giữ nguyên toàn bộ 534 dòng trùng lặp và 108 giá trị viện phí âm).
- **CASE 2 (Cleaned Data)**: Huấn luyện trên dữ liệu sạch (Đã loại bỏ hoàn toàn 534 dòng lặp và làm mịn cột viện phí bằng kỹ thuật `clip(lower=0)`).

### Bảng đối chứng hiệu năng thực nghiệm (Data Quality Benchmarking)

| Thuật toán (Model) | CASE 1: Dữ liệu thô (Accuracy / F1 Macro) | CASE 2: Dữ liệu sạch (Accuracy / F1 Macro) | Độ chênh lệch ($\Delta$ Accuracy) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | 33.21% / 0.3271 | 33.64% / 0.3343 | **+ 0.43%** |
| **Decision Tree** | 33.10% / 0.3196 | 33.11% / 0.3241 | **+ 0.01%** |
| **Random Forest** | **44.52% / 0.4447** | **43.81% / 0.4379** | **- 0.71%** |
| **Gradient Boosting** | 32.58% / 0.3230 | 33.07% / 0.3299 | **+ 0.49%** |

> 🧐 **Giải mã nghịch lý toán học từ Kỹ sư AI**: 
> Tại sao thuật toán mạnh như **Random Forest** lại bị sụt giảm $0.71\%$ độ chính xác sau khi ta dọn sạch dữ liệu trùng lặp?
> Bản chất các dòng trùng lặp khi nằm trong tập dữ liệu huấn luyện (Train Set) sẽ hoạt động như một cơ chế tăng cường dữ liệu ẩn (*Implicit Oversampling*). Mô hình Cây quyết định khi gặp một cấu trúc lặp đi lặp lại nhiều lần trong một tập dữ liệu ngẫu nhiên (nhiễu hoàn toàn) sẽ có xu hướng học vẹt cấu trúc đó để tạo ra các nút lá phân rã giả tạo. Khi ta xóa bỏ trùng lặp, "bức màn" này bị hạ xuống, đưa mô hình về đúng thực tế hỗn loạn của dữ liệu nhiễu.
>
> **Kết luận**: Hiệu năng biến động không đáng kể ($< 1\%$) chứng minh định luật tối cao của Machine Learning: **"Garbage in, Garbage out"**. Việc làm sạch dữ liệu đơn thuần không thể cứu vãn được mô hình nếu bản thân dữ liệu gốc đã hoàn toàn mất đi tính nhân quả.

---

## 🚀 7. Advanced Improvements & Model Tuning

Mặc dù dữ liệu mang tính ngẫu nhiên, hệ thống vẫn triển khai đầy đủ các phân hệ cải tiến nâng cao chuyên sâu nhằm minh chứng năng lực thiết kế kiến trúc MLOps:

### 7.1. Các kỹ thuật tối ưu hóa nâng cao
1. **Advanced Feature Engineering (+6 Biến phái sinh)**: Trích xuất các biến mang tính chất tri thức chuyên ngành bao gồm `Age_Group` (Nhóm tuổi), `High_Billing` (Biến cờ hiệu cho hóa đơn kịch trần), `Long_Stay` (Số ngày nằm viện dài hạn), `Is_Weekend`, `Quarter`, và biến tương tác đặc trưng `Cond_Med_Interact`.
   - *Kết quả*: Giúp Random Forest bám vào các cấu trúc phi tuyến tính nhân tạo, đẩy Accuracy lên **~43.7%** (Tăng mạnh **+9.2%** so với baseline thô ban đầu).
2. **XGBoost & LightGBM Deployment**: Triển khai các thuật toán Boosting mạnh mẽ, cấu hình tham số giám sát tập Validation (`eval_set=X_val`) kết hợp cơ chế dừng sớm (`early_stopping_rounds`) để giám sát chặt chẽ hàm mất mát qua từng cây nhằm triệt tiêu hiện tượng Overfitting.
   - *Kết quả*: XGBoost đạt **36.7%**, LightGBM đạt **36.3%** trên tập kiểm thử độc lập.
3. **Hyperparameter Tuning**: Thực thi quét không gian tham số diện rộng thông qua `RandomizedSearchCV` (20 combinations × 5-Fold Cross-Validation) để tinh chỉnh các tham số `max_depth`, `n_estimators`, và `min_samples_split`.

### 7.2. Bảng Đánh giá Hiệu năng Tổng hợp & Lựa chọn Mô hình (Final Model Selection Benchmarking)
> **Lưu ý**: Sau khi thử nghiệm diện rộng, dự án trích xuất 3 mô hình đại diện xuất sắc nhất cho 3 kiến trúc thuật toán cốt lõi (Linear, Bagging, Boosting) để tiến hành đánh giá chuyên sâu và bốc ra mô hình tối ưu nhất đưa vào ứng dụng thực tế (`Streamlit`).

| Model | Train Accuracy | Test Accuracy | F1 Macro | CV Mean ± Std | Time (s) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 45.2% | **38.4%** | 0.3810 | 0.3750 ± 0.012 | 2.5s |
| **XGBoost** | 52.1% | 37.1% | 0.3690 | 0.3620 ± 0.015 | 4.1s |
| **Logistic Regression** | 34.5% | 34.2% | 0.3310 | 0.3390 ± 0.005 | 0.8s |

> 🏆 **Mô hình vô địch (Best Model)**: Mô hình tinh chỉnh tối ưu lưu trữ tại file `best_model_random_forest.pkl` được lựa chọn để đóng gói nhờ khả năng kiểm soát nhiễu tốt nhất, biên độ Overfitting thấp và duy trì độ ổn định cao qua các vòng Cross-Validation. Dự án nghiêm túc **không** sử dụng các kỹ thuật tự sinh số liệu giả tạo để cố tình "ép điểm" Accuracy lên cao, nhằm đảm bảo tuyệt đối tính toàn vẹn dữ liệu (*Data Integrity*).

---

## 💻 8. Code Snippets & Usage Guide

Dự án được thiết kế hoàn toàn theo kiến trúc hướng đối tượng (OOP), đóng gói thành các Class độc lập giúp bạn có thể dễ dàng tái sử dụng toàn bộ pipeline tiền xử lý và huấn luyện chỉ với vài dòng code ngắn.

### 🛠️ 8.1. Sử dụng Pipeline Tiền xử lý dữ liệu tự động
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

### 🧪 8.2. Huấn luyện đa mô hình song song & Đánh giá tổng hợp
Đoạn mã minh họa cách gọi lớp ModelTrainer để kích hoạt chuỗi huấn luyện các kiến trúc thuật toán, vẽ đường cong học tập Bootstrap và xuất báo cáo hiệu năng:
```python
from src.model_trainer import ModelTrainer

# 1. Khởi tạo bộ huấn luyện đa mô hình
trainer = ModelTrainer()

# 2. Huấn luyện đồng thời các thuật toán kết hợp truyền tập Val để giám sát đường cong học tập
trainer.fit(X_train, y_train, X_val, y_val, X_test, y_test, run_bootstrap=True)

# 3. Trích xuất bảng báo cáo hiệu năng tổng hợp (Final Benchmarking)
trainer.summary()

```


---
## 🛠️ 9. Hướng dẫn cài đặt & Chạy dự án (Installation & Usage)

### 1. Cài đặt môi trường
Khuyến khích tạo môi trường ảo (Virtual Environment) và cài đặt các thư viện phụ thuộc:
```bash
git clone [https://github.com/](https://github.com/)[TÊN_GITHUB_CỦA_BẠN]/healthcare-analysis-ai.git
cd healthcare-analysis-ai
pip install -r requirements.txt
```
---
