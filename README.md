# 🏥 Healthcare ML Pipeline: From Data Diagnosis to Predictive Modeling & Serving

Dataset gốc: [Healthcare Dataset (Kaggle)](https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset)

---

## 📋 Table of Contents
1.  [Introduction](#-1-introduction)
2.  [Tech Stack](#-2-tech-stack)
3.  [Project Structure](#-3-project-structure)
4.  [Data Diagnosis & Hypothesis Testing](#-4-data-diagnosis--hypothesis-testing)
5.  [Experimental Results: CASE 1 vs CASE 2](#-5-experimental-results-case-1-vs-case-2)
6.  [Model Operational Architecture & Tuning](#-6-model-operational-architecture--tuning)
7.  [ETL/ELT Pipeline & Automation](#-7-etlelt-pipeline--automation)
8.  [Streamlit App — Serving Layer](#-8-streamlit-app--serving-layer)
9.  [Usage Guide](#-9-usage-guide)
10. [Production Readiness & Engineering Notes](#-10-production-readiness--engineering-notes)

---

##  1. Introduction
Dự án nghiên cứu và xây dựng pipeline hướng đối tượng (OOP) nhằm tự động hóa quy trình tiền xử lý dữ liệu, kiểm định giả thuyết thống kê, và huấn luyện/đánh giá hiệu năng của nhiều mô hình Machine Learning (`Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM`) trên bộ dữ liệu y tế gồm *55,500 hồ sơ bệnh án*. Xây dựng *End-to-End Machine Learning Pipeline*, tập trung vào `việc chẩn đoán chất lượng dữ liệu chuyên sâu, phát hiện các điểm bất thường (outliers/anomalies)` và `xây dựng bộ công cụ (toolkit) tái sử dụng để tối ưu hóa quy trình từ xử lý dữ liệu đến huấn luyện mô hình`.

---

## ⚙️ 2. Tech Stack
- *Language*: Python 3.10+
- *Core ML*: Scikit-Learn, XGBoost, LightGBM, SciPy
- *Data Engineering*: Pandas, NumPy
- *Pipeline & Storage*: ETL/ELT, SQLite (warehouse cho audit trail & prediction log), Joblib (artifact serialization)
- *Serving*: Streamlit (giao diện dự đoán + dashboard giám sát)
- *QA & Automation*: Pytest, Logging 

---

##  3. Cấu trúc thư mục (Project Structure)
```text
healthcare_ml/
│
├── notebooks/                   # Nghiên cứu và thực nghiệm từng bước trên Jupyter
│   ├── 01_eda.ipynb             # Khám phá dữ liệu & Kiểm định Chi-square
│   ├── 02_processing.ipynb      # Pipeline tiền xử lý & Chia tách 80/10/10
│   └── 03_modeling.ipynb        # Huấn luyện mô hình gốc & Các phân hệ cải thiện
│
├── src/                         # Bộ công cụ OOP (Reusable Toolkit)
│   ├── data_inspector.py        # Kiểm định giả thuyết (Chi², MI)
│   ├── visualizer.py            # Tự động hóa biểu đồ
│   ├── preprocessor.py          # Pipeline clean/split/encode/scale
│   └── model_trainer.py         # Huấn luyện đa mô hình & Reporting
│
├── etl/                         # Pipeline ETL/ELT — điều phối lại toolkit ở src/ thành 1 quy trình chạy được
│   ├── config.py                # Đường dẫn, hằng số, danh mục categorical dùng chung
│   ├── extract.py               # Extract: đọc + validate schema dữ liệu nguồn
│   ├── transform.py             # Transform: DataInspector (QA) + AutoPreprocessor
│   ├── load.py                  # Load: lưu artifact + ghi warehouse (SQLite)
│   └── pipeline.py              # Orchestrator: extract → transform → train → load
│
├── app/                         # Serving layer
│   └── streamlit_app.py         # Giao diện dự đoán + dashboard giám sát
│
├── test/
│   └── test_preprocessor.py
│
└── requirements.txt
```

---

##  4. Data Quality Diagnosis & Hypothesis Testing
Quét dữ liệu thô qua bộ lọc `DataInspector` thu được các phát hiện cốt lõi đặc trưng sau:

- **Xử lý đặc trưng**: Loại bỏ các biến hành chính nhiễu `(Name, Doctor, Hospital, Room Number)`. Trích xuất đặc trưng lâm sàng: `Length of Stay (Số ngày nằm viện)`.

- **Bất thường dữ liệu**: Toán học Boxplot báo 0 dòng ngoại lai vì `Billing Amount` tuân theo phân phối đều từ vài trăm đến $50,000$ USD. Tuy nhiên, phân tích logic phát hiện 108 dòng có viện phí âm phi lý (`Xử lý bằng clip(lower=0)`) và 534 dòng trùng lặp tuyệt đối (`Xử lý bằng drop_duplicates`).

- **Tính mùa vụ**: Dữ liệu chu kỳ 5 năm (2019-2024) phẳng lỳ theo từng tháng, hoàn toàn không có tính mùa vụ (Seasonality) hay đột biến bệnh lý.

- **Kiểm định Chi-square ($\chi^2$)**: Kiểm định tính độc lập giữa các biến categorical với nhãn mục tiêu `Test Results`. **Tất cả features trả về p-value > 0.05.**

 *Insight*: Phân phối đều của dữ liệu số học và kết quả kiểm định độc lập khẳng định nhãn mục tiêu đã bị gán ngẫu nhiên cơ học (Pure Noise). Bộ dữ liệu không tồn tại mối quan hệ nhân quả y khoa thực tế (Dấu vết dữ liệu giả lập bằng máy).

<img width="828" height="273" alt="image" src="https://github.com/user-attachments/assets/ce6a5c36-f93c-4e1c-a5d7-af0028fcbf89" />

*Hình 1: Bằng chứng trực quan khẳng định Pure Noise. Bên trái: Toàn bộ p-value từ kiểm định Chi-square độc lập đều > 0.05 (đường nét đứt đỏ). Bên phải: Điểm Mutual Information xấp xỉ 0 trên mọi feature.*

>  Đây cũng là lý do phần ETL/ELT phía dưới không cố "ép" accuracy cao hơn bằng mọi giá — mục tiêu chuyển sang chứng minh **hạ tầng đúng** (pipeline tái tạo được, không leakage, có audit trail) thay vì chạy đua theo một chỉ số vô nghĩa trên dữ liệu nhiễu.

---

##  5. Experimental Results: CASE 1 vs CASE 2
Đối chứng hiệu năng mô hình giữa dữ liệu thô và dữ liệu đã làm sạch để đánh giá tác động của nhiễu dữ liệu.

| Thuật toán | CASE 1: Thô (Acc/F1) | CASE 2: Sạch (Acc/F1) | Δ Accuracy |
| :--- | :---: | :---: | :---: |
| *Logistic Regression* | 33.2% / 0.33 | 33.6% / 0.33 | +0.4% |
| *Decision Tree* | 33.1% / 0.32 | 33.1% / 0.32 | 0.0% |
| *Random Forest* | **44.5% / 0.44** | **43.8% / 0.44** | -0.7% |
| *Gradient Boosting* | 32.6% / 0.32 | 33.1% / 0.33 | +0.5% |

**Insight:** Sự sụt giảm nhẹ của *Random Forest* sau khi xóa trùng lặp do mất cơ chế *Implicit Oversampling* (nhân bản nhiễu). Hiệu năng không đổi (<1%) khẳng định dữ liệu gốc thiếu tính nhân quả; làm sạch không thể bù đắp được thiếu hụt tín hiệu y khoa.

---

##  6. Model Operational Architecture & Tuning

### 6.1. Kiến trúc suy luận (Inference Flow)
* *Pipeline*: AutoPreprocessor (Trích xuất đặc trưng ➔ Encoding ➔ Scaling). Lưu ý: *StandardScaler* được fit trên tập Train để tránh *Data Leakage*.
* *Ensemble Strategy*: Sử dụng *Random Forest* (Bagging, 100 estimators) nhờ khả năng kháng nhiễu thông qua cơ chế bỏ phiếu đám đông, giúp ổn định hóa các quyết định sai lệch.

### 6.2. Benchmarking (Final Selection)
| Mô hình | Test Accuracy | F1 Macro | Độ ổn định (CV) | Thế mạnh |
| :--- | :---: | :---: | :---: | :--- |
| *Logistic Regression* | 34.2% | 0.331 | Cao | Độ trễ thấp (<0.1s) |
| *XGBoost* | 37.1% | 0.369 | Trung bình | Kiểm soát Loss qua Early Stopping |
| *Random Forest* | **38.4%** | **0.381** | Cao | *Kháng nhiễu tốt nhất* |

**Best Model**: `best_model_random_forest.pkl`. Ưu tiên độ ổn định trên tập kiểm thử độc lập thay vì chạy đua Accuracy ảo trên dữ liệu nhiễu.

<img width="620" height="172" alt="image" src="https://github.com/user-attachments/assets/9f78db36-989c-4e01-9d9e-4f48c8ea9a02" />

*Hình 2: So sánh hiệu năng Test Accuracy, F1-Macro và Validation Accuracy giữa Baseline và các mô hình Refined sau Feature Engineering. Random Forest cho thấy sự cải thiện mạnh nhất nhờ feature interactions.*

### 6.3. Tối ưu hóa (Advanced Optimization)
* *Feature Engineering*: Thêm 6 biến chuyên ngành (Age_Group, Long_Stay, Cond_Med_Interact...) nâng hiệu năng Random Forest lên *43.7%* (+9.2% so với baseline).
* *Hyperparameter Tuning*: Sử dụng `RandomizedSearchCV (5-Fold CV)` tối ưu max_depth và min_samples_split để chặn Overfitting.

### 6.4. Model Robustness & Stability Analysis
- *Cross-Validation*: Sử dụng 5-Fold CV để đảm bảo mô hình không bị quá phụ thuộc vào phân chia dữ liệu Train/Test ngẫu nhiên.
- *Ensemble Stability*: Cơ chế Bagging của Random Forest giúp giảm thiểu ảnh hưởng của nhiễu (outliers) và làm mịn các quyết định của cây đơn lẻ.
- *Error Analysis*: Mô hình được đánh giá thông qua cả Accuracy và F1-Macro để đảm bảo tính bền vững trên cả các lớp dữ liệu mất cân bằng.

<img width="619" height="443" alt="image" src="https://github.com/user-attachments/assets/6e428a41-1768-4bcc-a5fd-75c6f7d1bf39" />

*Hình 3: Confusion Matrix của Random Forest (Best Model) trên tập Test. Mô hình không bị bias mạnh vào một class nào cụ thể (phân phối normalized recall khá đồng đều), củng cố thêm giả thuyết "đoán ngẫu nhiên" trên dữ liệu Pure Noise.*

<img width="617" height="221" alt="image" src="https://github.com/user-attachments/assets/0b2c75b6-0758-4afd-ae56-0ed5d5130c04" />

*Hình 4: Bootstrap Learning Curves cho thấy độ ổn định của Test Accuracy qua các kích thước tập Train khác nhau. Khoảng tin cậy (vùng bóng) rất hẹp khẳng định mô hình ổn định và không bị phụ thuộc vào phân chia dữ liệu ngẫu nhiên.*

### 6.5 Challenges & Future Work
**Challenges & Solutions**:
- *Data Quality*: Dữ liệu y tế được xác định có độ nhiễu cao (Pure Noise). Thay vì cố gắng tối ưu hóa chỉ số Accuracy một cách khiên cưỡng, tôi tập trung thiết lập Pipeline Architecture chuẩn mực, đảm bảo tính bền vững và khả năng kiểm soát dữ liệu đầu vào.
- *Engineering Mindset*: Xây dựng hệ thống theo hướng Module hóa (Modular) và Tái sử dụng (Reusable), giúp Pipeline có thể tích hợp ngay lập tức vào các dự án thực tế khi có nguồn dữ liệu chất lượng cao hơn.

**Future Work**:
- *Scalability*: Triển khai Pipeline lên hạ tầng Cloud (AWS/GCP) sử dụng Docker/Kubernetes.
- *MLOps Integration*: Tích hợp công cụ quản lý thí nghiệm (MLflow/W&B) để theo dõi phiên bản model và so sánh kết quả thực nghiệm khoa học.
- *Advanced Modeling*: Thử nghiệm các kiến trúc Deep Learning hoặc Tabular Transformers để khai thác sâu hơn các đặc trưng dữ liệu phức tạp.

---

## 🔄 7. ETL/ELT Pipeline & Automation

Toolkit ở `src/` đã đúng về mặt logic (chống leakage, OOV handling, CV...), nhưng notebook là quy trình chạy tay, từng bước. Package `etl/` **điều phối lại đúng các class đó** thành một pipeline chạy được bằng một lệnh, có thể lặp lại mỗi khi có dữ liệu mới, và có nơi lưu vết (audit trail) — thứ mà notebook không có.

```bash
python -m etl.pipeline
```

Một lần chạy thực hiện:

1. **Extract** (`etl/extract.py`) — đọc `healthcare_dataset.csv`, validate schema tối thiểu (đủ cột bắt buộc, không rỗng) — fail fast trước khi dữ liệu lỗi lan sang các bước sau.
2. **Transform** (`etl/transform.py`) — chạy `DataInspector` để log chất lượng dữ liệu (tái sử dụng insight ở mục 4), sau đó `AutoPreprocessor.fit_transform()` — **không viết lại** logic tiền xử lý gốc, chỉ orchestrate.
3. **Train** — tái sử dụng `ModelTrainer` để huấn luyện & so sánh các mô hình bằng Stratified K-Fold CV (tập model hiện có trong `ModelTrainer.DEFAULT_MODELS`; XGBoost/LightGBM đang ở `03_modeling.ipynb` — thêm vào dict này là cách nhanh nhất để pipeline tự động chạy luôn cả 2 model đó).
4. **Load** (`etl/load.py`) —
   - lưu `preprocessor.pkl` + toàn bộ model `.pkl` — **một artifact duy nhất** dùng chung cho notebook, test và Streamlit app, tránh lệch giữa lúc train và lúc serving;
   - lưu `metrics_summary.json`;
   - ghi lịch sử chạy (số dòng, best model, accuracy, timestamp) vào **SQLite warehouse** (`data/warehouse.db`, bảng `pipeline_runs`).

---

## 8. Streamlit App — Serving Layer

```bash
streamlit run app/streamlit_app.py
```

---

##  9. Usage Guide

Dự án được thiết kế hoàn toàn theo kiến trúc hướng đối tượng (OOP), đóng gói thành các Class độc lập giúp bạn có thể dễ dàng tái sử dụng toàn bộ pipeline tiền xử lý và huấn luyện chỉ với vài dòng code ngắn.

### 9.1. Cài đặt
```bash
git clone https://github.com/BaoVo1126/healthcare_prediction.git
cd healthcare_prediction
pip install -r requirements.txt
```

### 9.2. Dùng trực tiếp AutoPreprocessor (như trước)
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

print(f"Kích thước tập huấn luyện chuẩn hóa (Train Set): {X_train.shape}")
```

### 9.3. Chạy toàn bộ pipeline tự động + giao diện (mới)
```bash
python -m etl.pipeline              # extract → transform → train → load artifact + warehouse
streamlit run app/streamlit_app.py  # mở giao diện dự đoán tại localhost:8501
```

---

##  10. Production Readiness & Engineering Notes

**Đã hoàn thiện**:
- Logging hệ thống, `transform()` method cho inference, xử lý OOV categories, Unit tests (pytest) cho các module chính.
- Pipeline ETL/ELT tự động hóa toàn bộ quy trình extract → transform → train → load, với artifact tái tạo được (reproducible) thay vì file `.pkl` tĩnh nằm im trong repo.
- SQLite warehouse làm audit trail: lịch sử mỗi lần train (`pipeline_runs`) và nhật ký mọi lượt dự đoán thực tế (`predictions_log`).
- Giao diện Streamlit làm serving layer cho người dùng cuối, tách biệt rõ với phần nghiên cứu (`notebooks/`) và toolkit lõi (`src/`).

**Roadmap kỹ thuật**:
- Triển khai MLflow tracking để theo dõi phiên bản model và so sánh thí nghiệm khoa học một cách hệ thống hơn (hiện `metrics_summary.json` + SQLite mới chỉ đáp ứng nhu cầu cơ bản).
- Dockerize toàn bộ pipeline + app, triển khai CI/CD với GitHub Actions.
- Bổ sung FastAPI cho serving dạng API (Streamlit phục vụ tốt cho demo/người dùng cuối, nhưng không thay thế được một REST API cho hệ thống khác gọi vào).
- Đưa XGBoost/LightGBM (đang ở `03_modeling.ipynb`) vào `ModelTrainer.DEFAULT_MODELS` để pipeline tự động benchmark đủ cả 2 model này thay vì chỉ 4 model cơ bản.
