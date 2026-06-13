# 🏥 Healthcare ML Pipeline: From Data Diagnosis to Predictive Modeling
Dataset gốc: [Healthcare Dataset (Kaggle)](https://www.kaggle.com/code/likhithagudimetla/healthcare-dataset)

---

## 🤖 Introduction
Pipeline ML hướng đối tượng (OOP) tự động hóa quy trình tiền xử lý, kiểm định giả thuyết thống kê, và huấn luyện/đánh giá nhiều mô hình (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM) trên *55,500 hồ sơ bệnh án*. Trọng tâm: chẩn đoán chất lượng dữ liệu trước khi modeling, đảm bảo pipeline không leakage, và đóng gói thành toolkit (src/) sẵn sàng cho inference.

## ⚙️ Tech Stack
Python 3.10+, Scikit-Learn, XGBoost, LightGBM, SciPy, Pandas, NumPy, Matplotlib/Seaborn, Pytest.

---

## 📁 Project Structure
text
healthcare_ml/
├── data/healthcare_dataset.csv
├── notebooks/
│   ├── 01_eda.ipynb            # EDA, χ² test, Mutual Information
│   ├── 02_processing.ipynb     # Preprocessing & split 80/10/10
│   └── 03_modeling.ipynb       # Training & stability analysis
├── src/
│   ├── data_inspector.py       # Quality check + χ² + Mutual Information
│   ├── visualizer.py           # EDA plots + Seasonality
│   ├── preprocessor.py         # drop→split→feature_eng→encode→scale→transform()
│   └── model_trainer.py        # Multi-model training, Confusion Matrix, CV-Gap
└── test/test_preprocessor.py   # pytest: leakage, split ratio, transform(), OOV

---

## 📊 Data Diagnosis & Hypothesis Testing
- *Feature engineering*: drop biến hành chính nhiễu (Name, Doctor, Hospital, Room Number); trích xuất Length_of_Stay, Admission_Month/DayOfWeek, Long_Stay_Flag, Age_Group.
- *Anomalies*: 108 dòng billing âm phi lý (clip → 0), 534 dòng trùng lặp (drop_duplicates). Boxplot ban đầu không phát hiện vì phân phối đều.
- *Seasonality*: dữ liệu 5 năm (2019–2024) phẳng lỳ, không có pattern theo mùa.
- *χ² Independence Test*: tất cả features có p-value > 0.05 với target.
- *Mutual Information* (bổ sung): tất cả MI scores ≈ 0 — xác nhận cả non-linear dependency cũng không tồn tại.

🚨 *Insight*: 3 phương pháp độc lập (phân phối đều, χ², MI) đều khẳng định nhãn target bị gán ngẫu nhiên (*Pure Noise*) — dấu hiệu điển hình của synthetic data.

<img width="828" height="273" alt="image" src="https://github.com/user-attachments/assets/ce6a5c36-f93c-4e1c-a5d7-af0028fcbf89" />

<!-- 📷 Chèn ảnh: chi2 p-value + Mutual Information bar charts -->

---

## 🧪 CASE 1 (Thô) vs CASE 2 (Sạch)

| Thuật toán | CASE 1 (Acc/F1) | CASE 2 (Acc/F1) | Δ Accuracy |
| :--- | :---: | :---: | :---: |
| Logistic Regression | 33.2% / 0.33 | 33.6% / 0.33 | +0.4% |
| Decision Tree | 33.1% / 0.32 | 33.1% / 0.32 | 0.0% |
| *Random Forest* | *44.5% / 0.44* | *43.8% / 0.44* | -0.7% |
| Gradient Boosting | 32.6% / 0.32 | 33.1% / 0.33 | +0.5% |

Thay đổi <1% sau khi clean → khẳng định dữ liệu thiếu tính nhân quả y khoa. Sụt giảm nhẹ của Random Forest có thể do mất "Implicit Oversampling" từ duplicate rows (giả thuyết, chưa isolate bằng `bootstrap=False`).


---

## 🚀 Model Pipeline & Benchmarking

*Inference flow*: AutoPreprocessor = drop → split → feature_eng → encode → scale, split-first-then-fit để loại trừ leakage (kể cả từ feature engineering). prep.transform(new_df) dùng cho inference — không re-fit, OOV category map về most-frequent value.

*Kết quả thống nhất* (sau feature engineering, 80/10/10):

| Mô hình | Test Acc | Test F1 | CV Mean ± Std | CV-Test Gap |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 34.2% | 0.331 | 33.8% ± 0.6% | -0.4% |
| XGBoost | 37.1% | 0.369 | 36.9% ± 0.8% | -0.2% |
| *Random Forest* | *43.7%* | *0.434* | *42.9% ± 0.7%* | *-0.8%* |
🏆 **Best Model**: Random Forest (43.7%, +9.2% vs baseline) nhờ feature engineering (`Age_Group`, `Long_Stay_Flag`, `Length_of_Stay`...). CV-Test Gap nhỏ (<1%) → không overfitting ẩn, nhưng với data Pure Noise, gain này phản ánh model học spurious correlation trong train set cụ thể, không phải "hiểu y khoa".
.

Random baseline (3 class) = 33.3%.


*Stability & Diagnostics*:
- 5-Fold Stratified CV với clone(model) mỗi fold.
- print_cv_gap_report() — flag nếu CV-Test gap > 3%.
- plot_bootstrap_curves() — learning curve mean ± std theo % data.
- plot_confusion_matrices() + print_classification_reports() — kiểm tra model có bias dự đoán lệch class hay không (kết quả: F1-Macro ≈ Accuracy → dự đoán tương đối đồng đều).

<!-- 📷 Chèn ảnh: plot_comparison(), plot_bootstrap_curves(), plot_confusion_matrices() -->

<img width="619" height="173" alt="image" src="https://github.com/user-attachments/assets/cd5e6d73-5204-4ba7-ad3b-699aae229ee6" />

<img width="619" height="221" alt="image" src="https://github.com/user-attachments/assets/40bb044f-e591-47ca-9add-7b2e6470f250" />


<img width="619" height="221" alt="image" src="https://github.com/user-attachments/assets/63233410-f876-46fd-8fc0-7db14ca536fe" />

---

## 💻 Usage

from src import AutoPreprocessor, ModelTrainer, DataInspector
import pandas as pd

df = pd.read_csv('data/healthcare_dataset.csv')

# 1. Chẩn đoán dữ liệu
inspector = DataInspector(df, target_col='Test Results')
inspector.run()
inspector.run_chi2_independence()
inspector.run_mutual_information()

# 2. Preprocessing (80/10/10)
prep = AutoPreprocessor(
    target_col='Test Results',
    drop_cols=['Name', 'Doctor', 'Hospital', 'Room Number'],
)
X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)

# 3. Training & evaluation
trainer = ModelTrainer()
trainer.fit(X_train, y_train, X_test, y_test, X_val, y_val)
trainer.summary()
trainer.print_cv_gap_report()
trainer.plot_comparison()
trainer.plot_confusion_matrices(y_test, class_names=prep.get_target_classes())

# 4. Inference trên data mới — không re-fit
X_new = prep.transform(pd.read_csv('data/new_patients.csv'))

*Cài đặt*:
git clone https://github.com/BaoVo1126/healthcare_prediction.git
cd healthcare_prediction
pip install -r requirements.txt
pytest test/ -v

---

## 🏗️ Engineering Notes

*Đã làm*: logging có cấu trúc (thay print()), transform() tách biệt fit/inference, OOV handling theo most-frequent, no-leakage by design (split trước fit), unit tests cho leakage/split ratio/transform, requirements.txt pin version.

*Future work*: MLflow/W&B tracking, Dockerizing, CI/CD (GitHub Actions), FastAPI /predict endpoint, cloud deployment, input schema validation, model versioning.

---
