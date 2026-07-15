"""
streamlit_app.py — Giao diện phục vụ (serving layer) cho healthcare_ml.

Vai trò trong toàn bộ hệ thống:
    notebooks/  → nơi khám phá & chứng minh cách tiếp cận đúng
    src/        → logic tái sử dụng (preprocessing, training) đã được test
    etl/        → tự động hoá lại toàn bộ quy trình đó, sinh artifact
    app/ (đây)  → nơi một người KHÔNG biết code (bác sĩ, quản lý bệnh viện,
                  nhà tuyển dụng xem portfolio) có thể tự nhập dữ liệu và
                  thấy ngay mô hình hoạt động ra sao — biến một pipeline
                  chạy ngầm thành một sản phẩm ai cũng dùng được.

Chạy:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from etl.config import (
    PREPROCESSOR_PATH, BEST_MODEL_PATH, METRICS_PATH,
    GENDER_OPTIONS, BLOOD_TYPE_OPTIONS, CONDITION_OPTIONS,
    INSURANCE_OPTIONS, ADMISSION_OPTIONS, MEDICATION_OPTIONS,
)
from etl.load import Loader

st.set_page_config(
    page_title="Healthcare ML — Dự đoán kết quả xét nghiệm",
    page_icon="🏥",
    layout="wide",
)


# ─── Cache tài nguyên nặng (model, preprocessor, loader) ────────────
@st.cache_resource
def load_artifacts():
    if not PREPROCESSOR_PATH.exists() or not BEST_MODEL_PATH.exists():
        return None, None
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    return preprocessor, model


@st.cache_resource
def get_loader():
    return Loader()


preprocessor, model = load_artifacts()
loader = get_loader()

st.title("🏥 Healthcare ML — Dự đoán kết quả xét nghiệm")

if preprocessor is None or model is None:
    st.error(
        "Chưa tìm thấy artifact (preprocessor.pkl / best_model_random_forest.pkl).\n\n"
        "Hãy chạy pipeline trước:  `python -m etl.pipeline`"
    )
    st.stop()

tab_predict, tab_overview, tab_monitor = st.tabs(
    ["🔮 Dự đoán bệnh nhân", "📊 Tổng quan mô hình", "🛠️ ETL Pipeline Monitor"]
)

with tab_predict:
    st.subheader("Nhập thông tin bệnh nhân")

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Tuổi", min_value=0, max_value=120, value=45)
        gender = st.selectbox("Giới tính", GENDER_OPTIONS)
        blood_type = st.selectbox("Nhóm máu", BLOOD_TYPE_OPTIONS)
        condition = st.selectbox("Bệnh lý", CONDITION_OPTIONS)

    with col2:
        insurance = st.selectbox("Bảo hiểm", INSURANCE_OPTIONS)
        admission_type = st.selectbox("Loại nhập viện", ADMISSION_OPTIONS)
        medication = st.selectbox("Thuốc điều trị", MEDICATION_OPTIONS)
        billing = st.number_input("Chi phí (Billing Amount, USD)", min_value=0.0, value=25000.0, step=500.0)

    with col3:
        admission_date = st.date_input("Ngày nhập viện", value=date.today() - timedelta(days=5))
        discharge_date = st.date_input("Ngày xuất viện", value=date.today())

    if discharge_date < admission_date:
        st.warning("Ngày xuất viện đang nhỏ hơn ngày nhập viện — vui lòng kiểm tra lại.")

    if st.button("Dự đoán", type="primary", use_container_width=True):
        input_dict = {
            "Age": age,
            "Gender": gender,
            "Blood Type": blood_type,
            "Medical Condition": condition,
            "Date of Admission": admission_date.isoformat(),
            "Insurance Provider": insurance,
            "Billing Amount": billing,
            "Admission Type": admission_type,
            "Discharge Date": discharge_date.isoformat(),
            "Medication": medication,
        }
        input_df = pd.DataFrame([input_dict])

        try:
            X = preprocessor.transform(input_df)
            proba = model.predict_proba(X)[0]
            classes = preprocessor.decode_target(np.arange(len(proba)))
            pred_idx = int(np.argmax(proba))
            pred_class = classes[pred_idx]
            confidence = float(proba[pred_idx])

            st.success(f"**Kết quả dự đoán: {pred_class}**  (độ tin cậy {confidence*100:.1f}%)")

            proba_df = pd.DataFrame({"Nhãn": classes, "Xác suất (%)": (proba * 100).round(1)})
            st.bar_chart(proba_df.set_index("Nhãn"))

            st.caption(
                "⚠️ Model được train trên bộ dữ liệu Kaggle tổng hợp (synthetic) "
                "không có tín hiệu y khoa thật mạnh — accuracy dao động quanh mốc "
                "random baseline (~33%). Vì vậy công cụ này chỉ mang tính minh "
                "hoạ kỹ thuật (pipeline + serving), KHÔNG phải công cụ chẩn đoán."
            )

            loader.log_prediction(input_dict, pred_class, confidence, model_used="Random Forest")

        except Exception as e:
            st.error(f"Lỗi khi dự đoán: {e}")


with tab_overview:
    st.subheader("So sánh các mô hình đã train")
    st.markdown(
        "Bảng dưới đây được sinh tự động mỗi lần pipeline chạy lại "
        "(`etl/pipeline.py` → `models/metrics_summary.json`), nên luôn phản "
        "ánh đúng lần train gần nhất — không có tình trạng số liệu trong "
        "README bị lệch so với model thật đang chạy."
    )

    if METRICS_PATH.exists():
        with open(METRICS_PATH, encoding="utf-8") as f:
            metrics = pd.DataFrame(json.load(f)).set_index("Model")
        st.dataframe(metrics, use_container_width=True)
        st.bar_chart(metrics[["Test Acc"]])
    else:
        st.info("Chưa có metrics — hãy chạy `python -m etl.pipeline` trước.")

    st.divider()
    st.markdown(
        "**Vì sao dự án này vẫn có giá trị dù accuracy chỉ ~37%?** "
        "Đây là bộ dữ liệu Kaggle được sinh ngẫu nhiên (synthetic), nên bản "
        "thân bài toán gần như không có tín hiệu để học — điều quan trọng "
        "không phải là con số accuracy, mà là việc chứng minh được: pipeline "
        "chống rò rỉ dữ liệu (data leakage), có kiểm định chéo (CV) để phát "
        "hiện overfitting, và có hạ tầng đưa model vào sử dụng thực tế "
        "(ETL + serving)"
    )

with tab_monitor:
    st.subheader("Lịch sử chạy pipeline")
    runs = loader.read_recent_runs(limit=10)
    if runs.empty:
        st.info("Chưa có lần chạy pipeline nào được ghi lại.")
    else:
        st.dataframe(runs, use_container_width=True)

    st.subheader("Nhật ký dự đoán gần đây (từ chính app này)")
    preds = loader.read_recent_predictions(limit=20)
    if preds.empty:
        st.info("Chưa có dự đoán nào được thực hiện trong tab 'Dự đoán bệnh nhân'.")
    else:
        st.dataframe(preds, use_container_width=True)

    st.caption(
        "Hai bảng trên đến từ `data/warehouse.db` (SQLite) — đây là phần "
        "'Load' của ETL/ELT pipeline: mọi lần chạy training và mọi lượt "
        "dự đoán của người dùng đều được nạp vào warehouse để có thể theo "
        "dõi hiệu năng model theo thời gian (model monitoring) sau này."
    )
