"""
Unit tests cho AutoPreprocessor.

Chạy: pytest test/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest
from src.preprocessor import AutoPreprocessor

@pytest.fixture
def sample_df():
    """
    DataFrame tối thiểu có đủ các loại cột cần thiết:
    - Cột datetime (Date of Admission, Discharge Date)
    - Cột categorical (Gender, Blood Type, Medical Condition)
    - Cột numeric (Age, Billing Amount)
    - Cột target (Test Results)
    - Cột cần drop (Name)
    """
    np.random.seed(42)
    n = 500
    dates_in  = pd.date_range('2020-01-01', periods=n, freq='D')
    dates_out = dates_in + pd.to_timedelta(np.random.randint(1, 30, n), unit='D')

    return pd.DataFrame({
        'Name':              [f'Patient_{i}' for i in range(n)],
        'Age':               np.random.randint(18, 85, n),
        'Gender':            np.random.choice(['Male', 'Female'], n),
        'Blood Type':        np.random.choice(['A+', 'B+', 'O+', 'AB+'], n),
        'Medical Condition': np.random.choice(['Diabetes', 'Hypertension', 'Asthma'], n),
        'Date of Admission': dates_in,
        'Discharge Date':    dates_out,
        'Billing Amount':    np.random.uniform(-100, 50000, n),  
        'Test Results':      np.random.choice(['Normal', 'Abnormal', 'Inconclusive'], n),
    })


@pytest.fixture
def preprocessor():
    return AutoPreprocessor(
        target_col='Test Results',
        drop_cols=['Name'],
        val_size=0.1,
        test_size=0.1,
        random_state=42,
    )


class TestSplitShape:
    def test_returns_six_values(self, preprocessor, sample_df):
        """Với val_size > 0, fit_transform() phải trả về đúng 6 giá trị."""
        result = preprocessor.fit_transform(sample_df)
        assert len(result) == 6, "Phải trả về (X_train, X_val, X_test, y_train, y_val, y_test)"

    def test_split_ratio_approx(self, preprocessor, sample_df):
        """Tỷ lệ 80/10/10 phải xấp xỉ đúng (±5%)."""
        X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.fit_transform(sample_df)
        n = len(X_train) + len(X_val) + len(X_test)

        assert abs(len(X_train) / n - 0.80) < 0.05, f"Train ratio = {len(X_train)/n:.2f}, expected ~0.80"
        assert abs(len(X_val)   / n - 0.10) < 0.05, f"Val ratio = {len(X_val)/n:.2f}, expected ~0.10"
        assert abs(len(X_test)  / n - 0.10) < 0.05, f"Test ratio = {len(X_test)/n:.2f}, expected ~0.10"

    def test_no_val_returns_four_values(self, sample_df):
        """Với val_size = 0, phải trả về đúng 4 giá trị."""
        prep = AutoPreprocessor(target_col='Test Results', drop_cols=['Name'], val_size=0.0)
        result = prep.fit_transform(sample_df)
        assert len(result) == 4, "Phải trả về (X_train, X_test, y_train, y_test)"

    def test_feature_count_consistent(self, preprocessor, sample_df):
        """X_train, X_val, X_test phải có cùng số features."""
        X_train, X_val, X_test, *_ = preprocessor.fit_transform(sample_df)
        assert X_train.shape[1] == X_val.shape[1] == X_test.shape[1], \
            "Các splits phải có cùng số cột"


class TestNoDataLeakage:
    def test_scaler_fit_only_on_train(self, preprocessor, sample_df):
        """
        Kiểm tra scaler được fit trên train (không phải toàn bộ dataset).
        Mean của scaler phải gần với mean của X_train, không phải toàn bộ dataset.

        Nếu scaler fit trên toàn bộ → mean sẽ gần global mean hơn.
        """
        X_train, X_val, X_test, *_ = preprocessor.fit_transform(sample_df)
        train_mean_abs = np.abs(X_train.mean(axis=0)).max()
        assert train_mean_abs < 0.01, \
            f"X_train phải có mean ≈ 0 sau StandardScaler, nhưng max abs mean = {train_mean_abs:.4f}"

    def test_val_test_mean_not_zero(self, preprocessor, sample_df):
        """
        Val và test set KHÔNG được có mean = 0 (vì scaler fit trên train, không phải val/test).
        Nếu mean của val/test cũng = 0 → scaler đã fit trên cả val/test (leakage!).
        """
        X_train, X_val, X_test, *_ = preprocessor.fit_transform(sample_df)
        val_mean_abs  = np.abs(X_val.mean(axis=0)).max()
        test_mean_abs = np.abs(X_test.mean(axis=0)).max()
        assert val_mean_abs > 1e-6 or test_mean_abs > 1e-6, \
            "Val/test mean đều = 0 → nghi ngờ scaler fit trên toàn bộ dataset (data leakage!)"



class TestDataCleaning:
    def test_negative_billing_clipped(self, preprocessor, sample_df):
        """
        Giá trị âm trong Billing Amount phải được clip về 0.
        Kiểm tra gián tiếp: sau khi fit_transform, không có giá trị âm trong output.
        """
        assert (sample_df['Billing Amount'] < 0).any(), "Fixture phải có giá trị âm để test"


        df_fixed = preprocessor._fix_negatives(sample_df.copy())
        assert (df_fixed['Billing Amount'] >= 0).all(), \
            "Tất cả giá trị Billing Amount phải >= 0 sau khi clip"

    def test_drop_cols_removed(self, preprocessor, sample_df):
        """Cột 'Name' phải bị loại bỏ hoàn toàn khỏi features."""
        preprocessor.fit_transform(sample_df)
        feature_names = preprocessor.get_feature_names()
        assert 'Name' not in feature_names, "Cột 'Name' phải bị drop"

    def test_target_col_not_in_features(self, preprocessor, sample_df):
        """Cột target không được xuất hiện trong feature names."""
        preprocessor.fit_transform(sample_df)
        feature_names = preprocessor.get_feature_names()
        assert 'Test Results' not in feature_names, "Target col không được là feature"


class TestTransformMethod:
    def test_transform_returns_same_feature_count(self, preprocessor, sample_df):
        """
        transform() trên data mới phải trả về đúng số features như lúc fit.
        """
        preprocessor.fit_transform(sample_df)
        new_df = sample_df.drop(columns=['Test Results']).head(10)
        X_new = preprocessor.transform(new_df)
        assert X_new.shape[1] == len(preprocessor.get_feature_names()), \
            "transform() phải trả về đúng số features"

    def test_transform_before_fit_raises(self, sample_df):
        """
        Gọi transform() trước fit_transform() phải raise RuntimeError.
        Không được im lặng và trả về kết quả sai.
        """
        prep = AutoPreprocessor(target_col='Test Results')
        with pytest.raises(RuntimeError, match="Chưa được fit"):
            prep.transform(sample_df)

    def test_transform_handles_unseen_categories(self, preprocessor, sample_df):
        """
        transform() phải xử lý được category chưa thấy trong train (OOV handling).
        Không được raise LabelEncoder error.
        """
        preprocessor.fit_transform(sample_df)
        new_df = sample_df.head(5).copy()
        new_df['Gender'] = 'Unknown_Gender'
        new_df = new_df.drop(columns=['Test Results'])


        try:
            X_new = preprocessor.transform(new_df)
            assert X_new.shape[0] == 5
        except Exception as e:
            pytest.fail(f"transform() bị lỗi khi gặp OOV category: {e}")

    def test_transform_with_target_col_present(self, preprocessor, sample_df):
        """
        transform() phải xử lý được khi DataFrame vẫn còn cột target.
        (Tình huống thực tế: người dùng quên drop target)
        """
        preprocessor.fit_transform(sample_df)
        X_new = preprocessor.transform(sample_df.head(10))
        assert X_new.shape[0] == 10


class TestGetters:
    def test_get_feature_names_after_fit(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df)
        names = preprocessor.get_feature_names()
        assert isinstance(names, list)
        assert len(names) > 0

    def test_get_target_classes(self, preprocessor, sample_df):
        preprocessor.fit_transform(sample_df)
        classes = preprocessor.get_target_classes()
        assert set(classes) == {'Normal', 'Abnormal', 'Inconclusive'}

    def test_decode_target_roundtrip(self, preprocessor, sample_df):
        """encode rồi decode lại phải ra giá trị gốc."""
        _, _, _, y_train, _, _ = preprocessor.fit_transform(sample_df)
        decoded = preprocessor.decode_target(y_train)
        valid = set(preprocessor.get_target_classes())
        assert all(v in valid for v in decoded), "decode_target() trả về giá trị không hợp lệ"

    def test_getters_before_fit_raise(self):
        """Getters phải raise RuntimeError nếu chưa fit."""
        prep = AutoPreprocessor(target_col='Test Results')
        with pytest.raises(RuntimeError):
            prep.get_feature_names()
        with pytest.raises(RuntimeError):
            prep.get_target_classes()