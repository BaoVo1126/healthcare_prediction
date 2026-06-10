import pytest
import pandas as pd
import numpy as np
from src.preprocessor import AutoPreprocessor

@pytest.fixture
def sample_df():
    data = {
        'Name': ['A', 'B', 'C', 'D'],
        'Doctor': ['Dr.X', 'Dr.Y', 'Dr.X', 'Dr.Z'],
        'Hospital': ['H1', 'H2', 'H1', 'H2'],
        'Billing Amount': [100, -50, 200, -10],
        'Date of Admission': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
        'Discharge Date': ['2023-01-05', '2023-01-05', '2023-01-05', '2023-01-05'],
        'Test Results': ['Positive', 'Negative', 'Positive', 'Negative']
    }
    return pd.DataFrame(data)

def test_drop_cols(sample_df):
    prep = AutoPreprocessor(target_col='Test Results', drop_cols=['Name', 'Doctor'])
    X_train, X_test, y_train, y_test = prep.fit_transform(sample_df)
    
    assert 'Name' not in prep.get_feature_names()
    assert 'Doctor' not in prep.get_feature_names()

def test_fix_negatives(sample_df):
    prep = AutoPreprocessor(target_col='Test Results')
    df_fixed = prep._fix_negatives(sample_df.copy())
    
    assert (df_fixed['Billing Amount'] >= 0).all(), "Vẫn còn giá trị âm sau khi xử lý!"

def test_output_shapes(sample_df):
    prep = AutoPreprocessor(target_col='Test Results', val_size=0.25, test_size=0.25)
    X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(sample_df)
    
    assert X_train.shape[0] == 2
    assert X_val.shape[0] == 1
    assert X_test.shape[0] == 1