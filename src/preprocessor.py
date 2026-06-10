import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class AutoPreprocessor:
    """
    Pipeline tiền xử lý tự động: drop → fix → encode → feature_eng → scale → split.

    Hỗ trợ 2 chế độ split:
        - train/test (mặc định cũ):   val_size=0.0
        - train/val/test (mặc định):  val_size=0.1

    Cách dùng:
        from src.preprocessor import AutoPreprocessor

        # Train 80% | Val 10% | Test 10%
        prep = AutoPreprocessor(target_col='Test Results',
                                drop_cols=['Name', 'Doctor', 'Hospital'])
        X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)

        # Predict data mới:
        X_new = prep.transform(new_df)
    """

    def __init__(self, target_col, drop_cols=None,
                 val_size=0.1, test_size=0.1, random_state=42):
        self.target_col   = target_col
        self.drop_cols    = drop_cols or []
        self.val_size     = val_size    
        self.test_size    = test_size
        self.random_state = random_state

        self._le_dict     = {}        
        self._le_target   = LabelEncoder()
        self._scaler      = StandardScaler()
        self._feature_names = None
        self._cat_cols    = []

    def fit_transform(self, df):
        """
        Xử lý data và trả về splits đã scale.

        val_size > 0  → trả về X_train, X_val, X_test, y_train, y_val, y_test  (6 giá trị)
        val_size = 0  → trả về X_train, X_test, y_train, y_test                (4 giá trị)

        Fit tất cả encoder/scaler CHỈ trên train set để tránh data leakage.
        """
        df = df.copy()

        df = self._drop(df)

        df = self._fix_negatives(df)

        df = self._feature_engineering(df)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        use_stratify = y.nunique() <= 20

        if self.val_size > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if use_stratify else None
            )
            val_ratio_of_temp = self.val_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio_of_temp,
                random_state=self.random_state,
                stratify=y_temp if use_stratify else None
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if use_stratify else None
            )
            X_val, y_val = None, None

        self._cat_cols = X_train.select_dtypes(include='object').columns.tolist()
        for col in self._cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))
            for split in [X_val, X_test]:
                if split is not None:
                    split[col] = split[col].astype(str).apply(
                        lambda v: v if v in le.classes_ else le.classes_[0]
                    )
                    split[col] = le.transform(split[col])
            self._le_dict[col] = le

        y_train = self._le_target.fit_transform(y_train)
        y_test  = self._le_target.transform(y_test)
        if y_val is not None:
            y_val = self._le_target.transform(y_val)

        self._feature_names  = X_train.columns.tolist()
        X_train_sc = self._scaler.fit_transform(X_train)
        X_test_sc  = self._scaler.transform(X_test)
        X_val_sc   = self._scaler.transform(X_val) if X_val is not None else None


        n_total = len(X_train_sc) + (len(X_val_sc) if X_val_sc is not None else 0) + len(X_test_sc)
        print(f" Preprocessing xong!")
        print(f"   Tổng: {n_total:,} samples")
        print(f"   X_train : {X_train_sc.shape}  "
              f"({len(X_train_sc)/n_total*100:.0f}%)")
        if X_val_sc is not None:
            print(f"   X_val   : {X_val_sc.shape}   "
                  f"({len(X_val_sc)/n_total*100:.0f}%)")
        print(f"   X_test  : {X_test_sc.shape}   "
              f"({len(X_test_sc)/n_total*100:.0f}%)")
        print(f"   Features: {self._feature_names}")
        print(f"   Target classes: {list(self._le_target.classes_)}")

        if self.val_size > 0:
            return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test
        else:
            return X_train_sc, X_test_sc, y_train, y_test

    def transform(self, df):
        """Transform data mới để predict — dùng encoder/scaler đã fit."""
        df = df.copy()
        cols_drop = [c for c in self.drop_cols + [self.target_col] if c in df.columns]
        df.drop(columns=cols_drop, inplace=True)
        df = self._fix_negatives(df)
        df = self._feature_engineering(df)

        for col, le in self._le_dict.items():
            if col in df.columns:
                df[col] = df[col].astype(str).apply(
                    lambda v: v if v in le.classes_ else le.classes_[0]
                )
                df[col] = le.transform(df[col])

        for col in self._feature_names:
            if col not in df.columns:
                df[col] = 0
        df = df[self._feature_names]
        return self._scaler.transform(df)

    def _drop(self, df):
        cols = [c for c in self.drop_cols if c in df.columns]
        if cols:
            df.drop(columns=cols, inplace=True)
            print(f"     Dropped: {cols}")
        return df

    def _fix_negatives(self, df):
        """Clip giá trị âm trong các cột tiền tệ."""
        money_kw = ['billing', 'amount', 'price', 'cost', 'salary']
        for col in df.select_dtypes(include='number').columns:
            if any(kw in col.lower() for kw in money_kw) and (df[col] < 0).any():
                n = (df[col] < 0).sum()
                df[col] = df[col].clip(lower=0)
                print(f"    Clipped {n} giá trị âm → 0 trong '{col}'")
        return df

    def _feature_engineering(self, df):
        """Trích xuất features từ datetime columns."""
        if 'Date of Admission' in df.columns and 'Discharge Date' in df.columns:
            df['Date of Admission'] = pd.to_datetime(df['Date of Admission'])
            df['Discharge Date']    = pd.to_datetime(df['Discharge Date'])
            df['Length of Stay']    = (df['Discharge Date'] - df['Date of Admission']).dt.days
            df['Admission Month']   = df['Date of Admission'].dt.month
            df['Admission DayOfWeek'] = df['Date of Admission'].dt.dayofweek
            df.drop(columns=['Date of Admission', 'Discharge Date'], inplace=True)
        return df

    def get_feature_names(self):
        return self._feature_names

    def get_target_classes(self):
        return list(self._le_target.classes_)

    def decode_target(self, y_encoded):
        return self._le_target.inverse_transform(y_encoded)
