"""
AutoPreprocessor — pipeline tiền xử lý end-to-end cho healthcare dataset.

Thứ tự xử lý:
    drop → fix_negatives → split → feature_engineering (fit on train only)
    → encode (fit on train only) → scale (fit on train only)

Lý do feature_engineering được thực hiện SAU khi split:
    Nếu feature mới phụ thuộc vào thống kê của toàn bộ dataset (mean, median,
    group stats), việc tính trước khi split sẽ gây data leakage — thông tin
    từ val/test "rò rỉ" vào train. Tách ra sau split và fit_on_train đảm bảo
    pipeline hoàn toàn clean.
"""

import logging
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class AutoPreprocessor:
    """
    Pipeline tiền xử lý tự động: drop → fix → split → feature_eng → encode → scale.

    Hỗ trợ 2 chế độ split:
        - train/val/test (mặc định): val_size=0.1, test_size=0.1  → 80/10/10
        - train/test:                val_size=0.0, test_size=0.2  → 80/20

    Ví dụ sử dụng:
        from src.preprocessor import AutoPreprocessor

        prep = AutoPreprocessor(
            target_col='Test Results',
            drop_cols=['Name', 'Doctor', 'Hospital', 'Room Number']
        )

        # Fit + split (trả về 6 giá trị khi val_size > 0)
        X_train, X_val, X_test, y_train, y_val, y_test = prep.fit_transform(df)

        # Inference trên data mới — chỉ transform, không re-fit
        X_new = prep.transform(new_df)

        # Xem tên features và target classes
        print(prep.get_feature_names())
        print(prep.get_target_classes())
    """

    def __init__(
        self,
        target_col: str,
        drop_cols: list = None,
        val_size: float = 0.1,
        test_size: float = 0.1,
        random_state: int = 42,
    ):
        """
        Khởi tạo preprocessor.

        Parameters
        ----------
        target_col   : tên cột nhãn mục tiêu
        drop_cols    : danh sách cột cần loại (định danh hành chính)
        val_size     : tỷ lệ validation set (0.0 → bỏ qua val set)
        test_size    : tỷ lệ test set
        random_state : seed để reproducible
        """
        self.target_col   = target_col
        self.drop_cols    = drop_cols or []
        self.val_size     = val_size
        self.test_size    = test_size
        self.random_state = random_state

        self._le_dict       = {}         
        self._le_target     = LabelEncoder()
        self._scaler        = StandardScaler()
        self._feature_names = None     
        self._cat_cols      = []       
        self._most_freq_cat = {}      

        self._is_fitted = False           


    def fit_transform(self, df: pd.DataFrame):
        """
        Xử lý toàn bộ DataFrame và trả về các splits đã scale.

        Thứ tự: drop → fix_negatives → split → feature_engineering → encode → scale
        Tất cả encoder/scaler đều CHỈ được fit trên train set.

        Returns
        -------
        Nếu val_size > 0: (X_train, X_val, X_test, y_train, y_val, y_test)
        Nếu val_size = 0: (X_train, X_test, y_train, y_test)
        """
        df = df.copy()

        df = self._drop(df)

        df = self._fix_negatives(df)

        n_before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        n_dup = n_before - len(df)
        if n_dup > 0:
            logger.info("Đã loại %d duplicate rows", n_dup)

        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        use_stratify = y.nunique() <= 20

        if self.val_size > 0:
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if use_stratify else None,
            )
            val_ratio = self.val_size / (1.0 - self.test_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_ratio,
                random_state=self.random_state,
                stratify=y_temp if use_stratify else None,
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=self.test_size,
                random_state=self.random_state,
                stratify=y if use_stratify else None,
            )
            X_val, y_val = None, None

        X_train = self._feature_engineering_fit_transform(X_train)
        X_test  = self._feature_engineering_transform(X_test)
        if X_val is not None:
            X_val = self._feature_engineering_transform(X_val)


        self._cat_cols = X_train.select_dtypes(include='object').columns.tolist()
        for col in self._cat_cols:
            le = LabelEncoder()
            X_train[col] = le.fit_transform(X_train[col].astype(str))

            self._most_freq_cat[col] = le.classes_[
                np.bincount(le.transform(le.classes_)).argmax()
            ]

            for split in [X_val, X_test]:
                if split is not None:
                    split[col] = self._encode_with_oov(split[col], le, col)

            self._le_dict[col] = le

        y_train = self._le_target.fit_transform(y_train)
        y_test  = self._le_target.transform(y_test)
        if y_val is not None:
            y_val = self._le_target.transform(y_val)

        self._feature_names = X_train.columns.tolist()
        X_train_sc = self._scaler.fit_transform(X_train)
        X_test_sc  = self._scaler.transform(X_test)
        X_val_sc   = self._scaler.transform(X_val) if X_val is not None else None

        self._is_fitted = True


        n_total = len(X_train_sc) + (len(X_val_sc) if X_val_sc is not None else 0) + len(X_test_sc)
        logger.info("Preprocessing hoàn tất — tổng %d samples", n_total)
        self._print_split_summary(X_train_sc, X_val_sc, X_test_sc, n_total)

        if self.val_size > 0:
            return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test
        return X_train_sc, X_test_sc, y_train, y_test


    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Apply encoder/scaler đã fit lên DataFrame mới để inference.

        KHÔNG re-fit bất kỳ thứ gì — chỉ transform.
        Phù hợp để gọi trong serving/prediction pipeline.

        Parameters
        ----------
        df : DataFrame mới, có thể có hoặc không có cột target

        Returns
        -------
        np.ndarray — shape (n_samples, n_features), đã scale
        """
        if not self._is_fitted:
            raise RuntimeError(
                "AutoPreprocessor chưa được fit. Hãy gọi fit_transform() trước."
            )

        df = df.copy()

        cols_to_drop = [c for c in self.drop_cols + [self.target_col] if c in df.columns]
        if cols_to_drop:
            df.drop(columns=cols_to_drop, inplace=True)

        df = self._fix_negatives(df)
        df = self._feature_engineering_transform(df)

        for col, le in self._le_dict.items():
            if col in df.columns:
                df[col] = self._encode_with_oov(df[col], le, col)

        for col in self._feature_names:
            if col not in df.columns:
                logger.warning("Feature '%s' bị thiếu — điền 0", col)
                df[col] = 0
        df = df[self._feature_names]

        return self._scaler.transform(df)


    def _feature_engineering_fit_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Tạo features mới và lưu thống kê cần thiết từ train set.
        Với date features, không cần fit stats → gọi thẳng transform.
        """
        return self._feature_engineering_transform(X)

    def _feature_engineering_transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Áp dụng feature engineering — chỉ dùng thông tin của chính sample đó."""
        X = X.copy()

        if 'Date of Admission' in X.columns and 'Discharge Date' in X.columns:
            X['Date of Admission'] = pd.to_datetime(X['Date of Admission'], errors='coerce')
            X['Discharge Date']    = pd.to_datetime(X['Discharge Date'], errors='coerce')

            X['Length_of_Stay']      = (X['Discharge Date'] - X['Date of Admission']).dt.days
            X['Admission_Month']     = X['Date of Admission'].dt.month
            X['Admission_DayOfWeek'] = X['Date of Admission'].dt.dayofweek
            X['Long_Stay_Flag']      = (X['Length_of_Stay'] > 15).astype(int)

            X.drop(columns=['Date of Admission', 'Discharge Date'], inplace=True)

        if 'Age' in X.columns:
            X['Age_Group'] = pd.cut(
                X['Age'],
                bins=[0, 18, 35, 55, 75, 120],
                labels=[0, 1, 2, 3, 4],
            ).astype(float)

        return X


    def _drop(self, df: pd.DataFrame) -> pd.DataFrame:
        """Loại bỏ các cột định danh hành chính."""
        cols = [c for c in self.drop_cols if c in df.columns]
        if cols:
            df = df.drop(columns=cols)
            logger.info("Đã drop các cột: %s", cols)
        return df

    def _fix_negatives(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clip giá trị âm phi lý trong các cột tài chính về 0.
        Phát hiện tự động dựa trên tên cột.
        """
        money_kw = ['billing', 'amount', 'price', 'cost', 'salary', 'fee', 'charge']
        for col in df.select_dtypes(include='number').columns:
            if any(kw in col.lower() for kw in money_kw):
                n_neg = (df[col] < 0).sum()
                if n_neg > 0:
                    df[col] = df[col].clip(lower=0)
                    logger.info("Clipped %d giá trị âm → 0 trong cột '%s'", n_neg, col)
        return df

    def _encode_with_oov(self, series: pd.Series, le: LabelEncoder, col: str) -> pd.Series:
        """
        Encode một Series với xử lý OOV (Out-Of-Vocabulary).

        Chiến lược: map giá trị lạ về most-frequent class trong train set,
        thay vì classes_[0] (alphabetical first) — ít gây bias hơn.
        """
        known = set(le.classes_)
        most_freq = self._most_freq_cat.get(col, le.classes_[0])

        series = series.astype(str).apply(
            lambda v: v if v in known else most_freq
        )
        return le.transform(series)

    def _print_split_summary(self, X_train, X_val, X_test, n_total):
        """In bảng tóm tắt kích thước các splits."""
        print("\n" + "=" * 50)
        print("  Preprocessing xong!")
        print("=" * 50)
        print(f"  Tổng samples : {n_total:,}")
        print(f"  X_train      : {X_train.shape}  ({len(X_train)/n_total*100:.0f}%)")
        if X_val is not None:
            print(f"  X_val        : {X_val.shape}   ({len(X_val)/n_total*100:.0f}%)")
        print(f"  X_test       : {X_test.shape}   ({len(X_test)/n_total*100:.0f}%)")
        print(f"  Features     : {self._feature_names}")
        print(f"  Target classes: {list(self._le_target.classes_)}")
        print("=" * 50 + "\n")


    def get_feature_names(self) -> list:
        """Trả về danh sách tên features sau khi fit."""
        if not self._is_fitted:
            raise RuntimeError("Chưa fit. Hãy gọi fit_transform() trước.")
        return self._feature_names

    def get_target_classes(self) -> list:
        """Trả về danh sách nhãn target gốc (trước khi encode)."""
        if not self._is_fitted:
            raise RuntimeError("Chưa fit. Hãy gọi fit_transform() trước.")
        return list(self._le_target.classes_)

    def decode_target(self, y_encoded: np.ndarray) -> np.ndarray:
        """Chuyển ngược nhãn encoded về dạng string gốc."""
        return self._le_target.inverse_transform(y_encoded)