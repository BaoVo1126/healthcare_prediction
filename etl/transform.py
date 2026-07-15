import logging

from src.data_inspector import DataInspector
from src.preprocessor import AutoPreprocessor
from .config import TARGET_COL, DROP_COLS, RANDOM_STATE

logger = logging.getLogger(__name__)


class Transformer:

    def __init__(self, target_col: str = TARGET_COL, drop_cols: list = None):
        self.target_col = target_col
        self.drop_cols = drop_cols or DROP_COLS
        self.preprocessor = AutoPreprocessor(
            target_col=self.target_col,
            drop_cols=self.drop_cols,
            val_size=0.1,
            test_size=0.1,
            random_state=RANDOM_STATE,
        )

    def run_quality_check(self, df):
        logger.info("Transform: đang chạy data quality check...")
        try:
            DataInspector(df, target_col=self.target_col).run()
        except Exception as e:
            logger.warning("Transform: quality check gặp lỗi không nghiêm trọng: %s", e)

    def fit_transform(self, df):
        self.run_quality_check(df)

        X_train, X_val, X_test, y_train, y_val, y_test = self.preprocessor.fit_transform(df)

        logger.info(
            "Transform: hoàn tất — train=%s val=%s test=%s",
            X_train.shape, X_val.shape, X_test.shape,
        )

        return {
            "X_train": X_train, "X_val": X_val, "X_test": X_test,
            "y_train": y_train, "y_val": y_val, "y_test": y_test,
            "preprocessor": self.preprocessor,
        }
