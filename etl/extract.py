import logging
import pandas as pd

from .config import RAW_CSV_PATH, TARGET_COL

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Age", "Gender", "Blood Type", "Medical Condition",
    "Date of Admission", "Insurance Provider", "Billing Amount",
    "Admission Type", "Discharge Date", "Medication", TARGET_COL,
]


class Extractor:
    def __init__(self, source_path=None):
        self.source_path = source_path or RAW_CSV_PATH

    def extract(self) -> pd.DataFrame:
        logger.info("Extract: đang đọc dữ liệu từ %s", self.source_path)
        df = pd.read_csv(self.source_path)
        self._validate_schema(df)
        logger.info(
            "Extract: hoàn tất — %d dòng, %d cột", df.shape[0], df.shape[1]
        )
        return df

    def _validate_schema(self, df: pd.DataFrame):
        if df.empty:
            raise ValueError("Extract: nguồn dữ liệu rỗng.")

        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Extract: thiếu các cột bắt buộc: {missing}")

        logger.info("Extract: schema hợp lệ (%d cột bắt buộc đều có mặt)", len(REQUIRED_COLUMNS))
