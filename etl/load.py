import json
import logging
import sqlite3
from datetime import datetime, timezone

import joblib
import pandas as pd

from .config import WAREHOUSE_DB_PATH, PREPROCESSOR_PATH, METRICS_PATH, MODELS_DIR

logger = logging.getLogger(__name__)


class Loader:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or WAREHOUSE_DB_PATH)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS pipeline_runs (
                    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_at        TEXT NOT NULL,
                    rows_extracted INTEGER,
                    rows_train    INTEGER,
                    rows_val      INTEGER,
                    rows_test     INTEGER,
                    best_model    TEXT,
                    best_test_acc REAL,
                    status        TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions_log (
                    id            INTEGER PRIMARY KEY AUTOINCREMENT,
                    logged_at     TEXT NOT NULL,
                    input_json    TEXT NOT NULL,
                    predicted_class TEXT,
                    confidence    REAL,
                    model_used    TEXT
                )
            """)
            conn.commit()


    def save_preprocessor(self, preprocessor):
        joblib.dump(preprocessor, PREPROCESSOR_PATH)
        logger.info("Load: đã lưu preprocessor → %s", PREPROCESSOR_PATH)

    def save_metrics(self, metrics: dict):
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        logger.info("Load: đã lưu metrics tổng hợp → %s", METRICS_PATH)

    def log_pipeline_run(self, rows_extracted, split_shapes, best_model, best_test_acc, status="success"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO pipeline_runs
                   (run_at, rows_extracted, rows_train, rows_val, rows_test,
                    best_model, best_test_acc, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    rows_extracted,
                    split_shapes.get("train"), split_shapes.get("val"), split_shapes.get("test"),
                    best_model, best_test_acc, status,
                ),
            )
            conn.commit()
        logger.info("Load: đã ghi lịch sử pipeline run vào warehouse")


    def log_prediction(self, input_dict: dict, predicted_class: str, confidence: float, model_used: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO predictions_log
                   (logged_at, input_json, predicted_class, confidence, model_used)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(input_dict, ensure_ascii=False),
                    predicted_class, float(confidence), model_used,
                ),
            )
            conn.commit()

    def read_recent_runs(self, limit=10) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(
                f"SELECT * FROM pipeline_runs ORDER BY run_id DESC LIMIT {limit}", conn
            )

    def read_recent_predictions(self, limit=20) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql(
                f"SELECT * FROM predictions_log ORDER BY id DESC LIMIT {limit}", conn
            )
