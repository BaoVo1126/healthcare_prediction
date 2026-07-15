import logging
import time

from src.model_trainer import ModelTrainer
from .extract import Extractor
from .transform import Transformer
from .load import Loader
from .config import MODEL_FILENAMES, BEST_MODEL_PATH, MODELS_DIR

import joblib

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ETLPipeline:
    def __init__(self, source_path=None):
        self.extractor    = Extractor(source_path=source_path)
        self.transformer  = Transformer()
        self.loader       = Loader()

    def run(self, run_bootstrap: bool = False):
        t0 = time.time()
        logger.info("=" * 60)
        logger.info("  BẮT ĐẦU ETL PIPELINE — healthcare_ml")
        logger.info("=" * 60)

        df = self.extractor.extract()

     
        result = self.transformer.fit_transform(df)

  
        trainer = ModelTrainer()
        trainer.fit(
            result["X_train"], result["y_train"],
            result["X_test"], result["y_test"],
            result["X_val"], result["y_val"],
            run_bootstrap=run_bootstrap,
        )
        metrics_df = trainer.summary()

        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        for name, model in trainer.models.items():
            filename = MODEL_FILENAMES.get(name)
            if filename:
                joblib.dump(model, MODELS_DIR / filename)
                logger.info("Load: đã lưu model '%s' → %s", name, filename)

        best_model = trainer.get_best_model()
        joblib.dump(best_model, BEST_MODEL_PATH)
        logger.info("Load: đã lưu best model (%s) → %s", trainer.best_name, BEST_MODEL_PATH)

        self.loader.save_preprocessor(result["preprocessor"])
        self.loader.save_metrics(metrics_df.reset_index().to_dict(orient="records"))

        self.loader.log_pipeline_run(
            rows_extracted=len(df),
            split_shapes={
                "train": len(result["X_train"]),
                "val": len(result["X_val"]) if result["X_val"] is not None else 0,
                "test": len(result["X_test"]),
            },
            best_model=trainer.best_name,
            best_test_acc=trainer.results[trainer.best_name]["accuracy"],
        )

        elapsed = round(time.time() - t0, 1)
        logger.info("=" * 60)
        logger.info("  PIPELINE HOÀN TẤT trong %ss — best model: %s (acc=%.4f)",
                     elapsed, trainer.best_name, trainer.results[trainer.best_name]["accuracy"])
        logger.info("=" * 60)

        return {"trainer": trainer, "metrics": metrics_df, **result}


if __name__ == "__main__":
    ETLPipeline().run()
