"""
src — Healthcare ML Pipeline toolkit.

Modules:
    DataInspector    : kiểm tra chất lượng dữ liệu + χ² + MI tests
    AutoVisualizer   : tự động tạo toàn bộ EDA plots
    AutoPreprocessor : pipeline tiền xử lý end-to-end (fit/transform)
    ModelTrainer     : huấn luyện + đánh giá đa mô hình
"""
import logging


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

from .data_inspector import DataInspector
from .visualizer     import AutoVisualizer
from .preprocessor   import AutoPreprocessor
from .model_trainer  import ModelTrainer

__all__ = [
    "DataInspector",
    "AutoVisualizer",
    "AutoPreprocessor",
    "ModelTrainer",
]

__version__ = "1.1.0"