from pathlib import Path


BASE_DIR   = Path(__file__).resolve().parent.parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

RAW_CSV_PATH     = DATA_DIR / "healthcare_dataset.csv"
WAREHOUSE_DB_PATH = DATA_DIR / "warehouse.db"

PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"
METRICS_PATH       = MODELS_DIR / "metrics_summary.json"


MODEL_FILENAMES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree":       "decision_tree.pkl",
    "Random Forest":       "random_forest.pkl",
    "Gradient Boosting":   "gradient_boosting.pkl",
}
BEST_MODEL_PATH = MODELS_DIR / "best_model_random_forest.pkl"


TARGET_COL = "Test Results"
DROP_COLS  = ["Name", "Doctor", "Hospital", "Room Number"]

CLASS_NAMES = ["Abnormal", "Inconclusive", "Normal"]


GENDER_OPTIONS      = ["Male", "Female"]
BLOOD_TYPE_OPTIONS  = ["A+", "A-", "B+", "B-", "AB+", "AB-", "O+", "O-"]
CONDITION_OPTIONS   = ["Cancer", "Obesity", "Diabetes", "Asthma", "Hypertension", "Arthritis"]
INSURANCE_OPTIONS   = ["Blue Cross", "Medicare", "Aetna", "UnitedHealthcare", "Cigna"]
ADMISSION_OPTIONS   = ["Urgent", "Emergency", "Elective"]
MEDICATION_OPTIONS  = ["Paracetamol", "Ibuprofen", "Aspirin", "Penicillin", "Lipitor"]

RANDOM_STATE = 42
