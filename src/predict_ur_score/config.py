from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DATASET_PATH = DATA_DIR / "student-scores.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
RANDOM_STATE = 42
