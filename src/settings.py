import os.path
from pathlib import Path

BASE_PATH = Path(__file__).resolve().parent.parent


# MLflow
# TRACKING_URI = settings.MLFLOW_TRACKING_URI

# folder for data store
DATA_PATH = BASE_PATH / "data"

# folder for saving models
MODELS_PATH = BASE_PATH / "models"
if not os.path.isdir(MODELS_PATH):
    os.mkdir(MODELS_PATH)

# folder for saving others
STORAGE_PATH = BASE_PATH / "storage"
if not os.path.isdir(STORAGE_PATH):
    os.mkdir(STORAGE_PATH)

# folder for saving predictions
PRED_PATH = BASE_PATH / "predictions"
if not os.path.isdir(PRED_PATH):
    os.mkdir(PRED_PATH)
