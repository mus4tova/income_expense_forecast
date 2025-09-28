import pickle
import joblib
import numpy as np
from catboost import CatBoostRegressor

from src.settings import STORAGE_PATH, MODELS_PATH, PRED_PATH


class ModelPredictor:
    def __init__(self, dataset, mode):
        self.dataset = dataset
        self.mode = mode

    def predictor(self):
        with open(f"{STORAGE_PATH}/list_of_unique_users.pkl", "rb") as file:
            self.unique_users = pickle.load(file)

        loaded_scaler = joblib.load(
            f"{STORAGE_PATH}/scaler_column_amount_n26_currency.joblib"
        )

        if self.mode == "Difference":
            drop_col = "2_diff"
        else:
            drop_col = "2"
        X_pred = np.array(self.dataset.drop(columns=[drop_col, "user_id"]))

        model_ctb = CatBoostRegressor()
        model_ctb.load_model(f"{MODELS_PATH}/CatBoost_{self.mode}.cbm")
        model_lr = joblib.load(f"{MODELS_PATH}/LinReg_{self.mode}.joblib")

        y_pred_ctb = model_ctb.predict(X_pred)
        y_pred_ctb_scaled = loaded_scaler.inverse_transform(
            y_pred_ctb.reshape(-1, 1)
        ).flatten()
        self.save_predictions(y_pred_ctb_scaled, "CatBoost")

        y_pred_lr = model_lr.predict(X_pred)
        y_pred_lr_scaled = loaded_scaler.inverse_transform(
            y_pred_lr.reshape(-1, 1)
        ).flatten()
        self.save_predictions(y_pred_lr_scaled, "LinReg")

    def save_predictions(self, predictions: np.array, model_type: str):
        with open(f"{PRED_PATH}/{model_type}_{self.mode}.txt", "w") as f:
            for prediction, user in zip(predictions, self.unique_users):
                f.write(f"{user} {prediction}\n")
