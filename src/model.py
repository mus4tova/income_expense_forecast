import joblib
import optuna
import numpy as np
from loguru import logger
from catboost import CatBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.settings import MODELS_PATH


class Boosting:
    model: CatBoostRegressor

    def __init__(self, train_dataset, mode):
        self.mode = mode
        self.train_dataset = train_dataset

    def objective(self, trial):
        """Optuna optimization function"""
        X_train, y_train = self.train_dataset

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.1, random_state=0
        )

        # suggest hypermarameters
        learning_rate = trial.suggest_uniform("learning_rate", 0.001, 0.1)
        depth = trial.suggest_int("depth", 3, 12)
        min_data_in_leaf = trial.suggest_int("min_data_in_leaf", 10, 100)

        model = CatBoostRegressor(
            iterations=500,
            learning_rate=learning_rate,
            depth=depth,
            min_data_in_leaf=min_data_in_leaf,
            loss_function="RMSE",
            verbose=0,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), use_best_model=True)
        rmse = model.best_score_["validation"]["RMSE"]
        return rmse

    def model_trainer(self) -> CatBoostRegressor:
        # start optimization
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=1)

        best_params = study.best_params

        logger.info(f"Best hyperparams: {best_params}")

        # training model with best params
        X_train, y_train = self.train_dataset
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=best_params["learning_rate"],
            depth=best_params["depth"],
            min_data_in_leaf=best_params["min_data_in_leaf"],
            loss_function="RMSE",
            verbose=50,
        )
        model.fit(X_train, y_train, eval_set=(X_train, y_train), use_best_model=True)
        self.save_model(model)
        return model

    def save_model(self, model: CatBoostRegressor):
        model.save_model(f"{MODELS_PATH}/CatBoost_{self.mode}.cbm")


class LinearRegr:
    model: LinearRegression

    def __init__(self, train_dataset: tuple[np.array, np.array], mode: str):
        self.mode = mode
        self.train_dataset = train_dataset

    def model_trainer(self) -> LinearRegression:
        X_train, y_train = self.train_dataset
        model = LinearRegression()
        model.fit(X_train, y_train)
        self.save_model(model)
        return model

    def save_model(self, model: LinearRegression):
        joblib.dump(model, f"{MODELS_PATH}/LinReg_{self.mode}.joblib")
