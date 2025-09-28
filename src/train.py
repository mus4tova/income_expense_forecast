import pickle
import random
import numpy as np
import pandas as pd
from loguru import logger
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from src.model import Boosting, LinearRegr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.settings import STORAGE_PATH


class ModelTrainer:
    def __init__(self, dataset: pd.DataFrame, mode: str):
        self.dataset = dataset
        self.mode = mode

    def model_trainer(self):
        with open(f"{STORAGE_PATH}/list_of_unique_users.pkl", "rb") as file:
            self.unique_users = pickle.load(file)
        if self.mode == "Difference":
            target = "7_diff"
        else:
            target = "7"
        X_train, X_test, y_train, y_test = self.train_test_split(
            self.dataset, target=target
        )

        self.train_dataset = X_train, y_train
        self.test_dataset = X_test, y_test

        # 1
        model_lr = LinearRegr(self.train_dataset, self.mode).model_trainer()
        self.model_tester(model_lr, "LinReg")
        # 2
        model_ctb = Boosting(self.train_dataset, self.mode).model_trainer()
        self.model_tester(model_ctb, "CatBoost")

    def model_tester(
        self, model: [CatBoostRegressor | LinearRegression], model_type: str
    ):
        y_pred = model.predict(self.test_dataset[0])
        mse = mean_squared_error(self.test_dataset[1], y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.test_dataset[1], y_pred)
        r2 = r2_score(self.test_dataset[1], y_pred)

        logger.info(f"{self.mode} {model_type} Mean Squared Error: {mse}")
        logger.info(f"{self.mode} {model_type} Root Mean Squared Error: {rmse}")
        logger.info(f"{self.mode} {model_type} Mean Absolute Error: {mae}")
        logger.info(f"{self.mode} {model_type} R-squared: {r2}")

        self.regression_plot1(self.test_dataset[1], y_pred, model_type)
        self.regression_plot2(self.test_dataset[1], y_pred, model_type)

    def regression_plot1(self, y_test, y_pred, model_type: str):
        plt.figure(figsize=(6, 6))
        plt.scatter(y_test, y_pred, color="blue")
        plt.plot(
            [min(y_test), max(y_test)],
            [min(y_test), max(y_test)],
            color="red",
            linestyle="--",
        )
        plt.xlabel("Real values")
        plt.ylabel("Predicted values")
        plt.title("Real vs Pred")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"{STORAGE_PATH}/{self.mode}_{model_type}_1.png")

    def regression_plot2(
        self, y_test: pd.DataFrame, y_pred: pd.DataFrame, model_type: str
    ):
        plt.figure(figsize=(10, 6))
        # 100 points of real test data
        plt.plot([i for i in range(100)], y_test[:100])
        # 100 points of predicted data
        plt.plot([i for i in range(100)], y_pred[:100])
        plt.xlabel("Index of point")
        plt.ylabel("Pred/real values")
        plt.legend()
        plt.grid(True)
        # plt.show()
        plt.savefig(f"{STORAGE_PATH}/{self.mode}_{model_type}_2.png")

    def train_test_split(
        self, df: pd.DataFrame, target: str
    ) -> tuple[np.array, np.array, np.array, np.array]:
        # assign string names to all columns
        df.columns = df.columns.astype(str)

        # split users into train and test randomly
        split_size = int(len(self.unique_users) * 0.2)
        test_users = random.sample(self.unique_users, split_size)
        train_users = [item for item in self.unique_users if item not in test_users]

        # split dataset into train and test
        df_train = df[df["user_id"].isin(train_users)]
        df_test = df[df["user_id"].isin(test_users)]

        # split data into target and features
        X_train, X_test = df_train.drop(columns=[target, "user_id"]), df_test.drop(
            columns=[target, "user_id"]
        )
        y_train, y_test = df_train[target], df_test[target]
        return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
