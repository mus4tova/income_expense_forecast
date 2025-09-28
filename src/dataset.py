import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.settings import DATA_PATH, STORAGE_PATH


class DataPreprocessor:
    def __init__(self, mode: str):
        self.mode = mode
        self.scale = True
        self.common_processing()

    def common_processing(self):
        self.base = self.create_base_dataset()
        if self.mode != "Difference":
            self.in_ = self.create_subsample(self.base, "Income")
            self.out_ = self.create_subsample(self.base, "Outcome")
            self.combined = self.create_combined_dataset(self.in_, self.out_)

    def create_base_dataset(self) -> pd.DataFrame:
        df = self.merge_three_into_one()
        self.list_of_unique_users(df)
        df = self.encode_categorical_features(df)
        df = df.drop(
            columns=[
                "dataset_transaction",
                "dataset_user",
                "transaction_date",
                "mcc_group",
            ]
        )
        if self.scale:
            df = self.scaling(df)
        return df

    def create_income_dataset(self) -> pd.DataFrame:
        in_ = self.in_.merge(self.combined, on="user_id", how="outer")
        in_ = self.feature_engeneering2(in_)
        return in_

    def create_outcome_dataset(self) -> pd.DataFrame:
        out_ = self.out_.merge(self.combined, on="user_id", how="outer")
        out_ = self.feature_engeneering2(out_)
        return out_

    def create_difference_dataset(self) -> pd.DataFrame:
        # group the amount by user_id, by month, by direction (income and expenses) and sum them
        grouped_by_month_dir = (
            self.base[["user_id", "month", "direction", "amount_n26_currency"]]
            .groupby(["user_id", "month", "direction"])
            .agg("sum")
        )

        # convert them into rows
        diff_ = grouped_by_month_dir.pivot_table(
            index="user_id",
            columns=["month", "direction"],
            values="amount_n26_currency",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        # create a new column with the difference
        for month in [2, 3, 4, 5, 6, 7]:
            diff_[f"{month}_diff"] = diff_[(month, "In")] - diff_[(month, "Out")]

        # keep only user_id and the new difference columns
        diff_ = diff_[["user_id"] + [f"{month}_diff" for month in [2, 3, 4, 5, 6, 7]]]

        # get rid on several column levels
        diff_.columns = ["_".join(map(str, col)).strip() for col in diff_.columns]

        # group the rest "types" columns
        types_df = (
            self.base.groupby(["user_id"])
            .agg("sum")
            .drop(columns=["month"])
            .drop(columns=["amount_n26_currency", "direction"])
        )

        # merge them into one dataframe
        diff_ = diff_.merge(
            types_df, left_on="user_id_", right_on="user_id", how="outer"
        )

        diff_.columns = diff_.columns.str.rstrip("_")
        return diff_

    def create_combined_dataset(
        self, in_: pd.DataFrame, out_: pd.DataFrame
    ) -> pd.DataFrame:
        combined = in_.merge(out_, on="user_id", how="outer")
        combined.columns = combined.columns.str.replace("_x", "_out", regex=True)
        combined.columns = combined.columns.str.replace("_y", "_in", regex=True)
        for month in [2, 3, 4, 5, 6]:
            combined[f"in_out_{month}"] = (
                combined[f"{month}_in"] - combined[f"{month}_out"]
            )

        combined = combined[
            ["user_id", "in_out_2", "in_out_3", "in_out_4", "in_out_5", "in_out_6"]
        ]
        return combined

    def create_subsample(self, df: pd.DataFrame, mode: str) -> pd.DataFrame:
        if mode == "Outcome":
            dir = "Out"
        elif mode == "Income":
            dir = "In"
        slice = df[df.direction == dir].drop(columns=["direction"])
        res = self.data_cleaning(slice)
        res = self.feature_engeneering(res)
        return res

    def load_data_from_files(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        main = pd.read_csv(f"{DATA_PATH}/2016-09-19_79351_training.csv")
        mcc = pd.read_csv(f"{DATA_PATH}/mcc_group_definition.csv")
        types = pd.read_csv(f"{DATA_PATH}/transaction_types.csv")
        return main, mcc, types

    def scaling(self, df: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        # scale only numeric columns
        numeric_columns = df.select_dtypes(include=["number"]).columns.drop("month")
        for col in numeric_columns:
            df[col] = scaler.fit_transform(np.array(df[col]).reshape(-1, 1))
            if col == "amount_n26_currency":
                joblib.dump(scaler, f"{STORAGE_PATH}/scaler_column_{col}.joblib")
        return df

    def list_of_unique_users(self, df: pd.DataFrame):
        self.unique_users = pd.DataFrame(df["user_id"].unique(), columns=["user_id"])
        with open(f"{STORAGE_PATH}/list_of_unique_users.pkl", "wb") as file:
            unique_users_list = [user[0] for user in self.unique_users.values.tolist()]
            pickle.dump(unique_users_list, file)

    def merge_three_into_one(self) -> pd.DataFrame:
        main, mcc, types = self.load_data_from_files()
        df = (
            main.merge(types, left_on="transaction_type", right_on="type", how="left")
            .drop(columns=["type", "transaction_type"])
            .rename(columns={"explanation": "transaction_type"})
        )
        df = df.merge(mcc, on="mcc_group", how="left")
        df["mcc_group"] = df["mcc_group"].fillna(0)
        df["explanation"] = df["explanation"].fillna("non-card")
        df = df.rename(columns={"explanation": "mcc_explanation"})
        df["transaction_date"] = pd.to_datetime(df["transaction_date"])
        df["month"] = df["transaction_date"].dt.month
        return df

    def encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # transform the categorical data using one-hot encoding, as all categories are equally important
        one_hot_features = ["transaction_type", "agent", "mcc_explanation"]
        df = pd.get_dummies(df, columns=one_hot_features, dtype=int)
        return df

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        # group the income (or expenses) by user_id, month and sum them
        grouped_by_month = (
            df[["user_id", "month", "amount_n26_currency"]]
            .groupby(["user_id", "month"])
            .agg("sum")
        )

        # convert them into rows
        res = grouped_by_month.pivot_table(
            index="user_id",
            columns="month",
            values="amount_n26_currency",
            aggfunc="sum",
            fill_value=0,
        ).reset_index()

        # other features will be called "types"
        # sum the frequency of each type occurrence over the entire period for each user
        types_df = df.groupby(["user_id"]).agg("sum").drop(columns=["month"])

        # merge the table with grouped income (expenses) by month with the "types" table
        res = res.merge(types_df, on="user_id", how="outer").drop(
            columns="amount_n26_currency"
        )

        # make sure that all users are present in our table, if not then fill cells with 0
        res = res.merge(self.unique_users, on="user_id", how="outer").fillna(0)
        return res

    def feature_engeneering(self, df: pd.DataFrame) -> pd.DataFrame:
        month = [2, 3, 4, 5, 6]
        # new feature - average value of income (expenses)
        df["avg"] = df[month].mean(axis=1)

        # a feature that indicates zero income (expenses) for the last month
        df["one_empty_month"] = np.where((df[6] == 0), 1, 0)
        # for the last 2 month
        df["two_empty_month"] = np.where((df[5] == 0) & (df[6] == 0), 1, 0)

        # a feature that indicates a decrease in income (expenses) for the last month compared to the previous one
        df["desc_1_month"] = np.where(df[6] < df[5], 1, 0)
        # for the last 2 month
        df["desc_2_month"] = np.where((df[6] < df[5]) & (df[5] < df[4]), 1, 0)

        # features that indicates the difference between the income (expenses) of the month and the average
        for el in month:
            df[f"diff_avg_{el}"] = df[el] - df["avg"]
        return df

    def feature_engeneering2(self, df: pd.DataFrame) -> pd.DataFrame:
        # create two more features

        # a feature that indicates unprofitable last month
        df["one_unprofit_month"] = np.where((df["in_out_6"] < 0), 1, 0)
        # last 2 month
        df["two_unprofit_month"] = np.where(
            (df["in_out_6"] < 0) & (df["in_out_5"] < 0), 1, 0
        )
        return df
