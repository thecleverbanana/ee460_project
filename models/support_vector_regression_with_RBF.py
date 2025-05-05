import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(".."))

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, root_mean_squared_error
from utils.loader import load_data_with_logReturn
from utils.eval import evaluate_strategy_performance,calculate_average_pnl

class SVR_RBF:
    def __init__(self, csv_path, features, target, train_val_start, train_val_end, test_start, test_end,
                 loader_func=load_data_with_logReturn):
        self.csv_path = csv_path
        self.features = features
        self.target = target
        self.train_val_start = pd.Timestamp(train_val_start)
        self.train_val_end = pd.Timestamp(train_val_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)

        self.loader_func = loader_func

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = SVR(kernel='rbf', C=1.0, epsilon=0.001)

        self._load_and_split_data()

    def _load_and_split_data(self):
        data_clean = self.loader_func(self.csv_path)
        X = data_clean[self.features]
        y = data_clean[self.target]

        X_all = X.loc[self.train_val_start : self.train_val_end]
        y_all = y.loc[self.train_val_start : self.train_val_end]

        n = len(X_all)
        split_idx = int(n * 0.8)

        self.X_train = X_all.iloc[:split_idx]
        self.y_train = y_all.iloc[:split_idx]
        self.X_val = X_all.iloc[split_idx:]
        self.y_val = y_all.iloc[split_idx:]

        self.X_test = X.loc[self.test_start : self.test_end]
        self.y_test = y.loc[self.test_start : self.test_end]

    def train(self):
        X_train_scaled = self.scaler_X.fit_transform(self.X_train)

        self.model.fit(X_train_scaled, self.y_train)

    def predict(self):
        X_train_scaled = self.scaler_X.transform(self.X_train)
        X_val_scaled = self.scaler_X.transform(self.X_val)
        X_test_scaled = self.scaler_X.transform(self.X_test)

        y_train_pred = self.model.predict(X_train_scaled)
        y_val_pred = self.model.predict(X_val_scaled)
        y_test_pred = self.model.predict(X_test_scaled)

        return y_train_pred, y_val_pred, y_test_pred

    def evaluate(self):
        y_train_pred, y_val_pred, y_test_pred = self.predict()

        return {
            "Train R2": r2_score(self.y_train, y_train_pred),
            "Val R2": r2_score(self.y_val, y_val_pred),
            "Test R2": r2_score(self.y_test, y_test_pred),
            "Train RMSE": root_mean_squared_error(self.y_train, y_train_pred),
            "Val RMSE": root_mean_squared_error(self.y_val, y_val_pred),
            "Test RMSE": root_mean_squared_error(self.y_test, y_test_pred),
        }
    
    def run_trading_sim(self):
        _, y_val_pred, y_test_pred = self.predict()

        returns, capital, test_positions = evaluate_strategy_performance(
            self.y_val,
            y_val_pred,
            self.y_test,
            y_test_pred
        )

        pnl_result = calculate_average_pnl(test_positions, self.y_test)

        return {
            "Val Return": returns["Validation Cumulative Return"],
            "Val Sharpe": returns["Validation Sharpe Ratio"],
            "Test Return": returns["Test Cumulative Return"],
            "Test Sharpe": returns["Test Sharpe Ratio"],
            "Final Val Capital": capital["Final Val Capital"],
            "Final Test Capital": capital["Final Test Capital"],
            "Average PnL": pnl_result["average_pnl"],
            "Average PnL (%)": pnl_result["average_pnl_percent"]
        }

