import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(".."))

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import r2_score, root_mean_squared_error
from utils.loader import load_data_with_logReturn
from utils.eval import evaluate_strategy_performance,calculate_average_pnl

class MLPDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window):
        self.window = window
        self.features = feature_cols
        self.target = target_col
        self.X, self.y = self.create_features(df)

    def create_features(self, df):
        X_list, y_list = [], []
        for i in range(self.window, len(df) - 1):
            window_data = df.iloc[i - self.window:i][self.features].values
            X_list.append(window_data.flatten())
            y_list.append(df[self.target].iloc[i + 1])
        X_array = np.array(X_list)
        y_array = np.array(y_list)

        return torch.tensor(X_array, dtype=torch.float32), torch.tensor(y_array, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    

class MLPNet(nn.Module):
    def __init__(self, input_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)
    
class MLP_Regression:
    def __init__(self, csv_path, features, target, train_val_start, train_val_end, test_start, test_end,
                 loader_func=load_data_with_logReturn,
                 config=None):
        
        default_config = self._get_default_config()
        self.config = default_config if config is None else {**default_config, **config}

        self.csv_path = csv_path
        self.features = features
        self.target = target
        self.train_val_start = pd.Timestamp(train_val_start)
        self.train_val_end = pd.Timestamp(train_val_end)
        self.test_start = pd.Timestamp(test_start)
        self.test_end = pd.Timestamp(test_end)

        self.loader_func = loader_func

        self.device = self.config["device"]
        self.window = self.config["window"]
        self.shuffle_train_set = self.config["shuffle_train_set"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.dropout_rate = self.config["dropout_rate"]
        self.epochs = self.config["epochs"]

        self.scaler = StandardScaler()
        self._load_data_and_setup()
        self._init_model()

    def _get_default_config(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return {
            "window": 50,
            "batch_size": 64,
            "shuffle_train_set": False,
            "lr": 1e-3,
            "epochs": 20,
            "dropout_rate": 0.2,
            "device": device
        }

    def set_config(self, key, value):
        self.config[key] = value
        setattr(self, key, value)

    def _load_data_and_setup(self):
        df = self.loader_func(self.csv_path)
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.fit_transform(df_scaled[self.features])

        dataset = MLPDataset(df_scaled, self.features, self.target, self.window)
        all_dates = df.index.tolist()
        sample_end_dates = all_dates[self.window:len(dataset) + self.window]

        train_val_indices = [i for i, d in enumerate(sample_end_dates) if self.train_val_start <= d <= self.train_val_end]
        test_indices = [i for i, d in enumerate(sample_end_dates) if self.test_start <= d <= self.test_end]

        train_size = int(len(train_val_indices) * 0.8)
        train_indices = train_val_indices[:train_size]
        val_indices = train_val_indices[train_size:]

        self.train_loader = DataLoader(Subset(dataset, train_indices), batch_size=self.batch_size, shuffle=self.shuffle_train_set)
        self.val_loader = DataLoader(Subset(dataset, val_indices), batch_size=self.batch_size)
        self.test_loader = DataLoader(Subset(dataset, test_indices), batch_size=self.batch_size)

        self.y_train = [df[self.target].iloc[self.window + i + 1] for i in train_indices]
        self.y_val = [df[self.target].iloc[self.window + i + 1] for i in val_indices]
        self.y_test = [df[self.target].iloc[self.window + i + 1] for i in test_indices]

    def _init_model(self):
        input_size = len(self.features) * self.window
        self.model = MLPNet(input_size, self.dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device).unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.model(X_batch)
                loss = self.criterion(output, y_batch)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * X_batch.size(0)
            avg_train_loss = train_loss / len(self.train_loader.dataset)

            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device).unsqueeze(1)
                    val_pred = self.model(X_val)
                    loss = self.criterion(val_pred, y_val)
                    val_loss += loss.item() * X_val.size(0)
            avg_val_loss = val_loss / len(self.val_loader.dataset)

            print(f"Epoch {epoch+1}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f})")

    def predict(self):
        self.model.eval()
        preds_train, preds_val, preds_test = [], [], []

        with torch.no_grad():
            for X_batch, _ in self.train_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds_train.extend(y_pred.cpu().numpy().flatten())

            for X_batch, _ in self.val_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds_val.extend(y_pred.cpu().numpy().flatten())

            for X_batch, _ in self.test_loader:
                X_batch = X_batch.to(self.device)
                y_pred = self.model(X_batch)
                preds_test.extend(y_pred.cpu().numpy().flatten())

        return (
            np.array(preds_train),
            np.array(preds_val),
            np.array(preds_test)
        )

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

