import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import random
sys.path.append(os.path.abspath(".."))

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset

from sklearn.metrics import r2_score, root_mean_squared_error
from utils.loader import load_data_with_logReturn
from utils.eval import evaluate_strategy_performance_real_world,calculate_average_pnl
from utils.plotter import long_short_position_graph


class MLPDataset(Dataset):
    def __init__(self, df, feature_cols, target_col, window,stride):
        self.window = window
        self.stride = stride
        self.features = feature_cols
        self.target = target_col
        self.X, self.y = self.create_features(df)

    def create_features(self, df):
        X_list, y_list = [], []
        for i in range(self.window, len(df) - 1, self.stride): 
            # Slide a window of size `self.window` over the data with step size `self.stride`.
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
    def __init__(self, input_size, hidden_sizes, dropout_rate):
        super().__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, 1))  # Output layer
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
    
class MLP_Regression:
    def __init__(self, csv_path, features, target, 
                 train_val_test_start, train_val_test_end, 
                 loader_func=load_data_with_logReturn,
                 config=None):
        
        # Get default hyper-parameters if None in initialization
        default_config = self._get_default_config()
        self.config = default_config if config is None else {**default_config, **config}

        self.csv_path = csv_path
        self.features = features
        self.target = target
        self.train_val_test_start = pd.Timestamp(train_val_test_start)
        self.train_val_test_end = pd.Timestamp(train_val_test_end)

        self.loader_func = loader_func

        self.window = self.config["window"]
        self.stride = self.config["stride"]
        self.shuffle_dataset = self.config["shuffle_dataset"]
        self.batch_size = self.config["batch_size"]
        self.lr = self.config["lr"]
        self.dropout_rate = self.config["dropout_rate"]
        self.epochs = self.config["epochs"]
        self.device = self.config["device"]

        self.scaler = StandardScaler()
        self._load_data_and_setup()
        self._init_model()
        self.best_model_state = None

    def _get_default_config(self):
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        return {
            "window": 50,
            "stride": 50,
            "batch_size": 64,
            "shuffle_dataset": True,
            "lr": 1e-3,
            "epochs": 20,
            "dropout_rate": 0.2,
            "device": device
        }

    def set_config(self, key, value):
        self.config[key] = value
        setattr(self, key, value)

    def get_indices_by_date_range(self, sample_end_dates, start_date, end_date, name=""):
        """
        Return indices where sample_end_dates fall within [start_date, end_date].
        If no index is found, a warning is printed.
        """
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        indices = [
            i for i, date in enumerate(sample_end_dates)
            if start_date <= date <= end_date
        ]

        if len(indices) == 0:
            print(f"[Warning] No {name} samples in interval: {start_date.date()} ~ {end_date.date()}")

        return indices

    def _load_data_and_setup(self):
        df = self.loader_func(self.csv_path)
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.fit_transform(df_scaled[self.features])

        dataset = MLPDataset(df_scaled, self.features, self.target, self.window, self.stride)
        all_dates = df.index.tolist()
        sample_end_dates = [
            all_dates[i] for i in range(self.window, len(df) - 1, self.stride)
        ]  # offset by window size

        # --------------------------
        # Main (Train/Val/Test Split)
        # --------------------------
        main_indices = self.get_indices_by_date_range(
            sample_end_dates,
            self.train_val_test_start,
            self.train_val_test_end,
            name="Train/Val/Test"
        )

        if self.shuffle_dataset:
            random.seed(42)
            random.shuffle(main_indices)
        total = len(main_indices)
        train_end = int(total * 0.7)
        val_end = int(total * 0.85)

        train_indices = main_indices[:train_end]
        val_indices = main_indices[train_end:val_end]
        test_indices = main_indices[val_end:]

        self.train_loader = DataLoader(Subset(dataset, train_indices), batch_size=self.batch_size)
        self.val_loader = DataLoader(Subset(dataset, val_indices), batch_size=self.batch_size)
        self.test_loader = DataLoader(Subset(dataset, test_indices), batch_size=self.batch_size)

        self.y_train = [df[self.target].iloc[self.window + i + 1] for i in train_indices]
        self.y_val = [df[self.target].iloc[self.window + i + 1] for i in val_indices]
        self.y_test = [df[self.target].iloc[self.window + i + 1] for i in test_indices]

    def _init_model(self):
        input_size = len(self.features) * self.window
        hidden_sizes = self.config.get("hidden_sizes", [256, 128, 64, 32]) # Default
        self.model = MLPNet(input_size, hidden_sizes, self.dropout_rate).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train(self):
        best_val_loss = float("inf")
        best_model_state = None

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

            # Validation phase
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_val, y_val in self.val_loader:
                    X_val, y_val = X_val.to(self.device), y_val.to(self.device).unsqueeze(1)
                    val_pred = self.model(X_val)
                    loss = self.criterion(val_pred, y_val)
                    val_loss += loss.item() * X_val.size(0)

            avg_val_loss = val_loss / len(self.val_loader.dataset)

            # print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

            # Always track the best model on val loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict()
                self.best_model_state = best_model_state  # store in class

        # Restore best model weights after all epochs
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self):
            # Load best model weights if available
        if hasattr(self, "best_model_state") and self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
        else:
            print("[Warning] best_model_state is None — using current model weights.")

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
            np.array(preds_test),
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

class MLP_Simulation:
    def __init__(self, model, scaler, csv_path, features, target,
                 sim_start_date, sim_end_date,loader_func,
                 window,stride,batch_size,device):
        
        self.model = model.eval()  # Set to eval mode
        self.scaler = scaler
        self.csv_path = csv_path
        self.features = features
        self.target = target
        self.sim_start_date = pd.Timestamp(sim_start_date)
        self.sim_end_date = pd.Timestamp(sim_end_date)
        self.loader_func = loader_func
        self.window = window
        self.stride = stride
        self.batch_size = batch_size
        self.device = device

        self._load_and_prepare_simulation_data()

    def _load_and_prepare_simulation_data(self):
        # Load and scale the data
        df = self.loader_func(self.csv_path)
        df_scaled = df.copy()
        df_scaled[self.features] = self.scaler.transform(df_scaled[self.features])

        # Create dataset
        dataset = MLPDataset(df_scaled, self.features, self.target, self.window, self.stride)
        all_dates = df.index.tolist()
        sample_end_dates = [all_dates[i] for i in range(self.window, len(df) - 1, self.stride)]

        # Select samples within the simulation date range
        sim_indices = [
            i for i, date in enumerate(sample_end_dates)
            if self.sim_start_date <= date <= self.sim_end_date
        ]

        # Load the subset of data
        self.sim_loader = DataLoader(
            Subset(dataset, sim_indices), batch_size=self.batch_size
        )

        # Map back to real DataFrame indices (no +1 here!)
        df_indices = [self.window + i for i in sim_indices]
        self.X_real = df.iloc[df_indices].copy()
        self.X_real.index = df.index[df_indices]  # Ensure datetime index

        self.y_real = df[self.target].iloc[df_indices].copy()
        self.y_real.index = self.X_real.index  # Match index for plotting

    def run_simulation(self):
        preds = []

        with torch.no_grad():
            for X_batch, _ in self.sim_loader:
                X_batch = X_batch.to(self.device)
                pred = self.model(X_batch)
                preds.extend(pred.cpu().numpy().flatten())
        
        y_pred = np.array(preds)

        summary, capital, positions = evaluate_strategy_performance_real_world(
            self.y_real, y_pred
        )

        pnl_result = calculate_average_pnl(positions, self.y_real)

        result = {
            **summary,
            **capital,
            "Average PnL": pnl_result["average_pnl"],
            "Average PnL (%)": pnl_result["average_pnl_percent"]
        }

        y_true = pd.Series(self.y_real, index=self.X_real.index)
        fig = long_short_position_graph(self.X_real, y_true, y_pred, positions)

        return result, fig
