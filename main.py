import os
import csv
import datetime
from models.fir_regression import FIR_Regression
from models.support_vector_regression_with_RBF import SVR_RBF
from models.mlp import MLP_Regression
from utils.loader import load_data_with_logReturn

class BacktestRunner:
    def __init__(self, model_class, config, log_root="log", log_name="Expanding_Window_Log"):
        self.model_class = model_class
        self.model = model_class(**config)

        self.model_name = model_class.__name__
        self.log_dir = os.path.join(log_root, self.model_name)
        os.makedirs(self.log_dir, exist_ok=True)

        self.csv_path = os.path.join(self.log_dir, f"{self.model_name}_{log_name}.csv")
        self.config = config

    def run(self):
        self.model.train()

        eval_result = self.model.evaluate()
        trading_result = self.model.run_trading_sim()

        # self._print_eval_result(eval_result)
        # self._print_trading_result(trading_result)
        self._write_csv(eval_result,trading_result)

    def _write_csv(self, eval_result, trading_result):
        log_entry = {
            "train_val_start": self.config["train_val_start"],
            "train_val_end": self.config["train_val_end"],
            "test_start": self.config["test_start"],
            "test_end": self.config["test_end"],
            **eval_result,
            **trading_result
        }

        file_exists = os.path.isfile(self.csv_path)

        with open(self.csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())

            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

    def _print_eval_result(self, result):
        print("\nModel Evaluation:")
        for k, v in result.items():
            print(f"{k}: {v:.4f}")

    def _print_trading_result(self, result):
        print("\nStrategy Backtest Result:")
        for k, v in result.items():
            print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")


if __name__ == "__main__":
    features = [
        "Open", "High", "Low", "Close", "Volume",
        "LogReturn_Lag1", "LogReturn_Lag2", "LogReturn_Lag3",
        "LogReturn_Lag4", "LogReturn_Lag5"
    ]

    base_config = {
        "csv_path": "stocks/AAPL.csv",
        "features": features,
        "target": "LogReturn",
        # "train_val_start": "1981-12-31",
        "train_val_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-05-02",
        "loader_func": load_data_with_logReturn
    }

    mlp_extra_config = {
        "config": {
            "window": 50,
            "batch_size": 64,
            "lr": 1e-3,
            "epochs": 20,
            "dropout_rate": 0.2,
            "shuffle_train_set": False
        }
    }

    start_years = list(range(1981, 2020, 1)) 

    for year in start_years:
        start_date = f"{year}-12-31"
        base_config["train_val_start"] = start_date

        mlp_config = {**base_config, **mlp_extra_config}

        print(f"Running MLP_Regression with train_val_start = {start_date}")
        runner = BacktestRunner(MLP_Regression, mlp_config)
        runner.run()
