import os
import csv
import datetime
from models.fir_regression import FIR_Regression
from models.support_vector_regression_with_RBF import SVR_RBF
from models.mlp import MLP_Regression
from utils.loader import load_data_with_logReturn

# class BacktestRunner:
#     def __init__(self, model_class, config, log_root="log", log_name="Expanding_Window_Log"):
#         self.model_class = model_class
#         self.model = model_class(**config)

#         self.model_name = model_class.__name__
#         self.log_dir = os.path.join(log_root, self.model_name)
#         os.makedirs(self.log_dir, exist_ok=True)

#         self.csv_path = os.path.join(self.log_dir, f"{self.model_name}_{log_name}.csv")
#         self.config = config

#     def run(self):
#         self.model.train()

#         eval_result = self.model.evaluate()
#         trading_result = self.model.run_trading_sim()

#         # self._print_eval_result(eval_result)
#         # self._print_trading_result(trading_result)
#         self._write_csv(eval_result,trading_result)

#     def _write_csv(self, eval_result, trading_result):
#         log_entry = {
#             "train_val_start": self.config["train_val_start"],
#             "train_val_end": self.config["train_val_end"],
#             "test_start": self.config["test_start"],
#             "test_end": self.config["test_end"],
#             **eval_result,
#             **trading_result
#         }

#         file_exists = os.path.isfile(self.csv_path)

#         with open(self.csv_path, mode="a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=log_entry.keys())

#             if not file_exists:
#                 writer.writeheader()
#             writer.writerow(log_entry)

#     def _print_eval_result(self, result):
#         print("\nModel Evaluation:")
#         for k, v in result.items():
#             print(f"{k}: {v:.4f}")

#     def _print_trading_result(self, result):
#         print("\nStrategy Backtest Result:")
#         for k, v in result.items():
#             print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

class BacktestRunner:
    def __init__(self, model_class, model_config):
        self.model_class = model_class
        self.model_config = model_config
        self.model = model_class(**model_config)
        self.best_model = None
        self.eval_result = None
        self.sim_result = None
        self.sim_fig = None

    def train_optimal_model(self):
        self.model.train()
        self.best_model = self.model
        return self.best_model

    def evaluate_model(self):
        self.eval_result = self.model.evaluate()
        return self.eval_result

    def run_real_world_simulation(self, sim_class, sim_config):
        sim = sim_class(model=self.model.model,
                        scaler=self.model.scaler,
                        **sim_config)
        self.sim_result, self.sim_fig = sim.run_simulation()
        return self.sim_result, self.sim_fig

    def log_results(self, log_root="log", log_name=None):
        if self.eval_result is None or self.sim_result is None:
            raise ValueError("Must run evaluate_model and run_real_world_simulation before logging.")

        if log_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"log_{timestamp}"
        log_dir = os.path.join(log_root, self.model_class.__name__)
        os.makedirs(log_dir, exist_ok=True)
        csv_path = os.path.join(log_dir, f"{log_name}.csv")

        log_entry = {
            "train_val_start": self.model_config["train_val_test_start"],
            "train_val_end": self.model_config["train_val_test_end"],
            "test_start": self.model_config["train_val_test_start"],  
            "test_end": self.model_config["train_val_test_end"],
            **self.eval_result,
            **self.sim_result,
        }

        file_exists = os.path.isfile(csv_path)
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(log_entry)

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
            "train_val_test_start": "1981-12-31",
            "train_val_test_end": "1988-12-31",
            "loader_func": load_data_with_logReturn
        }

    mlp_extra_config = {
        "config": {
            "window": 50,
            "stride": 50,
            "batch_size": 64,
            "shuffle_train_set": False,
            "lr": 1e-3,
            "epochs": 20,
            "dropout_rate": 0.2,
        }
    }

    sim_config = {
        "csv_path": base_config["csv_path"],
        "features": base_config["features"],
        "target": base_config["target"],
        "sim_start_date": "2025-01-01",  
        "sim_end_date": "2025-05-02", 
        "loader_func": base_config["loader_func"],
        "window": 50,                  
        "stride": 1,                    
        "batch_size": 64,           
    }

    mlp_config = {**base_config}

    # start_years = list(range(1981, 2020, 1)) 

    # for year in start_years:
    #     start_date = f"{year}-12-31"
    #     base_config["train_val_start"] = start_date

    #     mlp_config = {**base_config, **mlp_extra_config}

    #     print(f"Running MLP_Regression with train_val_start = {start_date}")
    #     runner = BacktestRunner(MLP_Regression, mlp_config)
    #     runner.run()
    # runner = BacktestRunner(MLP_Regression, mlp_config)
    myMLP = MLP_Regression(**mlp_config)