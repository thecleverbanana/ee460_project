import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from utils.loader import load_data_with_logReturn
from svr import train_and_predict_rbf
from utils.eval import evaluate_strategy_performance,calculate_average_pnl

def plot_pnl_distribution(test_pnl, save_path):
    plt.figure(figsize=(10, 6))
    sns.histplot(test_pnl, kde=True, bins=30)
    plt.title('Distribution of Daily P&L on Test Data')
    plt.xlabel('Profit/Loss ($)')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved PnL distribution plot to {save_path}")

def plot_reconstructed_price_signals(X_test, y_test_pred, test_positions, save_path):
    # Signal masks
    long_signals = (test_positions == 1)
    short_signals = (test_positions == -1)

    # Dates
    test_dates = X_test.index
    plot_dates = test_dates[1:]

    # Reconstructed prices
    P_t = X_test["Close"][:-1]
    r_hat_t_plus_1 = y_test_pred[1:]
    P_hat_t_plus_1 = P_t * np.exp(r_hat_t_plus_1)

    # Plot
    plt.figure(figsize=(15, 6))
    plt.plot(test_dates, X_test["Close"], label="Actual Price", alpha=0.7)
    plt.plot(plot_dates, P_hat_t_plus_1, label="Reconstructed Price", linestyle="--", alpha=0.7)

    plt.scatter(plot_dates[long_signals[1:]], X_test["Close"].values[1:][long_signals[1:]],
                marker='^', color='green', label='Long Entry', zorder=5)
    plt.scatter(plot_dates[short_signals[1:]], X_test["Close"].values[1:][short_signals[1:]],
                marker='v', color='red', label='Short Entry', zorder=5)

    plt.title("AAPL 2023: Actual vs Reconstructed Price")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✓ Saved reconstructed price plot to {save_path}")

def main():
    data_clean = load_data_with_logReturn("test_stocks/AAPL.csv")

    features = [
        "Open", "High", "Low", "Close", "Volume",
        "LogReturn_Lag1", "LogReturn_Lag2", "LogReturn_Lag3", "LogReturn_Lag4", "LogReturn_Lag5"
    ]
    target = "LogReturn"

    X = data_clean[features]
    y = data_clean[target]

    val_start = pd.Timestamp("2022-01-01")
    val_end = pd.Timestamp("2022-12-31")
    test_start = pd.Timestamp("2023-01-01")
    test_end = pd.Timestamp("2023-12-31")

    for year in range(1980, 2022): 
        train_start = pd.Timestamp(f"{year}-01-01")
        train_end = pd.Timestamp("2021-12-31")

        X_train = X.loc[train_start:train_end]
        y_train = y.loc[train_start:train_end]
        X_val = X.loc[val_start:val_end]
        y_val = y.loc[val_start:val_end]
        X_test = X.loc[test_start:test_end]
        y_test = y.loc[test_start:test_end]

        if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
            print(f"Year {year} skipped due to insufficient data.")
            continue

        model_results = train_and_predict_rbf(X_train, y_train, X_val, y_val, X_test, y_test)

        returns, capital, test_positions = evaluate_strategy_performance(
            y_val=y_val,
            y_val_pred=model_results["y_val_pred"],
            y_test=y_test,
            y_test_pred=model_results["y_test_pred"]
        )
        pnl_result = calculate_average_pnl(test_positions, y_test)

        summary = {
            "Train Start": train_start,
            "Train End": train_end,
            "Val R2": model_results["metrics"]["Val R2"],
            "Test R2": model_results["metrics"]["Test R2"],
            "Validation Cumulative Return": returns["Validation Cumulative Return"],
            "Validation Sharpe Ratio": returns["Validation Sharpe Ratio"],
            "Test Sharpe": returns["Test Sharpe Ratio"],
            "Test Return": returns["Test Cumulative Return"],
            "Final Val Capital": capital["Final Val Capital"],
            "Final Test Capital": capital["Final Test Capital"],
            "Total Val Profit": capital["Total Val Profit"],
            "Total Test Profit": capital["Total Test Profit"],
            "Average PnL": pnl_result["average_pnl"],
            "Average PnL (%)": pnl_result["average_pnl_percent"]
        }

        os.makedirs("rbf_ewf/expanding", exist_ok=True)
        os.makedirs("rbf_ewf/expanding/plots", exist_ok=True)   

        pd.DataFrame([summary]).to_csv(f"rbf_ewf/expanding/{year}_metrics.csv", index=False)
        plot_path = f"rbf_ewf/expanding/plots/{year}_reconstructed_price.jpg"
        plot_reconstructed_price_signals(X_test, model_results["y_test_pred"], test_positions, plot_path)
        print(f"Saved metrics for training start year {year}")

if __name__ == "__main__":
    main()