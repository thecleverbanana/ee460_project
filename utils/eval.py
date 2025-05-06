def evaluate_strategy_performance(y_val, y_val_pred, y_test, y_test_pred, threshold=0.001, initial_capital=1_000_000):
    """
    Generate trading signals from predicted log returns and evaluate performance.

    Parameters:
        y_val (array-like): Actual log returns for the validation set
        y_val_pred (array-like): Predicted log returns for the validation set
        y_test (array-like): Actual log returns for the test set
        y_test_pred (array-like): Predicted log returns for the test set
        threshold (float): Signal threshold for generating long/short positions
        initial_capital (float): Initial trading capital

    Returns:
        - returns (dict): Cumulative returns and Sharpe ratios
        - capital (dict): Capital growth and total profit
        - val_positions (np.ndarray): Positions for validation set
        - test_positions (np.ndarray): Positions for test set
    """

    import numpy as np

    # Generate long/short/neutral positions
    val_positions = np.where(y_val_pred > threshold, 1, np.where(y_val_pred < -threshold, -1, 0))
    test_positions = np.where(y_test_pred > threshold, 1, np.where(y_test_pred < -threshold, -1, 0))

    # Strategy returns
    val_strategy_returns = val_positions * np.asarray(y_val)
    test_strategy_returns = test_positions * np.asarray(y_test)

    # Cumulative dollar returns
    val_dollar_returns = initial_capital * (np.exp(np.cumsum(val_strategy_returns)) - 1)
    test_dollar_returns = initial_capital * (np.exp(np.cumsum(test_strategy_returns)) - 1)

    # Cumulative return percentages
    val_cum_return = np.exp(np.cumsum(val_strategy_returns)) - 1
    test_cum_return = np.exp(np.cumsum(test_strategy_returns)) - 1

    # Sharpe ratios (annualized)
    val_sharpe = np.mean(val_strategy_returns) / np.std(val_strategy_returns) * np.sqrt(252)
    test_sharpe = np.mean(test_strategy_returns) / np.std(test_strategy_returns) * np.sqrt(252)

    # Return summary
    returns = {
        "Validation Cumulative Return": val_cum_return[-1],
        "Validation Sharpe Ratio": val_sharpe,
        "Test Cumulative Return": test_cum_return[-1],
        "Test Sharpe Ratio": test_sharpe
    }

    capital = {
        "Final Val Capital": val_dollar_returns[-1] + initial_capital,
        "Final Test Capital": test_dollar_returns[-1] + initial_capital,
        "Total Val Profit": val_dollar_returns[-1],
        "Total Test Profit": test_dollar_returns[-1]
    }

    return returns, capital,test_positions

def calculate_average_pnl(test_positions, y_test, initial_capital=1_000_000):
    """
    Calculate average PnL and percentage PnL from test positions and log returns.

    Parameters:
        test_positions (array-like): Position vector (1 = long, -1 = short, 0 = hold)
        y_test (array-like): Actual log returns for the test set
        initial_capital (float): Capital used per trade (default: $1,000,000)

    Returns:
        dict: {
            "average_pnl": float,
            "average_pnl_percent": float,
            "test_pnl": np.ndarray (PnL per time step)
        }
    """
    import numpy as np

    test_pnl = initial_capital * (np.exp(test_positions * np.asarray(y_test)) - 1)
    average_pnl = np.mean(test_pnl)
    average_pnl_percent = average_pnl / initial_capital * 100

    return {
        "average_pnl": average_pnl,
        "average_pnl_percent": average_pnl_percent,
        "test_pnl": test_pnl
    }

def reconstruct_price_and_signals(X_test, y_test_pred, test_positions):
    """
    Reconstruct predicted prices from predicted log returns and generate long/short markers.

    Parameters:
        X_test (pd.DataFrame): Test features (must include 'Close' column and datetime index)
        y_test_pred (array-like): Predicted log returns
        test_positions (array-like): Strategy positions (1 = long, -1 = short, 0 = hold)

    Returns:
        dict: {
            "P_hat_t_plus_1": np.ndarray of predicted prices,
            "P_t": np.ndarray of current prices,
            "long_signals": boolean array,
            "short_signals": boolean array,
            "plot_dates": pd.DatetimeIndex (for plotting x-axis)
        }
    """
    import numpy as np
    import pandas as pd

    # Ensure alignment of dimensions
    test_dates = X_test.index
    plot_dates = test_dates[1:]

    P_t = X_test["Close"].values[:-1]
    r_hat_t_plus_1 = y_test_pred[1:]

    # Reconstruct predicted price
    P_hat_t_plus_1 = P_t * np.exp(r_hat_t_plus_1)

    # Generate signal masks
    long_signals = (test_positions == 1)[1:]   # align to prediction shift
    short_signals = (test_positions == -1)[1:]

    return {
        "P_hat_t_plus_1": P_hat_t_plus_1,
        "P_t": P_t,
        "long_signals": long_signals,
        "short_signals": short_signals,
        "plot_dates": plot_dates
    }

def evaluate_strategy_performance_real_world(y_real, y_real_pred, threshold=0.001, initial_capital=1_000_000):
    """
    Evaluate trading performance on real-world data only.

    Parameters:
        y_real (array-like): Actual log returns
        y_real_pred (array-like): Predicted log returns
        threshold (float): Signal threshold for generating long/short positions
        initial_capital (float): Starting capital

    Returns:
        - summary (dict): Cumulative return, Sharpe ratio, final capital
        - capital (dict): Capital growth info
        - positions (np.ndarray): Strategy positions
    """
    import numpy as np

    # Generate positions
    positions = np.where(y_real_pred > threshold, 1,
                  np.where(y_real_pred < -threshold, -1, 0))

    # Strategy returns
    strategy_returns = positions * np.asarray(y_real)

    # Cumulative dollar returns
    dollar_returns = initial_capital * (np.exp(np.cumsum(strategy_returns)) - 1)

    # Cumulative return percentage
    cum_return = np.exp(np.cumsum(strategy_returns)) - 1

    # Sharpe ratio
    std = np.std(strategy_returns)
    if std == 0:
        sharpe = 0.0
    else:
        sharpe = np.mean(strategy_returns) / std * np.sqrt(252)

    summary = {
        "Real Cumulative Return": cum_return[-1],
        "Real Sharpe Ratio": sharpe
    }

    capital = {
        "Final Real Capital": dollar_returns[-1] + initial_capital,
        "Total Real Profit": dollar_returns[-1]
    }

    return summary, capital, positions
