import numpy as np
import matplotlib.pyplot as plt

def plot_actual_vs_predicted_prices(X_train, y_train, y_train_pred,
                                    X_val, y_val, y_val_pred,
                                    X_test, y_test, y_test_pred,
                                    figsize=(12, 10), save_path=None):
    """
    Plot actual vs predicted prices for Train, Validation, and Test sets.

    Parameters:
        X_train, X_val, X_test: DataFrames with a "Close" column
        y_train, y_val, y_test: pd.Series with true log returns
        y_train_pred, y_val_pred, y_test_pred: predicted log returns (same length as corresponding y)
        figsize: tuple for plot size
        save_path: optional path to save the plot (e.g., "figs/predicted_prices.jpg")
    """
    # Reconstruct predicted prices
    P_train = X_train["Close"].iloc[:-1].values
    P_val = X_val["Close"].iloc[:-1].values
    P_test = X_test["Close"].iloc[:-1].values

    P_train_pred = P_train * np.exp(y_train_pred[1:])
    P_val_pred = P_val * np.exp(y_val_pred[1:])
    P_test_pred = P_test * np.exp(y_test_pred[1:])

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=False)

    # Train
    axes[0].plot(y_train.index[1:], X_train["Close"].iloc[1:], label="Actual", alpha=0.7)
    axes[0].plot(y_train.index[1:], P_train_pred, label="Predicted", alpha=0.7)
    axes[0].set_title("Training Set: Actual vs Predicted Price")
    axes[0].set_ylabel("Price (USD)")
    axes[0].legend()
    axes[0].grid(True)

    # Validation
    axes[1].plot(y_val.index[1:], X_val["Close"].iloc[1:], label="Actual", alpha=0.7)
    axes[1].plot(y_val.index[1:], P_val_pred, label="Predicted", alpha=0.7)
    axes[1].set_title("Validation Set: Actual vs Predicted Price")
    axes[1].set_ylabel("Price (USD)")
    axes[1].legend()
    axes[1].grid(True)

    # Test
    axes[2].plot(y_test.index[1:], X_test["Close"].iloc[1:], label="Actual", alpha=0.7)
    axes[2].plot(y_test.index[1:], P_test_pred, label="Predicted", alpha=0.7)
    axes[2].set_title("Test Set: Actual vs Predicted Price")
    axes[2].set_xlabel("Date")
    axes[2].set_ylabel("Price (USD)")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✓ Plot saved to {save_path}")
    else:
        plt.show()
