
def train_and_predict_rbf(X_train, y_train, X_val, y_val, X_test, y_test, C=1.0, epsilon=0.001):
    """
    Standardize features and train an SVR model with RBF kernel, then predict on validation and test sets.

    Parameters:
        X_train (pd.DataFrame or np.ndarray): Training features
        y_train (pd.Series or np.ndarray): Training target
        X_val (pd.DataFrame or np.ndarray): Validation features
        X_test (pd.DataFrame or np.ndarray): Test features
        C (float): Regularization parameter for SVR
        epsilon (float): Epsilon-tube within which no penalty is associated in the training loss function

    Returns:
        dict: {
            "scaler": StandardScaler object used for normalization,
            "model": Trained SVR model,
            "y_val_pred": Predictions on validation set,
            "y_test_pred": Predictions on test set,
            "stats": Dictionary of basic statistics of predictions
        }
    """
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, root_mean_squared_error
    from sklearn.svm import SVR
    import numpy as np


    # Feature normalization
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # Train SVR model
    svr = SVR(kernel='rbf', C=C, epsilon=epsilon)
    svr.fit(X_train_scaled, y_train)

    # Predict
    # y_train_pred = svr.predict(X_train_scaled)
    y_val_pred = svr.predict(X_val_scaled)
    y_test_pred = svr.predict(X_test_scaled)

    # Basic prediction statistics
    stats = {
        "Val Pred Mean": np.mean(y_val_pred),
        "Val Pred Std": np.std(y_val_pred),
        "Test Pred Mean": np.mean(y_test_pred),
        "Test Pred Std": np.std(y_test_pred)
    }

    metrics = {
        # "Train R2": r2_score(y_train, y_train_pred),
        "Val R2": r2_score(y_val, y_val_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        # "Train RMSE": root_mean_squared_error(y_train, y_train_pred),
        "Val RMSE": root_mean_squared_error(y_val, y_val_pred),
        "Test RMSE": root_mean_squared_error(y_test, y_test_pred)
    }

    return {
        "scaler": scaler,
        "model": svr,
        "y_val_pred": y_val_pred,
        "y_test_pred": y_test_pred,
        "stats": stats,
        "metrics": metrics
    }
