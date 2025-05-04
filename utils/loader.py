def load_data_with_logReturn(filepath):
    """
    Load stock CSV and compute log return with lag features.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Processed dataframe with log returns and lag features.
    """
    import pandas as pd
    import numpy as np

    data = pd.read_csv(filepath, index_col="Date", parse_dates=True)
    # Compute log returns
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    # Add lagged log returns as features
    for lag in range(1, 6):
        data[f"LogReturn_Lag{lag}"] = data["LogReturn"].shift(lag)

    # Add percentage changes of other features
    data["Open"] = data["Open"]
    data["High"] = data["High"]
    data["Low"] = data["Low"]
    data["Close"] = data["Close"]
    data["Volume"] = data["Volume"]

    # Drop NaNs
    data.dropna(inplace=True)

    # Clean up metadata rows
    data_clean = data.copy()
    data_clean.index.name = "Date"

    # Convert index to datetime if not already
    data_clean.index = pd.to_datetime(data_clean.index)

    # Convert columns to numeric
    for col in data_clean.columns:
        data_clean[col] = pd.to_numeric(data_clean[col], errors='coerce')

    # Drop rows with missing values
    data_clean.dropna(inplace=True)

    return data_clean
