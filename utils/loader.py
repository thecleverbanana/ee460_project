def load_data(filepath):
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

def load_data_with_logReturn_black_swan(filepath):
    """
    Load stock CSV and compute log return with lag features.
    Remove rows during known black swan events.

    Parameters:
        filepath (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Cleaned dataframe with log returns and lag features.
    """
    import pandas as pd
    import numpy as np

    data = pd.read_csv(filepath, index_col="Date", parse_dates=True)

    # Compute log returns
    data["LogReturn"] = np.log(data["Close"] / data["Close"].shift(1))

    # Add lagged log returns as features
    for lag in range(1, 6):
        data[f"LogReturn_Lag{lag}"] = data["LogReturn"].shift(lag)

    # Drop initial NaNs caused by shift
    data.dropna(inplace=True)

    # Define black swan periods
    black_swan_periods = {
        "DotCom Bubble Crash": ("2000-03-01", "2002-10-01"),
        "2008 Financial Crisis": ("2008-09-01", "2009-06-01"),
        "Flash Crash": ("2010-05-06", "2010-05-07"),
        "Eurozone Crisis": ("2011-07-01", "2012-01-01"),
        "COVID-19 Crash": ("2020-02-15", "2020-04-30"),
        "SVB Collapse": ("2023-03-01", "2023-04-01"),
    }

    original_len = len(data)
    for name, (start, end) in black_swan_periods.items():
        mask = (data.index >= start) & (data.index <= end)
        removed_count = mask.sum()
        if removed_count > 0:
            print(f"[INFO] Removed {removed_count} rows for {name} ({start} to {end})")
        data = data[~mask]

    # Final cleanup
    data.index = pd.to_datetime(data.index)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()
    data.index.name = "Date"

    print(f"[INFO] Final dataset size: {len(data)} rows (removed {original_len - len(data)} total)")
    return data
