import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import zscore
import os
import io
import subprocess

PROCESSED_DATA_PATH = "uploads/processed_data.csv"

def convert_keys_to_str(d):
    """
    Recursively converts dictionary keys to strings if they are not already.
    This is useful for JSON serialization when keys are Timestamps.
    """
    if isinstance(d, dict):
        new_d = {}
        for k, v in d.items():
            new_key = str(k) if not isinstance(k, (str, int, float, bool)) and k is not None else k
            new_d[new_key] = convert_keys_to_str(v)
        return new_d
    elif isinstance(d, list):
        return [convert_keys_to_str(item) for item in d]
    else:
        return d

def load_data_from_df(df):
    """
    Generic data loader:
    - If headers are missing, assign default ones.
    - Tries to parse the first column as datetime; if successful, sets it as index.
    """
    # If columns are not strings, assign default column names.
    if not all(isinstance(col, str) for col in df.columns):
        df.columns = [f"col{i+1}" for i in range(len(df.columns))]
    
    # Try to auto-detect a datetime column.
    dt_col = None
    for col in df.columns:
        try:
            pd.to_datetime(df[col], errors="raise")
            dt_col = col
            break
        except Exception:
            continue

    if dt_col is not None:
        df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")
        df = df.dropna(subset=[dt_col])
        df = df.set_index(dt_col)
        df = df.sort_index()
    # Otherwise, leave index as is.
    return df

def handle_missing_values(df):
    """Interpolate missing values using time interpolation (if index is datetime)."""
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.to_series().fillna(method="ffill")
    return df.interpolate(method="time")

def handle_duplicates_timezone(df):
    """Remove duplicate index entries and remove timezone info if present."""
    if isinstance(df.index, pd.DatetimeIndex):
        df = df[~df.index.duplicated(keep="first")]
        df = df.tz_localize(None)
    return df

def resample_data(df, freq='D'):
    """Resample data from the first numeric column if the index is datetime."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return None
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return None
    value_col = numeric_cols[0]
    return df[value_col].resample(freq).mean().to_dict()

def check_stationarity(df):
    """Perform ADF test on the first numeric column, if possible."""
    if not isinstance(df.index, pd.DatetimeIndex) or df.empty:
        return {"ADF Statistic": None, "p-value": None, "Stationary": None}
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return {"ADF Statistic": None, "p-value": None, "Stationary": None}
    value_col = numeric_cols[0]
    result = adfuller(df[value_col].dropna())
    return {"ADF Statistic": result[0], "p-value": result[1], "Stationary": str(result[1] < 0.05)}

def detect_outliers(df):
    """Detect outliers using Z-score on the first numeric column."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return {"outliers": []}
    value_col = numeric_cols[0]
    df['zscore'] = zscore(df[value_col].dropna())
    outliers = df[np.abs(df['zscore']) > 3]
    return {"outliers": outliers[value_col].tolist()}

def feature_engineering(df):
    """Create rolling mean, rolling std, and first-order differencing for the first numeric column."""
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if not numeric_cols:
        return df
    value_col = numeric_cols[0]
    df['rolling_mean'] = df[value_col].rolling(window=7).mean()
    df['rolling_std'] = df[value_col].rolling(window=7).std()
    df['first_difference'] = df[value_col].diff()
    return df
import pandas as pd

PROCESSED_DATA_PATH = "./uploads/processed_data.csv"

def analyze_time_series(df):
    """
    Main processing function:
    - Loads the DataFrame generically (assigns headers if missing, and attempts to set a datetime index).
    - Handles missing values, duplicates, and timezone issues.
    - Computes summary statistics, stationarity, resampling, and outlier detection.
    - Performs feature engineering.
    - Saves the processed data to a CSV.
    Returns a JSON-friendly dictionary with the results.
    """
    df = load_data_from_df(df)
    df = handle_missing_values(df)
    df = handle_duplicates_timezone(df)
    
    # Compute descriptive statistics
    description = df.describe().to_dict()  # DataFrame -> Dict
    stationarity = check_stationarity(df)  # Already a Dict
    outliers = detect_outliers(df)  # Already a Dict
    daily = resample_data(df, freq='D')  # Already a Dict
    weekly = resample_data(df, freq='W')  # Already a Dict
    monthly = resample_data(df, freq='M')  # Already a Dict
    
    # Perform feature engineering
    df = feature_engineering(df)
    
    # Save the processed data to a CSV file
    df.to_csv(PROCESSED_DATA_PATH, index=True)
    
    # Convert non-serializable keys (like timestamps) to strings
    result = {
        "description": convert_keys_to_str(description),  # Convert Dict keys
        "stationarity": convert_keys_to_str(stationarity),  
        "resample": {
            "daily": convert_keys_to_str(daily),
            "weekly": convert_keys_to_str(weekly),
            "monthly": convert_keys_to_str(monthly)
        },
        "outliers": convert_keys_to_str(outliers),
        "message": "Time-series analysis completed successfully!",
        "csv_saved": PROCESSED_DATA_PATH
    }

    return result


if __name__ == "__main__":
    # For local testing, assume a sample CSV file named 'sample_data.csv' in the same directory.
    df_test = pd.read_csv("sample_data.csv", header=None)
    result = analyze_time_series(df_test)
    print(result)
    