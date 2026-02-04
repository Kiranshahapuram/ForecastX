import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


UPLOADS_DIR = "uploads"
PROCESSED_FILE = os.path.join(UPLOADS_DIR, "processed_data.csv")
FORECAST_CSV = os.path.join(UPLOADS_DIR, "full_forecast.csv")
FORECAST_PLOT = os.path.join(UPLOADS_DIR, "forecast_plot.png")


def _create_lag_features(data: pd.DataFrame, columns, lag: int = 10) -> pd.DataFrame:
    """Create lag features for the given numeric columns."""
    df_lagged = data.copy()
    for col in columns:
        for i in range(1, lag + 1):
            df_lagged[f"{col}_lag{i}"] = df_lagged[col].shift(i)
    df_lagged.dropna(inplace=True)
    return df_lagged


def _get_features_for_target(df: pd.DataFrame, target: str, lag: int = 10, time_features=None):
    """Return list of feature column names for a given target."""
    if time_features is None:
        time_features = []
    lag_features = [f"{target}_lag{i}" for i in range(1, lag + 1)]
    return lag_features + time_features


def run_forecast():
    """
    Run XGBoost-based time series forecasting on the processed data.

    Expects `uploads/processed_data.csv` produced by the preprocessing pipeline.
    Returns a JSON-serializable dict with metrics and forecast data, and
    saves a Matplotlib forecast plot to `uploads/forecast_plot.png`.
    """
    if not os.path.exists(PROCESSED_FILE):
        raise FileNotFoundError(
            f"Processed data file not found at {PROCESSED_FILE}. "
            f"Upload data first so the preprocessing step can generate it."
        )

    # ---------------------------
    # 1. Data Loading & Preprocessing
    # ---------------------------
    # Load processed data and auto-detect a datetime column.
    df = pd.read_csv(PROCESSED_FILE)

    if df.shape[1] < 2:
        raise ValueError("Processed data file does not contain enough columns for forecasting.")

    # Try to find a sensible datetime column by attempting coercive parsing and
    # requiring that at least half the values are valid timestamps.
    dt_col = None
    for col in df.columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        non_na_ratio = parsed.notna().mean()
        if non_na_ratio >= 0.5:
            dt_col = col
            df[col] = parsed
            break

    if dt_col is None:
        raise ValueError(
            "Could not detect a datetime column in processed data. "
            "Please ensure your input CSV has a proper datetime column."
        )

    df = df.dropna(subset=[dt_col])
    df = df.sort_values(dt_col)
    df = df.set_index(dt_col)
    df.index.name = "timestamp"

    # Fill missing values (forward and backward)
    df.ffill(inplace=True)
    df.bfill(inplace=True)

    # ---------------------------
    # 2. Feature Engineering
    # ---------------------------
    # Auto-detect numerical target columns (excluding time-based features)
    target_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not target_cols:
        raise ValueError("No numeric columns found in processed data to forecast.")

    # Add time-based features for forecasting
    df["hour"] = df.index.hour
    df["dayofweek"] = df.index.dayofweek
    df["month"] = df.index.month
    df["dayofyear"] = df.index.dayofyear

    # Apply lag features
    df = _create_lag_features(df, target_cols, lag=10)

    # Define time-based features
    time_features = ["hour", "dayofweek", "month", "dayofyear"]

    # ---------------------------
    # 3. Train-Test Split & Scaling
    # ---------------------------
    train, test = train_test_split(df, test_size=0.1, shuffle=False)

    # ---------------------------
    # 4. Model Training & Forecasting using XGBoost
    # ---------------------------
    metrics = {}
    predictions = {}
    actuals = {}

    for target in target_cols:
        features = _get_features_for_target(df, target, lag=10, time_features=time_features)

        if any(f not in df.columns for f in features):
            # Skip if necessary features are missing
            continue

        X = df[features]
        y = df[target]

        X_train, X_test = X.loc[train.index], X.loc[test.index]
        y_train, y_test = y.loc[train.index], y.loc[test.index]

        # Scaling
        scaler_X = StandardScaler()
        scaler_y = StandardScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)
        y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()

        # Train model
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=5,
            subsample=0.7,
            colsample_bytree=0.7,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train_scaled)
        y_pred_scaled = model.predict(X_test_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

        # Store results
        predictions[target] = y_pred.tolist()
        actuals[target] = y_test.tolist()
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        metrics[target] = {"MAE": mae, "RMSE": float(np.sqrt(mse)), "MAPE": float(mape)}

    if not predictions:
        raise ValueError("Forecasting failed: no suitable numeric targets were found.")

    # ---------------------------
    # 5. Plotting the Forecasts (Matplotlib)
    # ---------------------------
    os.makedirs(UPLOADS_DIR, exist_ok=True)

    fig, axs = plt.subplots(len(target_cols), 1, figsize=(14, 4 * len(target_cols)), constrained_layout=True)
    # When there is only one target, axs is a single Axes object
    if len(target_cols) == 1:
        axs = [axs]

    for i, target in enumerate(target_cols):
        if target not in predictions:
            continue
        axs[i].plot(test.index, test[target], label="Actual", linestyle="dashed", color="black")
        axs[i].plot(test.index, predictions[target], label="XGBoost Prediction", color="green")
        axs[i].set_title(f"Forecast for {target}")
        axs[i].set_xlabel("Timestamp")
        axs[i].set_ylabel(target)
        axs[i].legend()
        axs[i].grid()

    plt.savefig(FORECAST_PLOT)
    plt.close(fig)

    # ---------------------------
    # 6. Save Forecast Data to CSV
    # ---------------------------
    forecast_data = {"timestamp": [ts.isoformat() for ts in test.index]}
    for target, preds in predictions.items():
        forecast_data[f"{target}_forecast"] = preds

    forecast_df = pd.DataFrame(forecast_data)
    forecast_df.to_csv(FORECAST_CSV, index=False)

    # ---------------------------
    # 7. Build JSON-serializable response
    # ---------------------------
    result = {
        "message": "XGBoost forecasting completed successfully.",
        "metrics": metrics,
        "targets": list(predictions.keys()),
        "timestamps": forecast_data["timestamp"],
        "predictions": predictions,
        "actuals": actuals,
        "forecast_csv": FORECAST_CSV,
        "forecast_plot": FORECAST_PLOT,
    }
    return result


if __name__ == "__main__":
    # Manual test helper
    output = run_forecast()
    print(output)
