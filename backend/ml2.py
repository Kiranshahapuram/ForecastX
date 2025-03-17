import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------
# 1. Data Loading & Preprocessing
# ---------------------------
# Load dataset
file_path = "uploads/processed_data.csv"
df = pd.read_csv(file_path)

# Convert 'timestamp' to datetime and set as index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Fill missing values
df.fillna(method='ffill', inplace=True)
df.fillna(method='bfill', inplace=True)

# ---------------------------
# 2. Feature Engineering
# ---------------------------
# Auto-detect numerical target columns (excluding time-based features)
target_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Add time-based features for forecasting
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear

# Function to create lag features
def create_lag_features(data, columns, lag=10):
    df_lagged = data.copy()
    for col in columns:
        for i in range(1, lag+1):
            df_lagged[f'{col}_lag{i}'] = df_lagged[col].shift(i)
    df_lagged.dropna(inplace=True)  # Drop initial rows with NaN due to shifting
    return df_lagged

# Apply lag features
df = create_lag_features(df, target_cols, lag=10)

# Define time-based features
time_features = ['hour', 'dayofweek', 'month', 'dayofyear']

# ---------------------------
# 3. Train-Test Split & Scaling
# ---------------------------
train, test = train_test_split(df, test_size=0.1, shuffle=False)

# ---------------------------
# 4. Model Training & Forecasting using XGBoost
# ---------------------------
# Function to get feature columns for a given target
def get_features_for_target(df, target, lag=10, time_features=None):
    if time_features is None:
        time_features = []
    lag_features = [f'{target}_lag{i}' for i in range(1, lag+1)]
    return lag_features + time_features

# Store predictions & models
metrics = {}
predictions = {}
models = {}

# Train & forecast for each target column
for target in target_cols:
    features = get_features_for_target(df, target, lag=10, time_features=time_features)
    
    if any(f not in df.columns for f in features):  # Skip if necessary features are missing
        print(f"Skipping {target} due to missing lag features.")
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
    y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1,1)).ravel()
    
    # Train model
    model = XGBRegressor(
        objective='reg:squarederror', 
        n_estimators=1000, 
        learning_rate=0.01,  
        max_depth=5, 
        subsample=0.7, 
        colsample_bytree=0.7,
        random_state=42
    )
    model.fit(X_train_scaled, y_train_scaled)
    y_pred_scaled = model.predict(X_test_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1)).ravel()
    
    # Store results
    predictions[target] = y_pred
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mape = mean_absolute_percentage_error(y_test, y_pred)
    metrics[target] = {'MAE': mae, 'RMSE': np.sqrt(mse), 'MAPE': mape}
    models[target] = model

# ---------------------------
# 5. Plotting the Forecasts
# ---------------------------
fig, axs = plt.subplots(len(target_cols), 1, figsize=(14, 4 * len(target_cols)), constrained_layout=True)
for i, target in enumerate(target_cols):
    if target not in predictions:
        continue
    axs[i].plot(test.index, test[target], label="Actual", linestyle="dashed", color='black')
    axs[i].plot(test.index, predictions[target], label="XGBoost Prediction", color='green')
    axs[i].set_title(f"Forecast for {target}")
    axs[i].set_xlabel("Timestamp")
    axs[i].set_ylabel(target)
    axs[i].legend()
    axs[i].grid()
plt.show()

# ---------------------------
# 6. Print Evaluation Metrics
# ---------------------------
print("\nXGBoost Evaluation Metrics:")
for target, errs in metrics.items():
    print(f"{target}: MAE = {errs['MAE']:.4f}, RMSE = {errs['RMSE']:.4f}, MAPE = {errs['MAPE']:.2f}%")

# ---------------------------
# 7. Save Forecast Data to CSV (Backend Folder)
# ---------------------------
forecast_data = {'timestamp': test.index}
for target in predictions.keys():
    forecast_data[f'{target}_forecast'] = predictions[target]

forecast_df = pd.DataFrame(forecast_data)
forecast_file = "full_forecast.csv"
forecast_df.to_csv(forecast_file, index=False)
print(f"Forecast saved successfully at {forecast_file}!")
