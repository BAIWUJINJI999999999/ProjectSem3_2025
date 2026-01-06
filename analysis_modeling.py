import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

RAW_DATA_PATH = "data/sensor_data.csv"
PROCESSED_DATA_PATH = "data/processed_data.csv"
IMAGES_PATH = "images"
os.makedirs(IMAGES_PATH, exist_ok=True)

def load_and_preprocess_data():
    """Load raw data, complete denoising and normalization"""
    df = pd.read_csv(RAW_DATA_PATH, parse_dates=["timestamp"], index_col="timestamp")
    print("Basic information of raw data:")
    print(df.info())
    print("\nDescriptive statistics of raw data:")
    print(df.describe())

    df_denoised = df.rolling(window=5, center=True).mean().dropna()
    print("\nDescriptive statistics of denoised data:")
    print(df_denoised.describe())

    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df_denoised),
        index=df_denoised.index,
        columns=df_denoised.columns
    )

    df_scaled.to_csv(PROCESSED_DATA_PATH)
    print(f"\nProcessed data saved to: {PROCESSED_DATA_PATH}")
    return df, df_denoised, df_scaled, scaler

def plot_data(df, df_denoised):
    """Plot time series of raw vs denoised data"""
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df["temperature"], label="Original Temperature", color="orange", alpha=0.5)
    plt.plot(df_denoised.index, df_denoised["temperature"], label="Denoised Temperature", color="red")
    plt.title("Temperature Time Series (Original vs Denoised)")
    plt.xlabel("Time")
    plt.ylabel("Temperature (℃)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(df.index, df["humidity"], label="Original Humidity", color="lightblue", alpha=0.5)
    plt.plot(df_denoised.index, df_denoised["humidity"], label="Denoised Humidity", color="blue")
    plt.title("Humidity Time Series (Original vs Denoised)")
    plt.xlabel("Time")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{IMAGES_PATH}/time_series.png")
    print(f"Time series plot saved to: {IMAGES_PATH}/time_series.png")

def create_sequences(data, seq_length=24):
    """
    Convert time series data to supervised learning format
    seq_length: Input sequence length (previous 24 30-minute data points, i.e., 12 hours)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

def train_linear_regression(df_scaled, scaler):
    """Train Linear Regression model to predict humidity in 12 hours"""
    humidity_data = df_scaled["humidity"].values.reshape(-1, 1)
    seq_length = 24

    X, y = create_sequences(humidity_data, seq_length)
    X_flatten = X.reshape(X.shape[0], X.shape[1])  # Linear Regression requires 2D input

    X_train, X_test, y_train, y_test = train_test_split(X_flatten, y, test_size=0.3, shuffle=False)

    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)

    y_pred_lr = lr_model.predict(X_test)

    y_test_original = scaler.inverse_transform(y_test)
    y_pred_lr_original = scaler.inverse_transform(y_pred_lr)

    mae_lr = mean_absolute_error(y_test_original, y_pred_lr_original)
    rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr_original))
    print("\nLinear Regression Model Evaluation (Humidity Prediction):")
    print(f"MAE (Mean Absolute Error): {mae_lr:.2f} %")
    print(f"RMSE (Root Mean Squared Error): {rmse_lr:.2f} %")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, label="True Humidity", color="blue")
    plt.plot(y_pred_lr_original, label="Predicted Humidity (LR)", color="red", linestyle="--")
    plt.title("Linear Regression: True vs Predicted Humidity")
    plt.xlabel("Time Step")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{IMAGES_PATH}/lr_prediction.png")
    print(f"Linear Regression prediction plot saved to: {IMAGES_PATH}/lr_prediction.png")

    return lr_model, mae_lr, rmse_lr

def train_lstm(df_scaled, scaler):
    """Train lightweight LSTM model to predict humidity in 12 hours"""
    humidity_data = df_scaled["humidity"].values.reshape(-1, 1)
    seq_length = 24

    X, y = create_sequences(humidity_data, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=False)

    lstm_model = Sequential([
        LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2])),  # Lightweight: 32 neurons
        Dense(1)
    ])
    lstm_model.compile(optimizer="adam", loss="mse")

    print("\nStart training LSTM model...")
    lstm_model.fit(X_train, y_train, epochs=20, batch_size=8, validation_split=0.1, verbose=1)

    y_pred_lstm = lstm_model.predict(X_test)

    y_test_original = scaler.inverse_transform(y_test)
    y_pred_lstm_original = scaler.inverse_transform(y_pred_lstm)

    mae_lstm = mean_absolute_error(y_test_original, y_pred_lstm_original)
    rmse_lstm = np.sqrt(mean_squared_error(y_test_original, y_pred_lstm_original))
    print("\nLSTM Model Evaluation (Humidity Prediction):")
    print(f"MAE (Mean Absolute Error): {mae_lstm:.2f} %")
    print(f"RMSE (Root Mean Squared Error): {rmse_lstm:.2f} %")

    plt.figure(figsize=(10, 6))
    plt.plot(y_test_original, label="True Humidity", color="blue")
    plt.plot(y_pred_lstm_original, label="Predicted Humidity (LSTM)", color="green", linestyle="--")
    plt.title("LSTM: True vs Predicted Humidity")
    plt.xlabel("Time Step")
    plt.ylabel("Humidity (%)")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{IMAGES_PATH}/lstm_prediction.png")
    print(f"LSTM prediction plot saved to: {IMAGES_PATH}/lstm_prediction.png")

    lstm_model.save("code/lstm_model.h5")
    print("LSTM model saved to: code/lstm_model.h5")
    return lstm_model, mae_lstm, rmse_lstm

if __name__ == "__main__":
    df_raw, df_denoised, df_scaled, scaler = load_and_preprocess_data()
    plot_data(df_raw, df_denoised)

    lr_model, lr_mae, lr_rmse = train_linear_regression(df_scaled, scaler)

    lstm_model, lstm_mae, lstm_rmse = train_lstm(df_scaled, scaler)

    print("\n========== Model Comparison ==========")
    print(f"Linear Regression MAE: {lr_mae:.2f} | RMSE: {lr_rmse:.2f}")
    print(f"LSTM              MAE: {lstm_mae:.2f} | RMSE: {lstm_rmse:.2f}")
    print("======================================")
