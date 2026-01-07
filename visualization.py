import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta

def visualize_results(df, lr_test_pred, lr_future_pred, lstm_test_pred, lstm_future_pred):
    """
    Visualize raw data, denoised data and prediction results
    :param df: Preprocessed data DataFrame
    :param lr_test_pred: Linear Regression test predictions
    :param lr_future_pred: Linear Regression future predictions
    :param lstm_test_pred: LSTM test predictions
    :param lstm_future_pred: LSTM future predictions
    """

    plt.rcParams["figure.figsize"] = (16, 12)
    plt.rcParams["font.size"] = 10
    
    time_axis = pd.to_datetime(df["timestamp"])
    future_time = [time_axis.iloc[-1] + timedelta(minutes=30*i) for i in range(1, 25)]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    
    ax1.plot(time_axis, df["temperature"], label="Raw Temperature", alpha=0.5, color="orange")
    ax1.plot(time_axis, df["temperature_denoised"], label="Denoised Temperature", color="red")
    ax1.set_title("Raw vs Denoised Temperature")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Temperature (°C)")
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_axis, df["humidity"], label="Raw Humidity", alpha=0.5, color="lightblue")
    ax2.plot(time_axis, df["humidity_denoised"], label="Denoised Humidity", color="blue")
    ax2.set_title("Raw vs Denoised Humidity")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Humidity (%)")
    ax2.legend()
    ax2.grid(True)
    
    split_idx = int(len(df) * 0.8)
    ax3.plot(time_axis[:split_idx], df["humidity_denoised"][:split_idx], label="Training Set", color="blue")
    ax3.plot(time_axis[split_idx:], df["humidity_denoised"][split_idx:], label="Test Set", color="green")
    ax3.plot(time_axis[split_idx:], lr_test_pred, label="Linear Regression (Test)", color="red", linestyle="--")
    ax3.plot(future_time, lr_future_pred, label="Linear Regression (Future 12h)", color="orange", linestyle="--")
    ax3.set_title("Linear Regression Humidity Prediction")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Humidity (%)")
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(time_axis[:split_idx + len(lstm_test_pred)], df["humidity_denoised"][:split_idx + len(lstm_test_pred)], label="Actual Values", color="blue")
    ax4.plot(time_axis[split_idx + len(lstm_test_pred) - len(lstm_test_pred):split_idx + len(lstm_test_pred)], lstm_test_pred, label="LSTM Prediction (Test)", color="red", linestyle="--")
    ax4.plot(future_time, lstm_future_pred, label="LSTM Prediction (Future 12h)", color="orange", linestyle="--")
    ax4.set_title("LSTM Humidity Prediction")
    ax4.set_xlabel("Time")
    ax4.set_ylabel("Humidity (%)")
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig("prediction_results.png", dpi=300)
    print("Visualization results saved to prediction_results.png")
    plt.show()
