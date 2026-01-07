from data_collector import SensorDataCollector
from data_preprocessor import DataPreprocessor
from prediction_models import PredictionModel
from visualization import visualize_results
import pandas as pd

if __name__ == "__main__":
    # 1. Collect data (set use_raspberry_pi=True if using real hardware)
    collector = SensorDataCollector(use_raspberry_pi=False, collect_duration_hours=48, interval_minutes=30)
    df = collector.collect_data()
    
    # 2. Preprocess data
    preprocessor = DataPreprocessor(df)
    df = preprocessor.moving_average_denoise(window_size=3)
    df = preprocessor.normalize_data()
    
    # 3. Train prediction models
    model = PredictionModel(df, predict_hours=12)
    lr_test_pred, lr_future_pred = model.train_linear_regression()
    lstm_test_pred, lstm_future_pred = model.train_lstm()
    
    # 4. Visualize results
    visualize_results(df, lr_test_pred, lr_future_pred, lstm_test_pred, lstm_future_pred)
    
    # 5. Save key results
    result_df = pd.DataFrame({
        "timestamp": df["timestamp"],
        "humidity_denoised": df["humidity_denoised"],
        "temperature_denoised": df["temperature_denoised"]
    })
    result_df.to_csv("preprocessed_data.csv", index=False)
    print("Preprocessed data saved to preprocessed_data.csv")
