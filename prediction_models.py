import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings('ignore')

class PredictionModel:
    def __init__(self, df, predict_hours=12):
        """
        Initialize prediction models
        :param df: Preprocessed data DataFrame
        :param predict_hours: Number of hours to predict (default: 12 hours)
        """
        self.df = df
        self.predict_hours = predict_hours
        self.time_steps = 6
        self.lr_model = None
        self.lstm_model = None
    
    def _prepare_lr_data(self):
        """Prepare data for Linear Regression (use time index as feature to predict humidity)"""
        X = np.arange(len(self.df)).reshape(-1, 1)
        y = self.df["humidity_denoised"].values

        split_idx = int(len(X) * 0.8)
        self.X_train_lr, self.X_test_lr = X[:split_idx], X[split_idx:]
        self.y_train_lr, self.y_test_lr = y[:split_idx], y[split_idx:]
        return self.X_train_lr, self.X_test_lr, self.y_train_lr, self.y_test_lr
    
    def train_linear_regression(self):
        """Train Linear Regression model"""
        self._prepare_lr_data()
        self.lr_model = LinearRegression()
        self.lr_model.fit(self.X_train_lr, self.y_train_lr)
        
        test_pred = self.lr_model.predict(self.X_test_lr)

        future_idx = np.arange(len(self.df), len(self.df)+24).reshape(-1, 1)
        future_pred = self.lr_model.predict(future_idx)
        
        mse = mean_squared_error(self.y_test_lr, test_pred)
        r2 = r2_score(self.y_test_lr, test_pred)
        print(f"Linear Regression Evaluation: MSE={mse:.4f}, R²={r2:.4f}")
        
        return test_pred, future_pred
    
    def _create_sequences(self, data):
        """Create sequence data for LSTM (input: previous n time steps, output: next 1 time step)"""
        X, y = [], []
        for i in range(len(data) - self.time_steps):
            X.append(data[i:i+self.time_steps])
            y.append(data[i+self.time_steps])
        return np.array(X), np.array(y)
    
    def train_lstm(self):
        """Train LSTM model (predict humidity)"""

        humidity_data = self.df["humidity_norm"].values.reshape(-1, 1)

        X, y = self._create_sequences(humidity_data)

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        
        self.lstm_model = Sequential()
        self.lstm_model.add(LSTM(50, activation="relu", input_shape=(self.time_steps, 1)))
        self.lstm_model.add(Dense(1))
        self.lstm_model.compile(optimizer="adam", loss="mse")
        
        print("Training LSTM model...")
        self.lstm_model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))
        
        test_pred_norm = self.lstm_model.predict(X_test)

        test_pred = self.df["humidity_denoised"].max() * test_pred_norm + self.df["humidity_denoised"].min()
        y_test_original = self.df["humidity_denoised"].max() * y_test + self.df["humidity_denoised"].min()
        mse = mean_squared_error(y_test_original, test_pred)
        r2 = r2_score(y_test_original, test_pred)
        print(f"LSTM Model Evaluation: MSE={mse:.4f}, R²={r2:.4f}")
        
        last_sequence = humidity_data[-self.time_steps:]
        future_pred_norm = []
        for _ in range(24):
            seq = last_sequence.reshape(1, self.time_steps, 1)
            pred = self.lstm_model.predict(seq, verbose=0)
            future_pred_norm.append(pred[0][0])
            
            last_sequence = np.append(last_sequence[1:], pred, axis=0)

        future_pred = self.df["humidity_denoised"].max() * np.array(future_pred_norm) + self.df["humidity_denoised"].min()
        
        return test_pred, future_pred
