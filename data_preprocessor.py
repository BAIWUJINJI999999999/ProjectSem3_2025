import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, df):
        """
        Initialize data preprocessor
        :param df: Raw sensor data DataFrame
        """
        self.df = df.copy()
        self.scaler = MinMaxScaler(feature_range=(0, 1))  # Normalize to [0,1]
    
    def moving_average_denoise(self, window_size=3):
        """Denoise data using moving average method"""
        self.df["temperature_denoised"] = self.df["temperature"].rolling(window=window_size, center=True).mean()
        self.df["humidity_denoised"] = self.df["humidity"].rolling(window=window_size, center=True).mean()
        # Fill empty values at the start and end
        self.df.fillna(method="bfill", inplace=True)
        self.df.fillna(method="ffill", inplace=True)
        print("Moving average denoising completed")
        return self.df
    
    def normalize_data(self):
        """Normalize temperature and humidity data"""
        self.df["temperature_norm"] = self.scaler.fit_transform(self.df[["temperature_denoised"]])
        self.df["humidity_norm"] = self.scaler.fit_transform(self.df[["humidity_denoised"]])
        print("Data normalization completed (range: [0,1])")
        return self.df
