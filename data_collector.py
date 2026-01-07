import time
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class SensorDataCollector:
    def __init__(self, use_raspberry_pi=False, collect_duration_hours=48, interval_minutes=30):
        """
        Initialize the data collector
        :param use_raspberry_pi: Whether to use real Raspberry Pi hardware (False = generate simulated data)
        :param collect_duration_hours: Data collection duration in hours (default: 48 hours = 2 days)
        :param interval_minutes: Collection interval in minutes (default: 30 minutes)
        """
        self.use_raspberry_pi = use_raspberry_pi
        self.collect_duration = collect_duration_hours
        self.interval = interval_minutes
        self.data = []  # Store collected time, temperature, humidity
    
    def _read_sensor(self):
        """Read sensor data (real hardware or simulated)"""
        if self.use_raspberry_pi:
            # Raspberry Pi real sensor reading logic (requires RPi.GPIO library)
            # Note: Modify pin numbers according to your sensor wiring
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(17, GPIO.IN)  # Humidity sensor pin
            GPIO.setup(27, GPIO.IN)  # Temperature sensor pin (simulated, use DS18B20 for real)
            
            # Read raw values and convert to physical values (calibrate for your sensor)
            humidity_raw = GPIO.input(17)
            temperature_raw = GPIO.input(27)
            humidity = np.clip(random.uniform(20, 80) + humidity_raw*5, 0, 100)  # Humidity (%)
            temperature = np.clip(random.uniform(15, 35) + temperature_raw*2, 10, 40)  # Temperature (°C)
            
            GPIO.cleanup()
        else:
            # Generate simulated data (simulate 2 days without watering: humidity decreases slowly)
            base_humidity = 70 - (len(self.data) * 0.5)  # Humidity decreases over time
            humidity = np.clip(base_humidity + random.uniform(-3, 3), 10, 80)
            temperature = np.clip(25 + random.uniform(-2, 2), 15, 35)
        
        return temperature, humidity
    
    def collect_data(self):
        """Start data collection"""
        print(f"Starting data collection (Duration: {self.collect_duration} hours, Interval: {self.interval} minutes)...")
        start_time = datetime.now()
        end_time = start_time + timedelta(hours=self.collect_duration)
        
        while datetime.now() < end_time:
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            temp, humi = self._read_sensor()
            self.data.append([current_time, temp, humi])
            
            print(f"[{current_time}] Temperature: {temp:.2f}°C, Humidity: {humi:.2f}%")
            time.sleep(self.interval * 60)  # Wait for specified minutes
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.data, columns=["timestamp", "temperature", "humidity"])
        df.to_csv("sensor_data.csv", index=False)
        print("Data collection completed! Saved to sensor_data.csv")
        return df
