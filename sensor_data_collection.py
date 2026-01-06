import time
import csv
import os
from datetime import datetime
import Adafruit_DHT

SENSOR_TYPE = Adafruit_DHT.DHT11
GPIO_PIN = 4
COLLECTION_INTERVAL = 1800
DATA_SAVE_PATH = "data/sensor_data.csv"

os.makedirs(os.path.dirname(DATA_SAVE_PATH), exist_ok=True)

def init_csv():
    if not os.path.exists(DATA_SAVE_PATH):
        with open(DATA_SAVE_PATH, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "temperature", "humidity"])

def read_sensor_data():
    humidity, temperature = Adafruit_DHT.read_retry(SENSOR_TYPE, GPIO_PIN)
    if humidity is None or temperature is None:
        return None, None
    return round(temperature, 2), round(humidity, 2)

def save_data(temperature, humidity):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DATA_SAVE_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, temperature, humidity])
    print(f"[{timestamp}] data：temperature={temperature}℃，humidity={humidity}%")

if __name__ == "__main__":
    init_csv()
    print("Start...")
    try:
        while True:
            temp, hum = read_sensor_data()
            if temp is not None and hum is not None:
                save_data(temp, hum)
            else:
                print("Failed")
            time.sleep(COLLECTION_INTERVAL)
    except KeyboardInterrupt:
        print("\n stop：", DATA_SAVE_PATH)

