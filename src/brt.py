"""
Block-Recurrent Transformer Architecture. Example, not necessarily how we will do things.

This will use *real world* weather data to predict the weather. 
This file will be how we would do things *without* PyTorch, so this is like the "from scratch" method.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
from datetime import datetime, timedelta
import time
import os

from data_fetch.real_weather_data import *


def main():
    print(f"Recurrent Transformer (\"from scratch\") - Real Weather Data")

    # The cities we will fetch data from
    cities = [
        (40.7128, -74.0060, "New York"),
        (51.5074, -0.1278, "London"),
        (35.6762, 139.6503, "Tokyo"),
        (34.0522, -118.2437, "Los Angeles"),
        (-33.8688, 151.2093, "Sydney"),
        (38.8121, 77.6364, "Haymarket"),
        (25.0330, 121.5654, "Taipei"),
        (13.0843, 80.2705, "Chennai")
    ]

    all_temp_data = []

    for lat, lon, name in cities:
        weather_data = fetch_real_weather_data(lat, lon, days_back=365, location_name=name)
        if weather_data is not None:
            all_temp_data.append(weather_data["temperature"])
        time.sleep(0.5)  # be kind to the API
    
    if not all_temp_data:
        print("Failed to fetch weather data. Exiting.")
        return
    
    print(f"\nTotal data collected from {len(all_temp_data)} cities")


if __name__ == "__main__":
    main()
