import numpy as np
import requests
from datetime import datetime, timedelta
import time

def fetch_real_weather_data(latitude=40.7128, longitude=-74.0060,
                            days_back=365, location_name="New York"):
    """
    Fetch real historical weather data from Open-Meteo

    Args:
        latitude = Location latitude
        longitude = Location longitude
        days_back = Number of days of historical data to fetch
        location_name = Name of location for display
    """
    print(f"\nFetching real weather data for {location_name}...")
    print(f"  Coordinates: ({latitude}, {longitude})")
    print(f"  Time range: Last {days_back} days")

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    }

    print(f"   API Request: {url}")

    try: 
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()

        if "hourly" in data and "temperature_2m" in data["hourly"]:
            temperatures = np.array(data["hourly"]["temperature_2m"])
            humidity = np.array(data["hourly"]["relative_humidity_2m"])
            precipitation = np.array(data["hourly"]["precipitation"])
            wind_speed = np.array(data["hourly"]["wind_speed_10m"])
            timestamps = np.array(data["hourly"]["time"])

            print(f"   Successfully fetched {len(temperatures)} hourly data points")
            print(f"   Temperature range: {np.min(temperatures):.1f}" + r"$\degree$C " + f"to {np.max(temperatures):.1f}" + r"$\degree$")
            print(f"   Humidity range: {np.min(humidity):.1f}% to {np.max(humidity):.1f}%")

            return {
                "temperature": temperatures,
                "humidity": humidity,
                "precipitation": precipitation,
                "wind_speed": wind_speed,
                "timestamps": timestamps,
                "location": location_name,
                "coords": (latitude, longitude)
            }
        else:
            print("   Error: No data returned from API")
            return None
    
    except Exception as e:
        print(f"   Error fetching data: {e}")
        return None
    

def prepare_sequences(data, seq_length=48, pred_length=24):
    """
    Prepare sequences for training

    Args:
        data: Weather data array
        seq_length: Length of input sequence (hours)
        pred_length: Length of prediction sequence (hours)
    """
    sequences = []
    targets = []

    for i in range(len(data) - seq_length - pred_length):
        # Input: seq_length hours
        seq = data[i: i + seq_length]
        # Target: next pred_length hours
        target = data[i + seq_length:i + seq_length + pred_length]

        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)