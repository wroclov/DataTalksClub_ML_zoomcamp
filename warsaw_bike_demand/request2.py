import requests
import pandas as pd
import sqlite3
import joblib
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

# Load model artifacts
artifacts = joblib.load(
    PROJECT_DIR / "models" / "warsaw_bike_models.joblib"
)

# Load one real row from the processed dataset
conn = sqlite3.connect(
    PROJECT_DIR / "data" / "processed" / "bike_hourly.sqlite"
)

hourly_df = pd.read_sql_query(
    "SELECT * FROM hourly_station LIMIT 100",
    conn
)
conn.close()

# Use a row from the actual modeling test dataset if available.
# For this quick API test, use representative numeric feature values.
features = {
    "avg_available_bikes": 8.0,
    "min_available_bikes": 5.0,
    "stockout_rate": 0.0,
    "avg_free_racks": 10.0,
    "observations": 12,
    "latitude": 52.2297,
    "longitude": 21.0122,
    "hour": 12,
    "day_of_week": 2,
    "month": 9,
    "is_weekend": 0,
    "hour_sin": 0.0,
    "hour_cos": -1.0,
    "weekday_sin": 0.0,
    "weekday_cos": 1.0,
}

payload = {
    "regression": features,
    "classification": features,
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json=payload
)

print("Status:", response.status_code)
print("Response:")
print(response.json())