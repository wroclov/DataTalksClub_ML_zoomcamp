import requests

features = {
    "avg_available_bikes": 10.0,
    "min_available_bikes": 5.0,
    "stockout_rate": 0.0,
    "avg_free_racks": 8.0,
    "observations": 12,
    "latitude": 52.2297,
    "longitude": 21.0122,
    "hour": 14,
    "day_of_week": 2,
    "month": 9,
    "is_weekend": 0,
    "hour_sin": 0.5,
    "hour_cos": -0.866,
    "weekday_sin": 0.975,
    "weekday_cos": -0.223
}

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={
        "regression": features,
        "classification": features
    }
)

print(response.status_code)
print(response.json())