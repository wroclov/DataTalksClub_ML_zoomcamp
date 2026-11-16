
from pathlib import Path
from datetime import datetime
import sqlite3

import joblib
import numpy as np
import pandas as pd

from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
)


# ---------------------------------------------------------
# 1. Paths and configuration
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DB_FILE = PROJECT_DIR / "data" / "processed" / "bike_hourly.sqlite"
MODEL_DIR = PROJECT_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "warsaw_bike_models.joblib"

RANDOM_STATE = 42

if not DB_FILE.exists():
    raise FileNotFoundError(
        f"SQLite database not found: {DB_FILE}\n"
        "Make sure data/processed/bike_hourly.sqlite exists."
    )

print("Project directory:", PROJECT_DIR)
print("Database:", DB_FILE)


# ---------------------------------------------------------
# 2. Load hourly station data from SQLite
# ---------------------------------------------------------

query = """
SELECT
    hour AS timestamp,
    station_id,
    name,
    latitude,
    longitude,
    sum_available_bikes * 1.0 / observations
        AS avg_available_bikes,
    min_available_bikes,
    stockout_observations,
    stockout_observations * 1.0 / observations
        AS stockout_rate,
    sum_free_racks * 1.0 / observations
        AS avg_free_racks,
    observations
FROM hourly_station
ORDER BY timestamp, station_id
"""

with sqlite3.connect(DB_FILE) as conn:
    integrity = conn.execute(
        "PRAGMA integrity_check"
    ).fetchone()[0]

    if integrity != "ok":
        raise RuntimeError(
            f"SQLite integrity check failed: {integrity}"
        )

    hourly_df = pd.read_sql_query(query, conn)

hourly_df["timestamp"] = pd.to_datetime(
    hourly_df["timestamp"]
)

print("\nHourly dataset shape:", hourly_df.shape)
print("Unique stations:", hourly_df["station_id"].nunique())
print(
    "Date range:",
    hourly_df["timestamp"].min(),
    "to",
    hourly_df["timestamp"].max()
)


# ---------------------------------------------------------
# 3. Create next-hour regression dataset
# ---------------------------------------------------------

model_df = hourly_df.sort_values(
    ["station_id", "timestamp"]
).copy()

grouped = model_df.groupby("station_id", sort=False)

model_df["next_timestamp"] = (
    grouped["timestamp"].shift(-1)
)

model_df["target_next_hour"] = (
    grouped["avg_available_bikes"].shift(-1)
)

model_df["next_observations"] = (
    grouped["observations"].shift(-1)
)

# Keep consecutive hours with complete observations
model_df = model_df[
    (
        model_df["next_timestamp"]
        == model_df["timestamp"] + pd.Timedelta(hours=1)
    )
    & (model_df["observations"] == 12)
    & (model_df["next_observations"] == 12)
].copy()

# Time features
model_df["hour"] = model_df["timestamp"].dt.hour
model_df["day_of_week"] = (
    model_df["timestamp"].dt.day_of_week
)
model_df["month"] = model_df["timestamp"].dt.month

model_df["is_weekend"] = (
    model_df["day_of_week"] >= 5
).astype(int)

# Cyclical time encoding
model_df["hour_sin"] = np.sin(
    2 * np.pi * model_df["hour"] / 24
)
model_df["hour_cos"] = np.cos(
    2 * np.pi * model_df["hour"] / 24
)
model_df["weekday_sin"] = np.sin(
    2 * np.pi * model_df["day_of_week"] / 7
)
model_df["weekday_cos"] = np.cos(
    2 * np.pi * model_df["day_of_week"] / 7
)

print("\nRegression dataset shape:", model_df.shape)
print("Complete current observations:",
      model_df["observations"].eq(12).all())
print("Complete next observations:",
      model_df["next_observations"].eq(12).all())


# ---------------------------------------------------------
# 4. Define features and chronological split
# ---------------------------------------------------------

features = [
    "avg_available_bikes",
    "min_available_bikes",
    "stockout_rate",
    "avg_free_racks",
    "observations",
    "latitude",
    "longitude",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
]

target = "target_next_hour"

unique_times = np.sort(
    model_df["timestamp"].unique()
)

split_time = unique_times[
    int(len(unique_times) * 0.8)
]

train_df = model_df[
    model_df["timestamp"] < split_time
].copy()

test_df = model_df[
    model_df["timestamp"] >= split_time
].copy()

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

print("\nChronological split:", split_time)
print("Regression training rows:", len(X_train))
print("Regression testing rows:", len(X_test))


# ---------------------------------------------------------
# 5. Train regression model
# ---------------------------------------------------------

regression_model = XGBRegressor(
    n_estimators=200,
    max_depth=7,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:squarederror",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

regression_model.fit(X_train, y_train)

regression_predictions = regression_model.predict(X_test)

baseline_predictions = X_test["avg_available_bikes"]


def print_regression_metrics(name, actual, predicted):
    print(f"\n{name}")
    print(
        "MAE:",
        round(mean_absolute_error(actual, predicted), 4)
    )
    print(
        "RMSE:",
        round(
            np.sqrt(mean_squared_error(actual, predicted)),
            4
        )
    )
    print(
        "R2:",
        round(r2_score(actual, predicted), 4)
    )


print_regression_metrics(
    "Persistence baseline",
    y_test,
    baseline_predictions,
)

print_regression_metrics(
    "XGBoost regression",
    y_test,
    regression_predictions,
)


# ---------------------------------------------------------
# 6. Create next-hour stockout classification dataset
# ---------------------------------------------------------

classification_df = model_df.sort_values(
    ["station_id", "timestamp"]
).copy()

grouped = classification_df.groupby(
    "station_id",
    sort=False
)

classification_df["next_timestamp"] = (
    grouped["timestamp"].shift(-1)
)

classification_df["next_stockout_observations"] = (
    grouped["stockout_observations"].shift(-1)
)

classification_df["next_observations"] = (
    grouped["observations"].shift(-1)
)

# The target is whether the following hour has
# at least one zero-bike observation.
classification_df = classification_df[
    (
        classification_df["next_timestamp"]
        == classification_df["timestamp"]
        + pd.Timedelta(hours=1)
    )
    & (classification_df["observations"] == 12)
    & (classification_df["next_observations"] == 12)
].copy()

classification_df["target_stockout"] = (
    classification_df["next_stockout_observations"] > 0
).astype(int)

classification_features = features

print(
    "\nClassification dataset shape:",
    classification_df.shape
)
print(
    "Stockout prevalence:",
    round(
        classification_df["target_stockout"].mean() * 100,
        2
    ),
    "%"
)


# ---------------------------------------------------------
# 7. Chronological classification train/test split
# ---------------------------------------------------------

train_cls = classification_df[
    classification_df["timestamp"] < split_time
].copy()

test_cls = classification_df[
    classification_df["timestamp"] >= split_time
].copy()

X_train_cls = train_cls[classification_features]
y_train_cls = train_cls["target_stockout"]

X_test_cls = test_cls[classification_features]
y_test_cls = test_cls["target_stockout"]

print("Classification training rows:", len(X_train_cls))
print("Classification testing rows:", len(X_test_cls))
print(
    "Training stockout rate:",
    round(y_train_cls.mean() * 100, 2),
    "%"
)
print(
    "Testing stockout rate:",
    round(y_test_cls.mean() * 100, 2),
    "%"
)


# ---------------------------------------------------------
# 8. Select classification threshold on validation data
# ---------------------------------------------------------

train_times = np.sort(
    train_cls["timestamp"].unique()
)

validation_start = train_times[
    int(len(train_times) * 0.8)
]

fit_df = train_cls[
    train_cls["timestamp"] < validation_start
].copy()

validation_df = train_cls[
    train_cls["timestamp"] >= validation_start
].copy()

X_fit = fit_df[classification_features]
y_fit = fit_df["target_stockout"]

X_val = validation_df[classification_features]
y_val = validation_df["target_stockout"]

negative_count = (y_fit == 0).sum()
positive_count = (y_fit == 1).sum()

if positive_count == 0:
    raise ValueError(
        "No positive stockout examples in fitting data."
    )

validation_weight = negative_count / positive_count

validation_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=validation_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

validation_model.fit(X_fit, y_fit)

val_probabilities = validation_model.predict_proba(
    X_val
)[:, 1]

precision, recall, thresholds = precision_recall_curve(
    y_val,
    val_probabilities,
)

threshold_table = pd.DataFrame({
    "threshold": thresholds,
    "precision": precision[:-1],
    "recall": recall[:-1],
})

eligible = threshold_table[
    threshold_table["precision"] >= 0.50
]

if eligible.empty:
    raise ValueError(
        "No validation threshold achieved 50% precision."
    )

best_row = eligible.loc[
    eligible["recall"].idxmax()
]

selected_threshold = float(
    best_row["threshold"]
)

print("\nValidation period:")
print(
    validation_df["timestamp"].min(),
    "to",
    validation_df["timestamp"].max()
)
print(
    "Selected threshold:",
    round(selected_threshold, 4)
)
print(
    "Validation precision:",
    round(float(best_row["precision"]), 3)
)
print(
    "Validation recall:",
    round(float(best_row["recall"]), 3)
)


# ---------------------------------------------------------
# 9. Train final classifier on the full training period
# ---------------------------------------------------------

negative_count = (y_train_cls == 0).sum()
positive_count = (y_train_cls == 1).sum()

if positive_count == 0:
    raise ValueError(
        "No positive stockout examples in training data."
    )

final_weight = negative_count / positive_count

stockout_classifier = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    scale_pos_weight=final_weight,
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE,
)

stockout_classifier.fit(
    X_train_cls,
    y_train_cls
)

final_probabilities = (
    stockout_classifier.predict_proba(X_test_cls)[:, 1]
)

final_predictions = (
    final_probabilities >= selected_threshold
).astype(int)

print("\nFinal stockout classification report:")
print(
    classification_report(
        y_test_cls,
        final_predictions,
        target_names=["No stockout", "Stockout"],
        zero_division=0,
    )
)

print("Confusion matrix:")
print(
    confusion_matrix(y_test_cls, final_predictions)
)

print(
    "PR-AUC:",
    round(
        average_precision_score(
            y_test_cls,
            final_probabilities
        ),
        4
    )
)

print(
    "ROC-AUC:",
    round(
        roc_auc_score(
            y_test_cls,
            final_probabilities
        ),
        4
    )
)


# ---------------------------------------------------------
# 10. Save models and feature definitions
# ---------------------------------------------------------

model_artifacts = {
    "regression_model": regression_model,
    "stockout_classifier": stockout_classifier,
    "regression_features": features,
    "classification_features": classification_features,
    "stockout_threshold": selected_threshold,
    "created_at": datetime.now().isoformat(),
    "description": (
        "Warsaw bike-sharing demand and stockout prediction"
    ),
}

joblib.dump(model_artifacts, MODEL_FILE)

print("\nModels saved successfully.")
print("Model file:", MODEL_FILE)
print(
    "File size (MB):",
    round(MODEL_FILE.stat().st_size / (1024 ** 2), 2)
)


# ---------------------------------------------------------
# 11. Verify saved model can be loaded
# ---------------------------------------------------------

loaded_artifacts = joblib.load(MODEL_FILE)

sample_regression = loaded_artifacts[
    "regression_model"
].predict(X_test.head(5))

sample_stockout = loaded_artifacts[
    "stockout_classifier"
].predict_proba(X_test_cls.head(5))[:, 1]

print("\nModel reload test successful.")
print("Sample predicted bikes:", sample_regression)
print("Sample stockout probabilities:", sample_stockout)
print(
    "Saved threshold:",
    loaded_artifacts["stockout_threshold"]
)
