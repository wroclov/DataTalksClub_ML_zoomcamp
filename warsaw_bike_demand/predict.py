
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


# ---------------------------------------------------------
# 1. Load saved models
# ---------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent

MODEL_FILE = (
    PROJECT_DIR
    / "models"
    / "warsaw_bike_models.joblib"
)

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_FILE}. "
        "Run train.py first."
    )

artifacts = joblib.load(MODEL_FILE)

regression_model = artifacts["regression_model"]
stockout_classifier = artifacts["stockout_classifier"]

regression_features = artifacts["regression_features"]
classification_features = artifacts["classification_features"]

stockout_threshold = artifacts["stockout_threshold"]


# ---------------------------------------------------------
# 2. Create Flask application
# ---------------------------------------------------------

app = Flask(__name__)


# ---------------------------------------------------------
# 3. Home endpoint
# ---------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "message": "Warsaw Bike-Sharing Prediction API",
        "description": (
            "Predicts next-hour bike availability "
            "and stockout risk."
        ),
        "endpoints": {
            "health": "/health",
            "predict": "/predict"
        }
    })


# ---------------------------------------------------------
# 4. Health check
# ---------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "models_loaded": True
    })


# ---------------------------------------------------------
# 5. Prediction endpoint
# ---------------------------------------------------------

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "A valid JSON object is required."
        }), 400

    regression_input = data.get("regression")
    classification_input = data.get("classification")

    if not isinstance(regression_input, dict):
        return jsonify({
            "error": "Missing regression feature dictionary."
        }), 400

    if not isinstance(classification_input, dict):
        return jsonify({
            "error": "Missing classification feature dictionary."
        }), 400

    # Check required features
    missing_regression = [
        feature
        for feature in regression_features
        if feature not in regression_input
    ]

    missing_classification = [
        feature
        for feature in classification_features
        if feature not in classification_input
    ]

    if missing_regression or missing_classification:
        return jsonify({
            "error": "Missing required model features.",
            "missing_regression_features": missing_regression,
            "missing_classification_features": missing_classification
        }), 400

    try:
        # Prepare regression input
        X_regression = pd.DataFrame(
            [[
                regression_input[feature]
                for feature in regression_features
            ]],
            columns=regression_features
        )

        # Prepare classification input
        X_classification = pd.DataFrame(
            [[
                classification_input[feature]
                for feature in classification_features
            ]],
            columns=classification_features
        )

        # Check for non-numeric or invalid values
        X_regression = X_regression.apply(
            pd.to_numeric,
            errors="raise"
        )

        X_classification = X_classification.apply(
            pd.to_numeric,
            errors="raise"
        )

        # Generate predictions
        predicted_bikes = regression_model.predict(
            X_regression
        )[0]

        stockout_probability = (
            stockout_classifier.predict_proba(
                X_classification
            )[0, 1]
        )

        stockout_alert = (
            stockout_probability >= stockout_threshold
        )

        return jsonify({
            "predicted_available_bikes": round(
                float(predicted_bikes), 2
            ),
            "stockout_probability": round(
                float(stockout_probability), 4
            ),
            "stockout_risk_percent": round(
                float(stockout_probability * 100), 2
            ),
            "stockout_alert": bool(stockout_alert),
            "classification_threshold": float(
                stockout_threshold
            )
        })

    except (TypeError, ValueError) as error:
        return jsonify({
            "error": str(error)
        }), 400

    except Exception:
        app.logger.exception("Prediction failed")

        return jsonify({
            "error": "Prediction failed."
        }), 500


# ---------------------------------------------------------
# 6. Run application
# ---------------------------------------------------------

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
