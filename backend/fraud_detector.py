import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "final_invoice_fraud_model.pkl"
)

# Load saved model package
saved_objects = joblib.load(MODEL_PATH)

model = saved_objects["model"]
MODEL_NAME = saved_objects["model_name"]
FEATURE_COLUMNS = saved_objects["feature_columns"]
RISK_THRESHOLDS = saved_objects["risk_thresholds"]


def predict_invoice(invoice_data):
    """
    Predict invoice fraud using the saved Random Forest pipeline.

    invoice_data must contain all 24 model features.
    """

    # Arrange input in the exact training feature order
    input_df = pd.DataFrame(
        [invoice_data],
        columns=FEATURE_COLUMNS
    )

    # Predict fraud probability
    probabilities = model.predict_proba(input_df)[0]

    fraud_probability = float(probabilities[1]) * 100

    # Fraud classification
    prediction = int(model.predict(input_df)[0])

    # Risk classification using saved thresholds
    if fraud_probability <= RISK_THRESHOLDS["low_max"]:
        risk_level = "LOW"

    elif fraud_probability <= RISK_THRESHOLDS["medium_max"]:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "model_name": MODEL_NAME,
        "prediction": prediction,
        "fraud_probability": round(fraud_probability, 2),
        "risk_level": risk_level,
        "decision": "FRAUD" if prediction == 1 else "NOT FRAUD"
    }


if __name__ == "__main__":
    print("Model loaded successfully!")
    print("Model:", MODEL_NAME)
    print("Required features:", len(FEATURE_COLUMNS))
    print("Risk thresholds:", RISK_THRESHOLDS)