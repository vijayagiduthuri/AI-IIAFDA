from functools import lru_cache

import pandas as pd

from backend.config import (
    INVOICES_FILE,
    SUPPLIERS_FILE,
    DEPARTMENTS_FILE,
    BEHAVIOURAL_FILE
)

from backend.fraud_detector import FEATURE_COLUMNS


# --------------------------------------------------
# LOAD STRUCTURED DATASET
# --------------------------------------------------

@lru_cache(maxsize=1)
def load_structured_data():
    """
    Load and merge the original invoice dataset tables.

    Returns the complete structured invoice dataset.
    Does not modify the original Parquet files.

    The result is cached: the parquet files are read and merged only
    once per app run, not on every invoice or question.
    Callers must not modify the returned DataFrame in place
    (use .copy() first).
    """

    invoices = pd.read_parquet(INVOICES_FILE)
    suppliers = pd.read_parquet(SUPPLIERS_FILE)
    departments = pd.read_parquet(DEPARTMENTS_FILE)
    behavioural = pd.read_parquet(BEHAVIOURAL_FILE)

    # Validate key uniqueness
    if not invoices["invoice_id"].is_unique:
        raise ValueError("Duplicate invoice_id values found.")

    if not suppliers["supplier_id"].is_unique:
        raise ValueError("Duplicate supplier_id values found.")

    if not departments["department_id"].is_unique:
        raise ValueError("Duplicate department_id values found.")

    if not behavioural["invoice_id"].is_unique:
        raise ValueError("Duplicate behavioural invoice_id values found.")

    # Merge invoice and supplier information
    data = invoices.merge(
        suppliers,
        on="supplier_id",
        how="left",
        validate="many_to_one"
    )

    # Merge department information
    data = data.merge(
        departments,
        on="department_id",
        how="left",
        validate="many_to_one"
    )

    # Merge behavioural features
    data = data.merge(
        behavioural,
        on="invoice_id",
        how="left",
        validate="one_to_one"
    )

    return data


@lru_cache(maxsize=1)
def _invoice_index():
    """Map invoice_id -> row position, built once for fast lookups."""

    data = load_structured_data()

    return pd.Series(
        range(len(data)),
        index=data["invoice_id"].astype(str)
    )


def get_invoice_record(invoice_id):
    """
    Return one invoice from the structured dataset as a one-row
    DataFrame (a copy), or None if the invoice_id is not found.
    """

    index = _invoice_index()
    key = str(invoice_id)

    if key not in index.index:
        return None

    data = load_structured_data()

    return data.iloc[[int(index[key])]].copy()


def get_supplier_history(supplier_id, exclude_invoice_id=None, limit=10):
    """Return the supplier's most recent invoices from the dataset."""

    data = load_structured_data()

    history = data.loc[
        data["supplier_id"].astype(str) == str(supplier_id)
    ]

    if exclude_invoice_id is not None:
        history = history.loc[
            history["invoice_id"].astype(str) != str(exclude_invoice_id)
        ]

    columns = [
        column for column in [
            "invoice_id", "invoice_date", "invoice_amount",
            "invoice_type", "duplicate_invoice_flag",
            "split_invoice_flag", "late_night_submission_flag"
        ]
        if column in history.columns
    ]

    return (
        history.sort_values("invoice_date")
        .tail(limit)[columns]
        .copy()
    )


# --------------------------------------------------
# FEATURE ENGINEERING
# --------------------------------------------------

def prepare_model_features(data):
    """
    Create derived features using the original training formulas.
    """

    features_df = data.copy()

    features_df["invoice_date"] = pd.to_datetime(
        features_df["invoice_date"],
        errors="coerce"
    )

    # Date-based features
    features_df["invoice_year"] = (
        features_df["invoice_date"].dt.year
    )

    features_df["invoice_month"] = (
        features_df["invoice_date"].dt.month
    )

    features_df["invoice_day"] = (
        features_df["invoice_date"].dt.day
    )

    features_df["invoice_day_of_week"] = (
        features_df["invoice_date"].dt.day_of_week
    )

    features_df["is_weekend"] = (
        features_df["invoice_day_of_week"] >= 5
    ).astype(int)

    # Amount-based features
    features_df["amount_to_supplier_avg"] = (
        features_df["invoice_amount"] /
        features_df["supplier_avg_amount_90d"].replace(0, 1)
    )

    features_df["amount_to_supplier_overall_avg"] = (
        features_df["invoice_amount"] /
        features_df["avg_invoice_amount"].replace(0, 1)
    )

    features_df["amount_to_department_budget"] = (
        features_df["invoice_amount"] /
        features_df["annual_budget"].replace(0, 1)
    )

    return features_df


# --------------------------------------------------
# MODEL FEATURE VALIDATION
# --------------------------------------------------

def validate_model_features(features_df, verbose=False):
    """
    Validate and arrange the 24 model input features.
    Set verbose=True to print a summary (used when testing).
    """

    missing_features = [
        column
        for column in FEATURE_COLUMNS
        if column not in features_df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing model features: {missing_features}"
        )

    model_input = features_df[FEATURE_COLUMNS].copy()

    missing_values = int(
        model_input.isnull().sum().sum()
    )

    if verbose:
        print("\nMODEL FEATURE VALIDATION")
        print("-" * 50)
        print("Expected features:", len(FEATURE_COLUMNS))
        print("Available features:", model_input.shape[1])
        print("Missing feature names:", missing_features)
        print("Model input shape:", model_input.shape)
        print("Missing values:", missing_values)

    if missing_values > 0:
        raise ValueError(
            "Missing values found in model input. "
            "Check the original training preprocessing."
        )

    if verbose:
        print("Feature validation successful!")

    return model_input


# --------------------------------------------------
# MODEL PREDICTION TEST
# --------------------------------------------------

def test_model_prediction(model_input, original_data):
    """
    Run a prediction on one real invoice from the dataset.
    """

    from backend.fraud_detector import predict_invoice

    if model_input.empty:
        raise ValueError("No records available for prediction.")

    sample_invoice = model_input.iloc[0].to_dict()

    result = predict_invoice(sample_invoice)

    invoice_id = original_data.iloc[0]["invoice_id"]

    print("Invoice ID:", invoice_id)
    print(
        "Invoice Amount:",
        original_data.iloc[0]["invoice_amount"]
    )

    print("\nPrediction Results:")

    for key, value in result.items():
        print(f"{key}: {value}")

    return result


# --------------------------------------------------
# STANDALONE EXECUTION
# --------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("LOADING INVOICE DATASET")
    print("=" * 60)

    data = load_structured_data()

    print("Structured dataset shape:", data.shape)

    features_df = prepare_model_features(data)

    print(
        "Feature-engineered dataset shape:",
        features_df.shape
    )

    derived_features = [
        "invoice_year",
        "invoice_month",
        "invoice_day",
        "invoice_day_of_week",
        "is_weekend",
        "amount_to_supplier_avg",
        "amount_to_supplier_overall_avg",
        "amount_to_department_budget"
    ]

    print("\nDERIVED FEATURE SAMPLE")
    print("-" * 50)

    print(
        features_df[derived_features]
        .head(5)
        .to_string(index=False)
    )

    model_input = validate_model_features(features_df, verbose=True)

    prediction_result = test_model_prediction(
        model_input,
        data
    )

    print("\n" + "=" * 60)
    print("DATA LOADING AND MODEL TEST COMPLETED")
    print("=" * 60)