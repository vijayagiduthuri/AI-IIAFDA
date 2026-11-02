"""
Feature builder for NEW invoices (invoices that are not in the dataset).

The fraud model needs the same 24 features it was trained on. For an
invoice that is already in the dataset these come straight from the
parquet files. For a new invoice we rebuild them:

  From the invoice (OCR):
      invoice_amount (total), invoice_date, payment_terms

  From the supplier master table:
      supplier_country, supplier_age_days, supplier_risk_score,
      blacklisted_flag, avg_invoice_amount

  From the supplier's past invoices in the dataset:
      supplier_invoice_count_30d, supplier_avg_amount_90d, and the
      supplier's usual department and invoice type (these are not
      printed on the invoice)

  From all dataset amounts:
      invoice_amount_zscore (global mean / population std, as in the dataset)

  From the upload time:
      submission_hour, late_night_submission_flag

Every approximation made is returned in `assumptions`, so it can be
shown to the user and reported honestly.
"""

from functools import lru_cache

import pandas as pd

from backend.config import SUPPLIERS_FILE, DEPARTMENTS_FILE
from backend.data_loader import load_structured_data


# --------------------------------------------------
# CACHED REFERENCE TABLES
# --------------------------------------------------

@lru_cache(maxsize=1)
def _suppliers():
    table = pd.read_parquet(SUPPLIERS_FILE)
    table["supplier_id"] = table["supplier_id"].astype(str)
    return table.set_index("supplier_id")


@lru_cache(maxsize=1)
def _departments():
    table = pd.read_parquet(DEPARTMENTS_FILE)
    table["department_id"] = table["department_id"].astype(str)
    return table.set_index("department_id")


@lru_cache(maxsize=1)
def _history_by_supplier():
    """Supplier's invoices with parsed dates, grouped once for speed."""

    data = load_structured_data()

    history = data[[
        "invoice_id", "supplier_id", "department_id", "invoice_date",
        "invoice_amount", "invoice_type", "payment_terms",
        "submission_hour", "late_night_submission_flag"
    ]].copy()

    history["supplier_id"] = history["supplier_id"].astype(str)
    history["invoice_date"] = pd.to_datetime(
        history["invoice_date"], errors="coerce"
    )

    return {
        supplier_id: group.sort_values("invoice_date")
        for supplier_id, group in history.groupby("supplier_id")
    }


@lru_cache(maxsize=1)
def late_night_hours():
    """
    Hours that the dataset marks as late-night submissions,
    learned from the data itself (hour -> flag mostly 1).
    """

    data = load_structured_data()

    rate = (
        data.groupby("submission_hour")["late_night_submission_flag"]
        .mean()
    )

    return frozenset(int(hour) for hour, value in rate.items() if value >= 0.5)


@lru_cache(maxsize=1)
def amount_zscore_stats():
    """
    Mean and population std of invoice_amount over the whole dataset.
    The dataset's invoice_amount_zscore is computed against these
    global values (verified: exact match), not per supplier.
    """

    amounts = load_structured_data()["invoice_amount"]

    return float(amounts.mean()), float(amounts.std(ddof=0))


@lru_cache(maxsize=1)
def known_values():
    """Valid payment terms and currencies seen in the dataset."""

    data = load_structured_data()

    return {
        "payment_terms": frozenset(data["payment_terms"].astype(str).str.upper()),
        "currency": frozenset(data["currency"].astype(str).str.upper())
    }


# --------------------------------------------------
# PUBLIC HELPERS
# --------------------------------------------------

def get_supplier(supplier_id):
    """Return the supplier master record as a dict, or None if unknown."""

    if not supplier_id:
        return None

    suppliers = _suppliers()
    key = str(supplier_id)

    if key not in suppliers.index:
        return None

    record = suppliers.loc[key].to_dict()
    record["supplier_id"] = key

    return record


def supplier_invoice_history(supplier_id):
    """All dataset invoices of a supplier (empty DataFrame if none)."""

    return _history_by_supplier().get(str(supplier_id), pd.DataFrame())


def find_similar_dataset_invoices(supplier_id, amount, invoice_date,
                                  amount_tolerance=1.00, day_window=7):
    """
    Possible duplicates INSIDE the dataset: same supplier, same amount
    (within tolerance) and an invoice date within `day_window` days.
    """

    history = supplier_invoice_history(supplier_id)

    if history.empty or amount is None:
        return []

    matches = history.loc[
        (history["invoice_amount"] - float(amount)).abs() <= amount_tolerance
    ]

    date = pd.to_datetime(invoice_date, errors="coerce")

    if not pd.isna(date):
        matches = matches.loc[
            (matches["invoice_date"] - date).abs().dt.days <= day_window
        ]

    return [
        {
            "invoice_id": row["invoice_id"],
            "invoice_amount": float(row["invoice_amount"]),
            "invoice_date": (
                row["invoice_date"].strftime("%Y-%m-%d")
                if not pd.isna(row["invoice_date"]) else None
            ),
            "source": "dataset"
        }
        for _, row in matches.head(5).iterrows()
    ]


# --------------------------------------------------
# FEATURE BUILDER
# --------------------------------------------------

def build_new_invoice_record(extracted_fields, supplier, submission_time=None,
                             exclude_invoice_id=None):
    """
    Build a one-row DataFrame shaped like a dataset invoice record
    (same columns as load_structured_data()), ready for
    prepare_model_features() and the fraud model.

    exclude_invoice_id: leave this invoice out of the supplier history
    (only needed when testing with invoices that are in the dataset).

    Returns (record_df, assumptions).
    """

    assumptions = []

    supplier_id = supplier["supplier_id"]
    amount = float(extracted_fields["total"])
    invoice_date = pd.to_datetime(extracted_fields["invoice_date"])

    if submission_time is None:
        submission_time = pd.Timestamp.now()

    submission_hour = int(pd.Timestamp(submission_time).hour)

    history = supplier_invoice_history(supplier_id)

    if exclude_invoice_id is not None and not history.empty:
        # used by the evaluation script, so an invoice never sees itself
        history = history.loc[history["invoice_id"] != exclude_invoice_id]

    # ----------------------------------------------
    # Behavioural windows (30 / 90 days)
    # ----------------------------------------------

    anchor = invoice_date
    recent = history.loc[
        (history["invoice_date"] < anchor)
        & (history["invoice_date"] >= anchor - pd.Timedelta(days=90))
    ]

    if recent.empty and not history.empty:
        # The invoice is dated after the dataset's period (or the supplier
        # had a quiet spell): use the supplier's most recent activity.
        anchor = history["invoice_date"].max() + pd.Timedelta(days=1)
        assumptions.append(
            "No supplier invoices in the 90 days before this invoice date; "
            "30/90-day features use the supplier's most recent activity "
            f"(up to {history['invoice_date'].max():%Y-%m-%d})."
        )

    before = history.loc[history["invoice_date"] < anchor]

    window_30 = before.loc[before["invoice_date"] >= anchor - pd.Timedelta(days=30)]
    window_90 = before.loc[before["invoice_date"] >= anchor - pd.Timedelta(days=90)]

    count_30d = int(len(window_30))

    if len(window_90):
        avg_90d = float(window_90["invoice_amount"].mean())
    else:
        avg_90d = float(supplier.get("avg_invoice_amount") or amount)
        assumptions.append(
            "No supplier invoices in the 90-day window; the supplier's "
            "overall average amount was used instead."
        )

    # ----------------------------------------------
    # Amount z-score vs all dataset amounts (same as the dataset does)
    # ----------------------------------------------

    mean_amount, std_amount = amount_zscore_stats()
    zscore = float((amount - mean_amount) / std_amount) if std_amount > 0 else 0.0

    # ----------------------------------------------
    # Fields not printed on the invoice
    # ----------------------------------------------

    if not history.empty:
        department_id = str(history["department_id"].mode().iloc[0])
        invoice_type = str(history["invoice_type"].mode().iloc[0])
        assumptions.append(
            f"Department ({department_id}) and invoice type ({invoice_type}) "
            "are not printed on the invoice; the supplier's most frequent "
            "values were used."
        )
    else:
        department_id = None
        invoice_type = None
        assumptions.append(
            "Supplier has no invoice history; department and invoice type "
            "are unknown."
        )

    payment_terms = extracted_fields.get("payment_terms")

    if not payment_terms and not history.empty:
        payment_terms = str(history["payment_terms"].mode().iloc[0])
        assumptions.append(
            "Payment terms could not be read; the supplier's usual terms were used."
        )

    departments = _departments()

    if department_id is not None and department_id in departments.index:
        region = departments.loc[department_id, "region"]
        annual_budget = float(departments.loc[department_id, "annual_budget"])
    else:
        region = None
        annual_budget = None

    late_flag = int(submission_hour in late_night_hours())

    assumptions.append(
        f"Submission hour is the upload time ({submission_hour}:00)."
    )

    record = {
        "invoice_id": extracted_fields.get("invoice_id"),
        "supplier_id": supplier_id,
        "department_id": department_id,
        "invoice_date": invoice_date,
        "invoice_amount": amount,
        "currency": extracted_fields.get("currency"),
        "payment_terms": payment_terms,
        "invoice_type": invoice_type,
        "submission_hour": submission_hour,
        "image_path": None,
        "supplier_country": supplier.get("supplier_country"),
        "supplier_age_days": supplier.get("supplier_age_days"),
        "supplier_risk_score": supplier.get("supplier_risk_score"),
        "blacklisted_flag": supplier.get("blacklisted_flag"),
        "avg_invoice_amount": supplier.get("avg_invoice_amount"),
        "region": region,
        "annual_budget": annual_budget,
        "supplier_invoice_count_30d": count_30d,
        "supplier_avg_amount_90d": avg_90d,
        "invoice_amount_zscore": zscore,
        # Not model features; only the dataset can set these flags.
        "duplicate_invoice_flag": 0,
        "split_invoice_flag": 0,
        "late_night_submission_flag": late_flag
    }

    return pd.DataFrame([record]), assumptions