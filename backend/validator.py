"""
Invoice validation.

Checks are split into two kinds:

  HARD checks - a failure sends the invoice to MANUAL REVIEW.
  SOFT checks - reported to the user but do not block approval.

Two situations are handled:

  1. validate_invoice_fields()  - the invoice exists in the dataset.
     OCR fields are compared with the dataset record.
       HARD: invoice_id, supplier_id, supplier_country, invoice_date,
             currency, payment_terms, subtotal + tax = total
       SOFT: OCR total vs dataset invoice_amount (in this dataset the
             amount printed on the image often differs from
             invoice_amount - a known dataset quirk)

  2. validate_new_invoice()  - the invoice is NOT in the dataset.
     OCR fields are checked against the supplier master and basic rules.
       HARD: required fields readable, supplier_country matches the
             supplier master, valid date not in the future, known
             currency and payment terms, subtotal + tax = total
       SOFT: invoice number format

A field the OCR could not read is "not_read", not a mismatch.
"""

import re

import pandas as pd

# Tolerances (in invoice currency)
ARITHMETIC_TOLERANCE = 1.00   # subtotal + tax vs total on the same invoice
AMOUNT_TOLERANCE = 1.00       # OCR total vs dataset invoice_amount

INVOICE_ID_PATTERN = re.compile(r"^INV_\d{7}$")


def _normalise_text(value):
    """Upper-case and remove spaces so small OCR differences don't count."""

    return "".join(str(value).split()).upper()


def _compare_text(ocr_value, expected_value, severity="hard"):

    if ocr_value is None or str(ocr_value).strip() == "":
        status, match = "not_read", None
    elif expected_value is None or (
        not isinstance(expected_value, str) and pd.isna(expected_value)
    ):
        status, match = "not_read", None
    elif _normalise_text(ocr_value) == _normalise_text(expected_value):
        status, match = "match", True
    else:
        status, match = "mismatch", False

    return {
        "ocr_value": ocr_value,
        "dataset_value": None if expected_value is None else str(expected_value),
        "status": status,
        "match": match,
        "severity": severity
    }


def _result(ok, ocr_value, expected, severity="hard", difference=None):
    return {
        "ocr_value": ocr_value,
        "dataset_value": expected,
        "difference": difference,
        "status": "match" if ok else "mismatch",
        "match": ok,
        "severity": severity
    }


def _not_read(ocr_value, expected, severity="hard"):
    return {
        "ocr_value": ocr_value,
        "dataset_value": expected,
        "status": "not_read",
        "match": None,
        "severity": severity
    }


def _check_arithmetic(extracted_fields):
    """subtotal + tax must equal total on the invoice itself."""

    subtotal = extracted_fields.get("subtotal")
    tax = extracted_fields.get("tax")
    total = extracted_fields.get("total")

    if None in (subtotal, tax, total):
        return _not_read(
            f"subtotal={subtotal}, tax={tax}, total={total}",
            "subtotal + tax = total"
        )

    difference = round(subtotal + tax - total, 2)

    return _result(
        abs(difference) <= ARITHMETIC_TOLERANCE,
        f"{subtotal} + {tax} = {round(subtotal + tax, 2)}",
        f"total on invoice = {total}",
        difference=difference
    )


def _summarise(comparisons, extra_hard_issues=None):

    hard_issues = list(extra_hard_issues or [])
    soft_issues = []

    for field, result in comparisons.items():

        if result["status"] != "mismatch":
            continue

        issue = {
            "field": field,
            "ocr_value": result.get("ocr_value"),
            "dataset_value": result.get("dataset_value"),
            "difference": result.get("difference")
        }

        if result["severity"] == "hard":
            hard_issues.append(issue)
        else:
            soft_issues.append(issue)

    not_read = [
        field for field, result in comparisons.items()
        if result["status"] == "not_read"
    ]

    return {
        "status": "completed",
        "message": "Invoice validation completed.",
        # mismatch_count = blocking (hard) problems only
        "mismatch_count": len(hard_issues),
        "hard_issues": hard_issues,
        "soft_issues": soft_issues,
        "not_read_fields": not_read,
        "comparisons": comparisons
    }


# --------------------------------------------------
# 1. INVOICE THAT EXISTS IN THE DATASET
# --------------------------------------------------

def validate_invoice_fields(extracted_fields, dataset_record=None):
    """
    Validate OCR-extracted fields against the dataset record.

    dataset_record: the matching row (dict or pandas Series). If not
    given, it is looked up by invoice_id using the cached dataset.
    """

    invoice_id = extracted_fields.get("invoice_id")

    if not invoice_id:
        return {
            "status": "failed",
            "message": "Invoice ID could not be extracted.",
            "comparisons": {},
            "hard_issues": [],
            "soft_issues": [],
            "mismatch_count": 0
        }

    if dataset_record is None:
        from backend.data_loader import get_invoice_record

        found = get_invoice_record(invoice_id)
        dataset_record = None if found is None else found.iloc[0].to_dict()

    if dataset_record is None:
        return {
            "status": "not_found",
            "message": "Invoice ID not found in the dataset.",
            "comparisons": {},
            "hard_issues": [],
            "soft_issues": [],
            "mismatch_count": 0
        }

    record = dict(dataset_record)
    comparisons = {}

    for field in ["invoice_id", "supplier_id", "supplier_country",
                  "payment_terms", "currency"]:
        comparisons[field] = _compare_text(
            extracted_fields.get(field), record.get(field)
        )

    dataset_date = pd.to_datetime(record.get("invoice_date"), errors="coerce")
    comparisons["invoice_date"] = _compare_text(
        extracted_fields.get("invoice_date"),
        dataset_date.strftime("%Y-%m-%d") if not pd.isna(dataset_date) else None
    )

    comparisons["invoice_arithmetic"] = _check_arithmetic(extracted_fields)

    # SOFT: OCR total vs dataset invoice amount
    total = extracted_fields.get("total")
    dataset_amount = record.get("invoice_amount")

    if total is None or dataset_amount is None:
        comparisons["invoice_amount"] = _not_read(total, dataset_amount, "soft")
    else:
        difference = round(float(total) - float(dataset_amount), 2)
        comparisons["invoice_amount"] = _result(
            abs(difference) <= AMOUNT_TOLERANCE,
            total, float(dataset_amount), "soft", difference
        )

    return _summarise(comparisons)


# --------------------------------------------------
# 2. NEW INVOICE (NOT IN THE DATASET)
# --------------------------------------------------

def validate_new_invoice(extracted_fields, supplier=None,
                         known_currencies=None, known_payment_terms=None,
                         today=None):
    """
    Validate a new invoice against the supplier master and basic rules.

    supplier: supplier master record (dict) or None if not found.
    """

    comparisons = {}
    extra_hard = []

    # Required fields must be readable, otherwise the invoice
    # cannot be checked or scored.
    for field in ["invoice_id", "supplier_id", "invoice_date", "total"]:
        if extracted_fields.get(field) in (None, ""):
            extra_hard.append({
                "field": f"{field} (not readable)",
                "ocr_value": None,
                "dataset_value": "required for a new invoice",
                "difference": None
            })

    # Supplier country printed on the invoice vs supplier master
    if supplier is not None:
        comparisons["supplier_country"] = _compare_text(
            extracted_fields.get("supplier_country"),
            supplier.get("supplier_country")
        )

    # Invoice date: valid and not in the future
    raw_date = extracted_fields.get("invoice_date")
    if raw_date:
        date = pd.to_datetime(raw_date, errors="coerce")
        today = pd.Timestamp(today or pd.Timestamp.now()).normalize()

        if pd.isna(date):
            comparisons["invoice_date"] = _result(False, raw_date, "valid date")
        else:
            comparisons["invoice_date"] = _result(
                date <= today, raw_date, f"not after {today:%Y-%m-%d}"
            )

    # Currency and payment terms must be known values
    for field, allowed in [("currency", known_currencies),
                           ("payment_terms", known_payment_terms)]:
        value = extracted_fields.get(field)
        if value and allowed:
            comparisons[field] = _result(
                str(value).upper() in allowed,
                value, " / ".join(sorted(allowed))
            )

    comparisons["invoice_arithmetic"] = _check_arithmetic(extracted_fields)

    # SOFT: invoice number format
    invoice_id = extracted_fields.get("invoice_id")
    if invoice_id:
        comparisons["invoice_id_format"] = _result(
            bool(INVOICE_ID_PATTERN.match(str(invoice_id))),
            invoice_id, "INV_ + 7 digits", "soft"
        )

    return _summarise(comparisons, extra_hard_issues=extra_hard)