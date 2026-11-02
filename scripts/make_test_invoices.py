"""
Create NEW test invoice images (not in the dataset) for demos.

Run from the project folder:
    python scripts/make_test_invoices.py

Creates a folder  test_invoices/  with five invoices:

  1_normal_known_supplier.png   expected: APPROVE (low risk)
  2_unknown_supplier.png        expected: MANUAL REVIEW (UNKNOWN_SUPPLIER)
  3_blacklisted_supplier.png    expected: REJECT (BLACKLISTED_SUPPLIER)
  4_wrong_country.png           expected: MANUAL REVIEW (supplier_country mismatch)
  5_edited_total.png            expected: MANUAL REVIEW (invoice_arithmetic)

The images use the same layout as the dataset invoices, so the same
OCR and field extraction works on them.
"""

import sys
import uuid
from datetime import date
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from backend.new_invoice_features import (
    _suppliers,
    supplier_invoice_history,
    build_new_invoice_record
)
from backend.invoice_analyzer import _score

OUTPUT_DIR = PROJECT_DIR / "test_invoices"
SCORING_TIME = pd.Timestamp.now().normalize() + pd.Timedelta(hours=11)


# --------------------------------------------------
# IMAGE RENDERING
# --------------------------------------------------

def _font(size):
    for name in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                 "LiberationSans-Regular.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def money(value):
    return f"{value:,.2f}"


def render_invoice(path, supplier_id, country, invoice_id, invoice_date,
                   payment_terms, total, edit_total_to=None):
    """Draw an invoice in the dataset's text layout."""

    subtotal = round(total / 1.15, 2)
    tax = round(total - subtotal, 2)

    # three line items that add up to the subtotal
    shares = [0.24, 0.37]
    lines = []
    remaining = subtotal
    for i, (qty, share) in enumerate(zip([5, 3], shares), start=1):
        line_total = round(subtotal * share, 2)
        lines.append((i, qty, round(line_total / qty, 2), line_total))
        remaining = round(remaining - line_total, 2)
    lines.append((3, 5, round(remaining / 5, 2), remaining))

    printed_total = edit_total_to if edit_total_to is not None else total

    text = [
        "TAX INVOICE",
        f"Supplier ID: {supplier_id}",
        f"Country: {country}",
        f"Invoice ID: {invoice_id}",
        f"Invoice Date: {invoice_date}",
        f"Payment Terms: {payment_terms}",
        *[
            f"Item {i} Qty {qty} Unit {money(unit)} Line {money(line)}"
            for i, qty, unit, line in lines
        ],
        f"Subtotal: {money(subtotal)}",
        f"Tax (15%): {money(tax)}",
        f"Total: {money(printed_total)} ZAR",
    ]

    image = Image.new("RGB", (1400, 1000), "white")
    draw = ImageDraw.Draw(image)
    title_font, body_font = _font(44), _font(30)

    y = 60
    for index, line in enumerate(text):
        draw.text((70, y), line, fill="black",
                  font=title_font if index == 0 else body_font)
        y += 80 if index == 0 else 58

    image.save(path)


# --------------------------------------------------
# PICK SUPPLIERS FROM THE DATASET
# --------------------------------------------------

def typical_amount(supplier_id):
    history = supplier_invoice_history(supplier_id)
    return round(float(history["invoice_amount"].median()), 2)


def usual_terms(supplier_id):
    history = supplier_invoice_history(supplier_id)
    return str(history["payment_terms"].mode().iloc[0])


def pick_low_risk_supplier():
    """A non-blacklisted supplier whose typical invoice scores LOW."""

    suppliers = _suppliers().reset_index()
    candidates = suppliers.loc[suppliers["blacklisted_flag"] == 0]
    candidates = candidates.sort_values("supplier_risk_score").head(40)

    for _, row in candidates.iterrows():
        supplier = row.to_dict()
        amount = typical_amount(supplier["supplier_id"])
        fields = {
            "invoice_id": "INV_TEST",
            "total": amount,
            "invoice_date": date.today().isoformat(),
            "payment_terms": usual_terms(supplier["supplier_id"]),
            "currency": "ZAR",
        }
        record, _ = build_new_invoice_record(fields, supplier, SCORING_TIME)
        prediction = _score(record)

        if prediction and prediction["risk_level"] == "LOW":
            return supplier, amount, prediction

    raise RuntimeError("No low-risk supplier found among the candidates.")


def main():

    OUTPUT_DIR.mkdir(exist_ok=True)
    today = date.today().isoformat()
    suppliers = _suppliers().reset_index()

    good, good_amount, good_prediction = pick_low_risk_supplier()
    blacklisted = suppliers.loc[suppliers["blacklisted_flag"] == 1].iloc[0].to_dict()

    other_countries = suppliers.loc[
        suppliers["supplier_country"] != good["supplier_country"],
        "supplier_country"
    ]
    wrong_country = str(other_countries.iloc[0])

    plans = [
        ("1_normal_known_supplier.png", "INV_9000001", good["supplier_id"],
         good["supplier_country"], good_amount, None),
        ("2_unknown_supplier.png", "INV_9000002", str(uuid.uuid4()),
         "ZA", 4250.00, None),
        ("3_blacklisted_supplier.png", "INV_9000003", blacklisted["supplier_id"],
         blacklisted["supplier_country"], typical_amount(blacklisted["supplier_id"]), None),
        # different amounts, so these are not flagged as duplicates of invoice 1
        ("4_wrong_country.png", "INV_9000004", good["supplier_id"],
         wrong_country, round(good_amount * 0.9, 2), None),
        ("5_edited_total.png", "INV_9000005", good["supplier_id"],
         good["supplier_country"], round(good_amount * 1.1, 2),
         round(good_amount * 1.1 * 1.4, 2)),
    ]

    for filename, invoice_id, supplier_id, country, amount, edited in plans:
        terms = (
            usual_terms(supplier_id)
            if supplier_id in _suppliers().index else "NET30"
        )
        render_invoice(
            OUTPUT_DIR / filename, supplier_id, country, invoice_id,
            today, terms, amount, edit_total_to=edited
        )
        print(f"Created {filename:32s} supplier={supplier_id}  amount={amount}")

    print(
        f"\nInvoice 1 model check at 11:00: risk {good_prediction['fraud_probability']}"
        f" ({good_prediction['risk_level']})."
    )
    print(
        "Note: the app uses the real upload time. Uploading late at night "
        "can raise the risk score."
    )
    print(f"\nTest invoices saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()