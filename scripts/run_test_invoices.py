"""
End-to-end check of the five demo invoices in test_invoices/.

Runs each image through analyze_invoice() (the same call the app makes:
OCR -> fields -> validation -> model -> rules) and compares the final
decision with the expected one from make_test_invoices.py.

Run from the project folder:
    python scripts/run_test_invoices.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import tempfile

import pandas as pd

from backend import history_store
from backend.invoice_analyzer import analyze_invoice

# Use a temporary empty history database, so analyses saved earlier in the
# app (e.g. invoice 1 already approved) do not trigger the duplicate rules.
# The real data/invoice_history.db is not touched.
_temp_dir = tempfile.mkdtemp()
history_store.DATABASE_PATH = Path(_temp_dir) / "test_history.db"
history_store.initialize_history_db()

TEST_DIR = PROJECT_DIR / "test_invoices"

# Fixed daytime upload time, so the late-night flag does not depend on
# when the script is run.
SUBMISSION_TIME = pd.Timestamp.now().normalize() + pd.Timedelta(hours=11)

EXPECTED = {
    "1_normal_known_supplier.png": "APPROVE",
    "2_unknown_supplier.png": "MANUAL_REVIEW",
    "3_blacklisted_supplier.png": "REJECT",
    "4_wrong_country.png": "MANUAL_REVIEW",
    "5_edited_total.png": "MANUAL_REVIEW",
}


def normalise(decision):
    return str(decision or "").upper().replace(" ", "_")


def main():

    passed = 0

    for name, expected in EXPECTED.items():

        path = TEST_DIR / name
        print("\n" + "=" * 64)
        print(name)

        if not path.exists():
            print("  MISSING - run scripts/make_test_invoices.py first")
            continue

        result = analyze_invoice(file_path=str(path), submission_time=SUBMISSION_TIME)

        if result.get("status") != "completed":
            print("  analysis failed:", result.get("message"))
            continue

        disposition = result["disposition"]
        decision = normalise(disposition["disposition"])
        ok = decision == normalise(expected)
        passed += ok

        fields = result.get("extracted_fields", {})
        print(f"  supplier_id   {fields.get('supplier_id')}  (found: {result.get('supplier_found')})")
        print(f"  total         {fields.get('total')}   date {fields.get('invoice_date')}")
        print(f"  risk          {result.get('risk_score')}  {result.get('risk_level')}")
        print(f"  rules fired   {disposition.get('triggered_rules')}")
        print(f"  decision      {decision}   expected {expected}   {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(f"  reason        {disposition.get('reason')}")

    print("\n" + "=" * 64)
    print(f"{passed}/{len(EXPECTED)} test invoices gave the expected decision")


if __name__ == "__main__":
    main()
