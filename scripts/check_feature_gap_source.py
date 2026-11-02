"""
Where does the remaining new-invoice accuracy gap come from? (paper experiment)

Same 2000-invoice TEST sample as check_new_invoice_features.py (random_state=42).
Each invoice is rebuilt as a new invoice, then scored three ways:
  A. rebuilt as-is                              (what the app does)
  B. rebuilt + true department and invoice type (fields not on the invoice)
  C. B + true 30-day count and 90-day average   (all remaining differences removed)

Run from the project folder:
    python scripts/check_feature_gap_source.py 2000
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from backend.config import DATASET_DIR
from backend.data_loader import load_structured_data
from backend.new_invoice_features import get_supplier, build_new_invoice_record, _departments
from backend.invoice_analyzer import _score

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def main():
    data = load_structured_data()
    labels = pd.read_parquet(DATASET_DIR / "labels.parquet")[["invoice_id", "is_fraud"]]
    data = data.merge(labels, on="invoice_id", how="left")
    splits = pd.read_parquet(DATASET_DIR / "splits.parquet")
    data = data.merge(splits, on="invoice_id", how="left")
    data = data.loc[data["split"] == "test"]
    sample = data.sample(n=min(SAMPLE_SIZE, len(data)), random_state=42)
    departments = _departments()

    rows = []
    for count, (_, inv) in enumerate(sample.iterrows(), start=1):
        date = pd.to_datetime(inv["invoice_date"])
        fields = {"invoice_id": inv["invoice_id"], "total": float(inv["invoice_amount"]),
                  "invoice_date": date.strftime("%Y-%m-%d"),
                  "payment_terms": inv["payment_terms"], "currency": inv["currency"]}
        submitted = date.normalize() + pd.Timedelta(hours=int(inv["submission_hour"]))
        a, _ = build_new_invoice_record(fields, get_supplier(inv["supplier_id"]), submitted,
                                        exclude_invoice_id=inv["invoice_id"])
        a = a.iloc[0]

        b = a.copy()
        dept = str(inv["department_id"])
        b["department_id"] = inv["department_id"]
        b["invoice_type"] = inv["invoice_type"]
        if dept in departments.index:
            b["region"] = departments.loc[dept, "region"]
            b["annual_budget"] = float(departments.loc[dept, "annual_budget"])

        c = b.copy()
        c["supplier_invoice_count_30d"] = inv["supplier_invoice_count_30d"]
        c["supplier_avg_amount_90d"] = inv["supplier_avg_amount_90d"]

        row = {"invoice_id": inv["invoice_id"], "is_fraud": int(inv["is_fraud"])}
        for name, rec in [("A", a), ("B", b), ("C", c)]:
            p = _score(pd.DataFrame([rec]))
            row[f"pred_{name}"] = p["prediction"] if p else None
            row[f"score_{name}"] = p["fraud_probability"] if p else None
        p = _score(sample.loc[[inv.name]])
        row["pred_dataset"] = p["prediction"] if p else None
        rows.append(row)
        if count % 200 == 0:
            print(f"  processed {count}/{len(sample)}", flush=True)

    r = pd.DataFrame(rows).dropna()
    out = PROJECT_DIR / "data" / "feature_gap_source.csv"
    r.to_csv(out, index=False)

    print(f"\nFEATURE GAP SOURCE  ({len(r)} invoices)")
    print(f"   {'':45s}{'acc':>7s}{'prec':>7s}{'rec':>7s}{'F1':>7s}{'FP':>5s}{'FN':>5s}")
    for label, col in [("dataset features", "pred_dataset"),
                       ("A rebuilt (app)", "pred_A"),
                       ("B + true department / invoice type", "pred_B"),
                       ("C + true 30d count / 90d average", "pred_C")]:
        y, p = r["is_fraud"], r[col].astype(int)
        tn, fp, fn, tp = confusion_matrix(y, p).ravel()
        print(f"   {label:45s}{accuracy_score(y,p):>7.3f}{precision_score(y,p,zero_division=0):>7.3f}"
              f"{recall_score(y,p,zero_division=0):>7.3f}{f1_score(y,p,zero_division=0):>7.3f}{fp:>5d}{fn:>5d}")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
