"""
How well does the NEW-invoice path work?  (experiment for the paper)

Takes a random sample of dataset TEST-split invoices and treats each one
as if it were a new invoice:
  - the model features are rebuilt from the supplier master and the
    supplier's other invoices (exactly what happens for a new upload)
  - these rebuilt features are compared with the dataset's real values
  - the model is run on both, and both are compared with the true label

Run from the project folder:
    python scripts/check_new_invoice_features.py            (500 invoices)
    python scripts/check_new_invoice_features.py 2000       (2000 invoices)

Results are printed and saved to  data/new_invoice_feature_check.csv
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.config import DATASET_DIR
from backend.data_loader import load_structured_data
from backend.new_invoice_features import get_supplier, build_new_invoice_record
from backend.invoice_analyzer import _score

SAMPLE_SIZE = int(sys.argv[1]) if len(sys.argv) > 1 else 500


def main():

    data = load_structured_data()
    labels = pd.read_parquet(DATASET_DIR / "labels.parquet")[["invoice_id", "is_fraud"]]
    data = data.merge(labels, on="invoice_id", how="left")

    splits_file = DATASET_DIR / "splits.parquet"
    if splits_file.exists():
        splits = pd.read_parquet(splits_file)
        data = data.merge(splits, on="invoice_id", how="left")
        data = data.loc[data["split"] == "test"]
        print("Using the TEST split only.")

    sample = data.sample(n=min(SAMPLE_SIZE, len(data)), random_state=42)
    rows = []

    for count, (_, inv) in enumerate(sample.iterrows(), start=1):

        supplier = get_supplier(inv["supplier_id"])
        date = pd.to_datetime(inv["invoice_date"])

        fields = {
            "invoice_id": inv["invoice_id"],
            "total": float(inv["invoice_amount"]),
            "invoice_date": date.strftime("%Y-%m-%d"),
            "payment_terms": inv["payment_terms"],
            "currency": inv["currency"],
        }

        submitted = date.normalize() + pd.Timedelta(hours=int(inv["submission_hour"]))
        rebuilt, _ = build_new_invoice_record(
            fields, supplier, submitted, exclude_invoice_id=inv["invoice_id"]
        )
        rebuilt = rebuilt.iloc[0]

        original_prediction = _score(sample.loc[[inv.name]])
        rebuilt_prediction = _score(pd.DataFrame([rebuilt]))

        rows.append({
            "invoice_id": inv["invoice_id"],
            "is_fraud": int(inv["is_fraud"]),
            "count_30d_dataset": inv["supplier_invoice_count_30d"],
            "count_30d_rebuilt": rebuilt["supplier_invoice_count_30d"],
            "avg_90d_dataset": inv["supplier_avg_amount_90d"],
            "avg_90d_rebuilt": rebuilt["supplier_avg_amount_90d"],
            "zscore_dataset": inv["invoice_amount_zscore"],
            "zscore_rebuilt": rebuilt["invoice_amount_zscore"],
            "late_dataset": inv["late_night_submission_flag"],
            "late_rebuilt": rebuilt["late_night_submission_flag"],
            "department_match": str(inv["department_id"]) == str(rebuilt["department_id"]),
            "invoice_type_match": str(inv["invoice_type"]) == str(rebuilt["invoice_type"]),
            "score_dataset": original_prediction["fraud_probability"] if original_prediction else np.nan,
            "score_rebuilt": rebuilt_prediction["fraud_probability"] if rebuilt_prediction else np.nan,
            "level_dataset": original_prediction["risk_level"] if original_prediction else None,
            "level_rebuilt": rebuilt_prediction["risk_level"] if rebuilt_prediction else None,
            "pred_dataset": original_prediction["prediction"] if original_prediction else np.nan,
            "pred_rebuilt": rebuilt_prediction["prediction"] if rebuilt_prediction else np.nan,
        })

        if count % 100 == 0:
            print(f"  processed {count}/{len(sample)}")

    results = pd.DataFrame(rows)
    output = PROJECT_DIR / "data" / "new_invoice_feature_check.csv"
    output.parent.mkdir(exist_ok=True)
    results.to_csv(output, index=False)

    scored = results.dropna(subset=["pred_dataset", "pred_rebuilt"])

    def corr(a, b):
        return results[[a, b]].astype(float).corr().iloc[0, 1]

    print("\n" + "=" * 64)
    print(f"NEW-INVOICE FEATURE CHECK  ({len(results)} invoices)")
    print("=" * 64)

    print("\n1. How close are the rebuilt features to the dataset's values?")
    print(f"   30-day invoice count   correlation {corr('count_30d_dataset', 'count_30d_rebuilt'):.3f}"
          f"   mean abs. difference {(results['count_30d_dataset'] - results['count_30d_rebuilt']).abs().mean():.2f}")
    print(f"   90-day average amount  correlation {corr('avg_90d_dataset', 'avg_90d_rebuilt'):.3f}")
    print(f"   amount z-score         correlation {corr('zscore_dataset', 'zscore_rebuilt'):.3f}")
    print(f"   late-night flag        agreement   {(results['late_dataset'] == results['late_rebuilt']).mean():.1%}")
    print(f"   department guessed     correct     {results['department_match'].mean():.1%}")
    print(f"   invoice type guessed   correct     {results['invoice_type_match'].mean():.1%}")

    print("\n2. Does the model give the same answer?")
    print(f"   same risk level (LOW/MEDIUM/HIGH)  {(scored['level_dataset'] == scored['level_rebuilt']).mean():.1%}")
    print(f"   same fraud / not-fraud prediction  {(scored['pred_dataset'] == scored['pred_rebuilt']).mean():.1%}")
    print(f"   mean abs. risk score difference    {(scored['score_dataset'] - scored['score_rebuilt']).abs().mean():.2f}")

    print("\n3. Accuracy against the true labels")
    print(f"   {'':22s}{'accuracy':>10s}{'precision':>11s}{'recall':>9s}{'F1':>7s}")
    for name, column in [("dataset features", "pred_dataset"),
                         ("rebuilt (new-invoice)", "pred_rebuilt")]:
        y_true, y_pred = scored["is_fraud"], scored[column].astype(int)
        print(f"   {name:22s}"
              f"{accuracy_score(y_true, y_pred):>10.3f}"
              f"{precision_score(y_true, y_pred, zero_division=0):>11.3f}"
              f"{recall_score(y_true, y_pred, zero_division=0):>9.3f}"
              f"{f1_score(y_true, y_pred, zero_division=0):>7.3f}")

    print(f"\nDetailed results saved to: {output}")


if __name__ == "__main__":
    main()