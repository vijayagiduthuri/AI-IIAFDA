
import pandas as pd
from pathlib import Path

DATASET_DIR = Path(
    r"D:\archive (5)\Procument Invoice Fraud Dataset v1"
)

IMAGE_DIR = DATASET_DIR / "images"

# Load the original dataset files
invoices = pd.read_parquet(DATASET_DIR / "invoices.parquet")
suppliers = pd.read_parquet(DATASET_DIR / "suppliers.parquet")
behavioural = pd.read_parquet(
    DATASET_DIR / "behavioural_features.parquet"
)
labels = pd.read_parquet(DATASET_DIR / "labels.parquet")
images = pd.read_parquet(
    DATASET_DIR / "images_metadata.parquet"
)

# Join the dataset using the actual invoice and supplier IDs
df = invoices.merge(
    suppliers,
    on="supplier_id",
    how="left"
)

df = df.merge(
    behavioural,
    on="invoice_id",
    how="left"
)

df = df.merge(
    labels[["invoice_id", "is_fraud", "fraud_type"]],
    on="invoice_id",
    how="left"
)

df = df.merge(
    images,
    on="invoice_id",
    how="inner",
    suffixes=("_invoice", "_metadata")
)

# Identify the actual image path
df["image_path"] = df["image_path_metadata"].fillna(
    df["image_path_invoice"]
)

# Select low-risk, non-fraudulent candidates
candidates = df[
    (df["is_fraud"] == 0)
    & (df["blacklisted_flag"] == 0)
    & (df["duplicate_invoice_flag"] == 0)
    & (df["split_invoice_flag"] == 0)
    & (df["late_night_submission_flag"] == 0)
    & (df["supplier_risk_score"] <= 0.30)
    & (df["image_tamper_flag"] == 0)
].copy()

# Calculate the difference between the original invoice amount
# and the amount extracted from its image metadata
candidates["amount_difference"] = (
    candidates["invoice_amount"]
    - candidates["ocr_total_extracted"]
).abs()

# Retain candidates with matching OCR and dataset amounts
candidates = candidates[
    candidates["ocr_total_extracted"].notna()
    & (candidates["amount_difference"] <= 1.00)
].copy()

# Resolve actual image files
def resolve_image(path):
    if pd.isna(path):
        return None

    filename = Path(str(path)).name
    candidate = IMAGE_DIR / filename

    return str(candidate) if candidate.is_file() else None

candidates["resolved_image_path"] = (
    candidates["image_path"].apply(resolve_image)
)

candidates = candidates[
    candidates["resolved_image_path"].notna()
].copy()

# Sort by lowest supplier risk and closest amount match
candidates = candidates.sort_values(
    ["supplier_risk_score", "amount_difference"]
)

print("\n" + "=" * 70)
print("CLEAN INVOICE CANDIDATES")
print("=" * 70)

print("Total image-backed candidates:", len(candidates))

columns = [
    "invoice_id",
    "invoice_amount",
    "ocr_total_extracted",
    "amount_difference",
    "supplier_risk_score",
    "blacklisted_flag",
    "duplicate_invoice_flag",
    "split_invoice_flag",
    "late_night_submission_flag",
    "image_tamper_flag",
    "is_fraud",
    "resolved_image_path"
]

if not candidates.empty:
    print("\nTop 15 candidates:")
    print(
        candidates[columns]
        .head(15)
        .to_string(index=False)
    )

    output_file = (
        Path(__file__).resolve().parent.parent / "data"
        / "approve_candidates.csv"
    )

    candidates[columns].to_csv(output_file, index=False)

    print("\nCandidate list saved to:", output_file)

else:
    print("\nNo candidates matched all the selected conditions.")
