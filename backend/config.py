import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Main dataset location. Set DATASET_DIR in the .env file to run the
# app on another machine; the default is the original development path.
DATASET_DIR = Path(
    os.getenv(
        "DATASET_DIR",
        r"D:\archive (5)\Procument Invoice Fraud Dataset v1"
    )
)

# Application directories
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
CHROMA_DIR = BASE_DIR / "chroma_db"
OUTPUT_DIR = BASE_DIR / "data"

# Dataset files
INVOICES_FILE = DATASET_DIR / "invoices.parquet"
SUPPLIERS_FILE = DATASET_DIR / "suppliers.parquet"
DEPARTMENTS_FILE = DATASET_DIR / "departments.parquet"
BEHAVIOURAL_FILE = DATASET_DIR / "behavioural_features.parquet"
IMAGES_METADATA_FILE = DATASET_DIR / "images_metadata.parquet"
IMAGES_DIR = DATASET_DIR / "images"

# Warn only when something is missing (no output when all is fine).
_missing = [
    path.name
    for path in [
        INVOICES_FILE,
        SUPPLIERS_FILE,
        DEPARTMENTS_FILE,
        BEHAVIOURAL_FILE,
        IMAGES_METADATA_FILE
    ]
    if not path.exists()
]

if _missing:
    print(
        f"WARNING: dataset files not found in {DATASET_DIR}: "
        + ", ".join(_missing)
        + ". Set DATASET_DIR in the .env file."
    )
