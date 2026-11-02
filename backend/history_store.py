import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

# --------------------------------------------------
# DATABASE CONFIGURATION
# --------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"

DATA_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_PATH = DATA_DIR / "invoice_history.db"

# Duplicate detection settings
DUPLICATE_AMOUNT_TOLERANCE = 1.00   # same amount within this value
DUPLICATE_DAY_WINDOW = 7            # invoice dates within this many days


# --------------------------------------------------
# JSON SERIALIZATION
# --------------------------------------------------

def json_serializer(value):
    """Convert pandas, NumPy, and datetime values into JSON-safe values."""

    if hasattr(value, "isoformat"):
        return value.isoformat()

    if hasattr(value, "item"):
        return value.item()

    return str(value)


def _connect():
    return sqlite3.connect(DATABASE_PATH)


# --------------------------------------------------
# DATABASE INITIALIZATION
# --------------------------------------------------

def initialize_history_db():
    """
    Create the invoice history table if it does not exist, and add the
    columns used for duplicate detection to older databases.
    """

    with _connect() as connection:

        connection.execute("""
            CREATE TABLE IF NOT EXISTS invoice_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                invoice_id TEXT,
                filename TEXT,
                disposition TEXT,
                reason TEXT,
                risk_score REAL,
                risk_level TEXT,
                mismatch_count INTEGER,
                retrieved_count INTEGER,
                analysis_json TEXT NOT NULL,
                analyzed_at_utc TEXT NOT NULL
            )
        """)

        existing_columns = {
            row[1]
            for row in connection.execute(
                "PRAGMA table_info(invoice_history)"
            )
        }

        new_columns = {
            "supplier_id": "TEXT",
            "invoice_amount": "REAL",
            "invoice_date": "TEXT"
        }

        added = False

        for column, column_type in new_columns.items():
            if column not in existing_columns:
                connection.execute(
                    f"ALTER TABLE invoice_history "
                    f"ADD COLUMN {column} {column_type}"
                )
                added = True

        if added:
            _backfill_duplicate_columns(connection)

        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_invoice "
            "ON invoice_history (invoice_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_supplier "
            "ON invoice_history (supplier_id)"
        )

        connection.commit()


def _extract_key_fields(result):
    """Get supplier, amount and date for duplicate checks."""

    structured = result.get("structured_invoice") or {}
    extracted = result.get("extracted_fields") or {}

    supplier_id = structured.get("supplier_id") or extracted.get("supplier_id")

    amount = structured.get("invoice_amount")
    if amount is None:
        amount = extracted.get("total")

    invoice_date = structured.get("invoice_date") or extracted.get("invoice_date")
    invoice_date = pd.to_datetime(invoice_date, errors="coerce")

    return (
        str(supplier_id) if supplier_id is not None else None,
        float(amount) if amount is not None else None,
        invoice_date.strftime("%Y-%m-%d") if not pd.isna(invoice_date) else None
    )


def _backfill_duplicate_columns(connection):
    """Fill the new columns for rows saved before they existed."""

    rows = connection.execute(
        "SELECT id, analysis_json FROM invoice_history "
        "WHERE supplier_id IS NULL"
    ).fetchall()

    for record_id, analysis_json in rows:
        try:
            supplier_id, amount, invoice_date = _extract_key_fields(
                json.loads(analysis_json)
            )
        except (ValueError, TypeError):
            continue

        connection.execute(
            "UPDATE invoice_history "
            "SET supplier_id = ?, invoice_amount = ?, invoice_date = ? "
            "WHERE id = ?",
            (supplier_id, amount, invoice_date, record_id)
        )


# --------------------------------------------------
# DUPLICATE DETECTION
# --------------------------------------------------

def check_for_duplicates(invoice_id, supplier_id=None,
                         invoice_amount=None, invoice_date=None):
    """
    Check the analysis history for:
      1. the same invoice_id analysed before (repeated submission)
         - and whether any of those earlier analyses was APPROVED
      2. a different invoice_id from the same supplier with the same
         amount and an invoice date within DUPLICATE_DAY_WINDOW days
         (possible duplicate billing under a new invoice number)
    """

    result = {
        "same_invoice_seen": False,
        "previous_count": 0,
        "previous_records": [],
        "previously_approved": False,
        "approved_record_id": None,
        "similar_invoices": []
    }

    with _connect() as connection:

        connection.row_factory = sqlite3.Row

        previous = connection.execute("""
            SELECT id, disposition, analyzed_at_utc
            FROM invoice_history
            WHERE invoice_id = ?
            ORDER BY id
        """, (str(invoice_id),)).fetchall()

        if previous:
            result["same_invoice_seen"] = True
            result["previous_count"] = len(previous)
            result["previous_records"] = [dict(row) for row in previous]

            approved = [
                row for row in previous
                if row["disposition"] == "APPROVE"
            ]

            if approved:
                result["previously_approved"] = True
                result["approved_record_id"] = approved[0]["id"]

        if supplier_id is None or invoice_amount is None:
            return result

        candidates = connection.execute("""
            SELECT id, invoice_id, invoice_amount, invoice_date, disposition
            FROM invoice_history
            WHERE supplier_id = ?
              AND invoice_id != ?
              AND ABS(invoice_amount - ?) <= ?
        """, (
            str(supplier_id),
            str(invoice_id),
            float(invoice_amount),
            DUPLICATE_AMOUNT_TOLERANCE
        )).fetchall()

    current_date = pd.to_datetime(invoice_date, errors="coerce")
    seen_ids = set()

    for row in candidates:

        if row["invoice_id"] in seen_ids:
            continue

        other_date = pd.to_datetime(row["invoice_date"], errors="coerce")

        if not pd.isna(current_date) and not pd.isna(other_date):
            if abs((current_date - other_date).days) > DUPLICATE_DAY_WINDOW:
                continue

        seen_ids.add(row["invoice_id"])
        result["similar_invoices"].append(dict(row))

    return result


# --------------------------------------------------
# SAVE INVOICE ANALYSIS
# --------------------------------------------------

def save_invoice_analysis(result, filename):
    """Save a completed invoice analysis without losing its full result."""

    if result.get("status") != "completed":
        raise ValueError("Only completed invoice analyses can be saved.")

    disposition = result.get("disposition") or {}
    validation = result.get("validation") or {}
    rag_context = result.get("rag_context") or {}

    extracted_fields = result.get("extracted_fields") or {}
    structured_invoice = result.get("structured_invoice") or {}

    invoice_id = (
        extracted_fields.get("invoice_id")
        or structured_invoice.get("invoice_id")
    )

    supplier_id, invoice_amount, invoice_date = _extract_key_fields(result)

    analysis_json = json.dumps(
        result,
        default=json_serializer,
        ensure_ascii=False
    )

    with _connect() as connection:

        connection.execute("""
            INSERT INTO invoice_history (
                invoice_id,
                filename,
                disposition,
                reason,
                risk_score,
                risk_level,
                mismatch_count,
                retrieved_count,
                analysis_json,
                analyzed_at_utc,
                supplier_id,
                invoice_amount,
                invoice_date
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            str(invoice_id) if invoice_id is not None else None,
            filename,
            disposition.get("disposition"),
            disposition.get("reason"),
            result.get("risk_score"),
            result.get("risk_level"),
            validation.get("mismatch_count", 0),
            rag_context.get("retrieved_count", 0),
            analysis_json,
            datetime.now(timezone.utc).isoformat(),
            supplier_id,
            invoice_amount,
            invoice_date
        ))

        connection.commit()

    return True


# --------------------------------------------------
# FETCH INVOICE HISTORY
# --------------------------------------------------

def get_invoice_history():
    """Return saved invoice analyses, newest first."""

    with _connect() as connection:

        connection.row_factory = sqlite3.Row

        rows = connection.execute("""
            SELECT
                id,
                invoice_id,
                filename,
                disposition,
                reason,
                risk_score,
                risk_level,
                mismatch_count,
                retrieved_count,
                analyzed_at_utc
            FROM invoice_history
            ORDER BY id DESC
        """).fetchall()

    return [dict(row) for row in rows]


# --------------------------------------------------
# FETCH COMPLETE INVOICE ANALYSIS
# --------------------------------------------------

def get_invoice_analysis(record_id):
    """Retrieve the complete saved analysis using its database ID."""

    with _connect() as connection:

        row = connection.execute("""
            SELECT analysis_json
            FROM invoice_history
            WHERE id = ?
        """, (record_id,)).fetchone()

    if row is None:
        return None

    return json.loads(row[0])


# --------------------------------------------------
# CLEAR HISTORY (useful before a demo)
# --------------------------------------------------

def clear_invoice_history():
    """Delete all saved analyses."""

    with _connect() as connection:
        connection.execute("DELETE FROM invoice_history")
        connection.commit()


# --------------------------------------------------
# INITIALIZE DATABASE
# --------------------------------------------------

initialize_history_db()