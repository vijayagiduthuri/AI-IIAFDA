
import os
import json
import hashlib

from datetime import datetime, timezone

import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer

from backend.config import CHROMA_DIR
from backend.data_loader import load_structured_data


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

COLLECTION_NAME = "invoice_knowledge"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 500

CHECKPOINT_DIR = os.path.join(
    os.path.dirname(CHROMA_DIR),
    "rag_checkpoints"
)

INDEX_CHECKPOINT_FILE = os.path.join(
    CHECKPOINT_DIR,
    "rag_index_progress.json"
)

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# --------------------------------------------------
# INITIALIZE CHROMADB
# --------------------------------------------------

def get_chroma_collection():
    """Connect to the existing persistent ChromaDB collection."""

    client = chromadb.PersistentClient(
        path=CHROMA_DIR
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={
            "description": "Invoice intelligence knowledge base",
            "embedding_model": EMBEDDING_MODEL_NAME
        }
    )

    return collection


# --------------------------------------------------
# LOAD EMBEDDING MODEL
# --------------------------------------------------

_embedding_model = None


def get_embedding_model():
    """Load the embedding model once and reuse it."""

    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


# --------------------------------------------------
# CONVERT INVOICE RECORDS INTO DOCUMENTS
# --------------------------------------------------

def invoice_to_document(row):
    """Convert one structured invoice record into searchable text."""

    fields = [
        "invoice_id",
        "supplier_id",
        "department_id",
        "invoice_date",
        "invoice_amount",
        "currency",
        "payment_terms",
        "invoice_type",
        "submission_hour",
        "supplier_country",
        "supplier_age_days",
        "supplier_risk_score",
        "blacklisted_flag",
        "avg_invoice_amount",
        "region",
        "annual_budget",
        "supplier_invoice_count_30d",
        "supplier_avg_amount_90d",
        "invoice_amount_zscore",
        "duplicate_invoice_flag",
        "split_invoice_flag",
        "late_night_submission_flag",
        "image_path"
    ]

    parts = []

    for field in fields:

        if field not in row.index:
            continue

        value = row[field]

        if pd.isna(value):
            continue

        parts.append(f"{field}: {value}")

    return "\n".join(parts)


# --------------------------------------------------
# CHECKPOINT MANAGEMENT
# --------------------------------------------------

def load_index_checkpoint():

    if not os.path.exists(INDEX_CHECKPOINT_FILE):
        return {"last_completed_batch": 0}

    with open(
        INDEX_CHECKPOINT_FILE,
        "r",
        encoding="utf-8"
    ) as file:
        return json.load(file)


def save_index_checkpoint(data):

    temp_file = INDEX_CHECKPOINT_FILE + ".tmp"

    with open(
        temp_file,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(data, file, indent=4)

    os.replace(
        temp_file,
        INDEX_CHECKPOINT_FILE
    )


# --------------------------------------------------
# DOCUMENT IDENTIFIERS
# --------------------------------------------------

def create_document_id(invoice_id):
    """Create a stable ChromaDB document ID."""

    return hashlib.sha256(
        str(invoice_id).encode("utf-8")
    ).hexdigest()


# --------------------------------------------------
# CHROMADB SEMANTIC RETRIEVAL
# --------------------------------------------------

def _query_chromadb(query_text, top_k=5):

    if not query_text or not query_text.strip():
        return []

    if not isinstance(top_k, int) or not 1 <= top_k <= 50:
        raise ValueError("top_k must be between 1 and 50.")

    collection = get_chroma_collection()

    collection_count = collection.count()

    if collection_count == 0:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        [query_text],
        convert_to_numpy=True
    )[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_count),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    retrieved_records = []

    ids = results.get("ids", [[]])[0]
    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    for i, document in enumerate(documents):

        metadata = (
            metadatas[i]
            if metadatas and metadatas[i]
            else {}
        )

        retrieved_records.append({
            "document_id": ids[i],
            "invoice_id": metadata.get("invoice_id"),
            "document": document,
            "distance": (
                float(distances[i])
                if distances and distances[i] is not None
                else None
            )
        })

    return retrieved_records


# --------------------------------------------------
# PIPELINE A: INVOICE DECISION RETRIEVAL
# --------------------------------------------------

def retrieve_invoice_decision_context(
    invoice_data,
    risk_score,
    top_k=5
):
    """
    Retrieve the exact invoice and related historical records
    for the invoice decision workflow.
    """

    invoice_id = str(
        invoice_data.get("invoice_id", "")
    )

    if not invoice_id:
        raise ValueError(
            "invoice_id is required for invoice decision retrieval."
        )

    collection = get_chroma_collection()

    exact_id = create_document_id(invoice_id)

    exact_result = collection.get(
        ids=[exact_id],
        include=[
            "documents",
            "metadatas"
        ]
    )

    exact_documents = exact_result.get("documents") or []

    invoice_context = []

    if exact_documents:

        invoice_context.append({
            "invoice_id": invoice_id,
            "document": exact_documents[0],
            "context_type": "current_invoice"
        })

    query_fields = [
        "invoice_id",
        "supplier_id",
        "department_id",
        "invoice_amount",
        "currency",
        "payment_terms",
        "invoice_type",
        "supplier_country",
        "supplier_risk_score",
        "blacklisted_flag",
        "avg_invoice_amount",
        "supplier_invoice_count_30d",
        "supplier_avg_amount_90d",
        "invoice_amount_zscore",
        "duplicate_invoice_flag",
        "split_invoice_flag",
        "late_night_submission_flag"
    ]

    query_parts = []

    for field in query_fields:

        value = invoice_data.get(field)

        if value is not None:
            query_parts.append(
                f"{field}: {value}"
            )

    query_parts.append(
        f"risk_score: {risk_score}"
    )

    semantic_query = (
        "Find relevant invoice records with similar supplier details, "
        "invoice amounts, risk characteristics, behavioural indicators "
        "and possible fraud patterns.\n"
        + "\n".join(query_parts)
    )

    related_records = _query_chromadb(
        semantic_query,
        top_k=top_k + 1
    )

    related_records = [
        record
        for record in related_records
        if str(record.get("invoice_id")) != invoice_id
    ][:top_k]

    context_parts = []

    for item in invoice_context:

        context_parts.append(
            "[CURRENT INVOICE]\n"
            + item["document"]
        )

    for item in related_records:

        context_parts.append(
            f"[RELATED INVOICE: {item['invoice_id']}]\n"
            + item["document"]
        )

    return {
        "retrieval_type": "invoice_decision",
        "invoice_id": invoice_id,
        "risk_score": risk_score,
        "current_invoice": invoice_context,
        "related_invoices": related_records,
        "context": "\n\n".join(context_parts),
        "retrieved_count": (
            len(invoice_context) + len(related_records)
        )
    }


# --------------------------------------------------
# PIPELINE B: USER QUERY RETRIEVAL
# --------------------------------------------------

def retrieve_query_context(user_query, top_k=5):
    """
    Retrieve query-specific context for Ask AI.

    This workflow is independent of invoice decision retrieval.
    """

    if not user_query or not user_query.strip():

        raise ValueError(
            "Please provide a natural-language question."
        )

    retrieved_records = _query_chromadb(
        user_query.strip(),
        top_k=top_k
    )

    context_parts = []

    for record in retrieved_records:

        context_parts.append(
            f"[INVOICE: {record['invoice_id']}]\n"
            f"{record['document']}"
        )

    return {
        "retrieval_type": "user_query",
        "user_query": user_query,
        "retrieved_records": retrieved_records,
        "context": "\n\n".join(context_parts),
        "retrieved_count": len(retrieved_records)
    }


# --------------------------------------------------
# MANDATORY INVOICE FIELDS
# --------------------------------------------------

MANDATORY_INVOICE_FIELDS = [
    "invoice_id",
    "supplier_id",
    "department_id",
    "invoice_date",
    "invoice_amount",
    "currency",
    "payment_terms",
    "invoice_type"
]


def find_invoices_with_missing_mandatory_fields(sample_size=20):
    """
    Check every invoice for missing mandatory fields.

    Detects null values and empty/whitespace-only strings.
    """

    if not isinstance(sample_size, int) or sample_size < 0:
        raise ValueError("sample_size cannot be negative.")

    data = load_structured_data()

    missing_columns = [
        field
        for field in MANDATORY_INVOICE_FIELDS
        if field not in data.columns
    ]

    if missing_columns:

        raise ValueError(
            f"Mandatory columns missing from dataset: {missing_columns}"
        )

    missing_mask = pd.DataFrame(index=data.index)

    for field in MANDATORY_INVOICE_FIELDS:

        values = data[field]

        missing_mask[field] = (
            values.isna()
            | values.astype("string")
            .str.strip()
            .eq("")
            .fillna(False)
        )

    invoice_has_missing = missing_mask.any(axis=1)

    affected_rows = data.loc[
        invoice_has_missing
    ].copy()

    affected_rows["missing_fields"] = (
        missing_mask.loc[invoice_has_missing]
        .apply(
            lambda row: [
                field
                for field in MANDATORY_INVOICE_FIELDS
                if row[field]
            ],
            axis=1
        )
    )

    missing_counts = {
        field: int(missing_mask[field].sum())
        for field in MANDATORY_INVOICE_FIELDS
    }

    sample_columns = [
        field
        for field in [
            "invoice_id",
            "supplier_id",
            "department_id",
            "invoice_date",
            "invoice_amount",
            "currency",
            "payment_terms",
            "invoice_type",
            "missing_fields"
        ]
        if field in affected_rows.columns
    ]

    sample_records = (
        affected_rows[sample_columns]
        .head(sample_size)
        .to_dict(orient="records")
    )

    return {
        "analysis_type": "missing_mandatory_fields",
        "total_invoices_checked": int(len(data)),
        "invoices_with_missing_fields": int(
            invoice_has_missing.sum()
        ),
        "invoices_without_missing_fields": int(
            (~invoice_has_missing).sum()
        ),
        "missing_counts_by_field": missing_counts,
        "sample_size": len(sample_records),
        "sample_records": sample_records
    }


# --------------------------------------------------
# STRUCTURED QUERY ENGINE
# --------------------------------------------------

ALLOWED_QUERY_OPERATIONS = {
    "count",
    "sum",
    "average",
    "minimum",
    "maximum",
    "unique_count",
    "group_count",
    "group_aggregate",
    "multi_aggregate",
    "list"
}

ALLOWED_FILTER_OPERATORS = {
    "eq",
    "ne",
    "gt",
    "gte",
    "lt",
    "lte",
    "contains",
    "in",
    "not_in",
    "is_null",
    "not_null"
}


def _validate_limit(value, default=20):

    if value is None:
        value = default

    if type(value) is not int or not 1 <= value <= 100:
        raise ValueError("Limit must be an integer between 1 and 100.")

    return value


def _validate_group_fields(group_by, columns):

    if not isinstance(group_by, list) or not group_by:
        raise ValueError("A non-empty group_by list is required.")

    if len(group_by) > 3:
        raise ValueError("A maximum of three grouping fields is allowed.")

    if len(set(group_by)) != len(group_by):
        raise ValueError("Duplicate grouping fields are not allowed.")

    for field in group_by:

        if not isinstance(field, str) or field not in columns:
            raise ValueError(f"Unknown grouping field: {field}")

    return group_by



def _apply_structured_filters(data, filters):
    """Apply validated numeric, date, and text filters."""

    if not isinstance(filters, list) or len(filters) > 10:
        raise ValueError("A maximum of 10 filters is allowed.")

    date_fields = {
        "invoice_date"
    }

    for condition in filters:

        if not isinstance(condition, dict):
            raise ValueError("Each filter must be an object.")

        field = condition.get("field")
        operator = condition.get("operator")
        value = condition.get("value")

        if not isinstance(field, str) or field not in data.columns:
            raise ValueError(f"Unknown dataset field: {field}")

        if operator not in ALLOWED_FILTER_OPERATORS:
            raise ValueError(f"Unsupported filter operator: {operator}")

        series = data[field]

        if operator == "is_null":
            mask = series.isna()

        elif operator == "not_null":
            mask = series.notna()

        elif operator in {"eq", "ne"}:

            if value is None:
                raise ValueError("eq/ne filters require a value.")

            if field in date_fields:
                date_series = pd.to_datetime(
                    series, errors="coerce"
                )
                date_value = pd.to_datetime(value, errors="coerce")

                if pd.isna(date_value):
                    raise ValueError("Invalid date filter value.")

                mask = (
                    date_series == date_value
                    if operator == "eq"
                    else date_series != date_value
                )

            else:
                mask = (
                    series.astype("string") == str(value)
                    if operator == "eq"
                    else series.astype("string") != str(value)
                )

        elif operator in {"gt", "gte", "lt", "lte"}:

            if field in date_fields:
                comparable_series = pd.to_datetime(
                    series, errors="coerce"
                )
                try:
                    comparison_value = pd.to_datetime(
                        value, errors="raise"
                    )
                except (TypeError, ValueError):
                    raise ValueError("Date filters require a valid date value.")

            else:
                comparable_series = pd.to_numeric(
                    series, errors="coerce"
                )
                try:
                    comparison_value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        "Numeric filters require a numeric value."
                    )

            if operator == "gt":
                mask = comparable_series > comparison_value

            elif operator == "gte":
                mask = comparable_series >= comparison_value

            elif operator == "lt":
                mask = comparable_series < comparison_value

            else:
                mask = comparable_series <= comparison_value

        elif operator == "contains":

            if value is None:
                raise ValueError("contains requires a value.")

            mask = series.astype("string").str.contains(
                str(value),
                case=False,
                na=False,
                regex=False
            )

        elif operator in {"in", "not_in"}:

            if not isinstance(value, list) or len(value) > 100:
                raise ValueError(
                    "in/not_in requires a list of at most 100 values."
                )

            if field in date_fields:
                comparable_series = pd.to_datetime(
                    series, errors="coerce"
                )
                comparison_values = pd.to_datetime(
                    value, errors="coerce"
                )
                mask = comparable_series.isin(comparison_values)

            else:
                mask = series.isin(value)

            if operator == "not_in":
                mask = ~mask

        data = data.loc[mask].copy()

    return data



def execute_structured_query(plan):
    """
    Execute a controlled AI-generated query plan against
    the complete structured invoice dataset.

    Only predefined pandas operations are executed.
    No AI-generated Python or SQL is executed.
    """

    if not isinstance(plan, dict):
        raise ValueError("Query plan must be a dictionary.")

    operation = plan.get("operation")

    if operation not in ALLOWED_QUERY_OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")

    data = load_structured_data()
    total_invoices = len(data)

    data = _apply_structured_filters(
        data,
        plan.get("filters", [])
    )

    matching_invoices = len(data)

    # ------------------------------------------
    # COUNT
    # ------------------------------------------

    if operation == "count":
        result = {
            "count": int(matching_invoices)
        }

    # ------------------------------------------
    # GROUP COUNT
    # ------------------------------------------

    elif operation == "group_count":
        group_by = _validate_group_fields(
            plan.get("group_by"),
            data.columns
        )

        grouped = (
            data.groupby(group_by, dropna=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )

        limit = _validate_limit(plan.get("limit"))

        result = {
            "groups": grouped.head(limit).to_dict(orient="records"),
            "group_count": int(len(grouped))
        }

    # ------------------------------------------
    # SINGLE-COLUMN AGGREGATIONS
    # ------------------------------------------

    elif operation in {
        "sum",
        "average",
        "minimum",
        "maximum",
        "unique_count"
    }:
        field = plan.get("field")

        if not isinstance(field, str) or field not in data.columns:
            raise ValueError(f"Unknown aggregation field: {field}")

        series = data[field]

        if operation == "unique_count":
            value = int(series.nunique(dropna=True))

        else:
            numeric = pd.to_numeric(series, errors="coerce")

            if operation == "sum":
                value = float(numeric.sum())

            elif operation == "average":
                value = float(numeric.mean()) if numeric.notna().any() else None

            elif operation == "minimum":
                value = float(numeric.min()) if numeric.notna().any() else None

            else:
                value = float(numeric.max()) if numeric.notna().any() else None

        result = {
            "field": field,
            "value": value
        }

    # ------------------------------------------
    # MULTIPLE AGGREGATIONS
    # ------------------------------------------

    elif operation == "multi_aggregate":
        aggregations = plan.get("aggregations")

        if not isinstance(aggregations, list) or not aggregations:
            raise ValueError("Multi-aggregation requires a non-empty aggregations list.")

        if len(aggregations) > 10:
            raise ValueError("A maximum of 10 aggregations is allowed.")

        allowed_aggregations = {
            "sum",
            "average",
            "minimum",
            "maximum",
            "unique_count"
        }

        aggregation_results = []

        for item in aggregations:
            if not isinstance(item, dict):
                raise ValueError("Each aggregation must be a dictionary.")

            agg_operation = item.get("operation")
            field = item.get("field")

            if agg_operation not in allowed_aggregations:
                raise ValueError(
                    f"Unsupported aggregation: {agg_operation}"
                )

            if not isinstance(field, str) or field not in data.columns:
                raise ValueError(f"Unknown aggregation field: {field}")

            series = data[field]

            if agg_operation == "unique_count":
                value = int(series.nunique(dropna=True))

            else:
                numeric = pd.to_numeric(series, errors="coerce")

                if agg_operation == "sum":
                    value = float(numeric.sum())

                elif agg_operation == "average":
                    value = (
                        float(numeric.mean())
                        if numeric.notna().any() else None
                    )

                elif agg_operation == "minimum":
                    value = (
                        float(numeric.min())
                        if numeric.notna().any() else None
                    )

                else:
                    value = (
                        float(numeric.max())
                        if numeric.notna().any() else None
                    )

            aggregation_results.append({
                "operation": agg_operation,
                "field": field,
                "value": value
            })

        result = {
            "aggregations": aggregation_results
        }

    # ------------------------------------------
    # GROUPED AGGREGATION
    # ------------------------------------------

    elif operation == "group_aggregate":
        group_by = _validate_group_fields(
            plan.get("group_by"),
            data.columns
        )

        aggregate = plan.get("aggregate", "sum")
        field = plan.get("field")

        if aggregate not in {"sum", "mean", "min", "max", "count"}:
            raise ValueError("Unsupported grouped aggregation.")

        if aggregate == "count" and field is None:
            grouped = (
                data.groupby(group_by, dropna=False)
                .size()
                .reset_index(name="result")
            )

        else:
            if not isinstance(field, str) or field not in data.columns:
                raise ValueError(
                    f"Aggregation '{aggregate}' requires a valid field. "
                    f"Received: {field}"
                )

            if aggregate == "count":
                grouped = (
                    data.groupby(group_by, dropna=False)[field]
                    .count()
                    .reset_index(name="result")
                )

            else:
                numeric = pd.to_numeric(data[field], errors="coerce")

                working = data[group_by].copy()
                working["_value"] = numeric

                grouped = (
                    working.groupby(group_by, dropna=False)["_value"]
                    .agg(aggregate)
                    .reset_index(name="result")
                )

        grouped = grouped.sort_values(
            "result",
            ascending=False
        )

        limit = _validate_limit(plan.get("limit"))

        result = {
            "groups": grouped.head(limit).to_dict(orient="records"),
            "group_count": int(len(grouped))
        }

    # ------------------------------------------
    # LIST RECORDS
    # ------------------------------------------

    elif operation == "list":
        columns = plan.get("columns") or ["invoice_id"]

        if not isinstance(columns, list) or not columns:
            raise ValueError("List operation requires columns.")

        if len(columns) > 15:
            raise ValueError("A maximum of 15 output columns is allowed.")

        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate output columns are not allowed.")

        invalid_columns = [
            column for column in columns
            if not isinstance(column, str) or column not in data.columns
        ]

        if invalid_columns:
            raise ValueError(
                f"Unknown output columns: {invalid_columns}"
            )

        sort_by = plan.get("sort_by")

        if sort_by is not None:
            if not isinstance(sort_by, str) or sort_by not in data.columns:
                raise ValueError(f"Unknown sort field: {sort_by}")

            sort_order = plan.get("sort_order", "desc")

            if sort_order not in {"asc", "desc"}:
                raise ValueError("sort_order must be asc or desc.")

            data = data.sort_values(
                sort_by,
                ascending=(sort_order == "asc"),
                na_position="last"
            )

        limit = _validate_limit(plan.get("limit"))

        shown = min(limit, matching_invoices)

        result = {
            "records": (
                data[columns]
                .head(limit)
                .to_dict(orient="records")
            ),
            "returned_records": shown,
            # Without this the LLM sometimes describes the shown rows as
            # if they were all matching invoices.
            "note": (
                f"Only the first {shown} of {int(matching_invoices):,} "
                "matching invoices are shown"
                + (f", sorted by {sort_by} ({plan.get('sort_order', 'desc')})" if sort_by else "")
                + ". These rows are NOT representative of all matching invoices."
            )
        }

        if sort_by is not None and pd.api.types.is_numeric_dtype(data[sort_by]):
            values = data[sort_by].dropna()
            result["sort_field_summary_over_all_matching"] = {
                "field": sort_by,
                "min": float(values.min()),
                "median": float(values.median()),
                "mean": float(values.mean()),
                "max": float(values.max()),
                "std": float(values.std()),
            }

    else:
        raise ValueError(f"Operation not implemented: {operation}")

    return {
        "analysis_type": "full_dataset",
        "operation": operation,
        "total_invoices_checked": int(total_invoices),
        "matching_invoices": int(matching_invoices),
        "result": result
    }

# --------------------------------------------------
# INDEXING FOUNDATION
# --------------------------------------------------

def index_invoice_batch(rows, collection, model):
    """Convert and embed one batch of invoice records."""

    documents = []
    document_ids = []
    metadatas = []

    for _, row in rows.iterrows():

        invoice_id = str(row["invoice_id"])

        document = invoice_to_document(row)

        if not document.strip():
            continue

        documents.append(document)

        document_ids.append(
            create_document_id(invoice_id)
        )

        metadatas.append({
            "invoice_id": invoice_id
        })

    if not documents:
        return 0

    embeddings = model.encode(
        documents,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True
    )

    collection.upsert(
        ids=document_ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    return len(documents)


# --------------------------------------------------
# COMPLETE DATASET INDEXING
# --------------------------------------------------

def index_all_invoices(batch_size=BATCH_SIZE):
    """
    Index structured invoices in resumable batches.

    Existing records are skipped.
    This function is provided for initial indexing or
    explicitly requested index maintenance only.
    """

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    print("\nLoading structured invoice dataset...")

    data = load_structured_data()

    total_invoices = len(data)

    collection = get_chroma_collection()
    model = get_embedding_model()

    initial_count = collection.count()

    print("\n" + "=" * 60)
    print("INVOICE RAG INDEXING")
    print("=" * 60)

    print("Total invoices:", total_invoices)
    print("Already indexed:", initial_count)
    print("Batch size:", batch_size)

    newly_indexed = 0
    skipped = 0
    failed_batches = 0

    for start in range(0, total_invoices, batch_size):

        end = min(
            start + batch_size,
            total_invoices
        )

        batch = data.iloc[start:end]

        batch_ids = [
            create_document_id(invoice_id)
            for invoice_id in batch["invoice_id"].astype(str)
        ]

        existing = set(
            collection.get(ids=batch_ids)["ids"]
        )

        pending_positions = [
            position
            for position, document_id in enumerate(batch_ids)
            if document_id not in existing
        ]

        skipped += len(batch) - len(pending_positions)

        if pending_positions:

            pending_batch = batch.iloc[
                pending_positions
            ]

            try:

                indexed = index_invoice_batch(
                    pending_batch,
                    collection,
                    model
                )

                newly_indexed += indexed

            except Exception as error:

                failed_batches += 1

                print(
                    f"\nBatch {start}:{end} failed: {error}"
                )

                raise

        checkpoint = {
            "collection_name": COLLECTION_NAME,
            "total_invoices": total_invoices,
            "newly_indexed_this_run": newly_indexed,
            "skipped_existing": skipped,
            "collection_count": collection.count(),
            "last_completed_batch_end": end,
            "failed_batches": failed_batches,
            "updated_at_utc": datetime.now(
                timezone.utc
            ).isoformat()
        }

        save_index_checkpoint(checkpoint)

        print(
            f"Processed {end:,}/{total_invoices:,} | "
            f"Newly indexed: {newly_indexed:,} | "
            f"Collection total: {collection.count():,}"
        )

    print("\n" + "=" * 60)
    print("RAG INDEXING COMPLETED")
    print("=" * 60)

    print("Newly indexed this run:", newly_indexed)
    print("Skipped existing:", skipped)
    print("Total documents in ChromaDB:", collection.count())

    return collection.count()


print("RAG engine module loaded successfully.")
