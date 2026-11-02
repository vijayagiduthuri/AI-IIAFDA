import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag_engine import (
    retrieve_invoice_decision_context,
    retrieve_query_context
)

# Test Pipeline A
sample_invoice = {
    "invoice_id": "INV_0000000",
    "supplier_id": "SUP_0000",
    "invoice_amount": 3565.57,
    "currency": "ZAR"
}

decision_context = retrieve_invoice_decision_context(
    invoice_data=sample_invoice,
    risk_score=14.95,
    top_k=5
)

print("PIPELINE A")
print("Retrieved count:", decision_context["retrieved_count"])
print("Current invoice found:",
      len(decision_context["current_invoice"]))
print("Related invoices:",
      len(decision_context["related_invoices"]))
print("\nContext preview:")
print(decision_context["context"][:2000])


# Test Pipeline B
query_context = retrieve_query_context(
    "Show invoices with unusual supplier risk scores.",
    top_k=5
)

print("\nPIPELINE B")
print("Retrieved count:", query_context["retrieved_count"])
print("\nContext preview:")
print(query_context["context"][:2000])