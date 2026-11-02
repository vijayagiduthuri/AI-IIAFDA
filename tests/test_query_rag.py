import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from backend.rag_engine import retrieve_query_context
from backend.ai_explainer import answer_invoice_query


query = "Show invoices with unusual supplier risk scores."

print("=" * 65)
print("CONVERSATIONAL INVOICE INTELLIGENCE TEST")
print("=" * 65)

# 1. Retrieve query-specific context
retrieved_context = retrieve_query_context(
    user_query=query,
    top_k=5
)

print("\nRetrieved records:", retrieved_context["retrieved_count"])

for record in retrieved_context["retrieved_records"]:
    print("\nInvoice ID:", record["invoice_id"])
    print("Distance:", record["distance"])
    print(record["document"][:500])

# 2. Generate conversational answer
answer = answer_invoice_query(
    user_query=query,
    retrieved_context=retrieved_context
)

print("\n" + "=" * 65)
print("GROQ CONVERSATIONAL ANSWER")
print("=" * 65)
print(answer)
