"""
Check the "Ask AI" feature with questions whose correct answer is known.

Each question goes through the same steps as the app:
  plan_invoice_query -> execute_structured_query and/or retrieve_query_context
  -> answer_invoice_query / answer_hybrid_invoice_query
The correct value is computed directly with pandas, so the AI answer
can be compared with it.

Run from the project folder:
    python scripts/check_ask_ai.py
"""

import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from backend.data_loader import load_structured_data
from backend.rag_engine import execute_structured_query, retrieve_query_context
from backend.ai_explainer import (
    plan_invoice_query, answer_invoice_query, answer_hybrid_invoice_query
)


def ask(question):
    plan = plan_invoice_query(question)
    route = plan["route"]
    structured, retrieved = None, None
    if route in {"structured", "hybrid"}:
        structured = execute_structured_query(plan)
    if route in {"rag", "hybrid"}:
        retrieved = retrieve_query_context(plan.get("rag_query") or question, top_k=5)
    if route == "rag":
        answer = answer_invoice_query(user_query=question, retrieved_context=retrieved)
    else:
        answer = answer_hybrid_invoice_query(user_query=question,
                                             structured_result=structured,
                                             retrieved_context=retrieved)
    return plan, structured, retrieved, answer


def main():
    d = load_structured_data()
    top_risk = d.sort_values("supplier_risk_score", ascending=False).head(3)

    checks = [
        ("How many invoices are in the dataset?", f"{len(d):,}"),
        ("What is the average invoice amount?", f"{d['invoice_amount'].mean():,.2f}"),
        ("How many duplicate invoices exist?", f"{int(d['duplicate_invoice_flag'].sum()):,}"),
        ("How many invoices come from blacklisted suppliers?",
         f"{int((d['blacklisted_flag'] == 1).sum()):,}"),
        ("How many invoices were submitted late at night?",
         f"{int(d['late_night_submission_flag'].sum()):,}"),
        ("Which supplier country has the most invoices?",
         f"{d['supplier_country'].value_counts().idxmax()} "
         f"({d['supplier_country'].value_counts().max():,})"),
        ("What is the highest invoice amount?", f"{d['invoice_amount'].max():,.2f}"),
        ("Show the 3 invoices with the highest supplier risk scores.",
         ", ".join(f"{r.invoice_id} ({r.supplier_risk_score})" for r in top_risk.itertuples())),
        ("Show invoices with unusual supplier risk scores.", "(open question - RAG, judge by eye)"),
    ]

    for question, truth in checks:
        print("\n" + "=" * 70)
        print("Q:", question)
        print("Correct:", truth)
        try:
            plan, structured, retrieved, answer = ask(question)
            print("Route:", plan["route"], "| operation:", plan.get("operation"))
            if retrieved:
                print("Retrieved:", retrieved.get("retrieved_count"))
            text = answer if isinstance(answer, str) else str(answer)
            print("AI answer:", text.strip()[:700])
        except Exception as error:
            print("ERROR:", type(error).__name__, error)


if __name__ == "__main__":
    main()
