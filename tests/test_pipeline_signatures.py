import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import inspect

from backend.ai_explainer import (
    plan_invoice_query,
    answer_hybrid_invoice_query
)

from backend.rag_engine import execute_structured_query

print("=" * 60)
print("PIPELINE B FUNCTION CHECK")
print("=" * 60)

print("Query planner:", inspect.signature(plan_invoice_query))
print("Query answer:", inspect.signature(answer_hybrid_invoice_query))
print("Structured executor:", inspect.signature(execute_structured_query))