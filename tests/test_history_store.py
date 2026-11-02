import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


from backend.history_store import (
    DATABASE_PATH,
    get_invoice_history
)

print("Database exists:", DATABASE_PATH.exists())
print("Database path:", DATABASE_PATH)

history = get_invoice_history()

print("Previously saved invoice records:", len(history))
