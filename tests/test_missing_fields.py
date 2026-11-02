import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.rag_engine import find_invoices_with_missing_mandatory_fields

result = find_invoices_with_missing_mandatory_fields()

print("Invoices checked:", result["total_invoices_checked"])
print("Invoices with missing fields:", result["invoices_with_missing_fields"])
print("Invoices without missing fields:", result["invoices_without_missing_fields"])

print("\nMissing counts by field:")
for field, count in result["missing_counts_by_field"].items():
    print(field, ":", count)

print("\nSample affected invoices:")
for record in result["sample_records"]:
    print(record)