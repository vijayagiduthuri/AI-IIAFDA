import pandas as pd
from pathlib import Path

from backend.invoice_processor import (
    extract_invoice_text,
    extract_invoice_fields
)

from backend.validator import (
    validate_invoice_fields,
    validate_new_invoice
)

from backend.data_loader import (
    get_invoice_record,
    get_supplier_history,
    prepare_model_features,
    validate_model_features
)

from backend.new_invoice_features import (
    get_supplier,
    build_new_invoice_record,
    find_similar_dataset_invoices,
    known_values
)

from backend.fraud_detector import predict_invoice

from backend.rag_engine import retrieve_invoice_decision_context

from backend.decision_engine import determine_invoice_disposition

from backend.history_store import check_for_duplicates

from backend.ai_explainer import generate_invoice_explanation, AIRateLimitError


def _clean_record(record_df):
    """One-row DataFrame -> JSON-friendly dict (dates as YYYY-MM-DD)."""

    return {
        key: (
            value.strftime("%Y-%m-%d")
            if isinstance(value, pd.Timestamp) and not pd.isna(value)
            else value
        )
        for key, value in record_df.iloc[0].to_dict().items()
    }


def _score(record_df):
    """Run the fraud model on a one-row record. Returns None if not possible."""

    features_df = prepare_model_features(record_df)

    try:
        model_input = validate_model_features(features_df)
    except ValueError:
        return None

    return predict_invoice(model_input.iloc[0].to_dict())


def analyze_invoice(file_path=None, uploaded_file=None, submission_time=None):
    """
    Full invoice pipeline:

      1. OCR
      2. Field extraction
      3. Is the invoice in the dataset?
           YES -> validate against the dataset record, score it
           NO  -> NEW INVOICE: validate against the supplier master,
                  rebuild the model features from the supplier's history,
                  score it (unknown supplier -> not scored)
      4. Duplicate check (analysis history + dataset)
      5. RULE-BASED final decision  (same input -> same decision)
      6. Supplier history + RAG context
      7. LLM explanation of the decision (does not change it)
    """

    # 1. OCR
    ocr_result = extract_invoice_text(
        file_path=file_path,
        uploaded_file=uploaded_file
    )

    if ocr_result.get("status") != "success":
        return {
            "status": "ocr_failed",
            "message": "No readable text was found on the invoice.",
            "ocr_result": ocr_result
        }

    ocr_text = ocr_result.get("extracted_text", "")

    # 2. Extract invoice fields
    extracted_fields = extract_invoice_fields(ocr_text)
    invoice_id = extracted_fields.get("invoice_id")

    if not invoice_id:
        return {
            "status": "invoice_id_missing",
            "message": "The invoice number could not be read from the invoice.",
            "ocr_result": ocr_result,
            "extracted_fields": extracted_fields
        }

    # 3. Dataset invoice or new invoice?
    invoice_record = get_invoice_record(invoice_id)
    feature_assumptions = []
    dataset_similar = []

    if invoice_record is not None:

        # ---------- invoice exists in the dataset ----------
        invoice_source = "dataset"
        supplier_found = True
        structured_invoice = _clean_record(invoice_record)

        validation_result = validate_invoice_fields(
            extracted_fields,
            dataset_record=structured_invoice
        )

        prediction_result = _score(invoice_record)

    else:

        # ---------- NEW invoice ----------
        invoice_source = "new"
        supplier = get_supplier(extracted_fields.get("supplier_id"))
        supplier_found = supplier is not None
        allowed = known_values()

        validation_result = validate_new_invoice(
            extracted_fields,
            supplier=supplier,
            known_currencies=allowed["currency"],
            known_payment_terms=allowed["payment_terms"]
        )

        can_score = (
            supplier_found
            and extracted_fields.get("total") is not None
            and extracted_fields.get("invoice_date") is not None
        )

        if can_score:
            record_df, feature_assumptions = build_new_invoice_record(
                extracted_fields, supplier, submission_time
            )
            structured_invoice = _clean_record(record_df)
            prediction_result = _score(record_df)

            if prediction_result is None:
                feature_assumptions.append(
                    "Some model features could not be built, so the "
                    "invoice was not scored."
                )

            dataset_similar = find_similar_dataset_invoices(
                supplier["supplier_id"],
                extracted_fields.get("total"),
                extracted_fields.get("invoice_date")
            )

        else:
            prediction_result = None
            structured_invoice = {
                "invoice_id": invoice_id,
                "supplier_id": extracted_fields.get("supplier_id"),
                "invoice_date": extracted_fields.get("invoice_date"),
                "invoice_amount": extracted_fields.get("total"),
                "currency": extracted_fields.get("currency"),
                "payment_terms": extracted_fields.get("payment_terms"),
                "supplier_country": extracted_fields.get("supplier_country")
            }
            feature_assumptions.append(
                "The invoice could not be scored: "
                + ("supplier not found in the supplier master."
                   if not supplier_found
                   else "invoice total or date could not be read.")
            )

    risk_score = prediction_result["fraud_probability"] if prediction_result else None
    risk_level = prediction_result["risk_level"] if prediction_result else "NOT SCORED"

    # 4. Duplicate check against earlier analyses (+ dataset for new invoices)
    duplicate_check = check_for_duplicates(
        invoice_id=invoice_id,
        supplier_id=structured_invoice.get("supplier_id"),
        invoice_amount=structured_invoice.get("invoice_amount"),
        invoice_date=structured_invoice.get("invoice_date")
    )
    duplicate_check["dataset_similar_invoices"] = dataset_similar

    analysis_evidence = {
        "status": "completed",
        "invoice_source": invoice_source,
        "supplier_found": supplier_found,
        "filename": ocr_result.get("filename"),
        "ocr_text": ocr_text,
        "extracted_fields": extracted_fields,
        "validation": validation_result,
        "prediction": prediction_result,
        "duplicate_check": duplicate_check,
        "structured_invoice": structured_invoice,
        "feature_assumptions": feature_assumptions
    }

    # 5. Rule-based final decision
    decision = determine_invoice_disposition(analysis_evidence)

    # 6. Context for the explanation
    supplier_history = []
    rag_context = {"context": "", "retrieved_count": 0}

    if supplier_found and structured_invoice.get("supplier_id"):

        supplier_history = get_supplier_history(
            structured_invoice.get("supplier_id"),
            exclude_invoice_id=invoice_id,
            limit=10
        ).to_dict(orient="records")

        supplier_history = [
            {
                key: (
                    value.strftime("%Y-%m-%d")
                    if isinstance(value, pd.Timestamp) else value
                )
                for key, value in row.items()
            }
            for row in supplier_history
        ]

        try:
            rag_context = retrieve_invoice_decision_context(
                invoice_data=structured_invoice,
                risk_score=risk_score,
                top_k=5
            )
        except Exception as error:
            rag_context = {
                "context": "",
                "retrieved_count": 0,
                "error": str(error)
            }

    # 7. LLM explanation (never changes the decision)
    try:
        explanation = generate_invoice_explanation(
            analysis_result=analysis_evidence,
            decision=decision,
            retrieved_context=rag_context,
            supplier_history=supplier_history
        )
    except Exception as error:
        explanation = {
            "explanation": (
                (str(error) + " " if isinstance(error, AIRateLimitError) else
                 "The AI explanation could not be generated. ")
                + "The decision above is based on the fixed rules."
            ),
            "key_findings": [],
            "supplier_history_findings": [],
            "recommended_actions": [],
            "explanation_error": str(error)
        }

    disposition_result = {
        **explanation,
        "disposition": decision["disposition"],
        "reason": decision["reason"],
        "triggered_rules": decision["triggered_rules"],
        "all_reasons": decision.get("all_reasons", []),
        "validation_issues": decision["hard_issues"],
        "soft_issues": decision["soft_issues"],
        "risk_score": risk_score,
        "risk_level": risk_level,
        "retrieved_context_count": rag_context.get("retrieved_count", 0)
    }

    return {
        "status": "completed",
        "invoice_source": invoice_source,
        "supplier_found": supplier_found,
        "feature_assumptions": feature_assumptions,
        "filename": ocr_result.get("filename"),
        "ocr_text": ocr_text,
        "extracted_fields": extracted_fields,
        "validation": validation_result,
        "prediction": prediction_result,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "duplicate_check": duplicate_check,
        "supplier_history": supplier_history,
        "rag_context": rag_context,
        "disposition": disposition_result,
        "structured_invoice": structured_invoice
    }


# --------------------------------------------------
# TEST FROM THE COMMAND LINE
#   python -m backend.invoice_analyzer                (first dataset invoice)
#   python -m backend.invoice_analyzer path\to\file.png
# --------------------------------------------------

if __name__ == "__main__":

    import sys

    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        from backend.config import INVOICES_FILE, IMAGES_DIR

        invoices = pd.read_parquet(INVOICES_FILE)
        sample = invoices.loc[invoices["image_path"].notna()].iloc[0]
        image_path = IMAGES_DIR / Path(str(sample["image_path"])).name

    print("=" * 65)
    print("AI-POWERED INVOICE INTELLIGENCE & FRAUD DETECTION")
    print("=" * 65)
    print("\nAnalyzing:", image_path)

    result = analyze_invoice(file_path=image_path)

    print("\nAnalysis status:", result["status"])

    if result["status"] == "completed":

        print("Invoice source:", result["invoice_source"])
        print("Risk score:", result["risk_score"], "|", result["risk_level"])
        print("\nDECISION:", result["disposition"]["disposition"])
        print("Reason:", result["disposition"]["reason"])
        print("Rules fired:", result["disposition"]["triggered_rules"])

        print("\nVALIDATION")
        for field, comparison in result["validation"]["comparisons"].items():
            print(
                f"  {field:20s} {comparison['status']:9s} "
                f"({comparison['severity']})  "
                f"OCR={comparison['ocr_value']}  "
                f"expected={comparison['dataset_value']}"
            )

        if result["feature_assumptions"]:
            print("\nASSUMPTIONS")
            for note in result["feature_assumptions"]:
                print("  -", note)

    else:
        print(result)