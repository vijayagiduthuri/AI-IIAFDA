"""
Rule-based decision engine.

The final disposition (APPROVE / REJECT / MANUAL REVIEW) is decided here
by fixed rules, so the same evidence always gives the same decision.
The LLM is only used afterwards to explain this decision in words.
"""


def _flag(invoice, field):
    """Read a 0/1 flag from the structured invoice safely."""

    value = invoice.get(field)

    try:
        return int(float(value)) == 1
    except (TypeError, ValueError):
        return False


def determine_invoice_disposition(analysis_result):
    """
    Decide the final invoice disposition from:
      - validation results (hard and soft issues)
      - ML risk prediction
      - duplicate checks against previously analysed invoices
      - risk flags from the structured dataset

    Rules are checked in order. Every rule that fires is recorded in
    `triggered_rules`, and the most serious one decides the outcome.
    """

    if analysis_result.get("status") != "completed":
        return {
            "disposition": "MANUAL REVIEW",
            "reason": "Invoice analysis is incomplete.",
            "triggered_rules": ["ANALYSIS_INCOMPLETE"],
            "hard_issues": [],
            "soft_issues": []
        }

    prediction = analysis_result.get("prediction") or {}
    validation = analysis_result.get("validation", {}) or {}
    duplicate_check = analysis_result.get("duplicate_check", {}) or {}
    invoice = analysis_result.get("structured_invoice", {}) or {}
    is_new_invoice = analysis_result.get("invoice_source") == "new"
    supplier_found = analysis_result.get("supplier_found", True)

    risk_level = prediction.get("risk_level", "UNKNOWN")
    risk_score = prediction.get("fraud_probability")
    model_flagged = prediction.get("prediction") == 1

    hard_issues = validation.get("hard_issues", [])
    soft_issues = validation.get("soft_issues", [])

    reject_reasons = []
    review_reasons = []
    triggered = []

    # ----------------------------------------------
    # REJECT RULES
    # ----------------------------------------------

    if _flag(invoice, "blacklisted_flag"):
        triggered.append("BLACKLISTED_SUPPLIER")
        reject_reasons.append(
            "The supplier is blacklisted. Payments to blacklisted "
            "suppliers are not allowed."
        )

    if duplicate_check.get("previously_approved"):
        triggered.append("ALREADY_APPROVED_DUPLICATE")
        reject_reasons.append(
            "This invoice ID was already analysed and approved earlier "
            f"(record {duplicate_check.get('approved_record_id')}). "
            "Paying it again would be a duplicate payment."
        )

    # ----------------------------------------------
    # MANUAL REVIEW RULES
    # ----------------------------------------------

    if is_new_invoice and not supplier_found:
        triggered.append("UNKNOWN_SUPPLIER")
        review_reasons.append(
            "The supplier on this invoice is not in the supplier master "
            "(or its ID could not be read). New suppliers must be "
            "verified before payment."
        )

    if hard_issues:
        triggered.append("HARD_VALIDATION_FAILURE")
        review_reasons.append(
            "Validation failed for: "
            + ", ".join(issue["field"] for issue in hard_issues)
            + "."
        )

    if duplicate_check.get("same_invoice_seen") and not duplicate_check.get(
        "previously_approved"
    ):
        triggered.append("REPEATED_SUBMISSION")
        review_reasons.append(
            "This invoice ID has been submitted before "
            f"({duplicate_check.get('previous_count')} earlier analysis/analyses)."
        )

    similar = (
        duplicate_check.get("similar_invoices", [])
        + duplicate_check.get("dataset_similar_invoices", [])
    )

    if similar:
        triggered.append("POSSIBLE_DUPLICATE")
        similar_ids = ", ".join(
            str(item["invoice_id"]) for item in similar[:3]
        )
        review_reasons.append(
            "Another invoice from the same supplier has a matching amount "
            f"and a nearby date ({similar_ids})."
        )

    if _flag(invoice, "duplicate_invoice_flag"):
        triggered.append("DATASET_DUPLICATE_FLAG")
        review_reasons.append(
            "The invoice is marked as a possible duplicate in the dataset."
        )

    if _flag(invoice, "split_invoice_flag"):
        triggered.append("SPLIT_INVOICE_FLAG")
        review_reasons.append(
            "The invoice looks like part of a split invoice "
            "(several small invoices instead of one large one)."
        )

    if not prediction:
        triggered.append("RISK_NOT_SCORED")
        review_reasons.append(
            "The fraud model could not score this invoice "
            "(missing supplier or invoice data)."
        )
    elif risk_level == "HIGH":
        triggered.append("HIGH_RISK_SCORE")
        review_reasons.append(
            f"The fraud model gave a HIGH risk score ({risk_score})."
        )
    elif model_flagged:
        triggered.append("MODEL_FLAGGED")
        review_reasons.append(
            "The fraud detection model classified this invoice as suspicious."
        )
    elif risk_level == "MEDIUM":
        triggered.append("MEDIUM_RISK_SCORE")
        review_reasons.append(
            f"The fraud model gave a MEDIUM risk score ({risk_score})."
        )
    elif risk_level != "LOW":
        triggered.append("RISK_UNAVAILABLE")
        review_reasons.append("The risk level could not be determined.")

    # ----------------------------------------------
    # FINAL OUTCOME
    # ----------------------------------------------

    base = {
        "triggered_rules": triggered,
        "hard_issues": hard_issues,
        "soft_issues": soft_issues,
        "all_reasons": reject_reasons + review_reasons
    }

    if reject_reasons:
        return {
            "disposition": "REJECT",
            "reason": " ".join(reject_reasons),
            **base
        }

    if review_reasons:
        return {
            "disposition": "MANUAL REVIEW",
            "reason": " ".join(review_reasons),
            **base
        }

    reason = (
        f"Low fraud risk ({risk_score}), all required fields are valid "
        "and no duplicates were found."
    )

    if soft_issues:
        reason += (
            " Minor notes (not blocking): "
            + ", ".join(issue["field"] for issue in soft_issues)
            + "."
        )

    triggered.append("ALL_CHECKS_PASSED")

    return {
        "disposition": "APPROVE",
        "reason": reason,
        **base
    }