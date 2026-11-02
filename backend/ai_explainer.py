import os
import json
import re
from functools import lru_cache

from groq import Groq
from dotenv import load_dotenv


# --------------------------------------------------
# GROQ CONFIGURATION
# --------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "openai/gpt-oss-20b"


def get_groq_client():
    """Initialize the Groq client."""

    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Please configure it in your .env file."
        )

    return Groq(
        api_key=GROQ_API_KEY
    )


# Used in order when a model hits its free-tier rate limit
# (each Groq model has its own daily token quota).
GROQ_FALLBACK_MODELS = [
    GROQ_MODEL,
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# model -> time (epoch seconds) until which it is skipped
_rate_limited_until = {}


class AIRateLimitError(RuntimeError):
    """All Groq models are rate limited; message is user-friendly."""


def _retry_seconds(error):
    """Read 'try again in 1m35.4s' from a Groq rate-limit error."""

    match = re.search(r"try again in (?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", str(error))

    if not match or not any(match.groups()):
        return 60.0

    hours, minutes, seconds = (float(x) if x else 0.0 for x in match.groups())

    return hours * 3600 + minutes * 60 + seconds


def _call_groq(system_prompt, user_prompt, max_tokens=1500):
    """
    Send a prompt to Groq. If a model is rate limited, the next model
    in GROQ_FALLBACK_MODELS is tried; rate-limited models are skipped
    until their retry time has passed.
    """

    import time
    from groq import RateLimitError

    now = time.time()
    shortest_wait = None

    for model in GROQ_FALLBACK_MODELS:

        if _rate_limited_until.get(model, 0) > now:
            wait = _rate_limited_until[model] - now
            shortest_wait = wait if shortest_wait is None else min(shortest_wait, wait)
            continue

        try:
            return _call_groq_model(model, system_prompt, user_prompt, max_tokens)

        except RateLimitError as error:
            wait = _retry_seconds(error)
            _rate_limited_until[model] = time.time() + wait
            shortest_wait = wait if shortest_wait is None else min(shortest_wait, wait)

    minutes = max(1, round((shortest_wait or 60) / 60))

    raise AIRateLimitError(
        "The free AI usage limit (Groq) has been reached for now. "
        f"Please try again in about {minutes} minute(s). "
        "Invoice checks, risk scores and decisions still work; "
        "only the AI text is unavailable."
    )


def _call_groq_model(model, system_prompt, user_prompt, max_tokens):
    """One Groq request to a specific model."""

    client = get_groq_client()

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=max_tokens
    )

    if not response.choices:
        raise RuntimeError(
            f"Groq returned no choices. Response: {response}"
        )

    choice = response.choices[0]
    content = choice.message.content

    if not content or not content.strip():
        raise RuntimeError(
            "Groq returned empty content. "
            f"Finish reason: {choice.finish_reason}. "
            f"Usage: {response.usage}. "
            f"Message: {choice.message}"
        )

    # Some models (e.g. Qwen) prepend their reasoning in <think> tags.
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)

    # gpt-oss-120b sometimes adds citation marks such as 【full_dataset】.
    content = re.sub(r"【[^】]*】", "", content)

    return content.strip()


# --------------------------------------------------
# JSON RESPONSE PARSER
# --------------------------------------------------

def _extract_json_object(response_text):
    """Extract a JSON object from an LLM response."""

    cleaned = response_text.strip()

    cleaned = re.sub(
        r"^```(?:json)?\s*",
        "",
        cleaned
    )

    cleaned = re.sub(
        r"\s*```$",
        "",
        cleaned
    )

    start = cleaned.find("{")
    end = cleaned.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError(
            "The AI response did not contain valid JSON."
        )

    result = json.loads(
        cleaned[start:end + 1]
    )

    if not isinstance(result, dict):
        raise ValueError(
            "The AI response must be a JSON object."
        )

    return result


def _as_list(value):
    """Make sure an LLM field is a list of strings."""

    if value is None:
        return []

    if isinstance(value, list):
        return [
            item if isinstance(item, str) else json.dumps(item, default=str)
            for item in value
        ]

    return [str(value)]


# --------------------------------------------------
# PIPELINE A: EXPLAIN THE RULE-BASED DECISION
# --------------------------------------------------

@lru_cache(maxsize=1)
def _supplier_risk_range():
    """Typical supplier_risk_score values, so the LLM can judge one in context."""

    from backend.data_loader import load_structured_data

    scores = load_structured_data()["supplier_risk_score"].dropna()
    q = scores.quantile([0.25, 0.5, 0.75, 0.9])

    return (
        f"25% of invoices are below {q[0.25]:.3f}, median {q[0.5]:.3f}, "
        f"75% below {q[0.75]:.3f}, 90% below {q[0.9]:.3f}, max {scores.max():.3f}"
    )


@lru_cache(maxsize=1)
def _dataset_currency_label():
    """Currency code(s) used in the dataset, e.g. 'ZAR'."""

    from backend.data_loader import load_structured_data

    currencies = load_structured_data()["currency"].dropna().astype(str).unique()

    return ", ".join(sorted(currencies)[:5])


def generate_invoice_explanation(
    analysis_result,
    decision,
    retrieved_context,
    supplier_history=None
):
    """
    Explain a disposition that has ALREADY been decided by the
    rule-based decision engine (backend/decision_engine.py).

    The LLM does not choose APPROVE / REJECT / MANUAL REVIEW.
    It only explains the decision in clear language, using:
    - Structured invoice information
    - ML fraud risk score
    - Validation findings
    - The supplier's recent invoice history
    - Invoice-specific RAG context
    """

    if analysis_result.get("status") != "completed":
        raise ValueError(
            "A completed invoice analysis is required."
        )

    prediction = analysis_result.get("prediction") or {}

    evidence = {
        "invoice_source": (
            "NEW invoice (not in the dataset); model features were rebuilt "
            "from the supplier master and the supplier's past invoices"
            if analysis_result.get("invoice_source") == "new"
            else "invoice found in the dataset"
        ),
        "feature_assumptions": analysis_result.get("feature_assumptions", []),
        "filename": analysis_result.get("filename"),
        "extracted_fields": analysis_result.get("extracted_fields", {}),
        "structured_invoice": analysis_result.get("structured_invoice", {}),
        "fraud_risk_score": prediction.get("fraud_probability"),
        "risk_level": prediction.get("risk_level"),
        "blocking_validation_issues": decision.get("hard_issues", []),
        "non_blocking_notes": decision.get("soft_issues", []),
        "duplicate_check": analysis_result.get("duplicate_check", {}),
        "field_meanings": {
            "invoice_amount_zscore": (
                "amount compared with ALL invoices in the dataset "
                "(global mean and std), NOT with this supplier's invoices"
            ),
            "supplier_avg_amount_90d": "this supplier's average amount over the previous 90 days",
            "avg_invoice_amount": "this supplier's overall average amount (supplier master)",
            "supplier_risk_score": (
                "supplier master score from 0 to 1; across the dataset "
                + _supplier_risk_range()
                + ". Judge it against this range only."
            ),
            "HIGH_RISK_SCORE rule": (
                "fires when the fraud model's risk_level is HIGH "
                "(fraud_risk_score), not because of supplier_risk_score"
            )
        }
    }

    rag_context = retrieved_context.get("context", "")

    history_text = (
        json.dumps(supplier_history, indent=2, default=str)
        if supplier_history
        else "No earlier invoices from this supplier were found."
    )

    prompt = f"""
The invoice below has already been assessed by a rule-based
decision engine. The decision is FINAL and must not be changed.

FINAL DECISION: {decision.get("disposition")}
RULES THAT FIRED: {", ".join(decision.get("triggered_rules", []))}
RULE ENGINE REASON: {decision.get("reason")}

INVOICE ANALYSIS EVIDENCE:
{json.dumps(evidence, indent=2, default=str)}

THIS SUPPLIER'S RECENT INVOICES (from the dataset):
{history_text}

RELATED INVOICES RETRIEVED FROM THE KNOWLEDGE BASE:
{rag_context if rag_context else "No related invoices were retrieved."}

YOUR TASK:

1. Explain in plain language why this decision was reached,
   based on the rules that fired and the evidence.
2. Do NOT suggest a different decision.
3. A risk score is an indicator, not proof of fraud.
4. Non-blocking notes may be mentioned, but must NOT be called red
   flags and must NOT be described as causing or supporting the
   decision. Only the RULES THAT FIRED caused the decision.
5. Compare with the supplier's earlier invoices only where the
   data supports it (e.g. amount much higher than usual).
6. Do not invent facts, policies or history.
7. If the fraud risk score is missing, say the invoice was not
   scored and why; do not guess a score.
8. For a NEW invoice, mention briefly that some model inputs were
   estimated (see feature_assumptions).
9. Use field_meanings when describing a feature; do not reinterpret it.
10. supplier_invoice_count_30d and supplier_avg_amount_90d count the
    30/90 days BEFORE THE INVOICE DATE. If feature_assumptions say the
    windows use the supplier's most recent activity instead, describe
    them as "in the 30/90 days before the supplier's last recorded
    invoice" and never as "the last 30 days" or "recent".
11. recommended_actions must contain at least one concrete next step
    that fits the decision (e.g. for REJECT: do not pay, notify the
    procurement/compliance team).

Return ONLY valid JSON in this exact structure:

{{
    "explanation": "2-4 sentence plain-language explanation",
    "key_findings": [],
    "supplier_history_findings": [],
    "recommended_actions": []
}}

Use empty arrays when there is nothing to report.
"""

    system_prompt = (
        "You explain invoice decisions that have already been made. "
        "Never change the decision. Base every statement on the "
        "supplied evidence. Return only valid JSON."
    )

    response_text = _call_groq(
        system_prompt,
        prompt,
        max_tokens=1500
    )

    result = _extract_json_object(response_text)

    return {
        "explanation": str(result.get("explanation", "")).strip(),
        "key_findings": _as_list(result.get("key_findings")),
        "supplier_history_findings": _as_list(
            result.get("supplier_history_findings")
        ),
        "recommended_actions": _as_list(result.get("recommended_actions"))
    }


# --------------------------------------------------
# PIPELINE B: CONVERSATIONAL RAG
# --------------------------------------------------

def answer_invoice_query(
    user_query,
    retrieved_context
):
    """
    Answer a natural-language invoice question using
    query-specific ChromaDB retrieval context.
    """

    if not user_query or not user_query.strip():
        raise ValueError(
            "Please provide a question."
        )

    rag_context = retrieved_context.get(
        "context",
        ""
    )

    prompt = f"""
Answer the user's question using the retrieved invoice information.

USER QUESTION:
{user_query}

RETRIEVED INVOICE CONTEXT:
{rag_context if rag_context else "No relevant invoice records were retrieved."}

INSTRUCTIONS:

1. Answer the question directly and clearly.
2. Use only facts supported by the retrieved context.
3. Do not invent invoice details, supplier information,
   statistics or business rules.
4. If retrieved context is insufficient, say so clearly.
5. Mention invoice IDs when relevant.
6. Distinguish retrieved facts from observations.
7. Do not assume that retrieved records represent
   the complete invoice dataset.
8. Amounts are in the invoices' currency ({_dataset_currency_label()});
   never use "$" or another currency symbol. Do not write JSON blocks.
"""

    system_prompt = (
        "You are an invoice intelligence assistant. "
        "Answer questions accurately using the supplied "
        "retrieved invoice context. Acknowledge missing evidence."
    )

    return _call_groq(
        system_prompt,
        prompt,
        max_tokens=1500
    )


# --------------------------------------------------
# AI QUERY PLANNER
# --------------------------------------------------

ALLOWED_ROUTES = {
    "structured",
    "rag",
    "hybrid"
}

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


def _parse_query_plan(response_text):
    """Parse and validate the query plan returned by the LLM."""

    plan = _extract_json_object(
        response_text
    )

    route = plan.get("route")

    if route not in ALLOWED_ROUTES:
        raise ValueError(
            f"Unsupported query route: {route}"
        )

    if route in {"structured", "hybrid"}:

        operation = plan.get("operation")

        if operation not in ALLOWED_QUERY_OPERATIONS:
            raise ValueError(
                f"Unsupported query operation: {operation}"
            )

        filters = plan.get(
            "filters",
            []
        )

        if not isinstance(filters, list):
            raise ValueError(
                "Query filters must be a list."
            )

        plan["filters"] = filters

        # "What is the average invoice amount?" sometimes comes back
        # without a field; the invoice amount is the natural default.
        if operation in {"sum", "average", "minimum", "maximum"} and not plan.get("field"):
            plan["field"] = "invoice_amount"

    if route in {"rag", "hybrid"}:

        rag_query = plan.get(
            "rag_query"
        )

        if rag_query is not None and not isinstance(
            rag_query,
            str
        ):
            raise ValueError(
                "rag_query must be a string."
            )

    return plan


def plan_invoice_query(user_query):
    """
    Convert a natural-language question into a controlled
    JSON query plan.

    The AI plans the operation.
    Python validates and executes it.
    """

    if not user_query or not user_query.strip():
        raise ValueError(
            "Please enter a question."
        )

    system_prompt = """
You are an AI query planner for an invoice intelligence system.

Convert the user's natural-language question into a JSON query plan.

The system has:

1. A structured invoice dataset containing invoice,
   supplier, department and behavioural information.

2. A ChromaDB invoice knowledge base for semantic retrieval.

Choose exactly one route:

STRUCTURED:
For complete-dataset counts, sums, averages, minimums,
maximums, grouping, filtering and listing.

RAG:
For contextual questions that require semantic retrieval
of relevant invoice records.

HYBRID:
When both complete-dataset calculations and contextual
invoice evidence are required.

ALLOWED OPERATIONS:

count, sum, average, minimum, maximum, unique_count,
group_count, group_aggregate, list

ALLOWED FILTER OPERATORS:

eq, ne, gt, gte, lt, lte, contains, in, not_in,
is_null, not_null

AVAILABLE DATASET COLUMNS:

invoice_id
supplier_id
department_id
invoice_date
invoice_amount
currency
payment_terms
invoice_type
submission_hour
supplier_country
supplier_age_days
supplier_risk_score
blacklisted_flag
avg_invoice_amount
region
annual_budget
supplier_invoice_count_30d
supplier_avg_amount_90d
invoice_amount_zscore
duplicate_invoice_flag
split_invoice_flag
late_night_submission_flag
image_path

IMPORTANT RULES:

1. Use only the available dataset column names.

2. Questions asking how many, how much, average, highest,
   lowest, percentage, totals or counts generally require
   structured analysis.

3. For duplicate invoice counts, use:
   operation=count
   filter: duplicate_invoice_flag eq 1.

4. For unusually high supplier risk scores, use a list
   operation sorted by supplier_risk_score descending.
   Include invoice_id, supplier_id and supplier_risk_score.

5. For invoice counts by department, use group_count
   with group_by ["department_id"].

6. For grouped numerical summaries, use group_aggregate.
   Always provide all four required properties:
   - group_by: the requested grouping column(s)
   - field: the exact numerical dataset column being summarized
   - aggregate: sum, mean, min, max or count
   - sort_order: desc for highest and asc for lowest.

   Never leave field or aggregate missing.

   For an average supplier risk score by supplier, use:
   operation = group_aggregate
   group_by = ["supplier_id"]
   field = "supplier_risk_score"
   aggregate = "mean".

7. For contextual explanations about a particular invoice,
   use RAG or hybrid retrieval.

8. For questions requiring both numerical statistics and
   contextual examples, use hybrid.

9. Do not invent filter values or dataset columns.

10. Never generate Python, SQL or executable code.

11. Missing mandatory fields require a dedicated audit of
    all eight configured mandatory fields. Do not approximate
    that audit with a single is_null filter.

12. Return ONLY valid JSON.

EXAMPLE:

Question:
How many duplicate invoices exist?

JSON:
{
    "route": "structured",
    "operation": "count",
    "filters": [
        {
            "field": "duplicate_invoice_flag",
            "operator": "eq",
            "value": 1
        }
    ]
}

EXAMPLE:

Question:
Show the 10 suppliers with the highest average supplier risk score.

JSON:
{
    "route": "structured",
    "operation": "group_aggregate",
    "filters": [],
    "group_by": ["supplier_id"],
    "field": "supplier_risk_score",
    "aggregate": "mean",
    "sort_order": "desc",
    "limit": 10
}

EXAMPLE:

Question:
Show the 10 invoices with the highest supplier risk scores.

JSON:
{
    "route": "structured",
    "operation": "list",
    "filters": [],
    "columns": [
        "invoice_id",
        "supplier_id",
        "supplier_risk_score"
    ],
    "sort_by": "supplier_risk_score",
    "sort_order": "desc",
    "limit": 10
}

EXAMPLE:

Question:
What are the minimum and maximum invoice amounts in the complete dataset?

JSON:
{
    "route": "structured",
    "operation": "multi_aggregate",
    "aggregations": [
        {
            "operation": "minimum",
            "field": "invoice_amount"
        },
        {
            "operation": "maximum",
            "field": "invoice_amount"
        }
    ],
    "filters": []
}

For a RAG question, return:

{
    "route": "rag",
    "rag_query": "Relevant semantic retrieval query"
}

For a hybrid question, include the structured operation
and a relevant rag_query.
"""

    response_text = _call_groq(
        system_prompt,
        user_query.strip(),
        max_tokens=1200
    )

    try:
        return _parse_query_plan(response_text)

    except ValueError as error:
        # The LLM occasionally returns malformed JSON or an invalid plan.
        # Ask once more, telling it what went wrong.
        retry_prompt = (
            f"{user_query.strip()}\n\n"
            f"Your previous plan was rejected ({error}). "
            "Return ONLY one valid JSON object following the schema."
        )

        response_text = _call_groq(
            system_prompt,
            retry_prompt,
            max_tokens=1200
        )

        return _parse_query_plan(response_text)


# --------------------------------------------------
# HYBRID ANSWER GENERATION
# --------------------------------------------------

def answer_hybrid_invoice_query(
    user_query,
    structured_result=None,
    retrieved_context=None
):
    """
    Generate a grounded response using:
    - Complete structured dataset calculations
    - Optional ChromaDB semantic retrieval
    """

    structured_evidence = (
        json.dumps(
            structured_result,
            indent=2,
            default=str
        )
        if structured_result is not None
        else "No structured dataset analysis was performed."
    )

    rag_evidence = (
        json.dumps(
            retrieved_context,
            indent=2,
            default=str
        )
        if retrieved_context is not None
        else "No ChromaDB context was retrieved."
    )

    prompt = f"""
Answer the user's invoice question using the supplied evidence.

USER QUESTION:
{user_query}

FULL-DATASET STRUCTURED ANALYSIS:
{structured_evidence}

CHROMADB RETRIEVED CONTEXT:
{rag_evidence}

INSTRUCTIONS:

1. Answer the question directly and clearly.

2. Use structured results for complete-dataset numerical claims.

3. Use ChromaDB records only as supporting contextual evidence.

4. Never describe five retrieved records as representing
   all 300,000 invoices.

5. Clearly distinguish dataset-wide findings from
   retrieved invoice examples. For a "list" result, the listed
   records are only the top rows; never claim a value shown in them
   applies to all matching invoices. Use
   sort_field_summary_over_all_matching for dataset-wide figures.

6. Do not invent statistics, invoice records or business rules.

7. If evidence is insufficient, acknowledge the limitation.

8. A high risk score is not proof of fraud.

9. Keep the answer professional and concise.

10. Present results in plain sentences or a markdown table. Do NOT
    write JSON blocks or invent field names; the app already shows
    the real query plan.

11. Amounts are in the invoices' currency ({_dataset_currency_label()});
    never use "$" or another currency symbol.
"""

    system_prompt = (
        "You are a careful invoice intelligence assistant. "
        "Ground every statement in the supplied evidence. "
        "Distinguish dataset-wide calculations from retrieved examples."
    )

    return _call_groq(
        system_prompt,
        prompt,
        max_tokens=1800
    )


# --------------------------------------------------
# LEGACY FRAUD EXPLANATION
# --------------------------------------------------

def generate_fraud_explanation(analysis_result):
    """
    Explain the ML prediction without independently making
    the final invoice disposition.
    """

    if analysis_result.get("status") != "completed":
        raise ValueError(
            "A completed invoice analysis is required."
        )

    evidence = {
        "filename": analysis_result.get("filename"),
        "extracted_fields": analysis_result.get(
            "extracted_fields",
            {}
        ),
        "prediction": analysis_result.get(
            "prediction",
            {}
        ),
        "validation": analysis_result.get(
            "validation",
            {}
        ),
        "structured_invoice": analysis_result.get(
            "structured_invoice",
            {}
        )
    }

    prompt = f"""
Explain the following invoice analysis evidence clearly.

{json.dumps(evidence, indent=2, default=str)}

Explain the model risk score, validation findings,
and relevant observations.

Do not invent missing information.
A mismatch or high risk score does not independently prove fraud.
Recommend appropriate verification steps.
"""

    return _call_groq(
        "You are a factual invoice fraud analysis assistant.",
        prompt,
        max_tokens=1200
    )