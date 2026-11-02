import streamlit as st
import pandas as pd

from backend.invoice_analyzer import analyze_invoice

from backend.rag_engine import (
    retrieve_query_context,
    find_invoices_with_missing_mandatory_fields,
    execute_structured_query
)

from backend.ai_explainer import (
    answer_invoice_query,
    answer_hybrid_invoice_query,
    plan_invoice_query,
    AIRateLimitError
)

from backend.history_store import (
    save_invoice_analysis,
    get_invoice_history,
    get_invoice_analysis,
    clear_invoice_history
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI-IIAFDA",
    page_icon="📄",
    layout="wide"
)


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "analysis_result" not in st.session_state:
    st.session_state.analysis_result = None

if "ask_ai_response" not in st.session_state:
    st.session_state.ask_ai_response = None


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("AI-IIAFDA")

st.sidebar.caption(
    "Invoice Intelligence & Fraud Detection"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Invoice Analysis",
        "Ask AI",
        "Invoice History"
    ]
)


# --------------------------------------------------
# APPLICATION HEADER
# --------------------------------------------------

st.title(
    "AI-Powered Invoice Intelligence & Fraud Detection"
)

st.caption(
    "Intelligent invoice analysis and fraud detection system"
)

st.divider()


# ==================================================
# PAGE 1: INVOICE ANALYSIS
# ==================================================

if page == "Invoice Analysis":

    st.header("Invoice Analysis")

    st.write(
        "Upload an invoice to extract its details, "
        "validate the information, calculate fraud risk, "
        "and generate an AI-powered final disposition."
    )

    uploaded_file = st.file_uploader(
        "Upload an invoice",
        type=[
            "png",
            "jpg",
            "jpeg",
            "tif",
            "tiff",
            "bmp",
            "pdf"
        ],
        key="invoice_uploader"
    )

    if uploaded_file is not None:

        st.success(
            f"File selected: {uploaded_file.name}"
        )

        if st.button(
            "Analyze Invoice",
            type="primary"
        ):

            st.session_state.analysis_result = None

            try:

                with st.spinner(
                    "Processing invoice, calculating risk, "
                    "retrieving context, and generating AI decision..."
                ):

                    result = analyze_invoice(
                        uploaded_file=uploaded_file
                    )

                if result.get("status") != "completed":

                    st.error(
                        result.get(
                            "message",
                            "Invoice analysis could not be completed."
                        )
                    )

                    st.json(result)

                else:

                    st.session_state.analysis_result = result

                    save_invoice_analysis(
                        result=result,
                        filename=uploaded_file.name
                    )

                    st.success(
                        "Invoice analysis saved to database."
                    )

            except Exception as error:

                st.error(
                    "An error occurred during invoice analysis."
                )

                st.exception(error)

    # ----------------------------------------------
    # DISPLAY LATEST ANALYSIS
    # ----------------------------------------------

    result = st.session_state.analysis_result

    if result is not None:

        disposition = result.get(
            "disposition",
            {}
        )

        validation = result.get(
            "validation",
            {}
        )

        prediction = result.get(
            "prediction",
            {}
        )

        # ------------------------------------------
        # FINAL DISPOSITION
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Final Invoice Disposition"
        )

        final_decision = disposition.get(
            "disposition",
            "MANUAL REVIEW"
        )

        decision_reason = disposition.get(
            "reason",
            "No decision explanation available."
        )

        if final_decision == "APPROVE":

            st.success(
                f"### {final_decision}\n\n{decision_reason}"
            )

        elif final_decision == "REJECT":

            st.error(
                f"### {final_decision}\n\n{decision_reason}"
            )

        else:

            st.warning(
                f"### {final_decision}\n\n{decision_reason}"
            )

        if result.get("invoice_source") == "new":
            st.info(
                "**New invoice** — this invoice is not in the dataset. "
                "It was checked against the supplier master, and the fraud "
                "model inputs were rebuilt from the supplier's past invoices."
            )

        triggered_rules = disposition.get("triggered_rules", [])

        if triggered_rules:
            st.caption(
                "Decision rules triggered: "
                + ", ".join(triggered_rules)
            )

        st.caption(
            "The decision is made by fixed business rules using "
            "validation results, the ML risk score and duplicate checks. "
            "The AI only explains it. Final action requires authorized "
            "human review."
        )

        # ------------------------------------------
        # DUPLICATE CHECK
        # ------------------------------------------

        duplicate_check = result.get("duplicate_check", {}) or {}

        if duplicate_check.get("same_invoice_seen"):
            st.warning(
                "This invoice ID has already been analysed "
                f"{duplicate_check.get('previous_count')} time(s) before."
                + (
                    " It was APPROVED earlier."
                    if duplicate_check.get("previously_approved")
                    else ""
                )
            )

        for similar in (
            duplicate_check.get("similar_invoices", [])
            + duplicate_check.get("dataset_similar_invoices", [])
        ):
            st.warning(
                "Possible duplicate: invoice "
                f"{similar.get('invoice_id')} from the same supplier has "
                f"amount {similar.get('invoice_amount')} on "
                f"{similar.get('invoice_date')}."
            )

        # ------------------------------------------
        # FRAUD RISK ASSESSMENT
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Fraud Risk Assessment"
        )

        col1, col2, col3 = st.columns(3)

        prediction = prediction or {}

        risk_score = result.get(
            "risk_score",
            prediction.get("fraud_probability")
        )

        risk_level = result.get(
            "risk_level",
            prediction.get("risk_level", "UNKNOWN")
        )

        col1.metric(
            "Fraud Risk Score (0-100)",
            f"{float(risk_score):.2f}" if risk_score is not None else "N/A"
        )

        col2.metric(
            "Risk Level",
            risk_level
        )

        col3.metric(
            "Blocking Validation Issues",
            validation.get("mismatch_count", 0)
        )

        st.caption(
            "Risk assessment model: "
            f"{prediction.get('model_name', 'Not available')}"
        )

        if risk_score is not None:
            st.progress(
                max(
                    0.0,
                    min(float(risk_score) / 100, 1.0)
                )
            )

        feature_assumptions = result.get("feature_assumptions", [])

        if feature_assumptions:
            with st.expander(
                "How the model inputs were built for this new invoice"
            ):
                for note in feature_assumptions:
                    st.markdown(f"- {note}")

        # ------------------------------------------
        # AI-GENERATED FINDINGS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "AI Decision Explanation"
        )

        if disposition.get("explanation"):
            st.write(disposition["explanation"])

        key_findings = disposition.get(
            "key_findings",
            []
        )

        if key_findings:

            for finding in key_findings:
                st.markdown(f"- {finding}")

        else:

            st.info(
                "No additional findings were returned."
            )

        # ------------------------------------------
        # VALIDATION ISSUES
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Validation Issues"
        )

        validation_issues = disposition.get(
            "validation_issues",
            []
        )

        def describe_issue(issue):
            text = (
                f"**{issue.get('field')}** — OCR: {issue.get('ocr_value')}, "
                f"expected: {issue.get('dataset_value')}"
            )
            if issue.get("difference") is not None:
                text += f" (difference {issue.get('difference')})"
            return text

        if validation_issues:

            for issue in validation_issues:
                st.error(describe_issue(issue))

        else:

            st.success(
                "No blocking validation issues were detected."
            )

        for issue in disposition.get("soft_issues", []):
            st.info(
                "Note (not blocking): " + describe_issue(issue)
            )

        if validation.get("not_read_fields"):
            st.caption(
                "Fields the OCR could not read (checked against the "
                "dataset instead): "
                + ", ".join(validation["not_read_fields"])
            )

        # ------------------------------------------
        # RAG FINDINGS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Supplier History and Similar Invoices"
        )

        rag_context = result.get(
            "rag_context",
            {}
        )

        history_findings = disposition.get(
            "supplier_history_findings",
            []
        )

        st.markdown("**Comparison with this supplier's past invoices**")

        if history_findings:

            for finding in history_findings:
                st.markdown(f"- {finding}")

        else:

            st.info(
                "No specific supplier-history findings were identified."
            )

        supplier_history = result.get("supplier_history", [])

        if supplier_history:
            with st.expander(
                "View this supplier's recent invoices"
            ):
                st.dataframe(
                    pd.DataFrame(supplier_history),
                    width="stretch",
                    hide_index=True
                )

        related_invoices = rag_context.get("related_invoices", [])

        st.metric(
            "Similar invoices retrieved from the knowledge base",
            rag_context.get("retrieved_count", 0)
        )

        if related_invoices:
            with st.expander("View similar invoices"):
                for item in related_invoices:
                    st.markdown(f"**{item.get('invoice_id')}**")
                    st.text(str(item.get("document", ""))[:600])

        # ------------------------------------------
        # RECOMMENDED ACTIONS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Recommended Actions"
        )

        recommended_actions = disposition.get(
            "recommended_actions",
            []
        )

        if recommended_actions:

            for action in recommended_actions:
                st.markdown(f"- {action}")

        else:

            st.info(
                "No additional actions were recommended."
            )

        # ------------------------------------------
        # EXTRACTED INVOICE DETAILS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Extracted Invoice Details"
        )

        fields_df = pd.DataFrame(
            [
                {
                    "Field": str(field),
                    "Extracted Value": (
                        str(value)
                        if value is not None
                        else "Not available"
                    )
                }
                for field, value
                in result.get(
                    "extracted_fields",
                    {}
                ).items()
            ]
        )

        st.dataframe(
            fields_df,
            width="stretch",
            hide_index=True
        )

        # ------------------------------------------
        # DATASET VALIDATION COMPARISON
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Invoice Validation Comparison"
        )

        st.caption(
            "Hard checks block approval when they fail. Soft checks are "
            "shown for information only."
        )

        comparison_rows = []

        for field, comparison in validation.get(
            "comparisons",
            {}
        ).items():

            comparison_rows.append(
                {
                    "Field": str(field),
                    "OCR Value": str(
                        comparison.get("ocr_value")
                    ),
                    "Expected Value": str(
                        comparison.get("dataset_value")
                    ),
                    "Result": str(
                        comparison.get("status", "")
                    ).replace("_", " ").upper(),
                    "Check Type": str(
                        comparison.get("severity", "")
                    ).upper()
                }
            )

        if comparison_rows:

            comparison_df = pd.DataFrame(
                comparison_rows
            )

            st.dataframe(
                comparison_df,
                width="stretch",
                hide_index=True
            )

        else:

            st.info(
                "No field comparisons are available."
            )

        # ------------------------------------------
        # STRUCTURED INVOICE INFORMATION
        # ------------------------------------------

        st.divider()

        st.subheader(
            "Structured Invoice Information"
        )

        st.json(
            result.get(
                "structured_invoice",
                {}
            )
        )

        # ------------------------------------------
        # OCR TEXT
        # ------------------------------------------

        with st.expander(
            "View Extracted OCR Text"
        ):

            st.text(
                result.get(
                    "ocr_text",
                    ""
                )
            )


# ==================================================
# PAGE 2: ASK AI — HYBRID QUERY ENGINE
# ==================================================

elif page == "Ask AI":

    st.header("Ask AI")

    st.write(
        "Ask questions about invoices, suppliers, risk indicators, "
        "fraud patterns, and the complete invoice dataset."
    )

    user_query = st.text_area(
        "Enter your question",
        placeholder=(
            "How many duplicate invoices exist?\n"
            "What is the average invoice amount?\n"
            "Show the 10 invoices with the highest supplier risk scores."
        ),
        height=120,
        key="user_query"
    )

    if st.button(
        "Ask AI",
        type="primary"
    ):

        if not user_query.strip():

            st.warning(
                "Please enter a question."
            )

        else:

            try:

                # ----------------------------------
                # STEP 1: QUERY PLANNING
                # ----------------------------------

                with st.spinner(
                    "Understanding your question..."
                ):

                    plan = plan_invoice_query(
                        user_query.strip()
                    )

                route = plan["route"]

                structured_result = None
                retrieved_context = None

                # ----------------------------------
                # STEP 2: STRUCTURED ANALYSIS
                # ----------------------------------

                if route in {
                    "structured",
                    "hybrid"
                }:

                    with st.spinner(
                        "Analyzing the complete invoice dataset..."
                    ):

                        structured_result = execute_structured_query(
                            plan
                        )

                # ----------------------------------
                # STEP 3: SEMANTIC RETRIEVAL
                # ----------------------------------

                if route in {
                    "rag",
                    "hybrid"
                }:

                    rag_query = (
                        plan.get("rag_query")
                        or user_query.strip()
                    )

                    with st.spinner(
                        "Retrieving relevant invoice context from ChromaDB..."
                    ):

                        retrieved_context = retrieve_query_context(
                            user_query=rag_query,
                            top_k=5
                        )

                # ----------------------------------
                # STEP 4: ANSWER GENERATION
                # ----------------------------------

                with st.spinner(
                    "Generating your answer..."
                ):

                    if route == "rag":

                        response = answer_invoice_query(
                            user_query=user_query.strip(),
                            retrieved_context=retrieved_context
                        )

                    else:

                        response = answer_hybrid_invoice_query(
                            user_query=user_query.strip(),
                            structured_result=structured_result,
                            retrieved_context=retrieved_context
                        )

                # ----------------------------------
                # STEP 5: SAVE RESPONSE
                # ----------------------------------

                st.session_state.ask_ai_response = {
                    "query": user_query.strip(),
                    "response": response,
                    "route": route,
                    "plan": plan,
                    "structured_result": structured_result,
                    "retrieved_count": (
                        retrieved_context.get(
                            "retrieved_count",
                            0
                        )
                        if retrieved_context
                        else 0
                    ),
                    "retrieved_records": (
                        retrieved_context.get(
                            "retrieved_records",
                            []
                        )
                        if retrieved_context
                        else []
                    )
                }

            except AIRateLimitError as error:

                st.warning(str(error))

            except Exception as error:

                st.error(
                    "Unable to process your question."
                )

                st.exception(error)

    # ----------------------------------------------
    # DISPLAY LATEST ANSWER
    # ----------------------------------------------

    answer_data = st.session_state.ask_ai_response

    if answer_data is not None:

        st.divider()

        st.subheader(
            "AI Response"
        )

        st.markdown(
            answer_data["response"]
        )

        st.caption(
            f'Analysis route: {answer_data["route"].upper()}'
        )

        # ------------------------------------------
        # VIEW GENERATED QUERY PLAN
        # ------------------------------------------

        with st.expander(
            "View AI Query Plan"
        ):

            st.json(
                answer_data["plan"]
            )

        # ------------------------------------------
        # FULL-DATASET RESULTS
        # ------------------------------------------

        structured_result = answer_data.get(
            "structured_result"
        )

        if structured_result is not None:

            with st.expander(
                "View Full-Dataset Analysis Results"
            ):

                col1, col2 = st.columns(2)

                col1.metric(
                    "Invoices Checked",
                    f'{structured_result["total_invoices_checked"]:,}'
                )

                col2.metric(
                    "Matching Invoices",
                    f'{structured_result["matching_invoices"]:,}'
                )

                st.json(
                    structured_result["result"]
                )

        # ------------------------------------------
        # CHROMADB RETRIEVED RECORDS
        # ------------------------------------------

        if answer_data["retrieved_count"]:

            st.caption(
                "Retrieved ChromaDB records: "
                f'{answer_data["retrieved_count"]}'
            )

            with st.expander(
                "View Retrieved ChromaDB Context"
            ):

                for record in answer_data["retrieved_records"]:

                    st.markdown(
                        f'**Invoice ID:** {record["invoice_id"]}'
                    )

                    st.caption(
                        f'Distance: {record["distance"]}'
                    )

                    st.text(
                        record["document"]
                    )

                    st.divider()


# ==================================================
# PAGE 3: INVOICE HISTORY
# ==================================================

elif page == "Invoice History":

    st.header("Invoice History")

    history = get_invoice_history()

    if history:

        st.success(
            f"Total saved invoice analyses: {len(history)}"
        )

        history_df = pd.DataFrame(history)

        display_columns = [
            "id",
            "invoice_id",
            "filename",
            "disposition",
            "risk_score",
            "risk_level",
            "mismatch_count",
            "retrieved_count",
            "analyzed_at_utc"
        ]

        available_columns = [
            column
            for column in display_columns
            if column in history_df.columns
        ]

        st.dataframe(
            history_df[available_columns],
            width="stretch",
            hide_index=True
        )

        # ------------------------------------------
        # DOWNLOAD HISTORY
        # ------------------------------------------

        csv_data = history_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="Download Invoice History",
            data=csv_data,
            file_name="invoice_history.csv",
            mime="text/csv"
        )

        # ------------------------------------------
        # CLEAR HISTORY (before a demo)
        # ------------------------------------------

        with st.expander("Clear history"):
            st.write(
                "Deletes all saved analyses. Useful before a demo, "
                "because re-analysing a saved invoice is reported "
                "as a duplicate."
            )
            confirm = st.checkbox("Yes, delete all saved analyses")
            if st.button("Clear history", disabled=not confirm):
                clear_invoice_history()
                st.success("History cleared.")
                st.rerun()

        # ------------------------------------------
        # VIEW COMPLETE ANALYSIS
        # ------------------------------------------

        st.divider()

        st.subheader(
            "View Saved Invoice Analysis"
        )

        record_options = {
            (
                f'ID: {record["id"]} | '
                f'Invoice: {record["invoice_id"]} | '
                f'{record["disposition"]}'
            ): record["id"]
            for record in history
        }

        selected_record = st.selectbox(
            "Select an invoice record",
            options=list(record_options.keys())
        )

        if st.button(
            "View Full Analysis"
        ):

            record_id = record_options[
                selected_record
            ]

            full_analysis = get_invoice_analysis(
                record_id
            )

            if full_analysis:

                st.json(
                    full_analysis
                )

            else:

                st.warning(
                    "The complete analysis could not be retrieved."
                )

    else:

        st.info(
            "No saved invoice analyses found. "
            "Analyze an invoice to create your first history record."
        )