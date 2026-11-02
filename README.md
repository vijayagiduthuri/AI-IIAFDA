# IIAFDA — Intelligent Invoice Analysis and Fraud Detection App

A Streamlit app that reads an invoice (image or PDF), checks it, scores its fraud risk and decides **APPROVE / MANUAL REVIEW / REJECT**. An AI model explains each decision and answers questions about the invoice dataset.

## How it works

```
Invoice image / PDF
   │
   ├─ OCR (Tesseract) ──► invoice fields (ID, supplier, date, amounts…)
   ├─ Validation ───────► checks against the supplier master and the invoice arithmetic
   ├─ Fraud model ──────► Random Forest risk score 0–100 (LOW / MEDIUM / HIGH)
   ├─ Duplicate check ──► dataset + previously analysed invoices
   ├─ Decision rules ───► APPROVE / MANUAL REVIEW / REJECT   (fixed rules, not the AI)
   └─ AI (Groq) + RAG ──► plain-language explanation using similar invoices (ChromaDB)
```

- **Invoices already in the dataset** use their stored features.
- **New invoices** have their model features rebuilt from the supplier master and the supplier's past invoices. Every estimate is listed on screen.
- The **AI never makes the decision**. It only explains the decision made by the rules.

### App pages

| Page | What it does |
|---|---|
| Invoice Analysis | Upload an invoice and see the decision, risk score, validation results, AI explanation and similar invoices |
| Ask AI | Ask questions about the 300,000-invoice dataset, e.g. "What is the average invoice amount?" |
| Invoice History | Saved analyses: view, download as CSV, clear before a demo |

## Setup

1. **Python 3.13** and a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Tesseract OCR** (Windows installer: https://github.com/UB-Mannheim/tesseract/wiki). Add its folder (e.g. `C:\Program Files\Tesseract-OCR`) to PATH. Check with `tesseract --version`.
3. **Dataset:** the *Procurement Invoice Fraud Dataset v1* folder (parquet files + `images/`).
4. **Settings:** copy `.env.example` to `.env` and fill in:
   ```
   GROQ_API_KEY=...            # free key from https://console.groq.com
   DATASET_DIR=D:/path/to/Procument Invoice Fraud Dataset v1
   ```
5. **Knowledge base:** `chroma_db/` holds the indexed invoices. If it is missing, rebuild it (takes a long time, resumes if interrupted):
   ```
   python -c "from backend.rag_engine import index_all_invoices; index_all_invoices()"
   ```

## Run

```
streamlit run app.py
```

Before a demo: open **Invoice History → Clear history**. Otherwise re-uploading an invoice that was already approved is rejected as a duplicate payment, which is intended behaviour.

## Demo invoices

`test_invoices/` contains five new invoices (not in the dataset). Recreate them with `python scripts/make_test_invoices.py`.

| File | Expected decision |
|---|---|
| 1_normal_known_supplier.png | APPROVE |
| 2_unknown_supplier.png | MANUAL REVIEW (supplier not in master) |
| 3_blacklisted_supplier.png | REJECT (blacklisted supplier) |
| 4_wrong_country.png | MANUAL REVIEW (country mismatch) |
| 5_edited_total.png | MANUAL REVIEW (totals do not add up) |

## Check scripts

| Command | Checks |
|---|---|
| `python scripts/run_test_invoices.py` | The 5 demo invoices end-to-end (uses a temporary history, so saved analyses are not touched) |
| `python scripts/check_ask_ai.py` | Ask AI answers against values computed directly from the data |
| `python scripts/check_new_invoice_features.py 500` | How closely rebuilt new-invoice features match the dataset |

Quick component tests are in `tests/`, e.g. `python tests/test_history_store.py`.

## Project layout

```
app.py                 Streamlit app
backend/
  config.py            paths (dataset path read from .env)
  invoice_processor.py OCR and field extraction
  validator.py         validation against the supplier master
  new_invoice_features.py  feature rebuild for new invoices
  fraud_detector.py    loads the model, predicts risk
  decision_engine.py   APPROVE / MANUAL REVIEW / REJECT rules
  invoice_analyzer.py  runs the full pipeline
  rag_engine.py        ChromaDB retrieval + dataset queries
  ai_explainer.py      Groq prompts (explanations, Ask AI)
  history_store.py     SQLite history of analyses
frontend/components.py UI helpers
models/                trained fraud model (.pkl)
chroma_db/             vector knowledge base
data/                  history database and check-script outputs
scripts/               demo and check scripts
tests/                 quick component tests
test_invoices/         demo invoices
```

## Notes and limits

- **Groq free tier:** each model allows about 200,000 tokens per day. When one runs out, the app automatically switches to `openai/gpt-oss-120b`, then `qwen/qwen3.8-27b`. If all three are exhausted, the AI text is paused, but decisions and risk scores keep working.
- **Department and invoice type** are not printed on invoices. For new invoices the supplier's most frequent values are used, and the screen says so.
- Amounts are in **ZAR** (the dataset's only currency).
- A risk score is an indicator, not proof of fraud. Final action requires authorised human review.
