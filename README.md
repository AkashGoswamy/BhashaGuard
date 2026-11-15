# BhashaGuard

BhashaGuard is a Streamlit application for auditing factual claims in AI-generated answers against local PDFs and Tavily web search, and for comparing English and Hindi answers. Verdicts are **ACCEPTED**, **BLOCKED**, and **NEEDS REVIEW**; evidence strength is reported separately.

## Requirements

- Windows 11 and Python 3.11
- A Groq API key for claim extraction, interpretation, answer comparison, and PDF sample questions
- A Tavily API key for live web search (local PDF retrieval works without it)

Keys are entered in the sidebar, held only in Streamlit session state, and never saved in project files.

## Run on Windows PowerShell

From the project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that PowerShell window, then activate the environment.

## Local PDFs

Place PDFs in `data/documents/`; subfolders are scanned too. Use **PDF library → Refresh PDF index** after adding or removing files. Text and page references are cached under `data/index/` and refreshed when the source PDF list or file metadata changes. Scanned PDFs without extractable text are reported as requiring OCR; this app does not perform OCR.

## Benchmarks

Place labelled `.json` or `.csv` files in `data/benchmark/`. JSON may be a list of cases or `{ "cases": [...] }`. Each case needs a `label` or `expected_verdict` and a `predicted_verdict` (one of the three exact verdict labels). Optional fields: `language`, `expected_discrepancy`, and `predicted_discrepancy` (boolean-like values). This dashboard calculates metrics from supplied predictions; it does not run the auditor over the dataset automatically.

Example:

```json
{
  "cases": [
    {"claim":"Example claim", "expected_verdict":"ACCEPTED", "predicted_verdict":"ACCEPTED", "language":"English"}
  ]
}
```

## Deployment

For Streamlit Community Cloud, deploy this repository with `app.py` as the entry point and `requirements.txt` for dependencies. Hugging Face Spaces can run the same app using its Streamlit SDK template and the same requirements. Enter API keys in the app sidebar after launch. Never commit keys. Hosted deployments cannot read PDFs on your personal computer: include documents in the repository or provide them through an appropriate hosted storage workflow. The `.gitignore` excludes local PDFs and benchmark files by default.

## Limits

- Groq and Tavily calls require network access and valid keys. Live webpage text extraction may fail for JavaScript-only or access-restricted pages; such sources are marked as search-result-only and cannot on their own produce a conclusive verdict.
- PDF retrieval uses lightweight token overlap, not semantic embeddings. The index detects changed PDFs from size and modification time.
- LLM claim interpretation can still make mistakes. Treat NEEDS REVIEW as unresolved, and inspect the source passages before relying on a result.
- No metrics are displayed until labelled benchmark cases with both reference and predicted verdicts are supplied.
