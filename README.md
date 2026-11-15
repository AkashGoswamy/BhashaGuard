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

The supplied source JSON files are kept unchanged in `data/json/`. The Evaluation tab normalizes them in memory and shows 50 factual cases plus 10 bilingual pairs. The companion file marks all 60 as needing source verification, so they are **provisional candidates**, not verified ground truth. The imported set is split deterministically by scheme/source/page/claim with seed 42; the assignments and dataset versions are recorded in each run. No case is counted in verified evaluation until it is separately reviewed. Empty manual templates and schemas remain in `data/benchmark/`.

The Evaluation tab and CLI run the production extraction, retrieval, verification, and comparison functions. Runs are saved under `data/benchmark/results/` with predictions, evidence, timing, reported token usage, API failures, model and dataset versions. Runs are explicit; the CLI defaults to at most five cases to control API usage. Failed and missing predictions count as incorrect in accuracy and remain visible. Retrieval and citation metrics are unavailable for the imported candidates until their gold pages and passages are verified. Do not tune on the held-out split.

For evaluation without starting Streamlit:

```powershell
python -m core.benchmark_cli --split dev --kind claims --dataset imported --mode provisional --limit 5
python -m core.benchmark_cli --split dev --kind bilingual --dataset imported --mode provisional --limit 5
```

The CLI prompts for the Groq key without echoing it and optionally for a Tavily key. For the held-out evaluation, use `--split final` only after development is complete.

Run the offline evaluation tests with:

```powershell
python -m unittest discover -s tests -v
```

## Deployment

For Streamlit Community Cloud, deploy this repository with `app.py` as the entry point and `requirements.txt` for dependencies. Hugging Face Spaces can run the same app using its Streamlit SDK template and the same requirements. Enter API keys in the app sidebar after launch. Never commit keys. Hosted deployments cannot read PDFs on your personal computer: include documents in the repository or provide them through an appropriate hosted storage workflow. Benchmark run outputs are ignored by Git; the empty templates and schemas are intended to be versioned.

## Limits

- Groq and Tavily calls require network access and valid keys. Live webpage text extraction may fail for JavaScript-only or access-restricted pages; such sources are marked as search-result-only and cannot on their own produce a conclusive verdict.
- Local PDF retrieval uses a small BM25-style lexical ranker with exact phrase, number, and language boosts. It does not download or run a separate embedding model. The index detects changed PDFs from size and modification time.
- LLM claim interpretation can still make mistakes. Treat NEEDS REVIEW as unresolved, and inspect the source passages before relying on a result.
- Provisional candidate metrics are labeled as provisional. This repository currently has zero manually verified benchmark records, so verified accuracy and citation/retrieval scores are unavailable.
