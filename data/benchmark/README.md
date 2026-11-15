# BhashaGuard benchmark data

The JSON files in this directory are empty manual-review templates. The two supplied candidate datasets are preserved unchanged under `data/json/` and read through `core/source_data.py`. The companion dataset marks its factual records and bilingual pairs as needing source verification. The app therefore reports imported-data metrics as provisional, and verified mode currently has zero eligible cases.

## Claim dataset format

Add manually reviewed records to `dev.json` or `final.json` under `cases`. Keep the final split held out during prompt and code tuning. Do not put paraphrases of the same underlying fact in both splits. The planning target is in `benchmark_plan.json`.

Each case uses this schema:

```json
{
  "test_id": "ADIP-DEV-001",
  "scheme": "ADIP",
  "language": "English",
  "input_answer": "The AI answer text submitted to BhashaGuard.",
  "expected_claim": "One individually verifiable factual claim from the answer.",
  "expected_verdict": "NEEDS REVIEW",
  "source_pdf": "example.pdf",
  "page": 1,
  "original_page_number": "printed page 1",
  "evidence": "Exact passage transcribed from the cited PDF page.",
  "expected_explanation": "Why this passage supports, contradicts, or does not resolve the claim.",
  "difficulty": "easy",
  "category": "eligibility"
}
```

The machine-readable schema is in `claim_case.schema.json`.

The example is schema illustration only, not a benchmark case. `page` is the 1-based PDF page index used by retrieval; `original_page_number` is the page label printed in the source (or a note such as `unpaginated PDF page 1`). For a genuinely unsupported claim, use an empty `source_pdf`, `page: null`, `original_page_number: ""`, and `evidence: ""`; explain why the supplied corpus cannot verify it. For a claim with conflicting source passages, keep the primary citation in the base fields and place each further exact citation in optional `additional_sources` entries with its own PDF, page index, original page label, and passage. Use only `ACCEPTED`, `BLOCKED`, or `NEEDS REVIEW`.

## Bilingual consistency format

Add manually labeled cases to `dev_bilingual.json` or `final_bilingual.json`:

```json
{
  "test_id": "BI-DEV-001",
  "english_answer": "An English statement.",
  "hindi_answer": "हिंदी कथन।",
  "expected_discrepancy": true,
  "category": "number",
  "expected_notes": "Record the exact discrepancy and why it matters.",
  "source_pdf": "example.pdf",
  "page": 1,
  "evidence": "Exact reviewed source passage."
}
```

This measures translation consistency only. It does not establish factual correctness; the claim benchmark independently verifies facts.
The machine-readable bilingual schema is `bilingual_case.schema.json`.

## Running

Run the benchmark from the project root with `python -m core.benchmark_cli --split dev --kind claims`. Use `--split final` only for a held-out measurement; do not tune from its results. Runs are written under `data/benchmark/results/` without API keys. The runner uses the same extraction, retrieval, and verification functions as the app. A Groq key is required; Tavily is optional. Benchmark runs never index PDFs automatically: the saved local index is used if current, otherwise local retrieval returns no evidence. Refresh the PDF index explicitly in the PDF Library tab only when you choose to process the documents.

Every row must be checked against the original PDF. Verify the PDF filename, printed page number (not just the PDF viewer page counter), exact passage, language, conditions, dates, units, exceptions, and expected verdict. Do not mark a case verified based only on an LLM suggestion. Imported records remain provisional until that review is recorded in the reviewed benchmark data; the source JSON files are never rewritten by the application.
