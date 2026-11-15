import streamlit as st
import hashlib
import json
from pathlib import Path

from core.cross_language import compare_answers
from core.evaluation import load_dataset, latest_run, summarize_run
from core.source_data import candidate_overview, load_candidate_dataset
from core.benchmark import run_claim_benchmark
from core.bilingual_evaluation import run_bilingual_benchmark, calculate_bilingual_metrics
from core.extraction import extract_claims
from core.llm import clear_model_cache, validate_groq_key
from core.retrieval import index_pdfs, index_status, retrieve_local, generate_questions
from core.verification import verify_claim
from core.web_search import search_web, validate_tavily_key

st.set_page_config(page_title="BhashaGuard", page_icon="🛡️", layout="wide")


def apply_ui_style():
    st.markdown("""
    <style>
      .stApp { background: #f6f5ef; color: #27332f; }
      .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
      h1, h2, h3 { color: #263b35; letter-spacing: -0.025em; }
      h1 { font-size: 2.35rem !important; }
      [data-testid="stSidebar"] { background: #edf0e9; border-right: 1px solid #dce3db; }
      [data-testid="stSidebar"] h1 { font-size: 1.5rem !important; }
      .stButton button { border-radius: 10px; min-height: 2.65rem; font-weight: 600; }
      .stTextInput input, .stTextArea textarea { border-radius: 10px; border-color: #d5ddd5; background: #fffefa; }
      [data-testid="stMetric"] { background: #fffefa; border: 1px solid #e1e6df; border-radius: 12px; padding: 0.8rem 1rem; }
      .bg-hero { background: linear-gradient(120deg, #e9efe8, #f1eee4); border: 1px solid #dfe6dc; border-radius: 18px; padding: 1.25rem 1.5rem; margin: 0 0 1.2rem; }
      .bg-hero h1 { margin: 0 0 .3rem; }
      .bg-hero p { margin: 0; color: #52635c; font-size: 1.04rem; }
      .bg-verdict { display: inline-block; padding: .4rem .75rem; border-radius: 999px; font-weight: 700; letter-spacing: .02em; margin-bottom: .6rem; }
      .bg-accepted { color: #205c43; background: #e3f1e7; }
      .bg-blocked { color: #8e342b; background: #f8e7e3; }
      .bg-review { color: #795b18; background: #f7efd9; }
      .bg-panel-title { color: #52635c; font-size: .82rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
      @media (max-width: 700px) { .block-container { padding: 1rem 1rem 2rem; } .bg-hero { padding: 1rem; } }
    </style>
    """, unsafe_allow_html=True)


def set_sample_question(question):
    st.session_state["single_answer"] = question


def keys_panel():
    with st.sidebar:
        st.markdown("## 🛡️ BhashaGuard")
        st.caption("Evidence-grounded AI reliability auditor")
        st.divider()
        st.subheader("API configuration")
        groq = st.text_input("Groq API key", type="password", key="groq_key", help="Used to extract and assess claims.")
        tavily = st.text_input("Tavily API key", type="password", key="tavily_key", help="Used to search live web sources.")
        if st.button("Validate keys", width="stretch"):
            with st.spinner("Validating service connections…"):
                if groq:
                    ok, msg = validate_groq_key(groq)
                    st.session_state.groq_status = (ok, msg)
                else:
                    st.session_state.groq_status = (False, "Enter a Groq key to validate.")
                if tavily:
                    ok, msg = validate_tavily_key(tavily)
                    st.session_state.tavily_status = (ok, msg)
                else:
                    st.session_state.tavily_status = (False, "Enter a Tavily key to validate.")
        if st.button("Clear keys and session", width="stretch"):
            clear_model_cache()
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        for name, key in (("Groq", "groq_status"), ("Tavily", "tavily_status")):
            status = st.session_state.get(key)
            if status:
                st.caption(f"{name}: {'Connected' if status[0] else 'Unavailable'} — {status[1]}")
            else:
                st.caption(f"{name}: Not validated")
        st.info("Keys stay in this Streamlit session and are never written to files.")


def show_claim_result(claim, groq_key, tavily_key, language):
    with st.expander(f"{claim}", expanded=True):
        local = retrieve_local(claim, language)
        local_status = index_status()
        web, web_note = search_web(claim, language, tavily_key) if tavily_key else ([], "Tavily key not configured; live search unavailable.")
        verdict = verify_claim(claim, local + web, groq_key)
        verdict_key = verdict.get("verdict", "NEEDS REVIEW")
        badge_class = {"ACCEPTED": "bg-accepted", "BLOCKED": "bg-blocked"}.get(verdict_key, "bg-review")
        st.markdown(f'<span class="bg-verdict {badge_class}">{verdict_key}</span>', unsafe_allow_html=True)
        st.metric("Evidence strength", verdict["strength"])
        st.markdown('<div class="bg-panel-title">Assessment</div>', unsafe_allow_html=True)
        st.write(verdict["explanation"])
        if verdict.get("correction"):
            st.markdown("**Suggested correction**")
            st.write(verdict["correction"])
        elif verdict["verdict"] != "ACCEPTED":
            st.caption("A correction is not suggested because the evidence does not support one clearly.")
        st.caption(web_note)
        if local_status["stale"]:
            st.caption("Local PDF files changed since the last index. Refresh the PDF index before relying on local retrieval.")
        elif local_status["pdf_files"] and not local_status["indexed"]:
            st.caption("Local PDFs are not indexed yet. Use PDF library → Refresh PDF index to include them.")
        if local or web:
            st.markdown("**Evidence**")
            for item in local + web:
                metadata = " · ".join(x for x in [item.get("domain", ""), item.get("location", ""), item.get("date", ""), item.get("language", "")] if x)
                st.markdown(f"**{item.get('title', item.get('document', 'Source'))}** · {metadata}")
                if item.get("credibility") == "search result only":
                    st.caption("Search-result passage only — webpage could not be retrieved; not conclusive evidence.")
                st.write(item.get("text", ""))
                if item.get("url"):
                    st.markdown(f"[Open source]({item['url']})")
        else:
            st.info("No evidence was retrieved. Verdict remains NEEDS REVIEW; absence of evidence is not proof of error.")


def _pct(value):
    return "Not measured" if value is None else f"{value:.1%}"


def _number(value, suffix=""):
    return "Not measured" if value is None else f"{value:.3f}{suffix}"


def render_claim_benchmark(split, groq_key, tavily_key, data_source="imported", mode="provisional", case_limit=5):
    dataset = load_candidate_dataset(split, "claims", mode) if data_source == "imported" else load_dataset(split, "claims")
    local_status = index_status()
    if not local_status["indexed"]:
        st.info("No current local PDF index is available. Benchmarking will not index the document library automatically; refresh it explicitly in PDF Library if you want local retrieval included.")
    if split == "final":
        st.warning("Held-out final set: use this for final measurement only. Do not tune prompts or code from these results.")
    if data_source == "imported":
        st.warning("PROVISIONAL EVALUATION — source records have not been independently verified. Do not report these metrics as verified accuracy.")
    st.caption(f"Dataset: `{Path(dataset['path']).name}` · eligible {mode if data_source == 'imported' else 'manual'} cases in split: {len(dataset['cases'])} · schema issues: {len(dataset['errors'])}")
    if dataset["cases"]:
        st.caption("Loaded case allocation (descriptive only): " + " · ".join(
            f"{label}: {sum(x.get('expected_verdict') == label for x in dataset['cases'])}" for label in ("ACCEPTED", "BLOCKED", "NEEDS REVIEW")))
    if dataset["errors"]:
        with st.expander("Dataset validation issues", expanded=True):
            st.json(dataset["errors"])
    if not dataset["cases"]:
        st.info(f"No {mode if data_source == 'imported' else 'manual'} cases are eligible for this split. No benchmark was run.")
        return
    button = "Run held-out final evaluation" if split == "final" else "Run development benchmark"
    if st.button(button, key=f"run_claims_{split}", type="primary"):
        if not groq_key:
            st.error("Enter and validate a Groq key in the sidebar before running the pipeline.")
        else:
            try:
                with st.spinner("Running claim extraction, PDF/web retrieval, verification, and recording results…"):
                    st.session_state[f"claim_run_{data_source}_{mode}_{split}"] = run_claim_benchmark(dataset, groq_key, tavily_key, split, case_limit)
            except Exception as exc:
                st.error(f"Benchmark run failed: {exc}")
    run = st.session_state.get(f"claim_run_{data_source}_{mode}_{split}") or latest_run(split, "claims")
    if not run:
        st.info("No benchmark run has been saved for this split yet.")
        return
    expected_hash = hashlib.sha256(json.dumps(dataset["cases"], ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if run.get("dataset_hash") and run["dataset_hash"] != expected_hash:
        st.warning("The saved run belongs to an earlier version of this dataset. It is shown as historical and does not measure the currently loaded cases.")
    metrics = summarize_run(run)
    run_mode = str(run.get("evaluation_mode", "legacy/unknown"))
    st.caption(f"Saved run `{run.get('run_id', 'unknown')}` · {run.get('created_at', '')} · mode: {run_mode} · model: {run.get('model_id', 'unknown')} · {run.get('services', {})}")
    a, b, c, d = st.columns(4)
    a.metric("Dataset size", metrics["dataset_size"])
    b.metric("Completed", metrics["completed"])
    c.metric("Failed", metrics["failed"])
    accuracy_label = "Provisional accuracy" if "provisional" in run_mode else "Verified accuracy" if run_mode == "verified/manual" else "Accuracy (verification status unknown)"
    d.metric(accuracy_label, _pct(metrics["accuracy"]))
    if run.get("dataset_total_eligible") is not None:
        st.caption(f"This batch contains {metrics['dataset_size']} cases from {run['dataset_total_eligible']} eligible cases in the selected split.")
    st.caption(f"Accuracy denominator: {metrics['accuracy_denominator']} labeled cases, including failed/missing predictions as incorrect. This run completed {metrics['completed']} and failed {metrics['failed']} cases.")
    p, r = st.columns(2)
    p.metric("Overall precision", _pct(metrics["overall_precision"]))
    r.metric("Overall recall", _pct(metrics["overall_recall"]))
    st.caption("Precision uses completed predictions as its denominator. Recall and accuracy count failed cases as misses.")
    a, b, c, d = st.columns(4)
    a.metric("Macro F1", _pct(metrics["macro_f1"]))
    b.metric("False acceptance rate", _pct(metrics["false_acceptance_rate"]))
    c.metric("False blocking rate", _pct(metrics["false_blocking_rate"]))
    d.metric("Abstention rate", _pct(metrics["abstention_rate"]))
    st.caption("False acceptance = blocked claims predicted accepted. False blocking = accepted claims predicted blocked. Coverage = decisive verdicts / all labeled cases; failures are uncovered.")
    a, b = st.columns(2)
    a.metric("Coverage", _pct(metrics["coverage"]))
    b.metric("Selective accuracy", _pct(metrics["selective_accuracy"]))
    st.markdown("**Per-class performance**")
    st.dataframe(metrics["per_class"], width="stretch")
    st.markdown("**Confusion matrix** · rows are expected verdicts; columns are predicted verdicts")
    st.dataframe(metrics["confusion_matrix"], width="stretch")
    with st.expander("Performance by language, scheme, category, and difficulty"):
        for field, groups in metrics.get("by_group", {}).items():
            st.write(field.title())
            st.dataframe([{"group": name, **values} for name, values in groups.items()], width="stretch")
    st.markdown("**Retrieval and citations**")
    a, b, c, d, e = st.columns(5)
    a.metric("Recall@3", _pct(metrics["recall_at_3"]))
    b.metric("Recall@5", _pct(metrics["recall_at_5"]))
    c.metric("MRR", _pct(metrics["mrr"]))
    d.metric("Correct document", _pct(metrics["correct_document_retrieval"]))
    e.metric("Correct page", _pct(metrics["correct_page_retrieval"]))
    st.caption(f"Citation correctness: {_pct(metrics['citation_correctness'])} · eligible gold-citation cases: {metrics['eligible_cases']}. Provisional imported references are excluded.")
    st.caption(f"Retrieval metrics use {metrics['eligible_cases']} cases with a reviewed source PDF. No source references means not measured.")
    st.markdown("**Execution and API operations**")
    a, b, c, d, e = st.columns(5)
    a.metric("Average latency", _number(metrics["average_latency_seconds"], " s"))
    b.metric("Median latency", _number(metrics["median_latency_seconds"], " s"))
    c.metric("P95 latency", _number(metrics["p95_latency_seconds"], " s"))
    d.metric("API failures", metrics["api_failures"])
    e.metric("Execution time", _number(metrics["execution_seconds"], " s"))
    token_label = "Not reported" if metrics["tokens_used"] is None else f"{metrics['tokens_used']:,}"
    st.caption(f"PDF index time: {_number(metrics['pdf_indexing_seconds'], ' s')} | token usage: {token_label} | Groq/Tavily/page calls: {metrics['groq_calls']}/{metrics['tavily_calls']}/{metrics['webpage_calls']}")
    st.caption(f"Failed webpage retrievals: {metrics['web_retrieval_failures']}. Token counts are shown only when the provider reports them.")
    failures = [x for x in run.get("cases", []) if x.get("status") != "completed"]
    st.markdown("**Failed test cases**")
    if failures:
        st.dataframe([{k: x.get(k) for k in ("test_id", "failure_category", "status", "error", "expected_verdict", "predicted_verdict", "latency_seconds")} for x in failures], width="stretch")
    else:
        st.caption("No failed cases in this saved run.")
    if run.get("cases"):
        ids = [str(x.get("test_id", "unknown")) for x in run["cases"]]
        chosen = st.selectbox("Inspect a case and its evidence", ids, key=f"inspect_claim_{split}")
        case = next(x for x in run["cases"] if str(x.get("test_id", "unknown")) == chosen)
        with st.expander(f"Case details: {chosen}"):
            st.write("**Input answer**")
            st.write(case.get("input_answer", ""))
            st.write("**Expected claim**")
            st.write(case.get("expected_claim", ""))
            st.write(f"Expected: {case.get('expected_verdict')} · Predicted: {case.get('predicted_verdict')} · Strength: {case.get('predicted_strength', 'Not available')}")
            st.write(case.get("predicted_explanation", case.get("error", "")))
            st.write(f"Ground-truth citation: {case.get('source_pdf') or 'No PDF source'} · PDF page index: {case.get('page')} · Original page label: {case.get('original_page_number', '')}")
            if case.get("evidence"):
                st.write("**Ground-truth evidence passage**")
                st.write(case["evidence"])
            st.write("**Retrieved evidence and references**")
            st.json(case.get("retrieved_evidence", []))


def render_bilingual_benchmark(split, groq_key, data_source="imported", mode="provisional", case_limit=5):
    dataset = load_candidate_dataset(split, "bilingual", mode) if data_source == "imported" else load_dataset(split, "bilingual")
    if split == "final":
        st.warning("Held-out bilingual set: use for final measurement only; do not tune against it.")
    if data_source == "imported":
        st.warning("PROVISIONAL CONSISTENCY EVALUATION — pair labels and source references still need manual review.")
    st.caption(f"Dataset: `{Path(dataset['path']).name}` · eligible {mode if data_source == 'imported' else 'manual'} pairs: {len(dataset['cases'])} · schema issues: {len(dataset['errors'])}")
    if dataset["errors"]:
        with st.expander("Dataset validation issues", expanded=True): st.json(dataset["errors"])
    if not dataset["cases"]:
        st.info(f"No {mode if data_source == 'imported' else 'manual'} bilingual pairs are eligible for this split.")
        return
    button = "Run held-out bilingual evaluation" if split == "final" else "Run bilingual development benchmark"
    if st.button(button, key=f"run_bilingual_{split}", type="primary"):
        if not groq_key: st.error("Enter a Groq API key to run the consistency comparison.")
        else:
            try:
                with st.spinner("Comparing English and Hindi meanings…"):
                    st.session_state[f"bilingual_run_{data_source}_{mode}_{split}"] = run_bilingual_benchmark(dataset, groq_key, split, case_limit)
            except Exception as exc: st.error(f"Bilingual benchmark failed: {exc}")
    run = st.session_state.get(f"bilingual_run_{data_source}_{mode}_{split}") or latest_run(split, "bilingual")
    if not run:
        st.info("No bilingual benchmark run has been saved for this split yet.")
        return
    metrics = calculate_bilingual_metrics(run.get("cases", []))
    run_mode = str(run.get("evaluation_mode", "legacy/unknown"))
    st.caption(f"Saved run `{run.get('run_id', 'unknown')}` · mode: {run_mode} · model: {run.get('model_id', 'unknown')} · consistency labels only; this does not assess factual truth.")
    a,b,c,d = st.columns(4)
    a.metric("Dataset size", metrics["dataset_size"]); b.metric("Completed", metrics["completed"])
    c.metric("Failed", metrics["failed"]); d.metric("Provisional consistency accuracy" if "provisional" in run_mode else "Detection accuracy", _pct(metrics["accuracy"]))
    st.caption(f"Accuracy counts failed cases as incorrect · completed {metrics['completed']} of {metrics['dataset_size']} · coverage {_pct(metrics.get('coverage'))}")
    a,b,c = st.columns(3)
    a.metric("Discrepancy precision", _pct(metrics["precision"])); b.metric("Discrepancy recall", _pct(metrics["recall"]))
    c.metric("Discrepancy F1", _pct(metrics["f1"]))
    latencies = [x["latency_seconds"] for x in run.get("cases", []) if isinstance(x.get("latency_seconds"), (int, float))]
    usage = [x.get("api_usage", {}) for x in run.get("cases", [])]
    failures_count = sum(int(x.get("api_failures", 0)) for x in usage)
    token_rows = [x for x in usage if x.get("token_usage_available")]
    total_tokens = sum(int(x.get("prompt_tokens", 0)) + int(x.get("completion_tokens", 0)) for x in token_rows) if token_rows else None
    a,b,c,d = st.columns(4)
    a.metric("Average latency", _number(sum(latencies)/len(latencies) if latencies else None, " s"))
    b.metric("API failures", failures_count)
    c.metric("Groq calls", sum(int(x.get("groq_calls", 0)) for x in usage))
    d.metric("Tokens", "Not reported" if total_tokens is None else f"{total_tokens:,}")
    if metrics.get("per_class"):
        st.write("Per-class performance"); st.dataframe(metrics["per_class"], width="stretch")
    if metrics.get("confusion_matrix"): st.dataframe(metrics["confusion_matrix"], width="stretch")
    if metrics["by_category"]: st.dataframe(metrics["by_category"], width="stretch")
    failures = [x for x in run.get("cases", []) if x.get("status") != "completed"]
    if failures: st.dataframe([{k:x.get(k) for k in ("test_id", "category", "status", "error")} for x in failures], width="stretch")
    if run.get("cases"):
        selected = st.selectbox("Inspect bilingual case", [x["test_id"] for x in run["cases"]], key=f"inspect_bilingual_{split}")
        case = next(x for x in run["cases"] if x["test_id"] == selected)
        with st.expander(f"Bilingual details: {selected}"):
            st.write("**English**"); st.write(case.get("english_answer", ""))
            st.write("**Hindi**"); st.write(case.get("hindi_answer", ""))
            st.write(f"Expected discrepancy: {case.get('expected_discrepancy')} · Predicted: {case.get('predicted_discrepancy')}")
            st.json(case.get("detected_discrepancies", []))


def main():
    apply_ui_style()
    keys_panel()
    st.markdown("""
    <div class="bg-hero">
      <h1>🛡️ BhashaGuard</h1>
      <p>Check AI-generated claims against local documents and live sources, and compare English and Hindi answers for factual differences.</p>
    </div>
    """, unsafe_allow_html=True)
    groq_key = st.session_state.get("groq_key", "")
    tavily_key = st.session_state.get("tavily_key", "")
    pages = st.tabs(["Single-answer audit", "English–Hindi comparison", "PDF library", "Evaluation"])

    with pages[0]:
        language = st.selectbox("Answer language", ["English", "Hindi"])
        answer = st.text_area("AI-generated answer", height=190, key="single_answer")
        if st.button("Audit answer", type="primary"):
            if not answer.strip():
                st.error("Paste an answer to audit.")
            elif not groq_key:
                st.error("Add a Groq API key in the sidebar to extract and assess claims.")
            else:
                with st.spinner("Checking your answer against available evidence…"):
                    claims, err = extract_claims(answer, groq_key, language)
                    if err:
                        st.error(err)
                    elif not claims:
                        st.info("No factual claims were extracted.")
                    else:
                        st.subheader("Original answer")
                        st.write(answer)
                        st.subheader(f"Claim-level results ({len(claims)})")
                        for claim in claims:
                            show_claim_result(claim, groq_key, tavily_key, language)

    with pages[1]:
        mode = st.radio("Input type", ["English and Hindi answers", "Original and translation"], horizontal=True)
        left, right = st.columns(2)
        with left:
            english = st.text_area("English answer", height=220)
        with right:
            hindi = st.text_area("Hindi answer", height=220)
        if st.button("Compare and verify", type="primary"):
            if not english.strip() or not hindi.strip():
                st.error("Enter both answers.")
            elif not groq_key:
                st.error("Add a Groq API key in the sidebar to compare claims.")
            else:
                with st.spinner("Comparing meaning and verifying claims independently…"):
                    st.subheader("Answers side by side")
                    c1, c2 = st.columns(2)
                    c1.write(english)
                    c2.write(hindi)
                    result, err = compare_answers(english, hindi, groq_key, mode)
                    if err:
                        st.error(err)
                    else:
                        if result.get("claim_extraction_errors"):
                            st.warning("Could not extract claims for every language, so those claims could not be independently evidence checked: " + "; ".join(result["claim_extraction_errors"]))
                        st.subheader("Detected discrepancies")
                        if not result.get("discrepancies"):
                            st.success("No material discrepancies detected.")
                        for d in result.get("discrepancies", []):
                            st.markdown(f"**{d.get('importance', 'NEEDS REVIEW')} — {d.get('type', 'Difference')}**")
                            st.write(d.get("explanation", ""))
                            english_col, hindi_col = st.columns(2)
                            english_col.markdown("**English**")
                            english_col.write(d.get("english", ""))
                            hindi_col.markdown("**Hindi**")
                            hindi_col.write(d.get("hindi", ""))
                        st.subheader("Independent evidence checks")
                        for claim in result.get("english_claims", []):
                            st.markdown(f"**English:** {claim}")
                            show_claim_result(claim, groq_key, tavily_key, "English")
                        for claim in result.get("hindi_claims", []):
                            st.markdown(f"**Hindi:** {claim}")
                            show_claim_result(claim, groq_key, tavily_key, "Hindi")

    with pages[2]:
        st.subheader("Local PDF evidence")
        st.write("Add PDF documents to `data/documents/` and refresh the index. Files are discovered recursively.")
        if st.button("Refresh PDF index"):
            with st.spinner("Reading PDFs…"):
                summary = index_pdfs()
            st.success(f"Indexed {summary['chunks']} passages from {summary['documents']} documents.")
            if summary["scanned"]:
                st.warning("No extractable text found (possibly scanned PDFs requiring OCR): " + ", ".join(summary["scanned"]))
            if summary["errors"]:
                st.warning("Some PDFs could not be read: " + "; ".join(summary["errors"]))
        state = index_status()["summary"]
        status = index_status()
        st.caption(f"Available documents: {state['documents']} · passages: {state['chunks']}")
        if status["stale"]:
            st.warning("PDF files have changed since the saved index. Refresh the index to use these files.")
        elif status["pdf_files"] and not status["indexed"]:
            st.info("PDFs are discovered, but their text has not been indexed. Indexing runs only when you press the refresh button.")
        if state["documents"]:
            if status["indexed"]:
                if st.button("Generate document-based sample questions"):
                    if not groq_key:
                        st.error("Add a Groq key to generate questions grounded in your PDFs.")
                    else:
                        with st.spinner("Preparing questions from the indexed documents…"):
                            qs, err = generate_questions(groq_key)
                        st.session_state.sample_questions = qs
                        if err: st.error(err)
                for q in st.session_state.get("sample_questions", []):
                    st.button(f"{q['question']} — {q['document']}, page {q['page']}", key=f"sample_{q['document']}_{q['page']}_{q['question']}", on_click=set_sample_question, args=(q["question"],))
            else:
                st.caption("Refresh the index before generating questions from document text.")
        else:
            st.info("No PDFs found yet. Place documents in data/documents/ to enable local retrieval.")

    with pages[3]:
        st.subheader("Evaluation & Benchmarks")
        overview = candidate_overview()
        if overview.get("error"):
            st.error(overview["error"])
        st.warning("Accuracy from the imported candidate files is PROVISIONAL. The companion dataset marks all source records for verification; no record is currently accepted as verified ground truth.")
        a, b, c, d = st.columns(4)
        a.metric("Factual cases", overview["factual_total"])
        b.metric("Bilingual pairs", overview["bilingual_total"])
        c.metric("Verified records", overview["verified"])
        d.metric("Unresolved source status", overview["needs_source_verification"] + overview["unverifiable"] + overview["rejected"])
        st.caption(f"Candidate files: `{overview['primary_file']}` (primary) and `{overview['supplementary_file']}` (normalization notes) · version {overview['dataset_version']} · split seed {overview['split_seed']}.")
        st.caption(f"Status counts: {overview['verified']} VERIFIED · {overview['needs_source_verification']} NEEDS_SOURCE_VERIFICATION · {overview['unverifiable']} UNVERIFIABLE · {overview['rejected']} REJECTED.")
        with st.expander("Dataset composition and quality"):
            st.write(f"Factual labels available for provisional scoring: {overview['factual_labeled']} / {overview['factual_total']} · records missing a canonical label after normalization: {overview['invalid_or_missing_labels']}.")
            st.write(f"Duplicate IDs: primary {overview['duplicate_ids_primary']}, companion {overview['duplicate_ids_supplementary']} · unmatched primary IDs: {overview['unmatched_primary_ids']} · canonical label disagreements: {overview['supplementary_label_disagreements']}.")
            st.write(f"Verification-status disagreements between source files: {overview['verification_status_conflicts']}.")
            st.write("Factual cases by language"); st.dataframe([{"language": k, "cases": v} for k, v in overview["language"].items()], width="stretch")
            st.write("Cases by scheme"); st.dataframe([{"scheme": k, "cases": v} for k, v in overview["scheme"].items()], width="stretch")
            st.write("Provisional factual labels"); st.dataframe([{"verdict": k, "cases": v} for k, v in overview["verdict"].items()], width="stretch")
            st.write("Label source"); st.dataframe([{"source": k, "cases": v} for k, v in overview["label_provenance"].items()], width="stretch")
            st.write("Difficulty"); st.dataframe([{"difficulty": k, "cases": v} for k, v in overview["difficulty"].items()], width="stretch")
            st.write("Category"); st.dataframe([{"category": k, "cases": v} for k, v in overview["category"].items()], width="stretch")
            st.caption(f"Candidate source passages: {overview['source_quotes']} · filenames: {overview['records_with_source_file']} · page references: {overview['records_with_page_reference']}. Page/quote references have not been independently approved; retrieval and citation metrics remain unavailable for these imported records.")
        data_source = st.radio("Dataset", ["imported", "manual"], format_func=lambda x: "Imported candidate JSON (provisional)" if x == "imported" else "Manual verified templates", horizontal=True, key="evaluation_data_source")
        mode = st.radio("Evaluation mode", ["provisional", "verified"], format_func=lambda x: x.title(), horizontal=True, key="evaluation_mode") if data_source == "imported" else "verified"
        split = st.radio("Dataset split", ["dev", "final"], format_func=lambda x: "Development" if x == "dev" else "Held-out final", horizontal=True, key="evaluation_split")
        default_limit = 5
        case_limit = st.slider("Maximum cases per run (controls API usage)", 1, 20, default_limit, key="evaluation_case_limit")
        st.caption("Runs are explicit and use the production pipeline. Failed calls count as incorrect in accuracy. The held-out split is generated deterministically with seed 42; keep it for final measurement only.")
        claim_tab, bilingual_tab = st.tabs(["Claim verification", "English–Hindi consistency"])
        with claim_tab:
            render_claim_benchmark(split, groq_key, tavily_key, data_source, mode, case_limit)
        with bilingual_tab:
            render_bilingual_benchmark(split, groq_key, data_source, mode, case_limit)

    st.divider()
    st.caption("Verdicts: ACCEPTED means evidence supports the claim; BLOCKED means credible evidence contradicts it; NEEDS REVIEW means evidence is insufficient, conflicting, outdated, or ambiguous. Evidence strength is assessed separately.")


if __name__ == "__main__":
    main()
