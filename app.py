import streamlit as st

from core.cross_language import compare_answers
from core.evaluation import load_benchmarks, calculate_metrics
from core.extraction import extract_claims
from core.llm import validate_groq_key
from core.retrieval import index_pdfs, retrieve_local, generate_questions
from core.verification import verify_claim
from core.web_search import search_web, validate_tavily_key

st.set_page_config(page_title="BhashaGuard", page_icon="🛡️", layout="wide")


def set_sample_question(question):
    st.session_state["single_answer"] = question


def keys_panel():
    with st.sidebar:
        st.title("🛡️ BhashaGuard")
        st.caption("Evidence-grounded AI reliability auditor")
        st.subheader("API configuration")
        groq = st.text_input("Groq API key", type="password", key="groq_key", help="Used to extract and assess claims.")
        tavily = st.text_input("Tavily API key", type="password", key="tavily_key", help="Used to search live web sources.")
        if st.button("Validate keys", use_container_width=True):
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
        if st.button("Clear keys and session", use_container_width=True):
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
        web, web_note = search_web(claim, language, tavily_key) if tavily_key else ([], "Tavily key not configured; live search unavailable.")
        verdict = verify_claim(claim, local + web, groq_key)
        c1, c2 = st.columns(2)
        c1.metric("Verdict", verdict["verdict"])
        c2.metric("Evidence strength", verdict["strength"])
        st.write(verdict["explanation"])
        if verdict.get("correction"):
            st.markdown("**Suggested correction**")
            st.write(verdict["correction"])
        elif verdict["verdict"] != "ACCEPTED":
            st.caption("A correction is not suggested because the evidence does not support one clearly.")
        st.caption(web_note)
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


def main():
    keys_panel()
    st.title("BhashaGuard")
    st.write("Check AI-generated claims against local documents and live sources, and compare English and Hindi answers for factual differences.")
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
                with st.spinner("Extracting claims and retrieving evidence…"):
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
                st.subheader("Answers side by side")
                c1, c2 = st.columns(2)
                c1.write(english)
                c2.write(hindi)
                with st.spinner("Comparing meaning and verifying claims independently…"):
                    result, err = compare_answers(english, hindi, groq_key, mode)
                if err:
                    st.error(err)
                else:
                    if result.get("claim_extraction_errors"):
                        st.warning("Could not extract claims for every language, so those claims could not be independently evidence checked: " + "; ".join(result["claim_extraction_errors"]))
                    st.subheader("Detected discrepancies")
                    if not result.get("discrepancies"):
                        st.info("No material discrepancies detected.")
                    for d in result.get("discrepancies", []):
                        st.markdown(f"**{d.get('importance', 'NEEDS REVIEW')} — {d.get('type', 'Difference')}**")
                        st.write(d.get("explanation", ""))
                        st.caption(f"English: {d.get('english', '')}  ·  Hindi: {d.get('hindi', '')}")
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
        state = index_pdfs()
        st.caption(f"Available documents: {state['documents']} · passages: {state['chunks']}")
        if state["documents"]:
            if st.button("Generate document-based sample questions"):
                if not groq_key:
                    st.error("Add a Groq key to generate questions grounded in your PDFs.")
                else:
                    qs, err = generate_questions(groq_key)
                    st.session_state.sample_questions = qs
                    if err: st.error(err)
            for q in st.session_state.get("sample_questions", []):
                st.button(f"{q['question']} — {q['document']}, page {q['page']}", key=f"sample_{q['document']}_{q['page']}_{q['question']}", on_click=set_sample_question, args=(q["question"],))
        else:
            st.info("No PDFs found yet. Place documents in data/documents/ to enable local retrieval.")

    with pages[3]:
        st.subheader("Benchmark evaluation")
        cases = load_benchmarks()
        if not cases:
            st.info("No benchmark has been loaded. Add labelled JSON or CSV cases under data/benchmark/.")
        else:
            if st.button("Calculate benchmark metrics"):
                metrics = calculate_metrics(cases)
                if metrics:
                    st.metric("Overall claim-level accuracy", f"{metrics['accuracy']:.1%}")
                    st.write("Per-verdict precision / recall / F1")
                    st.dataframe(metrics["per_verdict"], use_container_width=True)
                    st.metric("BLOCKED precision", f"{metrics['blocked_precision']:.1%}" if metrics["blocked_precision"] is not None else "N/A")
                    st.caption("Confusion matrix rows are labelled verdicts; columns are predicted verdicts.")
                    st.dataframe(metrics["confusion_matrix"], use_container_width=True)
                    if metrics.get("language_accuracy"):
                        st.write("Accuracy by language")
                        st.dataframe(metrics["language_accuracy"], use_container_width=True)
                    if metrics.get("discrepancy"):
                        st.write("Cross-language discrepancy detection")
                        st.json(metrics["discrepancy"])
                else:
                    st.warning("Benchmark files were found, but no valid labelled claims were present.")

    st.divider()
    st.caption("Verdicts: ACCEPTED means evidence supports the claim; BLOCKED means credible evidence contradicts it; NEEDS REVIEW means evidence is insufficient, conflicting, outdated, or ambiguous. Evidence strength is assessed separately.")


if __name__ == "__main__":
    main()
