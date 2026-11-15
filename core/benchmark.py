"""Runs the production claim-audit functions against manually labeled cases."""
import json
import hashlib
import re
import time
from datetime import datetime, timezone

from core.extraction import extract_claims
from core.llm import track_api_usage
import core.llm as llm
from core.retrieval import index_status, retrieve_local
from core.verification import verify_claim
from core.web_search import search_web
from core.evaluation import RESULTS_DIR


def _tokens(text):
    return set(re.findall(r"[\w\u0900-\u097F]+", str(text or "").casefold(), re.UNICODE))


def _matching_claim(expected, claims):
    gold = _tokens(expected)
    ranked = []
    for claim in claims:
        found = _tokens(claim)
        ranked.append((len(gold & found) / max(1, len(gold | found)), claim))
    return max(ranked, default=(0, None), key=lambda x: x[0])


def run_claim_benchmark(dataset, groq_key, tavily_key="", split="dev", case_limit=None):
    if not groq_key:
        raise ValueError("A Groq API key is required to run the claim benchmark.")
    if split not in ("dev", "final"):
        raise ValueError("split must be 'dev' or 'final'.")
    run_start = time.perf_counter()
    # Never index a user's entire PDF library as a side effect of benchmarking.
    # The user can explicitly build/refresh the local index in the PDF Library UI.
    index_summary = index_status()["summary"]
    selected_cases = dataset.get("cases", [])[:case_limit] if case_limit else dataset.get("cases", [])
    results = []
    for case in selected_cases:
        start = time.perf_counter()
        record = {"test_id": case["test_id"], "scheme": case["scheme"], "language": case["language"],
                  "input_answer": case["input_answer"], "expected_claim": case["expected_claim"],
                  "expected_verdict": case["expected_verdict"], "source_pdf": case.get("source_pdf", ""),
                  "page": case.get("page"), "original_page_number": case.get("original_page_number", ""), "evidence": case.get("evidence", ""),
                  "additional_sources": case.get("additional_sources", []),
                  "expected_explanation": case.get("expected_explanation", ""),
                  "difficulty": case.get("difficulty", ""), "category": case.get("category", ""),
                  "verification_status": case.get("verification_status", "VERIFIED/manual"),
                  "candidate_source_pdf": case.get("candidate_source_pdf", ""),
                  "candidate_pdf_page_index": case.get("candidate_pdf_page_index"),
                  "candidate_printed_page": case.get("candidate_printed_page", ""),
                  "candidate_evidence": case.get("candidate_evidence", ""),
                  "predicted_verdict": None, "retrieved_evidence": [], "claim_results": [], "status": "failed"}
        with track_api_usage() as usage:
            claims, extraction_error = extract_claims(case["input_answer"], groq_key, case["language"])
            record["extracted_claims"] = claims
            if extraction_error:
                record["error"] = extraction_error
                record["failure_category"] = "claim extraction error"
            elif not claims:
                record["error"] = "No claims were extracted from the input."
                record["failure_category"] = "claim extraction error"
            else:
                claims = list(dict.fromkeys(str(c).strip() for c in claims if str(c).strip()))
                record["extracted_claims"] = claims
                match_score, claim = _matching_claim(case["expected_claim"], claims)
                record["matched_claim"] = claim
                record["claim_match_score"] = match_score
                if not claim or match_score < 0.12:
                    record["alignment_warning"] = "No close lexical match to the annotated claim; evaluated the extracted claims as a set."
                try:
                    for extracted in claims:
                        local = retrieve_local(extracted, case["language"])
                        web, note = search_web(extracted, case["language"], tavily_key) if tavily_key else ([], "Tavily key not configured; local evidence only.")
                        all_evidence = local + web
                        assessment = verify_claim(extracted, all_evidence, groq_key)
                        record["claim_results"].append({"claim": extracted, **assessment, "web_note": note, "retrieved_evidence": all_evidence})
                    failures = [x for x in record["claim_results"] if x["explanation"].startswith("Evidence assessment failed")]
                    if failures:
                        record["error"] = "; ".join(x["explanation"] for x in failures)
                        record["failure_category"] = "evidence assessment/API failure"
                    else:
                        verdicts = [x["verdict"] for x in record["claim_results"]]
                        record["predicted_verdict"] = "BLOCKED" if "BLOCKED" in verdicts else "NEEDS REVIEW" if "NEEDS REVIEW" in verdicts else "ACCEPTED"
                        strengths = [x["strength"] for x in record["claim_results"]]
                        record["predicted_strength"] = "INSUFFICIENT" if "INSUFFICIENT" in strengths else "MODERATE" if "MODERATE" in strengths else "STRONG"
                        record["predicted_explanation"] = " ".join(f"{x['claim']}: {x['verdict']} — {x['explanation']}" for x in record["claim_results"])
                        blocked = [x for x in record["claim_results"] if x["verdict"] == "BLOCKED" and x.get("correction")]
                        record["correction"] = blocked[0]["correction"] if len(blocked) == 1 else ""
                        record["retrieved_evidence"] = [e for x in record["claim_results"] for e in x["retrieved_evidence"]]
                        record["status"] = "completed"
                except Exception as exc:
                    record["error"] = f"Pipeline failed ({exc.__class__.__name__}): {exc}"
                    record["failure_category"] = "pipeline/API failure"
        record["api_usage"] = dict(usage)
        record["latency_seconds"] = round(time.perf_counter() - start, 3)
        results.append(record)
    dataset_hash = hashlib.sha256(json.dumps(dataset.get("cases", []), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    run = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "created_at": datetime.now(timezone.utc).isoformat(), "kind": "claims", "split": split,
           "dataset_path": dataset.get("path", ""), "dataset_size": len(selected_cases),
           "dataset_total_eligible": len(dataset.get("cases", [])), "dataset_hash": dataset_hash,
           "dataset_validation_errors": dataset.get("errors", []), "pdf_indexing_seconds": None,
           "pdf_index_summary": index_summary,
           "services": {"groq": "enabled", "tavily": "enabled" if tavily_key else "not configured"}, "cases": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{split}_claims_{run['run_id']}.json"
    run.update({"execution_seconds": round(time.perf_counter() - run_start, 3),
                "dataset_version": dataset.get("dataset_version", "manual-template"),
                "evaluation_mode": dataset.get("evaluation_mode", "verified/manual"),
                "split_seed": dataset.get("seed"), "split_fraction": dataset.get("split_fraction"),
                "split_assignment_hash": dataset.get("split_assignment_hash"),
                "split_assignments": dataset.get("split_assignments", {}),
                "model_id": llm.MODEL,
                "retrieval_configuration": "local lexical retrieval + optional Tavily",
                "prompt_version": "core-prompts-v1"})
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["result_path"] = str(out)
    return run
