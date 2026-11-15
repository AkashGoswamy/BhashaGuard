"""Dataset validation and honest claim-level benchmark metrics."""
import json
import statistics
import unicodedata
from pathlib import Path
from core.retrieval import DOCS, fitz

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"
RESULTS_DIR = BENCHMARK_DIR / "results"
LABELS = ["ACCEPTED", "BLOCKED", "NEEDS REVIEW"]
SPLITS = ("dev", "final")

REQUIRED_FIELDS = ("test_id", "scheme", "language", "input_answer", "expected_claim",
                   "expected_verdict", "source_pdf", "page", "original_page_number", "evidence",
                   "expected_explanation", "difficulty", "category")


def _find_pdf(name):
    relative = str(name).replace("\\", "/").strip("/")
    paths = list(DOCS.rglob("*.pdf"))
    matches = [p for p in paths if str(p.relative_to(DOCS)).replace("\\", "/").casefold() == relative.casefold()]
    if not matches and "/" not in relative:
        matches = [p for p in paths if p.name.casefold() == Path(relative).name.casefold()]
    return matches[0] if matches else None


def _citation_error(citation):
    source = citation.get("source_pdf", "")
    path = _find_pdf(source)
    if not path:
        return f"Source PDF not found under data/documents/: {source}"
    try:
        page_number = int(citation["page"])
        with fitz.open(path) as pdf:
            if page_number < 1 or page_number > len(pdf):
                return f"PDF page {page_number} is outside {path.name} ({len(pdf)} pages)."
            page_text = pdf[page_number - 1].get_text("text")
        expected = " ".join(unicodedata.normalize("NFC", str(citation.get("evidence", ""))).casefold().split())
        actual = " ".join(unicodedata.normalize("NFC", page_text).casefold().split())
        if not expected or expected not in actual:
            return f"Evidence passage was not found on PDF page {page_number} in {path.name}."
    except (OSError, ValueError, TypeError, KeyError) as exc:
        return f"Could not verify source {source}: {exc}"
    return None


def load_dataset(split="dev", kind="claims"):
    """Load one explicit split; preserve row-level validation errors."""
    if split not in SPLITS:
        return {"cases": [], "errors": [f"Unknown split: {split}"], "path": ""}
    suffix = "" if kind == "claims" else "_bilingual"
    path = BENCHMARK_DIR / f"{split}{suffix}.json"
    if not path.exists():
        return {"cases": [], "errors": [f"Dataset file not found: {path.name}"], "path": str(path)}
    try:
        content = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"cases": [], "errors": [f"Could not read {path.name}: {exc}"], "path": str(path)}
    if not isinstance(content, dict) or not isinstance(content.get("cases"), list):
        return {"cases": [], "errors": ["Dataset must be an object with a cases array."], "path": str(path)}
    cases, errors, ids = [], [], set()
    required = REQUIRED_FIELDS if kind == "claims" else (
        "test_id", "english_answer", "hindi_answer", "expected_discrepancy", "category")
    for i, row in enumerate(content["cases"], start=1):
        if not isinstance(row, dict):
            errors.append({"row": i, "error": "Case must be a JSON object."})
            continue
        missing = [key for key in required if key not in row]
        if missing:
            errors.append({"row": i, "test_id": row.get("test_id"), "error": "Missing fields: " + ", ".join(missing)})
            continue
        test_id = str(row["test_id"]).strip()
        if not test_id:
            errors.append({"row": i, "error": "test_id cannot be empty."})
            continue
        if test_id in ids:
            errors.append({"row": i, "test_id": test_id, "error": "Duplicate test_id."})
            continue
        ids.add(test_id)
        if kind == "claims":
            verdict = str(row.get("expected_verdict", "")).upper().strip()
            if verdict not in LABELS:
                errors.append({"row": i, "test_id": test_id, "error": f"Invalid expected_verdict: {verdict}"})
                continue
            if not str(row.get("expected_claim", "")).strip() or not str(row.get("input_answer", "")).strip():
                errors.append({"row": i, "test_id": test_id, "error": "input_answer and expected_claim must not be empty."})
                continue
            if row.get("language") not in ("English", "Hindi"):
                errors.append({"row": i, "test_id": test_id, "error": "language must be English or Hindi."})
                continue
            if not str(row.get("expected_explanation", "")).strip() or not str(row.get("category", "")).strip():
                errors.append({"row": i, "test_id": test_id, "error": "expected_explanation and category are required for manual review."})
                continue
            if row.get("difficulty") not in ("easy", "medium", "hard"):
                errors.append({"row": i, "test_id": test_id, "error": "difficulty must be easy, medium, or hard."})
                continue
            source = row.get("source_pdf", "")
            extra_sources = row.get("additional_sources", [])
            if not isinstance(source, str):
                errors.append({"row": i, "test_id": test_id, "error": "source_pdf must name the primary PDF or be empty."})
                continue
            if not isinstance(extra_sources, list) or any(not isinstance(x, dict) for x in extra_sources):
                errors.append({"row": i, "test_id": test_id, "error": "additional_sources must be a list of citation objects."})
                continue
            if verdict in ("ACCEPTED", "BLOCKED") and not source:
                errors.append({"row": i, "test_id": test_id, "error": "ACCEPTED/BLOCKED ground truth requires a source_pdf."})
                continue
            if source and (row.get("page") is None or not str(row.get("original_page_number", "")).strip() or not str(row.get("evidence", "")).strip()):
                errors.append({"row": i, "test_id": test_id, "error": "A cited source_pdf requires PDF page, original page label, and exact evidence passage."})
                continue
            if not source and (extra_sources or row.get("page") is not None or str(row.get("original_page_number", "")).strip() or str(row.get("evidence", "")).strip()):
                errors.append({"row": i, "test_id": test_id, "error": "Without a source_pdf, page must be null and source citation fields must be empty."})
                continue
            citations = ([{"source_pdf": source, "page": row.get("page"), "original_page_number": row.get("original_page_number"), "evidence": row.get("evidence", "")}] if source else []) + extra_sources
            citation_issue = None
            for citation in citations:
                if not isinstance(citation, dict) or not all(str(citation.get(key, "")).strip() for key in ("source_pdf", "original_page_number", "evidence")) or citation.get("page") is None:
                    citation_issue = "Each citation needs source_pdf, PDF page, original_page_number, and exact evidence."
                    break
                citation_issue = _citation_error(citation)
                if citation_issue:
                    break
            if citation_issue:
                errors.append({"row": i, "test_id": test_id, "error": "Source citation check failed: " + citation_issue})
                continue
            if row.get("page") is not None:
                try:
                    if int(row["page"]) < 1: raise ValueError
                except (ValueError, TypeError):
                    errors.append({"row": i, "test_id": test_id, "error": "page must be a positive PDF page number or null."})
                    continue
            row = {**row, "expected_verdict": verdict}
        else:
            if not isinstance(row["expected_discrepancy"], bool):
                errors.append({"row": i, "test_id": test_id, "error": "expected_discrepancy must be boolean."})
                continue
            if row.get("category") not in ("equivalent", "number", "eligibility", "exception", "negation", "date", "percentage", "other"):
                errors.append({"row": i, "test_id": test_id, "error": "Invalid bilingual category."})
                continue
            if not str(row.get("english_answer", "")).strip() or not str(row.get("hindi_answer", "")).strip():
                errors.append({"row": i, "test_id": test_id, "error": "Both language answers must not be empty."})
                continue
        cases.append(row)
    return {"cases": cases, "errors": errors, "path": str(path), "metadata": content.get("metadata", {})}


def calculate_metrics(cases):
    """Score every labeled case; failed or missing predictions count as incorrect."""
    eligible = [c for c in cases if c.get("expected_verdict") in LABELS]
    completed = [c for c in eligible if c.get("status", "completed") == "completed" and c.get("predicted_verdict") in LABELS]
    failed = len(eligible) - len(completed)
    total = len(eligible)
    if not total:
        return {"dataset_size": 0, "completed": 0, "failed": 0, "accuracy": None, "macro_f1": None,
                "overall_precision": None, "overall_recall": None,
                "per_class": {}, "confusion_matrix": {}, "false_acceptance_rate": None,
                "false_blocking_rate": None, "abstention_rate": None, "coverage": None, "selective_accuracy": None,
                "accuracy_denominator": 0}
    matrix = {actual: {pred: 0 for pred in LABELS + ["FAILED"]} for actual in LABELS}
    for case in completed:
        matrix[case["expected_verdict"]][case["predicted_verdict"]] += 1
    for case in eligible:
        if case not in completed:
            matrix[case["expected_verdict"]]["FAILED"] += 1
    per_class = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[actual][label] for actual in LABELS if actual != label)
        fn = sum(matrix[label][pred] for pred in LABELS + ["FAILED"] if pred != label)
        support = sum(matrix[label].values())
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        if not support and not (tp + fp):
            f1 = None
        elif precision is None or recall is None or not (precision + recall):
            f1 = 0.0
        else:
            f1 = 2 * precision * recall / (precision + recall)
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support, "predicted": tp + fp}
    f1s = [x["f1"] for x in per_class.values() if x["f1"] is not None]
    definitive = [c for c in completed if c["predicted_verdict"] in ("ACCEPTED", "BLOCKED")]
    blocked = sum(1 for c in eligible if c["expected_verdict"] == "BLOCKED")
    accepted = sum(1 for c in eligible if c["expected_verdict"] == "ACCEPTED")
    false_accepts = sum(1 for c in completed if c["expected_verdict"] == "BLOCKED" and c["predicted_verdict"] == "ACCEPTED")
    false_blocks = sum(1 for c in completed if c["expected_verdict"] == "ACCEPTED" and c["predicted_verdict"] == "BLOCKED")
    correct = sum(c["expected_verdict"] == c.get("predicted_verdict") for c in completed)
    return {
        "dataset_size": total, "completed": len(completed), "failed": failed, "accuracy_denominator": total,
        "accuracy": correct / total,
        "overall_precision": correct / len(completed) if completed else None,
        "overall_recall": correct / total,
        "macro_f1": sum(f1s) / len(f1s) if f1s else None,
        "per_class": per_class, "confusion_matrix": matrix,
        "false_acceptance_rate": false_accepts / blocked if blocked else None,
        "false_blocking_rate": false_blocks / accepted if accepted else None,
        "abstention_rate": sum(c.get("predicted_verdict") == "NEEDS REVIEW" for c in completed) / total,
        "coverage": len(definitive) / total,
        "selective_accuracy": sum(c["expected_verdict"] == c["predicted_verdict"] for c in definitive) / len(definitive) if definitive else None,
    }


def _gold_docs(case):
    return {Path(str(x.get("source_pdf", ""))).name.casefold() for x in _gold_citations(case) if x.get("source_pdf")}


def _gold_citations(case):
    refs = []
    if case.get("source_pdf"):
        refs.append({"source_pdf": case["source_pdf"], "page": case.get("page"), "evidence": case.get("evidence", "")})
    refs.extend(case.get("additional_sources", []))
    return refs


def _norm(text):
    return " ".join(str(text or "").casefold().split())


def calculate_retrieval_metrics(cases):
    eligible = [c for c in cases if _gold_docs(c)]
    if not eligible:
        return {"eligible_cases": 0, "recall_at_3": None, "recall_at_5": None,
                "mrr": None, "correct_document_retrieval": None, "correct_page_retrieval": None, "citation_correctness": None}
    hits3 = hits5 = doc_hits = page_hits = cite_hits = 0
    reciprocal_ranks = []
    for c in eligible:
        gold_docs = _gold_docs(c)
        evidence = c.get("retrieved_evidence", [])
        match_doc = lambda e: Path(str(e.get("document") or e.get("title") or "")).name.casefold() in gold_docs
        if any(match_doc(e) for e in evidence): doc_hits += 1
        matched = [e for e in evidence if match_doc(e)]
        refs = _gold_citations(c)
        page_match = any(
            str(e.get("page", "")) == str(ref.get("page", "")) and
            Path(str(e.get("document") or e.get("title") or "")).name.casefold() == Path(str(ref.get("source_pdf", ""))).name.casefold()
            for e in evidence for ref in refs
        )
        if page_match: page_hits += 1
        if any(match_doc(e) for e in evidence[:3]): hits3 += 1
        if any(match_doc(e) for e in evidence[:5]): hits5 += 1
        rank = next((i for i, item in enumerate(evidence, 1) if any(
            Path(str(item.get("document") or item.get("title") or "")).name.casefold() == Path(str(ref.get("source_pdf", ""))).name.casefold()
            and str(item.get("page", "")) == str(ref.get("page", "")) for ref in refs)), None)
        reciprocal_ranks.append(1 / rank if rank else 0.0)
        passage_match = any(
            _norm(e.get("text", "")) in _norm(ref.get("evidence", "")) or _norm(ref.get("evidence", "")) in _norm(e.get("text", ""))
            for e in evidence for ref in refs if e.get("text") and
            Path(str(e.get("document") or e.get("title") or "")).name.casefold() == Path(str(ref.get("source_pdf", ""))).name.casefold() and
            str(e.get("page", "")) == str(ref.get("page", ""))
        )
        if page_match and passage_match: cite_hits += 1
    n = len(eligible)
    return {"eligible_cases": n, "recall_at_3": hits3/n, "recall_at_5": hits5/n,
            "mrr": sum(reciprocal_ranks)/n,
            "correct_document_retrieval": doc_hits/n, "correct_page_retrieval": page_hits/n,
            "citation_correctness": cite_hits/n}


def summarize_run(run):
    cases = run.get("cases", [])
    verdict = calculate_metrics(cases)
    retrieval = calculate_retrieval_metrics(cases)
    latencies = [float(c["latency_seconds"]) for c in cases if isinstance(c.get("latency_seconds"), (int, float))]
    sorted_latencies = sorted(latencies)
    p95_latency = sorted_latencies[max(0, int(0.95 * len(sorted_latencies) + 0.999999) - 1)] if sorted_latencies else None
    usage = [c.get("api_usage", {}) for c in cases]
    failures = sum(int(u.get("api_failures", 0)) for u in usage)
    web_failures = sum(int(u.get("web_retrieval_failures", 0)) for u in usage)
    groq_calls = sum(int(u.get("groq_calls", 0)) for u in usage)
    tavily_calls = sum(int(u.get("tavily_calls", 0)) for u in usage)
    webpage_calls = sum(int(u.get("webpage_calls", 0)) for u in usage)
    token_cases = [u for u in usage if u.get("token_usage_available")]
    tokens = (sum(int(u.get("prompt_tokens", 0)) + int(u.get("completion_tokens", 0)) for u in token_cases) if token_cases else None)
    by_group = {}
    for field in ("language", "scheme", "category", "difficulty"):
        groups = sorted({str(c.get(field) or "Unspecified") for c in cases})
        by_group[field] = {}
        for label in groups:
            subset = [c for c in cases if str(c.get(field) or "Unspecified") == label]
            group_metrics = calculate_metrics(subset)
            by_group[field][label] = {"cases": group_metrics["dataset_size"],
                                      "completed": group_metrics["completed"],
                                      "accuracy": group_metrics["accuracy"],
                                      "macro_f1": group_metrics["macro_f1"]}
    return {**verdict, **retrieval, "by_group": by_group,
            "average_latency_seconds": sum(latencies)/len(latencies) if latencies else None,
            "median_latency_seconds": statistics.median(sorted_latencies) if sorted_latencies else None,
            "p95_latency_seconds": p95_latency,
            "execution_seconds": run.get("execution_seconds"),
            "pdf_indexing_seconds": run.get("pdf_indexing_seconds"),
            "api_failures": failures,
            "web_retrieval_failures": web_failures,
            "groq_calls": groq_calls, "tavily_calls": tavily_calls, "webpage_calls": webpage_calls,
            "tokens_used": tokens}


def latest_run(split="dev", kind="claims"):
    if split not in SPLITS or not RESULTS_DIR.exists():
        return None
    prefix = f"{split}_{kind}_"
    paths = sorted(RESULTS_DIR.glob(prefix + "*.json"), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for path in paths:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
    return None
