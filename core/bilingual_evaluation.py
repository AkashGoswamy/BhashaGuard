"""Separate English-Hindi consistency benchmark; consistency is not truth."""
import json
import hashlib
import time
from datetime import datetime, timezone

from core.cross_language import compare_answers
from core.evaluation import RESULTS_DIR
from core.llm import track_api_usage
import core.llm as llm


def calculate_bilingual_metrics(cases):
    completed = [c for c in cases if isinstance(c.get("predicted_discrepancy"), bool) and isinstance(c.get("expected_discrepancy"), bool)]
    if not completed:
        return {"dataset_size": len(cases), "completed": 0, "failed": len(cases), "precision": None,
                "recall": None, "f1": None, "accuracy": 0.0 if cases else None,
                "coverage": 0.0 if cases else None, "selective_accuracy": None, "per_class": {}, "by_category": {}}
    tp = sum(c["expected_discrepancy"] and c["predicted_discrepancy"] for c in completed)
    fp = sum(not c["expected_discrepancy"] and c["predicted_discrepancy"] for c in completed)
    fn = sum(c["expected_discrepancy"] and not c["predicted_discrepancy"] for c in completed)
    fn += sum(bool(c.get("expected_discrepancy")) for c in cases if not isinstance(c.get("predicted_discrepancy"), bool) and isinstance(c.get("expected_discrepancy"), bool))
    tn = sum(not c["expected_discrepancy"] and not c["predicted_discrepancy"] for c in completed)
    precision = tp/(tp+fp) if tp+fp else None
    recall = tp/(tp+fn) if tp+fn else None
    f1 = 2*precision*recall/(precision+recall) if precision is not None and recall is not None and precision+recall else (0.0 if tp+fn else None)
    eq_precision = tn/(tn+fn) if tn+fn else None
    eq_recall = tn/(tn+fp) if tn+fp else None
    eq_f1 = 2*eq_precision*eq_recall/(eq_precision+eq_recall) if eq_precision is not None and eq_recall is not None and eq_precision+eq_recall else (0.0 if tn+fp else None)
    categories = {}
    for category in sorted({str(c.get("category", "unspecified")) for c in completed}):
        subset = [c for c in completed if str(c.get("category", "unspecified")) == category]
        correct = sum(c["expected_discrepancy"] == c["predicted_discrepancy"] for c in subset)
        all_subset = [c for c in cases if str(c.get("category", "unspecified")) == category]
        categories[category] = {"cases": len(all_subset), "completed": len(subset), "correct": correct, "accuracy": correct/len(all_subset)}
    decisive = [c for c in completed if c.get("predicted_discrepancy") in (True, False)]
    correct_decisive = sum(c["expected_discrepancy"] == c["predicted_discrepancy"] for c in decisive)
    return {"dataset_size": len(cases), "completed": len(completed), "failed": len(cases)-len(completed),
            "precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp+tn)/len(cases) if cases else None,
            "coverage": len(decisive)/len(cases) if cases else None,
            "selective_accuracy": correct_decisive/len(decisive) if decisive else None,
            "per_class": {"DISCREPANCY": {"precision": precision, "recall": recall, "f1": f1},
                          "EQUIVALENT": {"precision": eq_precision, "recall": eq_recall, "f1": eq_f1}},
            "confusion_matrix": {"true_discrepancy": {"predicted_yes": tp, "predicted_no": fn},
                                 "no_discrepancy": {"predicted_yes": fp, "predicted_no": tn}}, "by_category": categories}


def run_bilingual_benchmark(dataset, groq_key, split="dev", case_limit=None):
    if not groq_key:
        raise ValueError("A Groq API key is required to run the bilingual benchmark.")
    if split not in ("dev", "final"):
        raise ValueError("split must be 'dev' or 'final'.")
    run_start = time.perf_counter()
    selected_cases = dataset.get("cases", [])[:case_limit] if case_limit else dataset.get("cases", [])
    results = []
    for case in selected_cases:
        start = time.perf_counter()
        row = {**case, "predicted_discrepancy": None, "detected_discrepancies": [], "status": "failed"}
        with track_api_usage() as usage:
            result, error = compare_answers(case["english_answer"], case["hindi_answer"], groq_key, case.get("input_type", "English and Hindi answers"))
            if error:
                row["error"] = error
            else:
                row["detected_discrepancies"] = result.get("discrepancies", [])
                row["predicted_discrepancy"] = bool(row["detected_discrepancies"])
                row["english_claims"] = result.get("english_claims", [])
                row["hindi_claims"] = result.get("hindi_claims", [])
                row["claim_extraction_errors"] = result.get("claim_extraction_errors", [])
                row["status"] = "completed"
        row["api_usage"] = dict(usage)
        row["latency_seconds"] = round(time.perf_counter()-start, 3)
        results.append(row)
    dataset_hash = hashlib.sha256(json.dumps(dataset.get("cases", []), ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    run = {"run_id": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"),
           "created_at": datetime.now(timezone.utc).isoformat(), "kind": "bilingual", "split": split,
           "dataset_path": dataset.get("path", ""), "dataset_size": len(selected_cases),
           "dataset_total_eligible": len(dataset.get("cases", [])),
           "dataset_hash": dataset_hash,
           "dataset_validation_errors": dataset.get("errors", []), "cases": results}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{split}_bilingual_{run['run_id']}.json"
    run.update({"execution_seconds": round(time.perf_counter()-run_start, 3),
                "dataset_version": dataset.get("dataset_version", "manual-template"),
                "evaluation_mode": dataset.get("evaluation_mode", "verified/manual"),
                "split_seed": dataset.get("seed"), "split_fraction": dataset.get("split_fraction"),
                "split_assignment_hash": dataset.get("split_assignment_hash"),
                "split_assignments": dataset.get("split_assignments", {}),
                "model_id": llm.MODEL, "prompt_version": "core-prompts-v1"})
    out.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")
    run["result_path"] = str(out)
    return run
