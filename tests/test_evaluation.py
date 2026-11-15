import unittest
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from core.bilingual_evaluation import calculate_bilingual_metrics
from core.evaluation import calculate_metrics, calculate_retrieval_metrics, load_dataset
from core.llm import record_api_call, track_api_usage
from core.source_data import candidate_overview, load_candidate_dataset
from core.retrieval import _rank_chunks
from core.benchmark import run_claim_benchmark, _matching_claim
from core.web_search import _relevant_passage


class EvaluationTests(unittest.TestCase):
    def test_empty_templates_are_not_metrics(self):
        for split in ("dev", "final"):
            self.assertEqual(load_dataset(split, "claims")["cases"], [])
            self.assertEqual(load_dataset(split, "bilingual")["cases"], [])
        metrics = calculate_metrics([])
        self.assertIsNone(metrics["accuracy"])
        self.assertIsNone(metrics["coverage"])

    def test_verdict_metrics_keep_failed_cases_visible(self):
        cases = [
            {"expected_verdict": "ACCEPTED", "predicted_verdict": "ACCEPTED", "status": "completed"},
            {"expected_verdict": "BLOCKED", "predicted_verdict": "ACCEPTED", "status": "completed"},
            {"expected_verdict": "NEEDS REVIEW", "predicted_verdict": "BLOCKED", "status": "completed"},
            {"expected_verdict": "BLOCKED", "predicted_verdict": "NEEDS REVIEW", "status": "completed"},
            {"expected_verdict": "ACCEPTED", "predicted_verdict": "NEEDS REVIEW", "status": "failed"},
        ]
        result = calculate_metrics(cases)
        self.assertEqual(result["dataset_size"], 5)
        self.assertEqual(result["completed"], 4)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["accuracy"], 0.2)
        self.assertAlmostEqual(result["false_acceptance_rate"], 0.5)
        self.assertEqual(result["coverage"], 0.6)
        self.assertEqual(result["abstention_rate"], 0.2)
        self.assertAlmostEqual(result["selective_accuracy"], 1/3)

    def test_imported_data_remains_provisional_and_splits_are_stable(self):
        overview = candidate_overview()
        self.assertEqual(overview["factual_total"], 50)
        self.assertEqual(overview["bilingual_total"], 10)
        self.assertEqual(overview["verified"], 0)
        self.assertEqual(overview["needs_source_verification"] + overview["unverifiable"], 60)
        self.assertEqual(overview["verdict"], {"ACCEPTED": 15, "BLOCKED": 25, "NEEDS REVIEW": 10})
        dev = load_candidate_dataset("dev", "claims", "provisional")
        final = load_candidate_dataset("final", "claims", "provisional")
        dev_ids = {x["test_id"] for x in dev["cases"]}
        final_ids = {x["test_id"] for x in final["cases"]}
        self.assertFalse(dev_ids & final_ids)
        self.assertEqual(dev_ids | final_ids, {x["test_id"] for x in dev["all_records"]})
        self.assertEqual(len(dev_ids), 33)
        self.assertEqual(len(final_ids), 17)
        self.assertEqual(load_candidate_dataset("dev", "claims", "verified")["cases"], [])
        manual_review_row = next(x for x in dev["all_records"] if x["test_id"] == "BG-007")
        self.assertEqual(manual_review_row["expected_verdict"], "NEEDS REVIEW")
        self.assertEqual(manual_review_row["label_provenance"], "Claude provisional normalization")
        self.assertEqual(manual_review_row["verification_status"], "NEEDS_SOURCE_VERIFICATION")
        bilingual_dev = load_candidate_dataset("dev", "bilingual", "provisional")
        bilingual_final = load_candidate_dataset("final", "bilingual", "provisional")
        self.assertEqual(len(bilingual_dev["cases"]), 7)
        self.assertEqual(len(bilingual_final["cases"]), 3)
        self.assertFalse({x["test_id"] for x in bilingual_dev["cases"]} & {x["test_id"] for x in bilingual_final["cases"]})

    def test_failed_prediction_counts_as_incorrect_and_is_not_coverage(self):
        result = calculate_metrics([
            {"expected_verdict": "ACCEPTED", "predicted_verdict": None, "status": "failed"},
            {"expected_verdict": "BLOCKED", "predicted_verdict": "BLOCKED", "status": "completed"},
        ])
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["failed"], 1)
        self.assertEqual(result["coverage"], 0.5)

    def test_macro_f1_keeps_supported_classes_with_no_predictions(self):
        result = calculate_metrics([
            {"expected_verdict": "ACCEPTED", "predicted_verdict": "BLOCKED", "status": "completed"},
            {"expected_verdict": "BLOCKED", "predicted_verdict": "BLOCKED", "status": "completed"},
        ])
        self.assertEqual(result["per_class"]["ACCEPTED"]["f1"], 0.0)
        self.assertAlmostEqual(result["macro_f1"], 1/3)

    def test_retrieval_metrics_use_only_source_labeled_cases(self):
        cases = [
            {"source_pdf": "scheme.pdf", "page": 4, "evidence": "Exact source passage", "retrieved_evidence": [
                {"document": "scheme.pdf", "page": 4, "text": "Exact source passage and nearby context."}]},
            {"source_pdf": "", "retrieved_evidence": []},
        ]
        result = calculate_retrieval_metrics(cases)
        self.assertEqual(result["eligible_cases"], 1)
        self.assertEqual(result["recall_at_3"], 1)
        self.assertEqual(result["correct_page_retrieval"], 1)
        self.assertEqual(result["citation_correctness"], 1)
        self.assertIsNone(calculate_retrieval_metrics([cases[1]])["recall_at_5"])

    def test_bilingual_detection_metrics(self):
        result = calculate_bilingual_metrics([
            {"expected_discrepancy": True, "predicted_discrepancy": True, "category": "number"},
            {"expected_discrepancy": True, "predicted_discrepancy": False, "category": "negation"},
            {"expected_discrepancy": False, "predicted_discrepancy": False, "category": "equivalent"},
        ])
        self.assertEqual(result["completed"], 3)
        self.assertEqual(result["precision"], 1)
        self.assertEqual(result["recall"], 0.5)
        self.assertIn("number", result["by_category"])

    def test_usage_tracker_counts_calls_and_reported_tokens(self):
        with track_api_usage() as usage:
            record_api_call("groq")
            record_api_call("tavily", failed=True)
            record_api_call("webpage", failed=True)
            record_api_call("tokens", prompt_tokens=13, completion_tokens=5)
        self.assertEqual(usage["groq_calls"], 1)
        self.assertEqual(usage["tavily_calls"], 1)
        self.assertEqual(usage["api_failures"], 2)
        self.assertEqual(usage["webpage_calls"], 1)
        self.assertEqual(usage["web_retrieval_failures"], 1)
        self.assertEqual(usage["prompt_tokens"] + usage["completion_tokens"], 18)
        self.assertTrue(usage["token_usage_available"])

    def test_hybrid_lexical_retrieval_ranks_exact_numbers_and_hindi(self):
        chunks = [
            {"document": "scheme.pdf", "page": 2, "language": "English", "text": "Income threshold is 2.5 lakh per year for this beneficiary."},
            {"document": "scheme.pdf", "page": 3, "language": "English", "text": "Applicants should submit the required form."},
            {"document": "scheme-hi.pdf", "page": 4, "language": "Hindi", "text": "लाभार्थी की वार्षिक आय 2.5 लाख से कम होनी चाहिए।"},
        ]
        english = _rank_chunks("income threshold 2.5 lakh", chunks, "English")
        self.assertEqual(english[0]["page"], 2)
        hindi = _rank_chunks("लाभार्थी की वार्षिक आय 2.5 लाख", chunks, "Hindi")
        self.assertEqual(hindi[0]["document"], "scheme-hi.pdf")

    def test_empty_benchmark_run_persists_metadata_without_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("core.benchmark.RESULTS_DIR", Path(directory)):
                run = run_claim_benchmark({"cases": [], "path": "empty.json"}, "test-key-must-not-be-saved")
                saved = Path(run["result_path"]).read_text(encoding="utf-8")
                parsed = json.loads(saved)
        self.assertEqual(parsed["cases"], [])
        self.assertIn("execution_seconds", parsed)
        self.assertNotIn("test-key-must-not-be-saved", saved)

    def test_claim_alignment_returns_score_then_claim(self):
        score, claim = _matching_claim("Annual income must not exceed 2.5 lakh", ["Annual income must not exceed 2.5 lakh per year."])
        self.assertGreater(score, 0.5)
        self.assertIn("Annual income", claim)

    def test_web_evidence_requires_relevant_page_passage_and_numbers(self):
        self.assertEqual(_relevant_passage("ADIP Scheme | Government of India | Search | Menu", "ADIP eligibility family income 30,000"), "")
        passage = _relevant_passage("ADIP limits family income to 30,000 per month. Applicants must meet other conditions.", "ADIP family income 30,000")
        self.assertIn("30,000", passage)


if __name__ == "__main__":
    unittest.main()
