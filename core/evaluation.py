import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = ROOT / "data" / "benchmark"
LABELS = ["ACCEPTED", "BLOCKED", "NEEDS REVIEW"]


def load_benchmarks():
    BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)
    cases = []
    for path in BENCHMARK_DIR.iterdir():
        try:
            if path.suffix.lower() == ".json":
                content = json.loads(path.read_text(encoding="utf-8-sig"))
                cases.extend(content if isinstance(content, list) else content.get("cases", []))
            elif path.suffix.lower() == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as f: cases.extend(csv.DictReader(f))
        except (OSError, json.JSONDecodeError, TypeError):
            continue
    valid = []
    for case in cases:
        if not isinstance(case, dict): continue
        expected = str(case.get("expected_verdict", case.get("label", ""))).upper().strip()
        predicted = str(case.get("predicted_verdict", "")).upper().strip()
        if expected in LABELS and predicted in LABELS: valid.append({**case, "expected_verdict": expected, "predicted_verdict": predicted})
    return valid


def calculate_metrics(cases):
    if not cases: return {}
    matrix = {x: {y: 0 for y in LABELS} for x in LABELS}
    lang = {}
    for c in cases:
        actual, pred = c["expected_verdict"], c["predicted_verdict"]
        matrix[actual][pred] += 1
        language = str(c.get("language", "")).strip()
        if language:
            entry = lang.setdefault(language, [0, 0]); entry[1] += 1; entry[0] += actual == pred
    n = len(cases); correct = sum(matrix[x][x] for x in LABELS)
    per = {}
    for label in LABELS:
        tp = matrix[label][label]
        fp = sum(matrix[a][label] for a in LABELS if a != label)
        fn = sum(matrix[label][p] for p in LABELS if p != label)
        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        per[label] = {"precision": precision, "recall": recall, "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0}
    blocked_den = sum(matrix[a]["BLOCKED"] for a in LABELS)
    disc_cases = [c for c in cases if str(c.get("expected_discrepancy", "")).lower() in ("true", "false", "1", "0") and str(c.get("predicted_discrepancy", "")).lower() in ("true", "false", "1", "0")]
    discrepancy = None
    if disc_cases:
        positive = lambda c, k: str(c[k]).lower() in ("true", "1")
        tp = sum(positive(c,"expected_discrepancy") and positive(c,"predicted_discrepancy") for c in disc_cases)
        fp = sum(not positive(c,"expected_discrepancy") and positive(c,"predicted_discrepancy") for c in disc_cases)
        fn = sum(positive(c,"expected_discrepancy") and not positive(c,"predicted_discrepancy") for c in disc_cases)
        discrepancy = {"precision": tp/(tp+fp) if tp+fp else 0, "recall": tp/(tp+fn) if tp+fn else 0, "labelled_cases": len(disc_cases)}
    return {"accuracy": correct/n, "per_verdict": per, "blocked_precision": per["BLOCKED"]["precision"] if blocked_den else None, "confusion_matrix": matrix, "language_accuracy": {k: {"accuracy": v[0]/v[1], "cases": v[1]} for k,v in lang.items()}, "discrepancy": discrepancy}
