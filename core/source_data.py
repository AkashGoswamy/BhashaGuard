"""Read-only normalization of the supplied candidate benchmark JSON files."""
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "json"
PRIMARY = DATA_DIR / "bhashaguard-eval-dataset.json"
SUPPLEMENTARY = DATA_DIR / "bhashaguard_benchmark_final.json"
SEED = 42
LABELS = {"ACCEPTED", "BLOCKED", "NEEDS REVIEW"}


def _read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_verdict(value):
    value = re.sub(r"\s+", " ", str(value or "").upper().strip())
    aliases = {"REVIEW": "NEEDS REVIEW"}
    value = aliases.get(value, value)
    return value if value in LABELS else None


def _scheme(value):
    text = str(value or "")
    for name in ("ADIP", "PMFME", "PMAY-U 2.0", "SEED", "NHB"):
        if name.casefold() in text.casefold() or (name == "NHB" and "horticulture" in text.casefold()):
            return {"NHB": "NHB Schemes"}.get(name, name)
    return text.strip() or "Unspecified"


def _source_key(row):
    source = row.get("source_filename") or row.get("original_source_filename") or ""
    if isinstance(source, dict):
        source = source.get("english_pdf") or next(iter(source.values()), "")
    if "`" in str(source) or "&" in str(source):
        return ""
    return Path(str(source)).name


def _candidate_records(kind):
    primary = _read(PRIMARY)
    supplementary = _read(SUPPLEMENTARY)
    if not isinstance(primary, dict) or not isinstance(supplementary, dict):
        raise ValueError("Dataset roots must be JSON objects.")
    secondary_rows = supplementary.get("unverified_cases", [])
    if not isinstance(secondary_rows, list):
        raise ValueError("Companion unverified_cases must be an array.")
    supplement_by_id = {str(r.get("id", r.get("pair_id", ""))): r for r in secondary_rows if isinstance(r, dict)}
    if kind == "claims":
        rows = primary.get("verified_benchmark_cases", [])
        record_type = "factual_case"
    elif kind == "bilingual":
        rows = primary.get("verified_bilingual_pairs", [])
        record_type = "bilingual_pair"
    else:
        raise ValueError("kind must be 'claims' or 'bilingual'.")
    if not isinstance(rows, list):
        raise ValueError(f"Primary {kind} records must be a JSON array.")
    normalized = []
    for index, source in enumerate(rows, 1):
        if not isinstance(source, dict):
            normalized.append({"test_id": f"MALFORMED-{kind}-{index}", "expected_verdict": None,
                               "expected_discrepancy": None, "expected_label_valid": False,
                               "scheme": "Unspecified", "verification_status": "UNVERIFIABLE",
                               "record_type": record_type, "data_quality_error": "Record is not a JSON object."})
            continue
        case_id = str(source.get("id") or source.get("pair_id") or "").strip()
        companion = supplement_by_id.get(case_id, {})
        # Companion labels are used only as provisional normalization when the
        # primary label is a non-canonical placeholder. Neither source is treated
        # as verified until a person checks it against the original PDF.
        if kind == "claims":
            expected = _canonical_verdict(source.get("expected_verdict"))
            if not expected:
                expected = _canonical_verdict(companion.get("expected_verdict"))
            atomic = source.get("atomic_claims") or []
            if not isinstance(atomic, list):
                atomic = []
            row = {
                "test_id": case_id, "scheme": _scheme(source.get("scheme")),
                "language": "Hindi" if str(source.get("language", "")).casefold().startswith("hindi") else "English",
                "input_answer": str(source.get("input_text", "")),
                "expected_claim": str(atomic[0] if atomic else source.get("input_text", "")),
                "expected_verdict": expected,
                "expected_verdict_provisional": expected is not None,
                "candidate_expected_verdict": expected,
                "source_pdf": "", "page": None, "original_page_number": "", "evidence": "",
                "candidate_source_pdf": _source_key(source), "candidate_pdf_page_index": source.get("pdf_page_index"),
                "candidate_printed_page": source.get("printed_page", ""), "candidate_evidence": source.get("evidence_quote", ""),
                "source_quote_verified": False,
                "verification_status": "NEEDS_SOURCE_VERIFICATION" if _source_key(source) and source.get("evidence_quote") else "UNVERIFIABLE",
                "expected_explanation": str(source.get("explanation", "")),
                "difficulty": "hard" if "difficult" in str(source.get("difficulty", "")).casefold() else "medium",
                "category": str(source.get("category", "unspecified")),
                "label_provenance": "NotebookLM candidate" if _canonical_verdict(source.get("expected_verdict")) else "Claude provisional normalization",
                "source_verification_note": str(companion.get("verification_note", "")),
            }
            if not row["input_answer"] or not row["expected_claim"]:
                row["data_quality_error"] = "Missing input or atomic claim."
        else:
            expected_raw = source.get("expected_result", companion.get("expected_result", ""))
            expected_raw = str(expected_raw or "").strip().upper()
            expected = True if expected_raw in {"DISCREPANCY", "MEANING DISCREPANCY"} else False if expected_raw in {"EQUIVALENT", "SEMANTICALLY EQUIVALENT"} else None
            sources = source.get("source_filename", {})
            row = {
                "test_id": case_id,
                "english_answer": str(source.get("english_text", "")), "hindi_answer": str(source.get("hindi_text", "")),
                "expected_discrepancy": expected, "category": "other",
                "expected_notes": str(source.get("explanation", source.get("discrepancy_description", ""))),
                "source_files": sources if isinstance(sources, dict) else {"reference": str(sources)},
                "source_page_reference": str(source.get("page_reference", "")),
                "candidate_evidence": str(source.get("evidence_quote", "")),
                "source_quote_verified": False,
                "verification_status": "NEEDS_SOURCE_VERIFICATION" if sources and source.get("evidence_quote") else "UNVERIFIABLE",
                "scheme": _scheme(source.get("scheme")),
            }
        if not row["test_id"]:
            row["data_quality_error"] = "Missing case ID."
        row["primary_claimed_verification_status"] = str(source.get("verification_status", "UNSPECIFIED"))
        row["supplementary_claimed_verification_status"] = str(companion.get("verification_status", "UNMATCHED"))
        row["expected_label_valid"] = row.get("expected_verdict") in LABELS if kind == "claims" else isinstance(row.get("expected_discrepancy"), bool)
        row["record_type"] = record_type
        normalized.append(row)
    return normalized, primary, supplementary


def _group_key(row, kind):
    if kind == "bilingual":
        return row["test_id"]  # keep both sides of a pair together
    source = row.get("candidate_source_pdf") or "no-source"
    page = row.get("candidate_pdf_page_index")
    claim = " ".join(re.findall(r"[\w\u0900-\u097f]+", row.get("expected_claim", "").casefold()))
    claim_hash = hashlib.sha1(claim.encode("utf-8")).hexdigest()[:10]
    if source != "no-source" and page is not None:
        return "|".join((row["scheme"], source, str(page)))
    return "|".join((row["scheme"], source, claim_hash))


def _assign_splits(rows, kind):
    by_scheme = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_scheme[row.get("scheme", "Unspecified")][_group_key(row, kind)].append(row)
    assignment = {}
    if kind == "bilingual":
        groups = [(scheme, key, records) for scheme, values in by_scheme.items() for key, records in values.items()]
        rng = random.Random(SEED + sum(ord(ch) for ch in kind))
        rng.shuffle(groups)
        target = round(sum(len(records) for _, _, records in groups) * 0.30)
        selected, count = set(), 0
        for scheme, key, records in groups:
            if count < target:
                selected.add((scheme, key)); count += len(records)
        for scheme, key, records in groups:
            for row in records: assignment[row["test_id"]] = "final" if (scheme, key) in selected else "dev"
        return assignment
    for scheme, groups in sorted(by_scheme.items()):
        keys = list(groups)
        rng = random.Random(SEED + sum(ord(ch) for ch in scheme + kind))
        rng.shuffle(keys)
        target = round(sum(len(groups[k]) for k in keys) * 0.30)
        selected, count = set(), 0
        for key in keys:
            if len(keys) > 1 and count < target and (not selected or abs((count + len(groups[key])) - target) <= abs(count - target)):
                selected.add(key); count += len(groups[key])
        for key in keys:
            split = "final" if key in selected else "dev"
            for row in groups[key]: assignment[row["test_id"]] = split
    return assignment


def load_candidate_dataset(split="dev", kind="claims", mode="provisional"):
    if split not in ("dev", "final"):
        return {"cases": [], "errors": [f"Unknown split: {split}"], "path": str(PRIMARY)}
    if mode not in ("provisional", "verified"):
        return {"cases": [], "errors": [f"Unknown evaluation mode: {mode}"], "path": str(PRIMARY)}
    try:
        rows, primary, supplementary = _candidate_records(kind)
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"cases": [], "errors": [f"Could not normalize source dataset ({exc.__class__.__name__})."],
                "path": str(PRIMARY), "kind": kind, "evaluation_mode": mode}
    assignments = _assign_splits(rows, kind)
    selected = []
    for row in rows:
        if assignments[row["test_id"]] != split:
            continue
        if mode == "verified" and row["verification_status"] != "VERIFIED":
            continue
        if not row["expected_label_valid"]:
            continue
        selected.append(row)
    version = f"{primary.get('version', 'unknown')}|{supplementary.get('version', 'unknown')}"
    assignment_hash = hashlib.sha256(json.dumps(assignments, sort_keys=True).encode()).hexdigest()
    return {"cases": selected, "errors": [], "path": str(PRIMARY), "kind": kind,
            "evaluation_mode": mode, "dataset_version": version, "split_assignments": assignments,
            "split_assignment_hash": assignment_hash, "seed": SEED, "split_fraction": "70/30 grouped by scheme/source/page/claim",
            "all_records": rows}


def candidate_overview():
    try:
        factual, primary, supplementary = _candidate_records("claims")
        bilingual, _, _ = _candidate_records("bilingual")
    except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
        return {"error": f"Could not read candidate datasets ({exc.__class__.__name__}).",
                "primary_file": PRIMARY.name, "supplementary_file": SUPPLEMENTARY.name,
                "dataset_version": "unavailable", "factual_total": 0, "bilingual_total": 0,
                "total": 0, "factual_labeled": 0, "factual_unlabeled": 0,
                "verified": 0, "needs_source_verification": 0, "unverifiable": 0, "rejected": 0,
                "language": {}, "scheme": {}, "verdict": {}, "difficulty": {}, "category": {},
                "invalid_or_missing_labels": 0, "duplicate_ids_primary": 0, "duplicate_ids_supplementary": 0,
                "unmatched_primary_ids": 0, "supplementary_label_disagreements": 0,
                "verification_status_conflicts": 0,
                "source_quotes": 0, "records_with_source_file": 0, "records_with_page_reference": 0,
                "label_provenance": {}, "split_seed": SEED, "provisional_only": True}
    all_rows = factual + bilingual
    raw_primary = primary.get("verified_benchmark_cases", []) + primary.get("verified_bilingual_pairs", [])
    primary_ids = [str(x.get("id") or x.get("pair_id") or "") for x in raw_primary if isinstance(x, dict)]
    secondary_rows = [x for x in supplementary.get("unverified_cases", []) if isinstance(x, dict)]
    secondary_ids = [str(x.get("id") or x.get("pair_id") or "") for x in secondary_rows]
    secondary_by_id = {str(x.get("id") or x.get("pair_id") or ""): x for x in secondary_rows}
    label_disagreements = 0
    for row in primary.get("verified_benchmark_cases", []):
        if not isinstance(row, dict):
            continue
        case_id = str(row.get("id", ""))
        primary_label = _canonical_verdict(row.get("expected_verdict"))
        secondary_label = _canonical_verdict(secondary_by_id.get(case_id, {}).get("expected_verdict"))
        if primary_label and secondary_label and primary_label != secondary_label:
            label_disagreements += 1
    def counts(rows, field):
        return dict(sorted(Counter(str(r.get(field, "Unspecified")) for r in rows).items()))
    valid_factual = [r for r in factual if r.get("expected_label_valid")]
    return {
        "primary_file": PRIMARY.name, "supplementary_file": SUPPLEMENTARY.name,
        "dataset_version": f"{primary.get('version', 'unknown')}|{supplementary.get('version', 'unknown')}",
        "factual_total": len(factual), "bilingual_total": len(bilingual), "total": len(all_rows),
        "factual_labeled": len(valid_factual), "factual_unlabeled": len(factual)-len(valid_factual),
        "verified": sum(r["verification_status"] == "VERIFIED" for r in all_rows),
        "needs_source_verification": sum(r["verification_status"] == "NEEDS_SOURCE_VERIFICATION" for r in all_rows),
        "unverifiable": sum(r["verification_status"] == "UNVERIFIABLE" for r in all_rows),
        "rejected": sum(r["verification_status"] == "REJECTED" for r in all_rows),
        "language": counts(factual, "language"), "scheme": counts(factual, "scheme"),
        "verdict": counts(valid_factual, "expected_verdict"),
        "difficulty": counts(factual, "difficulty"), "category": counts(factual, "category"),
        "invalid_or_missing_labels": len(factual)-len(valid_factual),
        "duplicate_ids_primary": len(primary_ids) - len(set(primary_ids)),
        "duplicate_ids_supplementary": len(secondary_ids) - len(set(secondary_ids)),
        "unmatched_primary_ids": len(set(primary_ids) - set(secondary_ids)),
        "supplementary_label_disagreements": label_disagreements,
        "verification_status_conflicts": sum(
            r.get("primary_claimed_verification_status") != r.get("supplementary_claimed_verification_status")
            for r in all_rows if r.get("supplementary_claimed_verification_status") != "UNMATCHED"),
        "source_quotes": sum(bool(r.get("candidate_evidence")) for r in factual),
        "records_with_source_file": sum(bool(r.get("candidate_source_pdf")) for r in factual),
        "records_with_page_reference": sum(r.get("candidate_pdf_page_index") is not None for r in factual),
        "label_provenance": counts(factual, "label_provenance"),
        "factual_split_assignments": _assign_splits(factual, "claims"),
        "bilingual_split_assignments": _assign_splits(bilingual, "bilingual"),
        "split_seed": SEED,
        "provisional_only": True,
    }
