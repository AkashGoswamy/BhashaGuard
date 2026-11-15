"""CLI for repeatable claim and bilingual benchmark runs.

Credentials are requested through getpass and are never saved in result files.
"""
import argparse
import getpass
import json
import sys

from core.benchmark import run_claim_benchmark
from core.bilingual_evaluation import run_bilingual_benchmark, calculate_bilingual_metrics
from core.evaluation import load_dataset, summarize_run
from core.source_data import load_candidate_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run BhashaGuard's production audit pipeline on a labeled benchmark split.")
    parser.add_argument("--split", choices=("dev", "final"), default="dev", help="Use final only for held-out measurement; do not tune from it.")
    parser.add_argument("--kind", choices=("claims", "bilingual"), default="claims")
    parser.add_argument("--dataset", choices=("imported", "manual"), default="imported", help="Use the supplied candidate JSON files or manually reviewed templates.")
    parser.add_argument("--mode", choices=("provisional", "verified"), default="provisional", help="Imported data supports provisional runs only until source review is recorded.")
    parser.add_argument("--limit", type=int, default=5, help="Run at most this many cases to control API usage; use 0 for all eligible cases.")
    args = parser.parse_args(argv)
    dataset = load_candidate_dataset(args.split, args.kind, args.mode) if args.dataset == "imported" else load_dataset(args.split, args.kind)
    if dataset["errors"]:
        print("Dataset validation issues:")
        for error in dataset["errors"]: print(f"- {error}")
        if not dataset["cases"]: return 2
    if not dataset["cases"]:
        print(f"No eligible {args.mode if args.dataset == 'imported' else 'manual'} cases in {dataset['path']}; no benchmark was run.")
        return 2
    if args.limit < 0:
        parser.error("--limit must be zero or greater.")
    groq_key = getpass.getpass("Groq API key (input hidden): ").strip()
    tavily_key = getpass.getpass("Tavily API key (optional, input hidden): ").strip() if args.kind == "claims" else ""
    if not groq_key:
        print("A Groq API key is required.", file=sys.stderr)
        return 2
    if args.kind == "claims":
        run = run_claim_benchmark(dataset, groq_key, tavily_key, args.split, args.limit or None)
        summary = summarize_run(run)
    else:
        run = run_bilingual_benchmark(dataset, groq_key, args.split, args.limit or None)
        summary = calculate_bilingual_metrics(run["cases"])
    print(json.dumps({"split": args.split, "kind": args.kind, "summary": summary, "result_path": run["result_path"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
