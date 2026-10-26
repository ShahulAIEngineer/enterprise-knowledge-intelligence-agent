
"""Offline baseline evaluation for the RAG application.

Run:
    python evaluation/evaluate.py evaluation/results.json

Checks answer keywords, required source citations, and unsupported-question
refusals. This is a lightweight baseline, not a full semantic evaluation.
"""

import json
import re
import sys
import unicodedata
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"


def normalize(text: str) -> str:
    """Normalize Unicode punctuation and whitespace for robust comparisons."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
    )
    return re.sub(r"\s+", " ", text).strip()


def is_refusal(answer: str) -> bool:
    """Recognize common refusals when evidence is insufficient."""
    text = normalize(answer)
    refusal_phrases = (
        "could not find supporting information",
        "cannot answer that question from the provided documents",
        "can't answer that question from the provided documents",
        "do not contain any information",
        "does not contain enough information",
        "not enough information",
        "cannot answer from the available documents",
        "can't answer from the available documents",
        "context does not support",
    )
    return any(phrase in text for phrase in refusal_phrases)


def evaluate(dataset_path: Path, results_path: Path) -> dict:
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    result_items = json.loads(results_path.read_text(encoding="utf-8"))
    results = {item["id"]: item for item in result_items}

    details = []
    answer_checks = 0
    source_checks = 0
    evaluated = 0

    for case in dataset:
        result = results.get(case["id"])
        if result is None:
            details.append({
                "id": case["id"],
                "status": "missing_result",
                "answer_ok": False,
                "source_ok": False,
            })
            continue

        evaluated += 1
        answer = str(result.get("answer", ""))
        normalized_answer = normalize(answer)
        sources = result.get("sources", [])
        required_source = case.get("required_source")
        expected_terms = case.get("expected_answer_contains", [])

        if case["id"] == "rag-003" or required_source is None:
            # Unsupported questions must be refused, not answered with guesses.
            answer_ok = is_refusal(answer)
        else:
            # All expected terms must be present after Unicode normalization.
            answer_ok = all(
                normalize(term) in normalized_answer
                for term in expected_terms
            )

        source_ok = (
            required_source in sources
            if required_source is not None
            else len(sources) == 0
        )

        answer_checks += int(answer_ok)
        source_checks += int(source_ok)

        details.append({
            "id": case["id"],
            "answer_ok": answer_ok,
            "source_ok": source_ok,
            "expected_source": required_source,
            "actual_sources": sources,
            "answer_preview": answer[:180],
        })

    return {
        "dataset_cases": len(dataset),
        "results_found": len(results),
        "results_evaluated": evaluated,
        "answer_match_rate": round(answer_checks / evaluated, 3)
        if evaluated else 0.0,
        "source_match_rate": round(source_checks / evaluated, 3)
        if evaluated else 0.0,
        "case_details": details,
        "note": (
            "Baseline checks only; use semantic relevance, groundedness, "
            "and adversarial tests for a fuller evaluation."
        ),
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python evaluation/evaluate.py evaluation/results.json"
        )

    results_path = Path(sys.argv[1])
    if not results_path.is_absolute():
        results_path = Path.cwd() / results_path

    if not DATASET_PATH.exists():
        raise SystemExit(f"Dataset not found: {DATASET_PATH}")
    if not results_path.exists():
        raise SystemExit(f"Results file not found: {results_path}")

    print(json.dumps(evaluate(DATASET_PATH, results_path), indent=2))
