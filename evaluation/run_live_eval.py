
import json
import urllib.error
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.json"
RESULTS_PATH = BASE_DIR / "results.json"
API_URL = "http://127.0.0.1:8000/ask"


def main():
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    results = []

    for case in dataset:
        payload = json.dumps(
            {"question": case["question"]}
        ).encode("utf-8")

        request = urllib.request.Request(
            API_URL,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                result = json.loads(response.read().decode("utf-8"))

            results.append(
                {
                    "id": case["id"],
                    "answer": result.get("answer", ""),
                    "sources": sorted(
                        {
                            citation.get("source", "")
                            for citation in result.get("citations", [])
                            if citation.get("source")
                        }
                    ),
                }
            )
            print(f"SUCCESS: {case['id']}")

        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            print(f"FAILED: {case['id']} â€” {exc}")
            raise

    RESULTS_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nSaved {len(results)} real API results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

