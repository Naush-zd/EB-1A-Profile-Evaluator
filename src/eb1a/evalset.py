import json
from pathlib import Path
from typing import Any
from .criteria import CriterionId


def load_cases(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory)
    cases = []
    for path in sorted(root.glob("*.json")):
        cases.append(json.loads(path.read_text(encoding="utf-8")))
    if not cases:
        raise ValueError(f"no evaluation cases found in {root}")
    for case in cases:
        if set(case.get("gold", {})) != {c.value for c in CriterionId}:
            raise ValueError(f"case {case.get('profile_id')} must label all 10 criteria")
    return cases


def compare_case(case: dict[str, Any], actual: dict[str, str]) -> dict[str, Any]:
    gold = case["gold"]
    rows = []
    for criterion in CriterionId:
        expected, predicted = gold[criterion.value], actual.get(criterion.value, "missing")
        rows.append({"criterion_id": criterion.value, "expected": expected, "actual": predicted, "correct": expected == predicted})
    return {"profile_id": case["profile_id"], "results": rows}


def summarize(comparisons: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for item in comparisons for row in item["results"]]
    correct = sum(row["correct"] for row in rows)
    per_criterion = {}
    for criterion in CriterionId:
        subset = [row for row in rows if row["criterion_id"] == criterion.value]
        per_criterion[criterion.value] = sum(row["correct"] for row in subset) / len(subset) if subset else 0.0
    return {
        "profiles": len(comparisons),
        "labels": len(rows),
        "accuracy": correct / len(rows) if rows else 0.0,
        "per_criterion_accuracy": per_criterion,
        "failures": [row for row in rows if not row["correct"]],
    }
