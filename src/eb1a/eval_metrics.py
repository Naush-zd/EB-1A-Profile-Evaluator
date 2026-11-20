from collections import Counter


def classification_metrics(rows: list[dict]) -> dict:
    labels = sorted({label for row in rows for label in (row["expected"], row["actual"])})
    matrix = {expected: {actual: 0 for actual in labels} for expected in labels}
    for row in rows:
        matrix[row["expected"]][row["actual"]] += 1
    return {"count": len(rows), "accuracy": sum(row["correct"] for row in rows) / len(rows) if rows else 0.0, "confusion_matrix": matrix}
