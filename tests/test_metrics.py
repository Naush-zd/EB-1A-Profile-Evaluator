from eb1a.eval_metrics import classification_metrics


def test_classification_metrics_confusion_matrix():
    result = classification_metrics([
        {"expected": "strong", "actual": "strong", "correct": True},
        {"expected": "strong", "actual": "partial", "correct": False},
    ])
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"]["strong"]["partial"] == 1
