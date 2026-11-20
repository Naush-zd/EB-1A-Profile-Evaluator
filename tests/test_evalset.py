import json
from pathlib import Path
from eb1a.evalset import load_cases, summarize


def test_eval_cases_cover_all_criteria():
    cases = load_cases(Path("data/eval"))
    assert len(cases) >= 5
    assert all(len(case["gold"]) == 10 for case in cases)


def test_summary_accuracy():
    result = summarize([{"profile_id": "x", "results": [{"criterion_id": "awards", "correct": True}]}])
    assert result["accuracy"] == 1.0


def test_run_evaluation_metrics(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    from eb1a.eval_runner import run_evaluation
    # run with demo source chunks
    res = run_evaluation("data/eval", "data/processed/chunks.jsonl", fast=True)
    assert "accuracy" in res
    assert "citation_validity_rate" in res
    assert "hallucinated_evidence_rate" in res
    assert res["citation_validity_rate"] >= 0.0
    assert res["hallucinated_evidence_rate"] >= 0.0
