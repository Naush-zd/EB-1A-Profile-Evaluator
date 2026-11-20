from eb1a.eval_runner import run_evaluation


def test_guardrail_report_is_present(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("EB1A_DISABLE_DOTENV", "1")
    result = run_evaluation("data/eval", "data/processed/uscis_chunks.jsonl", fast=True, retrieval_mode="lexical")
    assert result["guardrails"]["prompt_injection_cases"] >= 1
    assert result["guardrails"]["prompt_injection_detection_rate"] == 1.0
