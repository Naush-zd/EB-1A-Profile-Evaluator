# EB-1A Profile Evaluator

An evaluation-first RAG prototype that maps a synthetic candidate profile to the 10 EB-1A evidentiary criteria using USCIS Policy Manual guidance.

## Scope

This is a decision-support prototype, not legal advice and not a prediction of a USCIS outcome. It distinguishes:

1. **Criterion evidence**: whether the supplied profile appears to support each regulatory criterion.
2. **Overall merits**: a separate, conservative readiness heuristic that must not be confused with satisfying the initial three-criterion threshold.

The source of truth is USCIS Policy Manual, Volume 6, Part F, Chapter 2. The official USCIS page and updates should be re-fetched and fingerprinted during ingestion so the corpus is reproducible.

## Architecture

```text
PDF/TXT profile
    -> extraction + normalization + prompt-injection isolation
    -> evidence spans with stable IDs
    -> criterion-specific query builder
    -> hybrid retrieval (BM25 + vector) + optional reranker
    -> structured LLM assessment
    -> Pydantic validation + citation/evidence validators
    -> JSON / CLI / API / Streamlit UI
```

## Repository layout

```text
src/eb1a/
  criteria.py       # stable criterion IDs and retrieval queries
  schemas.py        # API/output contract
  ingest.py         # source download, cleaning, chunking, fingerprints
  retrieve.py       # hybrid retrieval interface
  evaluate.py       # model orchestration and validators
  cli.py            # local entry point
data/
  raw/              # downloaded official source, ignored by git
  processed/        # chunks + metadata, ignored by git
  eval/             # synthetic profiles and gold labels
tests/              # unit and contract tests
```

## Implementation order

1. Freeze the output schema and 10 criterion IDs.
2. Ingest the official USCIS chapter; preserve URL, section, page/heading, retrieval text, and content hash for every chunk.
3. Add deterministic profile extraction with evidence span IDs.
4. Implement BM25 and Sentence Transformer retrieval, then compare lexical, semantic, and hybrid retrieval metrics.
5. Add structured assessment with a strict “not found in profile” rule.
6. Build 5–8 synthetic profiles and hand-label every criterion.
7. Measure rating accuracy, citation validity, evidence grounding, and hallucinated-evidence rate.
8. Expose the pipeline through CLI first, then add FastAPI or Streamlit.

## Key decisions

- **Pydantic JSON contract** over free-form prose.
- **Stable chunk IDs and evidence span IDs** so every claim is auditable.
- **No unsupported inference**: missing resume facts produce `insufficient_info`, never a guessed rating.
- **Hybrid retrieval** uses BM25 plus Sentence Transformers embeddings. The deterministic hash-vector implementation is retained only as an explicitly named offline regression baseline.
- **Synthetic profiles only** in the repository.
- **Versioned corpus**: store source URL, retrieval date, content hash, and policy revision.
- **Local semantic retrieval**: Sentence Transformers with a swappable embedding model.
- **Optional neural reranking**: A Sentence Transformers cross-encoder can rerank the hybrid top-k candidates when `RERANKER_MODEL` is configured.

Install semantic retrieval with `python -m pip install -e ".[embeddings]"`. The default model is `all-MiniLM-L6-v2`; the first semantic query downloads it into the local model cache. Production retrieval requires the real embedding model and persistent index. Use `--hash-baseline` only for isolated regression tests.

For the optional reranking bonus, set `RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` and run `python -m eb1a retrieve data/processed/uscis_chunks.jsonl "awards for excellence" --hybrid --index data/processed/embeddings.json --rerank`. The cross-encoder reranks retrieved candidates; it does not replace BM25 or embeddings.

## Run target

The first milestone is a deterministic CLI that accepts `.txt` and `.pdf`, emits validated JSON, and runs without a UI:

```powershell
python -m eb1a assess data/eval/profile_awards.txt --corpus data/processed/uscis_chunks.jsonl --index data/processed/embeddings.json --retrieval-mode hybrid --out assessment.json
python -m eb1a eval data/eval --corpus data/processed/uscis_chunks.jsonl --index data/processed/embeddings.json --retrieval-mode hybrid --fast --out data/processed/evaluation.json
```

## Run locally

From PowerShell in the repository root:

```powershell
pip install -e .
$env:PYTHONPATH = "src"
$env:LLM_PROVIDER = "mock"
.\scripts\run_demo.ps1
pytest -q
```

Mock mode requires no API key. It runs the complete local flow using a synthetic USCIS-like source and a synthetic candidate profile. This is for pipeline testing only; replace the source with the official USCIS material before evaluating legal-grounding quality.

To use a hosted provider, configure the environment before running `assess`:

```powershell
$env:LLM_PROVIDER = "groq" # openai, groq, gemini, or bedrock
$env:GROQ_API_KEY = "..."   # use the matching provider variable
$env:LLM_MODEL = "openai/gpt-oss-20b" # active Groq model; use the provider's verified model list
python -m eb1a assess data/eval/profile_researcher.txt --corpus data/processed/uscis_chunks.jsonl --out assessment.json
```

For evaluation across the test set:

```powershell
python -m eb1a eval data/eval --corpus data/processed/uscis_chunks.jsonl --fast
```

Optional API:

```powershell
python -m pip install -e ".[api]"
python -m uvicorn eb1a.api:app --reload
```

Use `POST /assess` with a TXT/PDF upload and `GET /health` for a readiness check.

### Rate-Limit Mitigations & Developer Experience

Free-tier providers (especially Groq) impose strict limits:
- **Automatic Cooldown & Backoff**: Automatically parses the exact wait time in Groq's 429 response (`try again in Xs/ms/m`) or `Retry-After` header and retries with progressive backoff up to 5 attempts.
- **Disk Caching**: Structured LLM responses are cached under `.eb1a_cache/`. Re-running evaluations does not consume extra tokens or hit rate limits for identical prompts.
- **Fast Mode (`--fast`)**: Skips making LLM calls for criteria that have zero candidate evidence keywords in the resume (automatically sets `insufficient_info`), reducing LLM calls by 70%.
- **Clean Ingestion**: HTML navbars and scripts are stripped during ingestion, cutting prompt token consumption by 60%.
- **Model Choice**: Use active Groq models like `openai/gpt-oss-20b` or `openai/gpt-oss-120b`, or `gemini-2.0-flash` on Google Gemini for high throughput and lower rate-limit friction.

## LLM providers

Set `LLM_PROVIDER` to `mock`, `openai`, `groq`, `gemini`, or `bedrock`. Credentials are read only from environment variables: `OPENAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, or the standard AWS credential chain. The provider adapter returns validated structured objects; mock mode remains the deterministic offline default.

## User interface

The project includes a Streamlit interface. Start it from the repository root:

```powershell
python -m pip install -e ".[ui]"
$env:PYTHONPATH = "src"
python -m streamlit run streamlit_app.py
```

Open `http://localhost:8501`, keep the default corpus path `data/processed/uscis_chunks.jsonl`, upload a TXT/PDF profile, and click **Evaluate profile**. The interface loads `.env`, shows the configured provider before evaluation, and displays whether the result came from the production LLM. Provider or index failures are shown explicitly; production mode does not silently substitute heuristic output. It shows the readiness score, all criterion assessments, validation warnings, run metadata, and a downloadable JSON result. It never displays an accuracy percentage for an unlabeled upload.

## Evaluation Results & Failure Analysis

Run the labeled benchmark with:

```powershell
python -m eb1a eval data/eval --corpus data/processed/uscis_chunks.jsonl --index data/processed/embeddings.json --retrieval-mode hybrid --fast --out data/processed/eval_results.json
```

The command writes metrics together with corpus version, provider, model, embedding model, index version, prompt version, dataset version, timestamp, run ID, fallback count, and an `authoritative` flag. Results are not valid benchmark claims when mock or fallback mode was used. The dataset contains seven synthetic profiles, including an exactly-three-criteria profile and a prompt-injection profile.

| Metric | Score | Target / Requirement |
|---|---|---|
| Rating Accuracy | Written by the evaluation run | Requires labeled gold cases |
| Citation Validity Rate | Written by the evaluation run | Citation IDs and source metadata |
| Citation Quality Rate | Written by the evaluation run | Exact quote and locator match |
| Hallucinated Evidence Rate | Written by the evaluation run | Evidence quote must match profile |
| Retrieval Recall / MRR | Written by the evaluation run | BM25, semantic, and hybrid comparison |

The deterministic hashing vector search is an offline regression baseline, not a semantic embedding model. It is excluded from production retrieval and benchmark claims. Optional reranking should be reported separately from the BM25-plus-embedding baseline.

### Failure Analysis

The report produced by each run contains per-criterion results, a confusion matrix, and a `failure_analysis` list linking every incorrect decision to its profile evidence, retrieved chunks, citations, and reasoning. Review borderline `partial` versus `strong` decisions manually; subjective regulatory thresholds often require corroborating material such as national/international stature or geographic and industry wage data. Do not copy a previous run's percentages into product claims.

1. **Absence Classification (`insufficient_info` vs `missing`)**:
   - When profiles explicitly state "no evidence provided", models sometimes categorize the criterion as `missing` rather than `insufficient_info`. Both signify inadequate proof to satisfy the regulatory standard, but `insufficient_info` is preferred to prompt the user for additional documentation.

### Future Improvements
1. **Dynamic Web Search Tool (Agentic Bonus)**: Suggesting relevant open calls, conferences, and judging panels for candidate gaps.
2. **Re-ranking**: Add a cross-encoder after the BM25 plus Sentence Transformer baseline, then report before/after retrieval metrics.
3. **Independent LLM-as-Judge**: Add a separate judge run only after the labeled benchmark and citation-review protocol are stable.
