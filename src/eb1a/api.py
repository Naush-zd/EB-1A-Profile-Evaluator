"""Optional FastAPI interface over the CLI/service assessment pipeline."""
from pathlib import Path
import os
import tempfile

from .corpus import read_jsonl
from .evaluate import validate_grounding
from .service import assess_with_fallback
from .profile import load_profile
from .providers import _load_local_env, get_llm_provider

try:
    from fastapi import FastAPI, File, HTTPException, UploadFile
except ImportError:  # pragma: no cover - optional dependency
    FastAPI = None


def create_app():
    if FastAPI is None:
        raise RuntimeError("FastAPI is optional; install with pip install -e \".[api]\"")
    app = FastAPI(title="EB-1A Profile Evaluator", version="0.1.0")

    @app.get("/")
    def root():
        return {
            "service": "EB-1A Profile Evaluator",
            "status": "ok",
            "docs": "/docs",
            "health": "/health",
            "assessment_endpoint": "POST /assess",
        }

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/ingest/profile")
    async def ingest_profile(file: UploadFile = File(...)):
        if not file.filename or Path(file.filename).suffix.lower() not in {".txt", ".pdf"}:
            raise HTTPException(status_code=400, detail="upload a .txt or .pdf profile")
        data = await file.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="profile exceeds 10 MB limit")
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / Path(file.filename).name
            profile_path.write_bytes(data)
            try:
                return {"profile": load_profile(profile_path).model_dump(mode="json")}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/assess")
    async def assess(file: UploadFile = File(...), corpus: str = "data/processed/uscis_chunks.jsonl"):
        if not file.filename or Path(file.filename).suffix.lower() not in {".txt", ".pdf"}:
            raise HTTPException(status_code=400, detail="upload a .txt or .pdf profile")
        data = await file.read()
        if len(data) > 10 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="profile exceeds 10 MB limit")
        corpus_path = Path(corpus)
        if not corpus_path.exists():
            raise HTTPException(status_code=400, detail="knowledge-base corpus not found")
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / Path(file.filename).name
            profile_path.write_bytes(data)
            try:
                profile = load_profile(profile_path)
                chunks = read_jsonl(corpus_path)
                _load_local_env()
                provider = None if os.getenv("LLM_PROVIDER", "mock").lower() == "mock" else get_llm_provider()
                assessment = assess_with_fallback(profile, chunks, provider=provider, allow_heuristic_fallback=False, index_path=corpus_path.with_name("embeddings.json"), retrieval_mode="hybrid")
                errors = validate_grounding(assessment, profile, chunks)
                return {"assessment": assessment.model_dump(mode="json"), "validation_warnings": errors}
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            except Exception as exc:
                raise HTTPException(status_code=502, detail=str(exc)) from exc

    return app


app = create_app() if FastAPI is not None else None
