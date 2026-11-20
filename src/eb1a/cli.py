import argparse
import json
import os
from pathlib import Path
from .profile import load_profile
from .corpus import chunk_text, read_jsonl, write_jsonl
from .retrieve import search
from .evaluate import assess_profile, validate_grounding
from .ingest import download_source, ingest_source_text, IngestionError, validate_corpus
from .eval_runner import run_evaluation
from .knowledge_base import build_knowledge_base, inspect_knowledge_base
from .providers import get_llm_provider, _load_local_env
from .vector import hybrid_search, semantic_search, hash_vector_search, indexed_semantic_search
from .service import assess_with_fallback


def main() -> int:
    parser = argparse.ArgumentParser(prog="eb1a")
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)
    profile_cmd = sub.add_parser("ingest-profile")
    profile_cmd.add_argument("profile", type=Path)
    profile_cmd.add_argument("--out", type=Path, default=Path("profile.json"))
    source = sub.add_parser("ingest-source")
    source.add_argument("source", type=Path)
    source.add_argument("--url", required=True)
    source.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    source_url = sub.add_parser("ingest-source-url")
    source_url.add_argument("--url", required=True)
    source_url.add_argument("--out", type=Path, default=Path("data/raw/uscis_source"))
    source_url.add_argument("--chunks-out", type=Path, default=Path("data/processed/chunks.jsonl"))
    kb = sub.add_parser("build-kb")
    kb.add_argument("source", type=Path)
    kb.add_argument("--url", required=True)
    kb.add_argument("--out", type=Path, default=Path("data/processed/chunks.jsonl"))
    kb.add_argument("--manifest", type=Path, default=Path("data/processed/manifest.json"))
    kb.add_argument("--index", type=Path, default=Path("data/processed/embeddings.json"))
    kb.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("corpus", type=Path)
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=5)
    retrieve.add_argument("--hybrid", action="store_true")
    retrieve.add_argument("--semantic", action="store_true")
    retrieve.add_argument("--hash-baseline", action="store_true")
    retrieve.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    retrieve.add_argument("--index", type=Path)
    retrieve.add_argument("--rerank", action="store_true", help="Apply optional neural cross-encoder reranking")
    retrieve.add_argument("--reranker-model", default=os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    assess = sub.add_parser("assess")
    assess.add_argument("profile", type=Path)
    assess.add_argument("--corpus", type=Path, required=True)
    assess.add_argument("--out", type=Path, default=Path("assessment.json"))
    assess.add_argument("--fast", action="store_true", help="Skip LLM calls for criteria with no candidate evidence keywords")
    assess.add_argument("--index", type=Path)
    assess.add_argument("--retrieval-mode", choices=["lexical", "semantic", "hybrid"], default="hybrid")
    assess.add_argument("--mode", choices=["production", "test_mock"], default="production")
    inspect = sub.add_parser("inspect-corpus")
    inspect.add_argument("corpus", type=Path)
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("dataset", type=Path)
    evaluation.add_argument("--corpus", type=Path, required=True)
    evaluation.add_argument("--out", type=Path)
    evaluation.add_argument("--fast", action="store_true", help="Skip LLM calls for criteria with no candidate evidence keywords")
    evaluation.add_argument("--index", type=Path, default=Path("data/processed/embeddings.json"))
    evaluation.add_argument("--retrieval-mode", choices=["lexical", "semantic", "hybrid"], default="hybrid")
    verify = sub.add_parser("verify-provider")
    args = parser.parse_args()
    if args.command == "ingest-profile":
        document = load_profile(args.profile)
        args.out.write_text(document.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    elif args.command == "ingest-source":
        chunks = ingest_source_text(args.source.read_text(encoding="utf-8"), args.url, args.out)
        print(f"Wrote {len(chunks)} chunks to {args.out}")
    elif args.command == "ingest-source-url":
        source_path, digest = download_source(args.url, args.out)
        raw = source_path.read_bytes()
        if source_path.suffix.lower() == ".pdf":
            raise SystemExit("downloaded PDF source requires a PDF text extraction dependency before chunking")
        chunks = ingest_source_text(raw.decode("utf-8"), args.url, args.chunks_out)
        print(f"Downloaded {digest[:16]} and wrote {len(chunks)} chunks to {args.chunks_out}")
    elif args.command == "build-kb":
        manifest = build_knowledge_base(args.source.read_text(encoding="utf-8"), args.url, args.out, args.manifest, args.index, args.embedding_model)
        print(json.dumps(manifest.to_dict(), indent=2))
    elif args.command == "retrieve":
        chunks = read_jsonl(args.corpus)
        if args.hash_baseline:
            results = hash_vector_search(chunks, args.query, args.top_k)
        elif args.semantic:
            results = indexed_semantic_search(chunks, args.query, args.index, args.top_k, args.embedding_model) if args.index else semantic_search(chunks, args.query, args.top_k, args.embedding_model, allow_download=True)
        elif args.hybrid:
            if args.rerank:
                from .vector import indexed_hybrid_search
                if not args.index:
                    raise SystemExit("--rerank requires --index")
                results = indexed_hybrid_search(chunks, args.query, args.index, args.top_k, model_name=args.embedding_model, reranker_model=args.reranker_model)
            else:
                results = hybrid_search(chunks, args.query, args.top_k, model_name=args.embedding_model, allow_download=True)
        else:
            results = search(chunks, args.query, args.top_k)
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
    elif args.command == "assess":
        profile = load_profile(args.profile)
        chunks = read_jsonl(args.corpus)
        _load_local_env()
        provider = None if os.getenv("LLM_PROVIDER", "mock").lower() == "mock" else get_llm_provider()
        index_path = args.index or args.corpus.with_name("embeddings.json")
        if args.mode == "production" and provider is None:
            raise SystemExit("production assessment requires a real LLM provider; configure LLM_PROVIDER and its credentials, or explicitly use --mode test_mock")
        assessment = assess_with_fallback(profile, chunks, provider=provider, skip_empty_evidence=args.fast, allow_heuristic_fallback=args.mode == "test_mock", index_path=index_path, retrieval_mode="lexical" if args.mode == "test_mock" else args.retrieval_mode)
        errors = validate_grounding(assessment, profile, chunks)
        if errors:
            print(json.dumps({"errors": errors}, indent=2))
            return 1
        args.out.write_text(assessment.model_dump_json(indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")
    elif args.command == "inspect-corpus":
        report = inspect_knowledge_base(args.corpus)
        print(json.dumps(report, indent=2))
        return 1 if report["validation_errors"] else 0
    elif args.command == "eval":
        result = run_evaluation(args.dataset, args.corpus, fast=args.fast, index_path=args.index, retrieval_mode=args.retrieval_mode)
        rendered = json.dumps(result, indent=2)
        if args.out:
            args.out.write_text(rendered, encoding="utf-8")
            print(f"Wrote {args.out}")
        else:
            print(rendered)
    elif args.command == "verify-provider":
        provider = get_llm_provider()
        if not hasattr(provider, "verify_credentials"):
            print(json.dumps({"provider": os.getenv("LLM_PROVIDER", "mock"), "ok": True, "message": "offline provider requires no credentials"}, indent=2))
        else:
            print(json.dumps(provider.verify_credentials(), indent=2))
    return 0


# Compatibility alias for environments that installed the original Typer entry point.
app = main


if __name__ == "__main__":
    raise SystemExit(main())
