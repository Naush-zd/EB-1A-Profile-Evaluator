"""Optional Streamlit presentation layer for the stable service functions."""
import json
import os
import tempfile
from pathlib import Path

import streamlit as st

from eb1a.corpus import read_jsonl
from eb1a.evaluate import validate_grounding
from eb1a.profile import load_profile
from eb1a.providers import get_llm_provider, _load_local_env
from eb1a.service import assess_with_fallback


st.set_page_config(page_title="EB-1A Profile Evaluator", layout="wide")
st.title("EB-1A Profile Evaluator")
st.caption("Educational decision-support prototype. Not legal advice or a USCIS outcome prediction.")

corpus_path = st.sidebar.text_input("Knowledge-base chunks", "data/processed/uscis_chunks.jsonl")
fast_mode = st.sidebar.checkbox("Fast mode (only criteria with profile evidence)", value=True)
_load_local_env()
configured_provider = os.getenv("LLM_PROVIDER", "mock").lower()
configured_model = os.getenv("LLM_MODEL", "")
st.sidebar.info(f"Configured evaluator: {configured_provider}" + (f" / {configured_model}" if configured_model else ""))
if configured_provider == "mock":
    st.warning(
        "LLM_PROVIDER is set to `mock`, which has no production assessment path. "
        "Set LLM_PROVIDER to openai, groq, gemini, or bedrock (with the matching API key in `.env`) "
        "before evaluating a real profile."
    )

uploaded = st.file_uploader("Upload a TXT or PDF profile", type=["txt", "pdf"])

if uploaded and st.button("Evaluate profile", disabled=configured_provider == "mock"):
    corpus = Path(corpus_path)
    if not corpus.exists():
        st.error("Knowledge-base chunks were not found. Build the knowledge base first.")
    else:
        suffix = Path(uploaded.name).suffix or ".txt"
        with tempfile.TemporaryDirectory() as directory:
            profile_path = Path(directory) / f"profile{suffix}"
            profile_path.write_bytes(uploaded.getvalue())
            try:
                profile = load_profile(profile_path)
                chunks = read_jsonl(corpus)
                provider = get_llm_provider()
                assessment = assess_with_fallback(profile, chunks, provider=provider, skip_empty_evidence=fast_mode, allow_heuristic_fallback=False, index_path=corpus.with_name("embeddings.json"), retrieval_mode="hybrid")
                errors = validate_grounding(assessment, profile, chunks)
                if errors:
                    st.warning("Validation warnings: " + "; ".join(errors))
                st.metric("Readiness score", f"{assessment.overall.readiness_score}/100")
                st.metric("Criteria with evidence", assessment.overall.criteria_meeting_initial_threshold)
                st.write({"mode": assessment.run.mode if assessment.run else "unknown", "provider": assessment.run.provider if assessment.run else "unknown", "LLM calls": assessment.run.llm_calls if assessment.run else 0, "fallback used": assessment.run.fallback_used if assessment.run else False})
                if assessment.run and assessment.run.fallback_used:
                    st.warning(f"Fallback assessment used. The provider was unavailable or rate-limited. Reason: {assessment.run.fallback_reason}")
                st.subheader("Assessment JSON")
                st.code(assessment.model_dump_json(indent=2), language="json")
                st.download_button("Download assessment JSON", assessment.model_dump_json(indent=2), "assessment.json", "application/json")
            except ValueError as exc:
                st.error(f"Could not read the uploaded profile: {exc}")
            except Exception as exc:
                st.error(
                    f"Assessment failed: {exc}\n\n"
                    "This usually means the LLM provider was unavailable, rate-limited, or the embedding "
                    "index at `embeddings.json` is missing next to the corpus. Verify the provider with "
                    "`python -m eb1a verify-provider` and confirm the knowledge base was built with `build-kb`."
                )
