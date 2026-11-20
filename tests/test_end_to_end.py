import shutil
import uuid
from pathlib import Path
from eb1a.corpus import write_jsonl
from eb1a.eval_runner import run_evaluation
from eb1a.evaluate import assess_profile, validate_grounding
from eb1a.knowledge_base import build_knowledge_base, inspect_knowledge_base
from eb1a.profile import load_profile


CORPUS = """EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts."""


def test_complete_pipeline_end_to_end():
    root = Path(".test-tmp") / uuid.uuid4().hex
    root.mkdir(parents=True)
    try:
        source = root / "source.txt"
        source.write_text(CORPUS, encoding="utf-8")
        profile_file = root / "profile.txt"
        profile_file.write_text("Received a national award.", encoding="utf-8")
        chunks_file = root / "chunks.jsonl"
        build_knowledge_base(CORPUS, "uscis", chunks_file, root / "manifest.json")
        assert inspect_knowledge_base(chunks_file)["validation_errors"] == []
        profile = load_profile(profile_file)
        from eb1a.corpus import read_jsonl
        chunks = read_jsonl(chunks_file)
        assessment = assess_profile(profile, chunks, chunks[0].corpus_version)
        assert len(assessment.criteria) == 10
        assert validate_grounding(assessment, profile, chunks) == []
    finally:
        shutil.rmtree(root)
