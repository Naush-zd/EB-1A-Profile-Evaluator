from pathlib import Path
import shutil
import uuid
from eb1a.knowledge_base import build_knowledge_base, inspect_knowledge_base


TEXT = "EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts. " * 2


def test_knowledge_base_manifest_and_inspection():
    folder = Path(".test-tmp") / uuid.uuid4().hex
    folder.mkdir(parents=True)
    try:
        chunks_path = folder / "chunks.jsonl"
        manifest_path = folder / "manifest.json"
        manifest = build_knowledge_base(TEXT, "uscis", chunks_path, manifest_path)
        report = inspect_knowledge_base(chunks_path)
        assert manifest.chunk_count == report["chunk_count"]
        assert report["validation_errors"] == []
        assert manifest_path.exists()
    finally:
        shutil.rmtree(folder)
