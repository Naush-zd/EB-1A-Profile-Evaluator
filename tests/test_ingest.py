import json
from pathlib import Path
import shutil
import uuid
import pytest
from eb1a.ingest import IngestionError, ingest_source_text, validate_corpus, clean_source_text
from eb1a.profile import load_profile


CORPUS = """EB-1A extraordinary ability policy manual. Awards and prizes. Membership requires outstanding achievement. Published material about the person. Judging the work of others. Original contributions of major significance. Authorship of scholarly articles. Display at artistic exhibitions. Leading or critical role. High salary. Commercial success in the performing arts."""


def test_corpus_validation_covers_all_criteria():
    chunks = __import__("eb1a.corpus", fromlist=["chunk_text"]).chunk_text(CORPUS, "u", "t")
    assert validate_corpus(chunks) == []


def test_html_cleaning_removes_markup():
    cleaned = clean_source_text("<html><body><h1>Extraordinary ability</h1><script>bad()</script><p>Awards and prizes.</p></body></html>")
    assert "Extraordinary ability" in cleaned
    assert "bad()" not in cleaned


def test_source_ingestion_rejects_duplicates_or_incomplete_corpus():
    folder = Path(".test-tmp") / uuid.uuid4().hex
    folder.mkdir(parents=True)
    with pytest.raises(IngestionError):
        ingest_source_text("short text", "u", folder / "chunks.jsonl")
    shutil.rmtree(folder)


def test_invalid_utf8_profile_fails():
    folder = Path(".test-tmp") / uuid.uuid4().hex
    folder.mkdir(parents=True)
    p = folder / "bad.txt"
    p.write_bytes(b"\xff\xfe")
    try:
        with pytest.raises(ValueError, match="UTF-8"):
            load_profile(p)
    finally:
        shutil.rmtree(folder)


def test_pdf_without_dependency_has_clear_error():
    folder = Path(".test-tmp") / uuid.uuid4().hex
    folder.mkdir(parents=True)
    p = folder / "profile.pdf"
    p.write_bytes(b"not a pdf")
    try:
        with pytest.raises(ValueError, match="pypdf|could not read PDF"):
            load_profile(p)
    finally:
        shutil.rmtree(folder)
