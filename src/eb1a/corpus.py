import hashlib
import re
from datetime import datetime, timezone
from pathlib import Path
from .schemas import SourceChunk


def corpus_version(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", text.lower())


def chunk_text(text: str, source_url: str, retrieved_at: str | None = None, target_words: int = 700, overlap_words: int = 100) -> list[SourceChunk]:
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []
    retrieved_at = retrieved_at or datetime.now(timezone.utc).isoformat()
    version = corpus_version(text)
    sections = re.split(r"(?m)^\s*(?=(?:chapter|section|criterion|initial evidence|final merits)\b[^\n]*)", text, flags=re.I)
    chunks: list[SourceChunk] = []
    for section_index, section in enumerate(sections):
        section = section.strip()
        if not section:
            continue
        section_lines = [line.strip() for line in section.splitlines() if line.strip()]
        heading = section_lines[0][:240] if section_lines else f"section-{section_index}"
        words = section.split()
        step = max(target_words - overlap_words, 1)
        for offset in range(0, len(words), step):
            piece = " ".join(words[offset:offset + target_words]).strip()
            if not piece:
                continue
            chunk_hash = hashlib.sha256(piece.encode("utf-8")).hexdigest()
            chunks.append(SourceChunk(
                chunk_id=f"{version}-{len(chunks):04d}", text=piece, source_url=source_url,
                retrieved_at=retrieved_at, corpus_version=version,
                section=f"section-{section_index}", heading=heading, locator=f"words-{offset}-{offset + len(piece.split())}",
                content_hash=chunk_hash,
            ))
            if offset + target_words >= len(words):
                break
    return chunks


def write_jsonl(chunks: list[SourceChunk], path: str | Path) -> None:
    import json
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("\n".join(c.model_dump_json() for c in chunks) + "\n", encoding="utf-8")


def read_jsonl(path: str | Path) -> list[SourceChunk]:
    import json
    return [SourceChunk.model_validate(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
