import hashlib
import json
import urllib.request
from html.parser import HTMLParser
from datetime import datetime, timezone
from pathlib import Path
from .criteria import CRITERIA
from .corpus import chunk_text, write_jsonl


class IngestionError(RuntimeError):
    pass


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"}:
            self.skip_depth += 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript", "svg", "nav", "header", "footer", "aside", "form"} and self.skip_depth:
            self.skip_depth -= 1
        if tag in {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip_depth:
            self.parts.append(data)


def clean_source_text(text: str) -> str:
    if "<html" not in text.lower() and "<body" not in text.lower():
        return text
    parser = _TextExtractor()
    parser.feed(text)
    lines = [" ".join(line.split()) for line in "".join(parser.parts).splitlines()]
    return "\n".join(line for line in lines if line)


def download_source(url: str, output: str | Path) -> tuple[Path, str]:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
    except Exception as exc:
        raise IngestionError(f"failed to download source: {exc}") from exc
    if not data:
        raise IngestionError("downloaded source is empty")
    target.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    target.with_suffix(target.suffix + ".meta.json").write_text(json.dumps({
        "url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": digest
    }, indent=2), encoding="utf-8")
    return target, digest


def validate_corpus(chunks) -> list[str]:
    errors = []
    if not chunks:
        return ["corpus contains no chunks"]
    ids = [chunk.chunk_id for chunk in chunks]
    if len(ids) != len(set(ids)):
        errors.append("duplicate chunk IDs")
    hashes = [chunk.content_hash for chunk in chunks]
    if len(hashes) != len(set(hashes)):
        errors.append("duplicate chunk content")
    full_text = " ".join(chunk.text.lower() for chunk in chunks)
    required = ["extraordinary ability", "policy manual"]
    errors.extend(f"missing expected term: {term}" for term in required if term not in full_text)
    criterion_terms = {
        "awards": ["award", "prize"], "membership": ["membership"],
        "published_material": ["published material"], "judging": ["judge", "judging"],
        "original_contributions": ["original contributions"], "scholarly_articles": ["scholarly"],
        "exhibitions": ["exhibition"], "leading_critical_role": ["leading", "critical role"],
        "high_salary": ["high salary"], "commercial_success": ["commercial success"],
    }
    for criterion, terms in criterion_terms.items():
        if not any(term in full_text for term in terms):
            errors.append(f"criterion not represented: {criterion}")
    return errors


def ingest_source_text(text: str, url: str, output: str | Path):
    chunks = chunk_text(clean_source_text(text), url)
    errors = validate_corpus(chunks)
    if errors:
        raise IngestionError("corpus validation failed: " + "; ".join(errors))
    write_jsonl(chunks, output)
    return chunks
