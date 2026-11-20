import hashlib
import re
from pathlib import Path
from .schemas import ProfileDocument


INSTRUCTION_PATTERNS = [
    r"ignore (?:all|any|the|previous) instructions",
    r"system message",
    r"developer message",
    r"reveal (?:the )?(?:prompt|instructions)",
]


def _normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(line.rstrip() for line in text.splitlines()).strip()


def _suspicious(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if any(re.search(p, line, re.I) for p in INSTRUCTION_PATTERNS)]


def load_profile(path: str | Path) -> ProfileDocument:
    p = Path(path)
    raw = p.read_bytes()
    if not raw:
        raise ValueError("profile file is empty")
    page_text: list[str]
    if p.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            reader = PdfReader(str(p))
            page_text = [_normalize(page.extract_text() or "") for page in reader.pages]
        except ImportError as exc:
            raise ValueError("PDF support requires the optional pypdf dependency") from exc
        except Exception as exc:
            raise ValueError(f"could not read PDF: {exc}") from exc
        text = _normalize("\n\n".join(page_text))
        if not text:
            raise ValueError("PDF contains no extractable text; scanned PDFs are unsupported")
        pages = max(len(page_text), 1)
    elif p.suffix.lower() == ".txt":
        try:
            text = _normalize(raw.decode("utf-8"))
        except UnicodeDecodeError as exc:
            raise ValueError("profile is not valid UTF-8 text") from exc
        page_text = [text]
        pages = 1
    else:
        raise ValueError("unsupported profile format; use .txt or .pdf")
    if not text:
        raise ValueError("profile contains no readable text")
    digest = hashlib.sha256(raw).hexdigest()
    return ProfileDocument(
        profile_id=digest[:16], source_path=str(p), content=text,
        content_hash=digest, pages=pages, suspicious_instructions=_suspicious(text), page_text=page_text
    )
