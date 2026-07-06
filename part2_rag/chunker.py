import os
import re

from .logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTS = {".txt", ".md", ".pdf"}


def parse_txt(content: str) -> str:
    return content


def parse_md(content: str) -> str:
    text = re.sub(r"```[\s\S]*?```", "", content)
    text = re.sub(r"!\[.*?\]\(.*?\)", "", text)
    text = re.sub(r"\[([^\]]*?)\]\(.*?\)", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[*_~`>|]", "", text)
    return text


def parse_pdf(filepath: str) -> str:
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF is required. Install: pip install PyMuPDF")
    doc = fitz.open(filepath)
    text = "\n".join(page.get_text() for page in doc)
    doc.close()
    return text


def read_file(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".pdf":
        return parse_pdf(filepath)
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    if ext == ".md":
        content = parse_md(content)
    return content


def chunk_text(text: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    if not text.strip():
        return []
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if chunk.strip():
            chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


def ingest_document(filepath: str, chunk_size: int = 512, overlap: int = 64) -> list[str]:
    logger.info("Ingesting: %s", filepath)
    text = read_file(filepath)
    if not text.strip():
        logger.warning("Empty file: %s", filepath)
        return []
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    logger.info("File: %s | chars=%d | chunks=%d", os.path.basename(filepath), len(text), len(chunks))
    return chunks
