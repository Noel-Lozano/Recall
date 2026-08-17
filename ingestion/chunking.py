"""
Ingestion: turn a PDF into a list of text chunks ready for embedding.

WHY fixed-size chunking for Phase 0:
Fixed-size, character-count chunking is the simplest possible strategy —
it ignores sentence/paragraph boundaries entirely. It's a bad long-term
choice (it can cut a sentence in half, splitting meaning across two chunks),
but that's exactly why it belongs in Phase 0: it gives you a real number
(via eval/eval_harness.py) to beat once you implement smarter chunking
in Phase 2. You can't prove an improvement without a baseline.
"""

from pypdf import PdfReader


def extract_text(pdf_path: str) -> str:
    """Pull raw text out of a PDF, page by page."""
    reader = PdfReader(pdf_path)
    text_parts = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Split text into overlapping fixed-size chunks.

    WHY overlap: without it, a fact that spans a chunk boundary (e.g. a
    definition that starts at the end of one chunk and finishes at the
    start of the next) becomes unretrievable — neither chunk alone contains
    the full meaning. Overlap trades a bit of redundant storage for much
    better recall.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def process_pdf(pdf_path: str) -> list[str]:
    """End-to-end: PDF path -> list of text chunks."""
    text = extract_text(pdf_path)
    return chunk_text(text)