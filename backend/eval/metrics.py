"""
Retrieval metrics: Recall@k, Hit@k and MRR.

WHY relevance is decided by evidence text, not chunk IDs:
A test question's ground truth is a short phrase copied from your notes
(the "evidence"). A retrieved chunk counts as relevant if it contains that
phrase. Chunk IDs change every time you change chunking (Phase 2), so a
test set keyed on IDs would have to be relabelled after every experiment.
Evidence text survives any chunking strategy, which is exactly what you
need to compare strategies fairly.
"""

import re


def normalize(text: str) -> str:
    """
    Lowercase and collapse everything that isn't a letter/digit into single
    spaces. PDF text extraction is messy (line breaks mid-sentence,
    hyphenated words, odd quote characters), so comparing raw strings
    would mark real matches as misses.

    "Letter/digit" is Unicode-aware, so Greek (σ, λ), superscripts (²) and
    accented letters (é) survive; an ASCII-only filter would turn "résumé"
    into "r sum" and "σ²" into nothing at all.
    """
    text = text.lower()
    text = re.sub(r"-\s*\n\s*", "", text)  # re-join words hyphenated across lines
    text = re.sub(r"[\W_]+", " ", text)
    return text.strip()


def contains_evidence(chunk_text: str, evidence: str) -> bool:
    return normalize(evidence) in normalize(chunk_text)


def recall_at_k(retrieved: list[str], evidence: list[str], k: int) -> float:
    """
    Fraction of a question's evidence snippets that appear in at least one
    of the top-k retrieved chunks.

    WHY recall (and not precision) for retrieval: the LLM can ignore an
    irrelevant chunk, but it cannot answer from a chunk that was never
    retrieved. Missing evidence is the failure that actually breaks answers.
    """
    if not evidence:
        return 0.0
    top = retrieved[:k]
    found = sum(1 for e in evidence if any(contains_evidence(c, e) for c in top))
    return found / len(evidence)


def hit_at_k(retrieved: list[str], evidence: list[str], k: int) -> float:
    """1.0 if ANY evidence snippet shows up in the top-k, else 0.0."""
    top = retrieved[:k]
    return 1.0 if any(contains_evidence(c, e) for c in top for e in evidence) else 0.0


def reciprocal_rank(retrieved: list[str], evidence: list[str]) -> float:
    """
    1 / (rank of the first relevant chunk), or 0 if none was retrieved.
    Rewards putting the right chunk FIRST, not just somewhere in the top-k.
    """
    for rank, chunk in enumerate(retrieved, start=1):
        if any(contains_evidence(chunk, e) for e in evidence):
            return 1.0 / rank
    return 0.0
