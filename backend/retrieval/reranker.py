"""
Reranking: re-score the dense search's candidates with a cross-encoder.

WHY a second stage: the embedding model (bi-encoder) encodes the question
and each chunk SEPARATELY, then compares two vectors. That's fast enough to
search every chunk, but it can't see how specific question words line up
with specific chunk words. The eval showed the cost: the right chunk was
in the top 10 for ~95% of questions but in the top 3 for only ~75%.

A cross-encoder reads the question and a chunk TOGETHER in one forward
pass, so it judges relevance much more precisely. That's too slow to run
on every chunk, so the pipeline is: dense search narrows to ~20
candidates, the cross-encoder reorders them, and we keep the top k.

WHY ms-marco-MiniLM-L-6-v2: trained on MS MARCO (real search queries vs
passages, the same task as "question vs note chunk"), small (~90MB),
runs on CPU in well under a second for 20 pairs.
"""

from sentence_transformers import CrossEncoder

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# How many dense results to hand to the reranker. The reranker can only
# reorder what it's given, so this caps the best possible recall; more
# candidates means a higher ceiling but a slower query.
DEFAULT_CANDIDATES = 20


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Reorder search() results by cross-encoder score and keep the top_k.
        Each returned dict gets a `rerank_score` (higher = more relevant)."""
        if not candidates:
            return []
        scores = self.model.predict([(query, c["text"]) for c in candidates])
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        return [{**c, "rerank_score": float(s)} for c, s in ranked[:top_k]]
