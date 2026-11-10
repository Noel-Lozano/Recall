"""
Eval harness: measure retrieval quality (Recall@k, Hit@k, MRR) on your notes.

How it works:
  1. Chunks + embeds every PDF in eval/docs/ into a SEPARATE Chroma
     collection (never touches the notes you uploaded through the API).
     The collection is rebuilt from scratch every run so chunking or
     embedding changes always take effect.
  2. Runs every question in eval/test_set.json through VectorStore.search().
  3. Scores the retrieved chunks against each question's evidence snippets.
  4. Prints a summary and saves a JSON report to eval/results/.

WHY this only measures retrieval (no LLM calls): if the right chunk isn't
retrieved, no prompt can save the answer, so retrieval is the first number
to get right. It's also free and fast: no Groq rate limits.

Run from the backend/ folder:
    python eval/eval_harness.py
    python eval/eval_harness.py --k 1 3 5 10 --chunk-size 500 --overlap 100
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
BACKEND_DIR = EVAL_DIR.parent
sys.path.insert(0, str(BACKEND_DIR))  # so `ingestion` / `retrieval` import from anywhere

from ingestion.chunking import process_pdf  # noqa: E402
from retrieval.reranker import RERANK_MODEL, Reranker  # noqa: E402
from retrieval.vectorstore import EMBEDDING_MODEL, VectorStore  # noqa: E402
from metrics import contains_evidence, hit_at_k, normalize, recall_at_k, reciprocal_rank  # noqa: E402


def load_test_set(path: Path) -> list[dict]:
    if not path.exists():
        sys.exit(f"Test set not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data

    for i, q in enumerate(questions):
        if not q.get("question") or not q.get("evidence"):
            sys.exit(f"Test item #{i} needs a non-empty 'question' and 'evidence'.")
        if isinstance(q["evidence"], str):
            q["evidence"] = [q["evidence"]]
        q.setdefault("id", f"q{i + 1}")
        # WHY: evidence made only of symbols (e.g. "→", "≠") normalizes to "",
        # and "" is a substring of every chunk, so it would score as a perfect
        # rank-1 hit and silently inflate every metric.
        for e in q["evidence"]:
            if not normalize(e):
                sys.exit(
                    f"[{q['id']}] evidence {e!r} has no letters or digits, so it would match "
                    "every chunk. Use a phrase with words around it instead."
                )
    return questions


def build_index(docs_dir: Path, chunk_size: int, overlap: int) -> tuple[VectorStore, list[str]]:
    pdfs = sorted(docs_dir.glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs in {docs_dir}. Copy the notes your test questions are about there.")

    store = VectorStore(persist_dir=str(EVAL_DIR / ".chroma_eval"), user_id="eval")
    store.clear()

    all_chunks: list[str] = []
    for pdf in pdfs:
        chunks = process_pdf(str(pdf), chunk_size=chunk_size, overlap=overlap)
        store.add_chunks(chunks, source=pdf.name)
        all_chunks.extend(chunks)
        print(f"  indexed {pdf.name}: {len(chunks)} chunks")
    return store, all_chunks


def main():
    parser = argparse.ArgumentParser(description="Measure retrieval Recall@k on your notes.")
    parser.add_argument("--test-set", type=Path, default=EVAL_DIR / "test_set.json")
    parser.add_argument("--docs-dir", type=Path, default=EVAL_DIR / "docs")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--rerank", action="store_true", help="rerank dense candidates with a cross-encoder")
    parser.add_argument("--candidates", type=int, default=20, help="dense results handed to the reranker")
    parser.add_argument("--no-save", action="store_true", help="don't write a results JSON")
    args = parser.parse_args()

    ks = sorted(set(args.k))
    max_k = ks[-1]

    questions = load_test_set(args.test_set)
    print(f"Building eval index (chunk_size={args.chunk_size}, overlap={args.overlap})...")
    store, all_chunks = build_index(args.docs_dir, args.chunk_size, args.overlap)

    # Sanity check: evidence that isn't inside ANY chunk can never be
    # retrieved. Either the snippet has a typo / doesn't match the PDF's
    # extracted text, or the chunker split it across a boundary. That second
    # case is a real chunking failure, so it still counts as a miss.
    unfindable = [
        (q["id"], e)
        for q in questions
        for e in q["evidence"]
        if not any(contains_evidence(c, e) for c in all_chunks)
    ]

    reranker = None
    if args.rerank:
        print(f"Reranking dense top-{args.candidates} with a cross-encoder...")
        reranker = Reranker()
    num_candidates = max(args.candidates, max_k) if reranker else max_k

    per_question = []
    for q in questions:
        candidates = store.search(q["question"], top_k=num_candidates)
        result = {"id": q["id"], "question": q["question"]}
        if reranker:
            # Ceiling: the reranker can only reorder what dense search found
            dense = [c["text"] for c in candidates]
            result["candidate_recall"] = recall_at_k(dense, q["evidence"], len(dense))
            candidates = reranker.rerank(q["question"], candidates, top_k=max_k)
        retrieved = [c["text"] for c in candidates]
        result.update({
            **{f"recall@{k}": recall_at_k(retrieved, q["evidence"], k) for k in ks},
            **{f"hit@{k}": hit_at_k(retrieved, q["evidence"], k) for k in ks},
            "rr": reciprocal_rank(retrieved, q["evidence"]),
        })
        per_question.append(result)

    n = len(per_question)
    summary = {
        **{f"recall@{k}": sum(r[f"recall@{k}"] for r in per_question) / n for k in ks},
        **{f"hit@{k}": sum(r[f"hit@{k}"] for r in per_question) / n for k in ks},
        "mrr": sum(r["rr"] for r in per_question) / n,
    }
    if reranker:
        summary["candidate_recall"] = sum(r["candidate_recall"] for r in per_question) / n

    # ---- report ----
    print(f"\nRetrieval results over {n} questions")
    print("-" * 36)
    for k in ks:
        print(f"  Recall@{k:<3} {summary[f'recall@{k}']:.3f}    Hit@{k:<3} {summary[f'hit@{k}']:.3f}")
    print(f"  MRR        {summary['mrr']:.3f}")
    if reranker:
        print(f"  Ceiling: dense Recall@{num_candidates} = {summary['candidate_recall']:.3f}"
              " (the reranker can't recover evidence outside its candidates)")

    report_k = 3 if 3 in ks else ks[0]
    misses = [r for r in per_question if r[f"recall@{report_k}"] < 1.0]
    if misses:
        print(f"\nQuestions with evidence missing from the top {report_k}:")
        for r in misses:
            print(f"  [{r['id']}] recall@{report_k}={r[f'recall@{report_k}']:.2f}  {r['question']}")

    if unfindable:
        print("\nWARNING: these evidence snippets are not inside any chunk, so they can never be retrieved.")
        print("Check for typos against the PDF text, or the chunker split them across two chunks:")
        for qid, e in unfindable:
            print(f"  [{qid}] {e!r}")

    if not args.no_save:
        results_dir = EVAL_DIR / "results"
        results_dir.mkdir(exist_ok=True)
        out = results_dir / f"{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(json.dumps({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "chunk_size": args.chunk_size,
                "overlap": args.overlap,
                "embedding_model": EMBEDDING_MODEL,
                "rerank_model": RERANK_MODEL if reranker else None,
                "rerank_candidates": num_candidates if reranker else None,
                "k": ks,
                "num_questions": n,
                "num_chunks": len(all_chunks),
            },
            "summary": summary,
            "unfindable_evidence": [{"id": qid, "evidence": e} for qid, e in unfindable],
            "per_question": per_question,
        }, indent=2), encoding="utf-8")
        print(f"\nSaved {out.relative_to(BACKEND_DIR)}")


if __name__ == "__main__":
    main()
