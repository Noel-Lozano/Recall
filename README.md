# Study Assistant RAG — Phase 1 Baseline

A fully local RAG system: upload your notes (PDFs), ask questions, get grounded
answers with citations back to the source chunk.

## Why this stack

| Piece | Choice | Why |
|---|---|---|
| LLM | Llama 3.3 70B via Groq (free tier) | Genuinely free (no credit card, no expiring trial credit), fast LPU inference, OpenAI-compatible API shape — good for learning, rate-limited so not meant for production traffic |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, LOCAL) | ~80MB, fast on CPU, no GPU needed — kept local since embeddings don't need the horsepower an LLM does, and it's free |
| Vector DB | Chroma (persistent local mode) | Zero infra — just a folder on disk, no Docker/server to manage |
| API | FastAPI | Async support matters once LLM calls are in the mix; auto docs help you inspect your own endpoints |

This is a **hybrid architecture**: local embeddings + vector search, cloud LLM
for generation. The `generation/llm.py` module is the only place that knows
which provider is doing generation — everything else in the codebase is
unaware of it. That's intentional: this file has already been swapped
between Ollama, Claude, and Groq without touching anything else in the
project, which is the point of designing to an interface.

This is intentionally a **naive baseline** — fixed-size chunking, top-k vector
search, no reranking. That's on purpose (see Phase 0 in the project plan):
you need a working end-to-end pipeline before you can measure anything. The
`eval/` folder is where you'll prove — with numbers — that later changes
(better chunking, reranking, hybrid search) actually help.

## Setup

1. **Get a Groq API key:** sign up at https://console.groq.com (no credit
   card needed), create a key under API Keys.
2. **Set it as an environment variable** (don't hardcode it in the code —
   this is a real security habit, not just a formality):
   ```bash
   export GROQ_API_KEY="your-key-here"     # macOS/Linux
   setx GROQ_API_KEY "your-key-here"        # Windows (restart terminal after)
   ```
3. **Install Python dependencies:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r ../requirements.txt
   ```
4. **Run the server:**
   ```bash
   uvicorn main:app --reload
   ```

Note: the first run will download the embedding model (~80MB) automatically —
that part still runs fully locally and doesn't touch the API.

Groq's free tier is rate-limited (roughly 30 requests/minute, ~1,000/day
depending on the model) — plenty for development and eval runs, but if you
ever hit a 429 error, that's the rate limit, not a bug.

## Using it

- `POST /upload` — upload a PDF, it gets chunked, embedded, and stored
- `POST /query` — ask a question, get an answer + the source chunks it was grounded in
- Visit `http://localhost:8000/docs` for interactive API docs (FastAPI gives you this for free)

## Measuring retrieval (Phase 1)

`eval/eval_harness.py` scores retrieval without calling the LLM, so it costs
nothing and doesn't count against Groq's rate limit.

1. Copy the PDFs you want to test against into `eval/docs/` (git-ignored).
2. Replace the example questions in `eval/test_set.json` with real ones from
   your notes. Each question has an `evidence` list: short phrases copied
   word-for-word from the notes that contain the answer. A retrieved chunk
   counts as relevant if it contains one of those phrases, so the same test
   set keeps working when you change chunking.
3. Run it from `backend/`:
   ```bash
   python eval/eval_harness.py
   python eval/eval_harness.py --chunk-size 500 --overlap 100   # try a variant
   ```

It prints:
- **Recall@k:** the share of evidence phrases found in the top k chunks
- **Hit@k:** the share of questions with at least one evidence phrase in the top k
- **MRR:** how high the first relevant chunk ranks

Each run also saves a JSON report to `eval/results/` so you can compare runs.
The eval builds its own Chroma collection in `eval/.chroma_eval/` and rebuilds
it every run, so your uploaded notes are never touched.

If it warns that evidence is "not inside any chunk", either the phrase doesn't
match the text extracted from the PDF, or the chunker split it across two
chunks. The second case is a real chunking failure, and Phase 2 should fix it.

## Next steps (see the project roadmap)

- Phase 2: improve chunking (semantic instead of fixed-size), re-run the eval
  and compare against the Phase 1 baseline
- Phase 3: add reranking + stricter citation enforcement
- Phase 4 (stretch): web search routing, then multi-user hosting
