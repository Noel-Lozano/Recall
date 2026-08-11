# Study Assistant RAG — Phase 0 Baseline

A fully local RAG system: upload your notes (PDFs), ask questions, get grounded
answers with citations back to the source chunk.

## Why this stack

| Piece | Choice | Why |
|---|---|---|
| LLM | Claude Haiku 4.5 (Anthropic API) | Cheap, fast, and strong at following strict grounding/citation instructions — a local model would bottleneck iteration speed on 8-16GB RAM/no GPU |
| Embeddings | `all-MiniLM-L6-v2` (sentence-transformers, LOCAL) | ~80MB, fast on CPU, no GPU needed — kept local since embeddings don't need the horsepower an LLM does, and it's free |
| Vector DB | Chroma (persistent local mode) | Zero infra — just a folder on disk, no Docker/server to manage |
| API | FastAPI | Async support matters once LLM calls are in the mix; auto docs help you inspect your own endpoints |

This is a **hybrid architecture**: local embeddings + vector search, cloud LLM
for generation. The `generation/llm.py` module is the only place that knows
about Claude specifically — everything else in the codebase is unaware of
which model is doing generation. That's intentional: swapping to OpenAI, or
back to a local model later, means editing one file, not restructuring the
project.

This is intentionally a **naive baseline** — fixed-size chunking, top-k vector
search, no reranking. That's on purpose (see Phase 0 in the project plan):
you need a working end-to-end pipeline before you can measure anything. The
`eval/` folder is where you'll prove — with numbers — that later changes
(better chunking, reranking, hybrid search) actually help.

## Setup

1. **Get an Anthropic API key:** sign up at https://console.anthropic.com
   (includes free starter credit), create a key under Settings > API Keys.
2. **Set it as an environment variable** (don't hardcode it in the code —
   this is a real security habit, not just a formality):
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"     # macOS/Linux
   setx ANTHROPIC_API_KEY "your-key-here"        # Windows (restart terminal after)
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

## Using it

- `POST /upload` — upload a PDF, it gets chunked, embedded, and stored
- `POST /query` — ask a question, get an answer + the source chunks it was grounded in
- Visit `http://localhost:8000/docs` for interactive API docs (FastAPI gives you this for free)

## Next steps (see the project roadmap)

- Phase 1: build `eval/test_set.json` with real Q/A pairs from your own notes,
  run `eval/eval_harness.py` to get a retrieval-quality baseline number
- Phase 2: improve chunking (semantic instead of fixed-size), measure again
- Phase 3: add reranking + stricter citation enforcement
- Phase 4 (stretch): web search routing, then multi-user hosting