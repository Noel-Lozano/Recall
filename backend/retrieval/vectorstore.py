"""
Retrieval layer: embed text and store/query it in a local Chroma DB.

WHY this is wrapped in a class (an interface) instead of calling Chroma
directly from main.py: it means main.py never needs to know HOW retrieval
works, only THAT it can call add_chunks() and search(). If you swap Chroma
for another vector DB later (e.g. Qdrant, when you add multi-user hosting),
only this file changes — nothing else in the codebase needs to know.
This is the same reasoning behind wrapping the LLM in generation/llm.py.
"""

from pathlib import Path
from typing import cast

import chromadb
import numpy as np
from chromadb.api.types import Embedding, Metadata
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# WHY all-MiniLM-L6-v2: small (~80MB), fast on CPU, no GPU required,
# strong quality-to-size tradeoff for a first pass. Swap this string
# to try a bigger/better embedding model later and compare eval numbers.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# WHY an absolute path anchored to backend/: a relative "./chroma_data"
# resolves against whatever directory you launched from, so running
# uvicorn or the eval script from a different folder would silently
# create (and search) a second, empty database.
DEFAULT_PERSIST_DIR = str(Path(__file__).resolve().parent.parent / "chroma_data")


class VectorStore:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR, user_id: str = "default_user"):
        # WHY persist_dir instead of an in-memory client: you want your
        # notes to survive a server restart without re-uploading everything.
        # anonymized_telemetry=False: don't send usage data to Chroma.
        # (The "Failed to send telemetry event" noise is a separate posthog
        # version clash, fixed by the posthog pin in requirements.txt.)
        self.client = chromadb.PersistentClient(
            path=persist_dir, settings=Settings(anonymized_telemetry=False)
        )

        # WHY collection name includes user_id even though there's only one
        # user right now: this is the schema decision that makes adding
        # real multi-user auth later a config change, not a rewrite.
        self.collection_name = f"notes_{user_id}"
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def _embed(self, texts: list[str]) -> list[Embedding]:
        # WHY np.asarray + cast: encode() is typed as returning
        # ndarray | Tensor | list[Tensor] depending on its flags, so the type
        # checker can't tell that it's a numpy array here. asarray pins it
        # down, and the cast tells Pylance what tolist() actually returns.
        vectors = np.asarray(self.embedder.encode(texts, convert_to_numpy=True))
        return cast(list[Embedding], vectors.tolist())

    def add_chunks(self, chunks: list[str], source: str):
        if not chunks:
            return
        embeddings = self._embed(chunks)
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas: list[Metadata] = [
            {"source": source, "chunk_index": i} for i in range(len(chunks))
        ]
        # WHY upsert instead of add: re-uploading a file with the same name
        # would otherwise be silently skipped by Chroma (duplicate IDs), so
        # edits to your notes would never reach the index.
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def clear(self):
        """Delete every chunk in this collection (used by the eval harness
        to rebuild its index from scratch each run)."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.get_or_create_collection(self.collection_name)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        count = self.collection.count()
        if count == 0:
            return []
        results = self.collection.query(
            query_embeddings=self._embed([query]),
            n_results=min(top_k, count),
        )
        # Chroma types these as Optional (they're None if excluded via
        # `include=`); we always request them, but guard so it type-checks.
        docs = (results["documents"] or [[]])[0]
        metas = (results["metadatas"] or [[]])[0]
        # Flatten Chroma's nested result format into something easier to use
        return [
            {"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"]}
            for doc, meta in zip(docs, metas)
        ]