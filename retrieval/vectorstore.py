"""
Retrieval layer: embed text and store/query it in a local Chroma DB.

WHY this is wrapped in a class (an interface) instead of calling Chroma
directly from main.py: it means main.py never needs to know HOW retrieval
works, only THAT it can call add_chunks() and search(). If you swap Chroma
for another vector DB later (e.g. Qdrant, when you add multi-user hosting),
only this file changes — nothing else in the codebase needs to know.
This is the same reasoning behind wrapping the LLM in generation/llm.py.
"""

import chromadb
from sentence_transformers import SentenceTransformer

# WHY all-MiniLM-L6-v2: small (~80MB), fast on CPU, no GPU required,
# strong quality-to-size tradeoff for a first pass. Swap this string
# to try a bigger/better embedding model later and compare eval numbers.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, persist_dir: str = "./chroma_data", user_id: str = "default_user"):
        # WHY persist_dir instead of an in-memory client: you want your
        # notes to survive a server restart without re-uploading everything.
        self.client = chromadb.PersistentClient(path=persist_dir)

        # WHY collection name includes user_id even though there's only one
        # user right now: this is the schema decision that makes adding
        # real multi-user auth later a config change, not a rewrite.
        self.collection = self.client.get_or_create_collection(f"notes_{user_id}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def add_chunks(self, chunks: list[str], source: str):
        embeddings = self.embedder.encode(chunks).tolist()
        ids = [f"{source}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": source, "chunk_index": i} for i in range(len(chunks))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )
        # Flatten Chroma's nested result format into something easier to use
        return [
            {"text": doc, "source": meta["source"], "chunk_index": meta["chunk_index"]}
            for doc, meta in zip(results["documents"][0], results["metadatas"][0])
        ]