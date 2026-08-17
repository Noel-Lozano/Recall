"""
FastAPI app — Phase 0 baseline.

WHY no auth/user login yet: with one user (you), auth is complexity that
doesn't teach you anything about RAG quality. The user_id="default_user"
below is a placeholder that keeps the data model ready for real auth later
without a rewrite (see vectorstore.py comments).
"""

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, UploadFile, File

from ingestion.chunking import process_pdf
from retrieval.vectorstore import VectorStore
from generation.llm import generate_answer

app = FastAPI(title="Study Assistant RAG - Phase 0")

# Single shared store for now — becomes per-user later
store = VectorStore(user_id="default_user")


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Save the upload to a temp file since pypdf needs a file path
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    chunks = process_pdf(tmp_path)
    store.add_chunks(chunks, source=file.filename)
    Path(tmp_path).unlink()  # clean up temp file

    return {"filename": file.filename, "chunks_added": len(chunks)}


@app.post("/query")
async def query(question: str, top_k: int = 3):
    retrieved = store.search(question, top_k=top_k)
    if not retrieved:
        return {"answer": "No notes uploaded yet.", "sources": []}

    answer = generate_answer(question, retrieved)
    return {
        "answer": answer,
        "sources": [
            {"source": c["source"], "chunk_index": c["chunk_index"], "preview": c["text"][:150]}
            for c in retrieved
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok"}