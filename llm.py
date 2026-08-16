"""
Generation layer: send retrieved chunks + question to Claude via the
Anthropic API.

WHY this is wrapped in a function instead of called inline from main.py:
this is the payoff of that design decision — swapping from a local Ollama
model to a cloud API touched ONLY this file. main.py, vectorstore.py, and
everything else are unchanged. If you swap to OpenAI later, or back to a
local model for the multi-user hosting phase, same story.

WHY the prompt forces citations: an LLM given retrieved text will happily
blend it with its own outside knowledge unless explicitly told not to.
Forcing "only answer from the provided context, and cite which chunk"
is what turns this from a generic chatbot into something you can actually
trust and measure faithfulness on in eval/eval_harness.py.

Requires an ANTHROPIC_API_KEY environment variable (see README for setup).
"""

import os
from anthropic import Anthropic

MODEL_NAME = "claude-haiku-4-5-20251001"

client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

SYSTEM_PROMPT = """You are a study assistant. Answer the user's question using
ONLY the provided context chunks below. If the answer isn't in the context,
say "I don't have that in your notes" — do not use outside knowledge.

After your answer, list which chunk number(s) you used, like: [Sources: 1, 3]
"""


def generate_answer(question: str, retrieved_chunks: list[dict]) -> str:
    context = "\n\n".join(
        f"[Chunk {i}] (from {c['source']}):\n{c['text']}"
        for i, c in enumerate(retrieved_chunks)
    )

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    return response.content[0].text