"""
Generation layer: send retrieved chunks + question to Llama 3.3 70B via
Groq's free API.

WHY Groq specifically: genuinely free tier, OpenAI-compatible API shape, and fast LPU
inference. Tradeoff: rate-limited and only serves open-source models — so citation discipline may
be slightly less strict than Claude's. That's a good thing to actually
measure in eval/eval_harness.py rather than take my word for.

WHY the prompt forces citations: an LLM given retrieved text will happily
blend it with its own outside knowledge unless explicitly told not to.
Forcing "only answer from the provided context, and cite which chunk"
is what turns this from a generic chatbot into something you can actually
trust and measure faithfulness on in eval/eval_harness.py.

Requires a GROQ_API_KEY environment variable (see README for setup).
"""

import os
from groq import Groq

MODEL_NAME = "llama-3.3-70b-versatile"

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

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

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1000,
    )
    return response.choices[0].message.content