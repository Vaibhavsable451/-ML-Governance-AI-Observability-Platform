"""LLM access. Uses Claude when ANTHROPIC_API_KEY is set, else an offline extractive answerer."""
from __future__ import annotations

import re

from core import config

SYSTEM = ("You are an AI governance assistant. Answer ONLY from the provided context. "
          "Cite the source in square brackets, e.g. [model-risk-policy]. "
          "If the context is insufficient, say so plainly. Never invent numbers.")


def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def cost_usd(tokens_in: int, tokens_out: int) -> float:
    return round(tokens_in / 1e6 * config.LLM_PRICE_IN_PER_1M + tokens_out / 1e6 * config.LLM_PRICE_OUT_PER_1M, 6)


def _mock(question: str, contexts: list[dict]) -> str:
    q = set(re.findall(r"[a-z0-9]{3,}", question.lower()))
    scored = []
    for c in contexts:
        for s in re.split(r"(?<=[.!?])\s+|\n+", c["text"]):
            t = set(re.findall(r"[a-z0-9]{3,}", s.lower()))
            if t and len(t & q):
                scored.append((len(t & q) / (len(t) ** 0.5), s.strip(), c["source"]))
    scored.sort(key=lambda x: x[0], reverse=True)
    picked = scored[:3]
    if not picked:
        return "I could not find supporting evidence for that in the governance documents."
    return " ".join(f"{s} [{src}]" for _, s, src in picked)


def generate(question: str, contexts: list[dict]) -> dict:
    ctx = "\n\n".join(f"[{c['source']}] {c['text']}" for c in contexts)
    if config.ANTHROPIC_API_KEY:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
            resp = client.messages.create(
                model=config.LLM_MODEL, max_tokens=600, system=SYSTEM,
                messages=[{"role": "user", "content": f"Context:\n{ctx}\n\nQuestion: {question}"}])
            text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
            return {"text": text, "tokens_in": resp.usage.input_tokens, "tokens_out": resp.usage.output_tokens,
                    "provider": "anthropic"}
        except Exception:
            pass  # fall through to offline mode
    text = _mock(question, contexts)
    return {"text": text, "tokens_in": _est_tokens(SYSTEM + ctx + question), "tokens_out": _est_tokens(text),
            "provider": "mock"}
