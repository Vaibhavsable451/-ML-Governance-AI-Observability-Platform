"""Lightweight groundedness evaluation: how much of an answer is supported by retrieved context."""
from __future__ import annotations

import re

_STOP = set("the a an and or of to in on for with is are was were be by as at it this that from which "
            "has have had not can will should must may any all your our their its into than then also".split())


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z0-9.%/-]+", text.lower()) if len(t) > 2 and t not in _STOP}


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def evaluate_groundedness(answer: str, contexts: list[str], support_threshold: float = 0.5) -> dict:
    ctx_tokens = set().union(*(_tokens(c) for c in contexts)) if contexts else set()
    ctx_text = " ".join(contexts)
    sents = [s for s in _sentences(answer) if _tokens(s)]
    if not sents or not ctx_tokens:
        return {"groundedness": 0.0, "hallucination_score": 100.0 if answer.strip() else 0.0,
                "unsupported_sentences": sents, "unsupported_numbers": []}
    unsupported, supported = [], 0
    for s in sents:
        toks = _tokens(s)
        ratio = len(toks & ctx_tokens) / len(toks)
        if ratio >= support_threshold:
            supported += 1
        else:
            unsupported.append(s)
    nums = set(re.findall(r"\d+(?:\.\d+)?%?", answer))
    bad_nums = sorted(n for n in nums if n not in ctx_text)
    grounded = supported / len(sents)
    score = min(100.0, (1 - grounded) * 100 + 10 * len(bad_nums))
    return {"groundedness": round(grounded, 3), "hallucination_score": round(score, 1),
            "unsupported_sentences": unsupported, "unsupported_numbers": bad_nums}
