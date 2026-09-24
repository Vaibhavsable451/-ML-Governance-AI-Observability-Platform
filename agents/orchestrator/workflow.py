"""LangGraph orchestrator. Falls back to a sequential runner if langgraph is not installed.

START -> input_validator -> security -> pii -> governance -> retrieval -> generate_answer
      -> evaluation -> risk -> human_review -> respond -> END
(input_validator and security short-circuit to respond when the request is blocked)
"""
from __future__ import annotations

import time
from typing import Optional, TypedDict

from agents.evaluation.hallucination import evaluate_groundedness
from agents.governance import registry
from agents.governance.service import evidence_chunks
from agents.security.guardrails import BLOCK_THRESHOLD, MAX_QUERY_CHARS, detect_pii, redact, scan
from core import llm
from monitoring.metrics.recorder import metrics
from rag.retrieval.retriever import Retriever

try:
    from langgraph.graph import END, START, StateGraph
    HAS_LANGGRAPH = True
except Exception:  # pragma: no cover
    HAS_LANGGRAPH = False


class GovernanceState(TypedDict, total=False):
    query: str
    user: str
    model_id: Optional[str]
    sanitized_query: str
    blocked: bool
    block_reason: str
    pii_detected: bool
    pii_types: list
    injection_score: float
    jailbreak_score: float
    evidence: list
    retrieved_docs: list
    answer: str
    llm_usage: dict
    hallucination_score: float
    groundedness: float
    drift_score: float
    bias_score: float
    risk_score: float
    risk_level: str
    request_risk: float
    requires_human_approval: bool
    human_approval: str
    final_response: str
    trace: list


def _traced(name):
    def deco(fn):
        def wrapper(state: GovernanceState) -> dict:
            t0 = time.perf_counter()
            update = fn(state) or {}
            ms = round((time.perf_counter() - t0) * 1000, 2)
            metrics.record("agent_node_ms", ms, "Milliseconds", {"node": name})
            update["trace"] = list(state.get("trace", [])) + [{"node": name, "ms": ms}]
            return update
        wrapper.__name__ = name
        return wrapper
    return deco


@_traced("input_validator")
def input_validator(state):
    q = (state.get("query") or "").strip()
    if not q:
        return {"blocked": True, "block_reason": "empty query"}
    if len(q) > MAX_QUERY_CHARS:
        return {"blocked": True, "block_reason": "input too long"}
    return {"query": q, "blocked": False}


@_traced("security")
def security_agent(state):
    r = scan(state["query"])
    inj, jb = r.injection_score, r.jailbreak_score
    blocked = inj >= BLOCK_THRESHOLD or jb >= BLOCK_THRESHOLD or bool(r.policy_violations)
    if blocked:
        registry.log_event("security", "HIGH", {"blocked": True, "reasons": r.reasons, "user": state.get("user"),
                                                "injection_score": inj, "jailbreak_score": jb}, state.get("model_id"))
    return {"injection_score": inj, "jailbreak_score": jb, "blocked": blocked,
            "block_reason": ", ".join(r.reasons)}


@_traced("pii")
def pii_agent(state):
    pii = detect_pii(state["query"])
    return {"pii_detected": bool(pii), "pii_types": sorted({p["type"] for p in pii}),
            "sanitized_query": redact(state["query"], pii)}


@_traced("governance")
def governance_agent(state):
    mid = state.get("model_id")
    profile = registry.get_profile(mid) if mid else None
    if not profile:
        return {"evidence": []}
    return {"evidence": evidence_chunks(mid), "drift_score": profile["drift"]["drift_score"],
            "bias_score": profile["bias"]["bias_score"], "risk_score": profile["risk"]["score"],
            "risk_level": profile["risk"]["level"]}


@_traced("retrieval")
def retrieval_agent(state):
    docs = Retriever(state.get("evidence")).search(state["sanitized_query"], k=4)
    return {"retrieved_docs": docs}


@_traced("generate_answer")
def generate_answer(state):
    out = llm.generate(state["sanitized_query"], state.get("retrieved_docs", []))
    cost = llm.cost_usd(out["tokens_in"], out["tokens_out"])
    registry.log_usage("chat", out["tokens_in"], out["tokens_out"], cost)
    metrics.record("llm_tokens", out["tokens_in"] + out["tokens_out"], "Count")
    metrics.record("llm_cost_usd", cost, "None")
    return {"answer": out["text"], "llm_usage": {**out, "cost_usd": cost}}


@_traced("evaluation")
def evaluation_agent(state):
    ev = evaluate_groundedness(state["answer"], [d["text"] for d in state.get("retrieved_docs", [])])
    return {"hallucination_score": ev["hallucination_score"], "groundedness": ev["groundedness"]}


@_traced("risk")
def risk_agent(state):
    req = (30 if state.get("pii_detected") else 0) + state.get("hallucination_score", 0) * 0.5 \
        + state.get("injection_score", 0) * 40
    return {"request_risk": round(min(100.0, req), 1)}


@_traced("human_review")
def human_review(state):
    mid = state.get("model_id")
    profile = registry.get_profile(mid) if mid else None
    needs = bool(profile and profile["status"] == "PENDING_REVIEW") or state.get("request_risk", 0) >= 70
    return {"requires_human_approval": needs, "human_approval": "pending" if needs else "not_required"}


@_traced("respond")
def respond(state):
    if state.get("blocked"):
        final = f"Request blocked by governance guardrails ({state.get('block_reason', 'policy')})."
    else:
        final = state.get("answer", "")
        if state.get("hallucination_score", 0) > 70:
            final += "\n\nLow confidence: this answer is weakly supported by the retrieved documents."
        if state.get("pii_detected"):
            final += f"\n\nNote: sensitive data ({', '.join(state['pii_types'])}) was redacted before processing."
        if state.get("requires_human_approval") and state.get("model_id"):
            final += f"\n\nModel {state['model_id']} is pending human review."
    registry.log_event("chat", "INFO", {
        "user": state.get("user"), "blocked": bool(state.get("blocked")),
        "pii_types": state.get("pii_types", []), "hallucination_score": state.get("hallucination_score"),
        "request_risk": state.get("request_risk"), "trace": state.get("trace", [])}, state.get("model_id"))
    return {"final_response": final}


_NODES = [("input_validator", input_validator), ("security", security_agent), ("pii", pii_agent),
          ("governance", governance_agent), ("retrieval", retrieval_agent), ("generate_answer", generate_answer),
          ("evaluation", evaluation_agent), ("risk", risk_agent), ("human_review", human_review),
          ("respond", respond)]
_GATES = {"input_validator", "security"}


class _SequentialGraph:
    def invoke(self, state: dict) -> dict:
        state = dict(state)
        for name, fn in _NODES:
            state.update(fn(state))
            if name in _GATES and state.get("blocked"):
                state.update(respond(state))
                break
        return state


def build_graph():
    if not HAS_LANGGRAPH:
        return _SequentialGraph()
    g = StateGraph(GovernanceState)
    for name, fn in _NODES:
        g.add_node(name, fn)
    g.add_edge(START, "input_validator")
    route = lambda s: "end" if s.get("blocked") else "continue"  # noqa: E731
    g.add_conditional_edges("input_validator", route, {"continue": "security", "end": "respond"})
    g.add_conditional_edges("security", route, {"continue": "pii", "end": "respond"})
    names = [n for n, _ in _NODES]
    for a, b in zip(names[2:-1], names[3:]):
        g.add_edge(a, b)
    g.add_edge("respond", END)
    return g.compile()


_graph = None


def run_workflow(query: str, model_id: str | None = None, user: str = "anonymous") -> dict:
    global _graph
    if _graph is None:
        _graph = build_graph()
    with metrics.timer("chat_latency_ms"):
        state = _graph.invoke({"query": query, "model_id": model_id, "user": user, "trace": []})
    state["engine"] = "langgraph" if HAS_LANGGRAPH else "sequential"
    return state
