"""LLM guardrails: PII detection/redaction, prompt-injection and jailbreak scoring, policy checks."""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

MAX_QUERY_CHARS = 4000
BLOCK_THRESHOLD = 0.6

# Priority order matters: earlier types claim their spans first.
_PII_ORDER = ["CREDIT_CARD", "SSN", "AADHAAR", "PAN", "EMAIL", "PHONE"]
_PII_RE = {
    "CREDIT_CARD": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
    "SSN": re.compile(r"(?<!\d)\d{3}-\d{2}-\d{4}(?!\d)"),
    "AADHAAR": re.compile(r"(?<!\d)\d{4}\s\d{4}\s\d{4}(?!\d)|(?<!\d)\d{12}(?!\d)"),
    "PAN": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b"),
    "PHONE": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}|\d{5}[\s-]?\d{5})(?!\d)"),
}

_INJECTION = [
    (r"ignore (?:all |any )?(?:the )?(?:previous|prior|above|earlier) (?:instructions|prompts?|rules)", 0.9),
    (r"disregard (?:all |any )?(?:the )?(?:previous|prior|above|earlier)", 0.9),
    (r"(?:reveal|show|print|repeat|output|leak) (?:me )?(?:your|the) (?:system|hidden|initial|original) "
     r"(?:prompt|instructions|message)", 0.9),
    (r"do not (?:follow|obey) (?:your|the) (?:rules|guidelines|polic(?:y|ies)|instructions)", 0.8),
    (r"override (?:your|the) (?:safety|instructions|polic(?:y|ies)|guardrails)", 0.8),
    (r"new instructions\s*:", 0.5),
    (r"you are now (?:an? )?", 0.4),
    (r"<\|?(?:system|im_start)\|?>", 0.7),
    (r"\bsystem prompt\b", 0.4),
]
_JAILBREAK = [
    (r"\bDAN\b", 0.7), (r"developer mode", 0.7), (r"\bjailbreak", 0.8),
    (r"(?:without|no) (?:any )?(?:restrictions|filters|limitations)", 0.7),
    (r"\bunfiltered\b", 0.6), (r"pretend (?:you|that you) (?:have no|are not bound|can do anything)", 0.8),
    (r"act as (?:an? )?(?:unrestricted|evil|unfiltered)", 0.8),
]
_POLICY = [
    (re.compile(r"disable (?:the )?(?:guardrails?|monitoring|logging|audit)", re.I), "attempt to disable controls"),
    (re.compile(r"bypass (?:the )?(?:approval|review|governance)", re.I), "attempt to bypass governance"),
]


@dataclass
class ScanResult:
    pii: list = field(default_factory=list)          # [{"type","start","end"}]
    injection_score: float = 0.0
    jailbreak_score: float = 0.0
    policy_violations: list = field(default_factory=list)
    blocked: bool = False
    reasons: list = field(default_factory=list)
    sanitized_text: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["pii_types"] = sorted({p["type"] for p in self.pii})
        return d


def luhn_ok(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
        alt = not alt
    return total % 10 == 0


def detect_pii(text: str) -> list[dict]:
    found, claimed = [], []
    for kind in _PII_ORDER:
        for m in _PII_RE[kind].finditer(text):
            s, e = m.span()
            if any(s < ce and e > cs for cs, ce in claimed):
                continue
            if kind == "CREDIT_CARD":
                digits = re.sub(r"\D", "", m.group())
                if not (13 <= len(digits) <= 19 and luhn_ok(digits)):
                    continue
            claimed.append((s, e))
            found.append({"type": kind, "start": s, "end": e})
    return sorted(found, key=lambda x: x["start"])


def redact(text: str, pii: list[dict]) -> str:
    out, last = [], 0
    for p in pii:
        out.append(text[last:p["start"]])
        out.append(f"[{p['type']}]")
        last = p["end"]
    out.append(text[last:])
    return "".join(out)


def _score(text: str, patterns) -> float:
    prod = 1.0
    for pat, w in patterns:
        if re.search(pat, text, flags=re.I if pat != r"\bDAN\b" else 0):
            prod *= 1 - w
    return round(1 - prod, 3)


def scan(text: str) -> ScanResult:
    r = ScanResult()
    r.pii = detect_pii(text)
    r.sanitized_text = redact(text, r.pii)
    r.injection_score = _score(text, _INJECTION)
    r.jailbreak_score = _score(text, _JAILBREAK)
    r.policy_violations = [msg for rx, msg in _POLICY if rx.search(text)]
    if len(text) > MAX_QUERY_CHARS:
        r.policy_violations.append("input too long")
    if r.injection_score >= BLOCK_THRESHOLD:
        r.reasons.append("prompt injection")
    if r.jailbreak_score >= BLOCK_THRESHOLD:
        r.reasons.append("jailbreak attempt")
    r.reasons += r.policy_violations
    r.blocked = bool(r.reasons)
    return r
