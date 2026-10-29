"""Input and output guardrails.

Input (before any LLM call): length limits, prompt-injection patterns,
de-anonymisation requests, abuse/threat terms, spam. Cheap, deterministic rules.
A second line of defence comes from the safety flags the LLM returns in triage.

Output (before anything is displayed): PII re-scan and citation grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.config import MAX_FEEDBACK_CHARS

MIN_CHARS = 15

INJECTION = [
    # imperative "ignore/disregard/forget ... instructions" at the start of a sentence,
    # so "my manager tends to ignore previous instructions from HR" is still allowed
    r"(^|[.!?]\s+|[-:,]\s*|\b(please|kindly|also)\s+)(ignore|disregard|forget)\s+(all\s+|the\s+|any\s+|your\s+)?"
    r"(previous|prior|above|earlier|your|these|those|policy|handbook|rules|instructions|guidance|system)\b",
    r"\bignore\s+(all\s+)?(previous|prior)\s+(instructions|directions|rules|guidance|prompts?)\b(?!\s+from\s+hr)",
    r"\bforget\s+(all\s+|your\s+)?(rules|instructions)\b",
    r"\b(system|hidden)\s+(prompt|instructions?)\b.*\b(show|print|reveal|output|repeat|translate)\b",
    r"\b(show|print|reveal|output|repeat|translate)\b.*\b(system|hidden)\s+(prompt|instructions?)\b",
    r"\byou\s+are\s+now\b", r"\b(you\s+are\s+(now\s+)?in|enter|enable|activate)\s+developer\s+mode\b",
    r"\bjailbreak\b", r"\bDAN\b",
    r"\bsystem\s*override\b", r"#{2,}\s*system", r"\bsystem\s*:", r"</?\s*feedback\s*>",
    r"\bnew\s+(rule|instruction)s?\b.*\b(from|:)", r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\bact\s+as\s+(my|a|an|the)\s+(lawyer|database|admin|hacker)\b",
    r"\bselect\s+\*\s+from\b", r"\bapi\s+keys?\b", r"\bconfiguration\b.*\bmodel\s+name\b",
    r"\bset\s+escalate\s*=", r"\bclassify\s+this\s+as\b", r"\bmark\s+this\s+ticket\s+resolved\b",
    r"\buse\s+this\s+exact\s+output\b", r"\brespond\s+only\s+with\b",
    r"\brepeat\s+the\s+text\s+of\s+the\s+previous\b",
    r"\bdo\s+not\s+do\s+any\s+hr\s+classification\b",
    r"\bkeep\s+all\s+names\s+exactly\b",
]
DEANON = [
    r"\bwho\s+(wrote|submitted|complained|sent|raised)\b",
    r"\bwhich\s+(employee|person|team\s+member)\s+(wrote|submitted|complained)\b",
    r"\b(guess|reveal|identify|tell\s+me)\b.*\b(author|identity|who\s+(wrote|submitted))\b",
    r"\bticket\s+ids?\s+of\s+everyone\b", r"\bexact\s+timestamps?\b",
    r"\b(individually|word\s+for\s+word|original\s+(wording|text))\b",
    r"\b(if|whether|did)\s+[\w .']{1,40}?\s+submit(ted)?\s+(anything|any\s+feedback|feedback|a\s+complaint)\b",
    r"\bwhat\s+(exactly\s+)?did\s+(each|the\s+vp|the\s+head)\b.*\b(say|write)\b",
    r"\bwhich\s+team\s+does\s+the\s+author\b",
]
ABUSE = [
    r"\bidiots?\b", r"\bmorons?\b", r"\bstupid\s+(company|managers|survey|people)\b", r"\btrash\b",
    r"\bgarbage\b", r"\bshut\s+up\b", r"\bjoke\s+of\s+a\s+human\b",
    r"\bthie(f|ves)\s+and\s+a?\s*liars?\b", r"\bevery\s+single\s+one\s+of\s+them\b.*\b(thie|liar)",
    r"\b(smash|kill|hurt|beat)\b.*\b(him|her|them|laptops?|everyone|you)\b",
    r"\bwill\s+make\s+sure\b.*\bregrets?\b", r"\bsee\s+what\s+happens\s+to\s+him\b",
    r"\bhope\s+you\s+all\s+get\s+fired\b", r"\bf+u+c+k+", r"\bbastards?\b",
]
SPAM = [
    r"https?://|bit\.ly|www\.", r"\bcrypto\b", r"\b\d{3,}\s*%\s*returns?\b",
    r"\bearn\s+₹?\s*[\d,]+\b", r"\bwhatsapp\s+me\b", r"\bguaranteed\b.*\breturns?\b",
    r"\b(asdf\w*|qwerty\w*|lorem\s+ipsum)\b", r"\b(test\s+){2,}test\b",
]


@dataclass
class GuardResult:
    allowed: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def message(self) -> str:
        if self.allowed:
            return ""
        r = self.reasons[0]
        return {
            "too_short": "Please describe your concern in a little more detail.",
            "too_long": f"Please keep feedback under {MAX_FEEDBACK_CHARS} characters.",
            "prompt_injection": "This submission looks like an instruction to the system rather than feedback, so it was not accepted.",
            "deanonymisation": "Requests to identify other employees or their feedback cannot be processed. Everyone's feedback is anonymous.",
            "abuse_or_threat": "This submission contains abusive or threatening language. Please rephrase your concern; if anyone is in danger, contact Security or call emergency services.",
            "spam": "This submission looks like spam or a link, so it was not accepted.",
            "off_topic": "This doesn't look like workplace feedback. Please describe a concern or suggestion about work.",
        }.get(r, "This submission could not be accepted.")


def _any(patterns, text):
    return any(re.search(p, text, re.I | re.M) for p in patterns)


def check_input(text: str) -> GuardResult:
    t = text.strip()
    reasons = []
    letters = sum(c.isalpha() for c in t)
    if len(t) < MIN_CHARS or letters < 10:
        reasons.append("too_short")
    if len(t) > MAX_FEEDBACK_CHARS:
        reasons.append("too_long")
    if _any(INJECTION, t):
        reasons.append("prompt_injection")
    if _any(DEANON, t):
        reasons.append("deanonymisation")
    if _any(ABUSE, t):
        reasons.append("abuse_or_threat")
    if _any(SPAM, t):
        reasons.append("spam")
    return GuardResult(not reasons, reasons)


def check_llm_flags(flags: dict) -> GuardResult:
    """Second line of defence: safety flags returned by the triage call."""
    mapping = {"injection_attempt": "prompt_injection", "deanonymisation_attempt": "deanonymisation",
               "abusive_or_threat": "abuse_or_threat", "off_topic": "off_topic"}
    reasons = [mapping[k] for k, v in flags.items() if v and k in mapping]
    return GuardResult(not reasons, reasons)


# ------------------------------------------------------------------ output

def pii_clean(text: str) -> tuple[str, int]:
    """Re-scan any text before display. Returns (clean text, number of items removed)."""
    from app.pipeline.privacy import GENERIC, PLACEHOLDER, scrub
    r = scrub(text)
    return PLACEHOLDER.sub(lambda m: GENERIC[m.group(1)], r.text), len(r.spans)


def grounded(citations: list[str], retrieved_ids: list[str]) -> bool:
    """Every cited clause must be one we actually retrieved and showed the model."""
    return bool(citations) and all(c in retrieved_ids for c in citations)
