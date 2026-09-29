"""Trust boundary: detect text in the pack that tries to instruct the agent.

Flagged records are quarantined. They stay in the audit trail but are never used as
evidence and never placed in a prompt.
"""
from __future__ import annotations

import re

_PATTERNS = [
    r"\bignore\b.{0,40}\b(rules?|instructions?|evidence|previous|above|dictionary|notes?)\b",
    r"\b(ai|llm|model|assistant|translator|agent)s?\b.{0,60}\b(must|should|always|never)\b",
    r"\bnote to (any )?(ai|llm|model|assistant|translator|agent)",
    r"\binstruction(s)? for (ai|llm|model|assistant|translation system)",
    r"\bmark (every|all)\b.{0,40}\b(approved|subtitles?|lines?)\b",
    r"\bconfidence\s*(=|of|to)?\s*1(\.0)?\b",
    r"\b(do not|don't|never)\b.{0,30}\b(human review|review items?|send .{0,20} for review)\b",
    r"\bsystem (prompt|note|message)\b",
    r"\bdeprecated\b.{0,40}\brules?\b",
]
_RX = [re.compile(p, re.IGNORECASE) for p in _PATTERNS]


def injection_hits(text: str | None) -> list[str]:
    if not text:
        return []
    return [rx.pattern for rx in _RX if rx.search(text)]


def is_injection(text: str | None) -> bool:
    """Two independent pattern hits, to keep ordinary sentences like 'the model must...' out."""
    return len(injection_hits(text)) >= 2
