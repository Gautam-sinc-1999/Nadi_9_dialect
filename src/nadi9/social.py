"""Who is talking to whom: addressee status and relationship tags, from free-text descriptions.

Keyword-based on purpose: every result is an *assumption* that is shown to reviewers, never a fact.
Status values: higher | equal | lower | elder_sibling | unknown
"""
from __future__ import annotations

import re

HIGHER = {"boss", "employer", "headman", "chief", "elder", "elders", "father", "mother", "parent", "parents",
          "grandfather", "grandmother", "teacher", "guest", "sir", "master", "superior", "landlord"}
LOWER = {"child", "children", "son", "daughter", "kid", "employee", "worker", "servant", "student", "junior"}
EQUAL = {"equals", "equal", "friend", "friends", "neighbour", "neighbor", "spouse", "spouses", "peer",
         "peers", "colleague", "coworker", "classmate", "husband", "wife"}
ESTRANGED = re.compile(r"quarrel|estrang|angry|not speaking|feud|fell out|hurt|cold", re.I)
RECONCILED = re.compile(r"reconcil|made up|apologi|said sorry|forgave|forgiven|again", re.I)


def _rank(text: str) -> int | None:
    ws = set(re.findall(r"[a-z]+", text.lower()))
    if ws & HIGHER:
        return 2
    if ws & LOWER:
        return 0
    if ws & EQUAL:
        return 1
    return None


def addressee_status(speaker: str | None, addressee: str | None, relationship: str | None) -> tuple[str, str]:
    """Returns (status, reason)."""
    rel = (relationship or "").lower()
    adr = (addressee or "").lower()
    if re.search(r"sibling|brother|sister", rel + " " + adr):
        if re.search(r"\b(elder|older|big)\b", adr):
            return "elder_sibling", f"addressee is '{addressee}'"
        if re.search(r"\b(younger|little)\b", adr):
            return "lower", f"addressee is '{addressee}'"
    if re.search(r"\b(equals?|friends?|peers?|spouses)\b", rel):
        return "equal", f"relationship '{relationship}'"
    m = re.match(r"\s*(.+?)\s+to\s+(.+)", rel)
    if m:
        a, b = _rank(m.group(1)), _rank(m.group(2))
        if a is not None and b is not None:
            s = "higher" if b > a else "lower" if b < a else "equal"
            return s, f"relationship '{relationship}'"
        if b == 2:
            return "higher", f"relationship '{relationship}'"
    r = _rank(adr)
    if r is not None:
        return {2: "higher", 1: "equal", 0: "lower"}[r], f"addressee is '{addressee}'"
    return "unknown", "no usable relationship information"


AUTHORITY = {"headman", "chief", "boss", "employer", "teacher", "grandfather", "grandmother", "priest",
             "officer", "master", "landlord", "manager", "king", "queen", "elder"}
_KIN_PHRASE = re.compile(r"\b(elder|older|younger|big|little)\s+(brother|sister)\b", re.I)


def _mentions(desc: str, name: str) -> bool:
    return bool(re.search(rf"\b{re.escape(name)}\b", desc, re.I))


def _authority(desc: str) -> bool:
    return bool(set(re.findall(r"[a-z]+", _KIN_PHRASE.sub(" ", desc.lower()))) & AUTHORITY)


def character_link(s_name: str, s_desc: str, s_age: int | None,
                   a_name: str, a_desc: str, a_age: int | None) -> tuple[str, str, str, str]:
    """(relation, addressee status, method, reason) between two characters, from their descriptions."""
    s_desc, a_desc = s_desc or "", a_desc or ""
    # 1. kinship / work relation stated in either description
    for desc, other, speaker_side in ((s_desc, a_name, True), (a_desc, s_name, False)):
        if not _mentions(desc, other):
            continue
        d = desc.lower()
        if re.search(r"\b(sister|brother|sibling)\b", d):
            older = (a_age or 0) > (s_age or 0) if (a_age and s_age) else bool(re.search(r"younger", s_desc, re.I))
            return ("sibling", "elder_sibling" if older else "lower", "kinship",
                    f"siblings ({desc!r}); {a_name} is {'older' if older else 'younger'}")
        if re.search(r"\b(mother|father|parent)\b", d):
            return ("parent_child", "lower" if speaker_side else "higher", "kinship", f"{desc!r}")
        if re.search(r"\b(son|daughter|child)\b", d):
            return ("parent_child", "higher" if speaker_side else "lower", "kinship", f"{desc!r}")
        if re.search(r"\b(boss|employer|manager)\b", d):
            return ("employer", "lower" if speaker_side else "higher", "kinship", f"{desc!r}")
        if re.search(r"\bfriend", d):
            return ("friend", "equal", "kinship", f"{desc!r}")
    # 2. a role that carries authority
    sa, aa = _authority(s_desc), _authority(a_desc)
    if aa and not sa:
        return "authority", "higher", "role", f"{a_name}: {a_desc!r}"
    if sa and not aa:
        return "authority", "lower", "role", f"{s_name}: {s_desc!r}"
    # 3. age
    if s_age and a_age:
        d = a_age - s_age
        if d >= 20:
            return "elder", "higher", "age", f"{a_name} is {d} years older"
        if d <= -20:
            return "elder", "lower", "age", f"{a_name} is {-d} years younger"
        if abs(d) < 10:
            return "peer", "equal", "age", f"similar ages ({s_age}, {a_age})"
    return "unknown", "unknown", "none", "no relation stated and no usable ages"


def relation_tags(*texts: str | None) -> set[str]:
    blob = " ".join(t for t in texts if t)
    tags = set()
    if ESTRANGED.search(blob):
        tags.add("estranged")
    if RECONCILED.search(blob):
        tags.add("reconciled")
        if re.search(r"\b(now|again|after)\b", blob, re.I):  # the reconciliation came later
            tags.discard("estranged")
    return tags
