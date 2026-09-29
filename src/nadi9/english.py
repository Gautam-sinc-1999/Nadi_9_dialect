"""Tiny source-language normaliser (English), used only to compare glosses with example translations.

Deliberately crude and swappable: if the episode's source language changes, replace this module.
Both sides of every comparison go through the same function, so crude stems still line up.
"""
from __future__ import annotations

import re

STOP = {
    "the", "a", "an", "is", "are", "am", "was", "were", "be", "been", "i", "you", "he", "she", "we",
    "they", "it", "me", "him", "her", "us", "them", "to", "of", "and", "or", "in", "on", "at", "my",
    "your", "his", "our", "their", "will", "did", "do", "does", "for", "with", "this", "that", "as",
    "by", "from", "used", "when", "only", "form", "e", "g", "etc", "also", "one's",
}
IRREGULAR = {
    "went": "go", "gone": "go", "going": "go", "goes": "go", "came": "come", "coming": "come",
    "ate": "eat", "eaten": "eat", "eating": "eat", "saw": "see", "seen": "see", "seeing": "see",
    "did": "do", "done": "do", "said": "say", "says": "say", "left": "leave", "leaving": "leave",
    "knew": "know", "known": "know", "forgave": "forgive", "forgiven": "forgive", "gave": "give",
    "brought": "bring", "took": "take", "taken": "take", "sat": "sit", "children": "child",
    "people": "person", "men": "man", "women": "woman", "is": "be", "am": "be", "are": "be",
    "was": "be", "were": "be", "has": "have", "had": "have", "thanks": "thank",
}
NUMBERS = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "twenty": "20",
    "hundred": "100",
}


# Words that describe the same register. Kept small and explicit so a reviewer can audit it.
SYNONYMS = {
    "polite": "respectful", "honorific": "respectful", "respect": "respectful", "deference": "respectful",
    "deferential": "respectful", "courteous": "respectful",
    "informal": "familiar", "casual": "familiar", "intimate": "familiar",
}


def lemma(w: str) -> str:
    w = w.lower().strip("'\"")
    if w in NUMBERS:
        return NUMBERS[w]
    if w in IRREGULAR:
        return IRREGULAR[w]
    if w in SYNONYMS:
        return SYNONYMS[w]
    for suf, rep, min_len in (("ies", "y", 5), ("ied", "y", 5), ("ing", "", 6), ("ed", "", 5), ("es", "", 5), ("s", "", 4)):
        if w.endswith(suf) and len(w) >= min_len and not w.endswith("ss"):
            w = w[: -len(suf)] + rep
            if suf == "ing" and len(w) > 3 and w[-1] == w[-2]:
                w = w[:-1]
            break
    return SYNONYMS.get(w, w)


_PAST = {"did", "was", "were", "had", "yesterday", "ago", "last", "went", "came", "ate", "saw", "said",
         "left", "knew", "forgave", "gave", "brought", "took", "sat", "held", "told", "thought", "made"}
_FUTURE = {"will", "shall", "tomorrow", "'ll"}
_PRESENT = {"am", "is", "are", "do", "does", "has", "have", "'s", "'re", "'m", "now"}
_PARTICIPLES = {"come", "gone", "been", "done", "seen", "eaten", "left", "made", "taken", "given", "said",
                "known", "forgiven", "changed", "become", "run", "sat"}
_NOT_ED = {"need", "speed", "feed", "bed", "red", "seed", "indeed", "hundred", "wed"}
_AUX_Q = {"do", "does", "did", "will", "would", "can", "could", "is", "are", "am", "was", "were",
          "shall", "should", "have", "has"}
_NEG = {"not", "n't", "never", "no", "don't", "didn't", "doesn't", "won't", "can't"}


def tense_cues(text: str) -> set[str]:
    """Which tenses the English sentence signals. A set, because one line can hold two clauses."""
    toks = [t.lower() for t in re.findall(r"[a-zA-Z']+", text or "")]
    out = set()
    for t in toks:
        if t in _FUTURE or t.endswith("'ll"):
            out.add("future")
        elif t in _PAST or (t.endswith("ed") and len(t) > 4 and t not in _NOT_ED):
            out.add("past")
        elif t in _PRESENT:
            out.add("present")
    for a, b in zip(toks, toks[1:] + [""]):  # perfect: "has come", "have changed" -> also past
        if a in {"has", "have", "'ve", "haven't", "hasn't"} and (b in _PARTICIPLES or (b.endswith("ed") and len(b) > 3)):
            out.add("past")
    if "did" in toks and "future" not in out:
        out.discard("present")
    return out


def is_yes_no_question(text: str) -> bool:
    t = (text or "").strip()
    if not t.endswith("?"):
        return False
    last = re.split(r",", t)[-1].strip().lower()
    first = re.findall(r"[a-z']+", last)
    return bool(first) and first[0] in _AUX_Q


def has_negation(text: str) -> bool:
    return any(t in _NEG or t.endswith("n't") for t in re.findall(r"[a-zA-Z']+", (text or "").lower()))


def words(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+|\d+", text or "")


def content_lemmas(text: str) -> set[str]:
    return {lemma(w) for w in words(text) if w.isdigit() or (w.lower() not in STOP and len(w) > 1)}


def same(a: str, b: str) -> bool:
    """Tolerant comparison for crude stems: 'arriv' vs 'arrive'."""
    if a == b:
        return True
    return min(len(a), len(b)) >= 4 and (a.startswith(b) or b.startswith(a))


def overlap(needles: set[str], hay: set[str]) -> set[str]:
    return {n for n in needles if any(same(n, h) for h in hay)}
