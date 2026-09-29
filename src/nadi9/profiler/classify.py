"""Decide what each file *is* (dictionary, examples, grammar, ...).

Order: pack manifest -> structural heuristics -> one LLM call on a small sample -> UNCLASSIFIED.
A file is never silently forced into a role.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ..models import Role
from .discover import RawSource, norm_key
from .mapping import foreign_sentence_columns

ACCEPT = 0.6
MARGIN = 0.15

MANIFEST_TYPES = {
    "dictionary": Role.DICTIONARY, "lexicon": Role.DICTIONARY, "glossary": Role.DICTIONARY,
    "approved_examples": Role.EXAMPLES, "examples": Role.EXAMPLES, "parallel": Role.EXAMPLES,
    "grammar_note": Role.GRAMMAR, "grammar": Role.GRAMMAR,
    "expert_notes": Role.EXPERT_NOTES, "experts": Role.EXPERT_NOTES,
    "viewer_feedback": Role.FEEDBACK, "feedback": Role.FEEDBACK,
    "interview": Role.INTERVIEW, "audio_interview": Role.INTERVIEW,
    "episode": Role.EPISODE, "transcript": Role.EPISODE,
    "characters": Role.CHARACTERS, "event": Role.EVENT, "correction": Role.EVENT,
}

TERM_KEYS = {"term", "word", "headword", "lemma", "entry", "form"}
GLOSS_KEYS = {"gloss", "meaning", "definition", "sense", "english", "translation"}
SRC_KEYS = {"source", "english", "en", "sourcetext", "original"}
TGT_KEYS = {"target", "targettext", "translation", "dialect"}
TIME_KEYS = {"start", "starttime", "in", "timecodein", "begin"}
ASR_KEYS = {"asrconfidence", "confidence", "conf", "interviewergloss", "gloss", "transcript"}
FEEDBACK_KEYS = {"user", "username", "likes", "upvotes", "rating", "comment"}
EXPERT_KEYS = {"parentname", "parentrole", "expert", "expertid", "parentexpertid"}
EVENT_KEYS = {"eventid", "change", "correction", "trigger"}
GRAMMAR_WORDS = ["suffix", "prefix", "tense", "negation", "word order", "verb", "noun",
                 "plural", "particle", "subject", "object", "honorific", "politeness"]


@dataclass
class RoleDecision:
    role: Role
    confidence: float
    method: str
    reason: str


def from_manifest(src: RawSource, manifest: Optional[dict[str, Any]]) -> Optional[RoleDecision]:
    if not manifest:
        return None
    for s in manifest.get("sources", []):
        if s.get("path") == src.rel_path:
            role = MANIFEST_TYPES.get(str(s.get("type", "")).lower())
            if role:
                return RoleDecision(role, 0.95, "manifest", f"manifest type '{s.get('type')}'")
    return None


def heuristic_scores(src: RawSource) -> dict[Role, float]:
    name = src.rel_path.lower()
    keys = src.keys | {norm_key(k) for k in src.file_meta}
    sc: dict[Role, float] = {r: 0.0 for r in Role}

    def has(ks: set[str]) -> bool:
        return bool(keys & ks)

    if re.search(r"(^|/)readme", name):
        sc[Role.DOCUMENTATION] = 0.95
    if re.search(r"manifest", name) and src.format == "json":
        sc[Role.MANIFEST] = 0.9
    if name.startswith("events/") or "/events/" in name or re.search(r"correction|erratum", name):
        sc[Role.EVENT] += 0.7
    if has(EVENT_KEYS):
        sc[Role.EVENT] += 0.4

    if src.format == "audio":
        sc[Role.INTERVIEW] = 0.9
    if src.format in {"srt", "vtt"}:
        sc[Role.EPISODE] = 0.9

    if src.records:
        if has(TERM_KEYS) and has(GLOSS_KEYS) and not has(TIME_KEYS):
            sc[Role.DICTIONARY] += 0.75
        if has(SRC_KEYS) and (has(TGT_KEYS) or foreign_sentence_columns(src.records)):
            sc[Role.EXAMPLES] += 0.6 + (0.1 if has({"speaker", "context", "relationship"}) else 0)
        if has(TIME_KEYS) and has({"text", "line", "dialogue"}) and has({"speaker", "character"}):
            sc[Role.EPISODE] += 0.7
        if has(TIME_KEYS) and has(ASR_KEYS):
            sc[Role.INTERVIEW] += 0.7
        elif has(ASR_KEYS) and has({"transcript", "conf", "asrconfidence"}):
            sc[Role.INTERVIEW] += 0.4
        if has(FEEDBACK_KEYS):
            sc[Role.FEEDBACK] += 0.7
        if has(EXPERT_KEYS):
            sc[Role.EXPERT_NOTES] += 0.7
        if has({"name"}) and has({"description", "age", "bio"}) and not has({"text"}):
            sc[Role.CHARACTERS] += 0.6
    if src.records_key and re.search(r"segment|utterance|turn", src.records_key.lower()):
        sc[Role.INTERVIEW] += 0.5

    if src.text is not None and src.format not in {"srt", "vtt"}:
        low = src.text.lower()
        hits = sum(low.count(w) for w in GRAMMAR_WORDS)
        if hits >= 8:
            sc[Role.GRAMMAR] += min(0.9, 0.5 + hits / 60)
        if re.search(r"\bcharacters?\b", name):
            sc[Role.CHARACTERS] += 0.6
        # letter/email-style attributed notes ("From: ...", "Date: ...")
        if len(re.findall(r"(?im)^(from|author|by|date)\s*:", src.text)) >= 2:
            sc[Role.EXPERT_NOTES] += 0.25

    # For unstructured text the file name is most of the signal available, so it weighs more there.
    name_weight = 0.45 if src.text is not None and not src.records else 0.2
    for pat, role in [(r"dict|lexic|gloss|word.?list", Role.DICTIONARY), (r"example|parallel|corpus", Role.EXAMPLES),
                      (r"grammar", Role.GRAMMAR), (r"expert|specialist|linguist", Role.EXPERT_NOTES),
                      (r"feedback|comment|review", Role.FEEDBACK), (r"interview|recording", Role.INTERVIEW),
                      (r"episode|script", Role.EPISODE), (r"character|cast", Role.CHARACTERS)]:
        if re.search(pat, name):
            sc[role] += name_weight
    return {r: min(v, 1.0) for r, v in sc.items()}


def heuristic_decision(src: RawSource) -> tuple[Optional[RoleDecision], dict[Role, float]]:
    sc = heuristic_scores(src)
    ranked = sorted(sc.items(), key=lambda kv: kv[1], reverse=True)
    (r1, s1), (_, s2) = ranked[0], ranked[1]
    if s1 >= ACCEPT and s1 - s2 >= MARGIN:
        return RoleDecision(r1, round(s1, 2), "heuristic", f"structure/name score {s1:.2f} (next {s2:.2f})"), sc
    return None, sc


def llm_sample(src: RawSource) -> dict[str, Any]:
    """What the model sees: file name, keys and at most 3 short records. Never the whole file."""
    return {
        "source_id": src.source_id,
        "path": src.rel_path,
        "format": src.format,
        "keys": sorted(src.keys)[:30],
        "sample_records": [{k: str(v)[:80] for k, v in r.items()} for r in src.records[:3]],
        "text_head": (src.text or "")[:600],
        "allowed_roles": [r.value for r in Role],
    }
