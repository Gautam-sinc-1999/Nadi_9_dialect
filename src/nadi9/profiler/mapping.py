"""Map a file's own field names onto canonical fields, then build canonical records.

Order per field: exact synonym -> fuzzy match -> LLM proposal (validated) -> missing.
"""
from __future__ import annotations

import difflib
import re
from typing import Any, Callable, Optional

from ..models import (Character, EpisodeLine, Example, FieldMapping, LexEntry, Note, Role,
                      Segment)
from .discover import RawSource, norm_key, to_ms

# canonical field -> (synonyms, required)
SPECS: dict[Role, dict[str, tuple[list[str], bool]]] = {
    Role.DICTIONARY: {
        "native_id": (["id", "entryid", "ref", "no", "number"], False),
        "term": (["term", "word", "headword", "lemma", "entry", "form"], True),
        "gloss": (["gloss", "meaning", "definition", "sense", "english", "translation"], True),
        "pos": (["pos", "partofspeech", "category", "wordclass", "class", "type"], False),
        "notes": (["notes", "note", "comment", "comments", "remarks", "usage"], False),
    },
    Role.EXAMPLES: {
        "native_id": (["id", "exampleid", "ref", "no"], False),
        "source_text": (["source", "sourcetext", "english", "en", "original"], True),
        "target_text": (["nadi9", "target", "targettext", "translation", "dialect"], True),
        "speaker": (["speaker", "from"], False),
        "addressee": (["addressee", "listener", "to"], False),
        "relationship": (["relationship", "relation"], False),
        "context": (["context", "situation", "scene", "notes"], False),
        "status": (["status", "approval"], False),
    },
    Role.EPISODE: {
        "line_id": (["id", "lineid", "subtitleid", "cue", "index"], False),
        "start": (["start", "starttime", "in", "timecodein", "begin"], True),
        "end": (["end", "endtime", "out", "timecodeout", "finish"], True),
        "text": (["text", "line", "dialogue", "sourcetext", "subtitle"], True),
        "speaker": (["speaker", "character"], False),
        "addressee": (["addressee", "listener", "to"], False),
        "scene": (["scene", "sceneid"], False),
        "notes": (["notes", "note", "direction", "context"], False),
    },
    Role.INTERVIEW: {
        "native_id": (["seg", "segment", "id"], False),
        "text": (["text", "transcript", "asrtext", "utterance"], False),
        "gloss": (["interviewergloss", "gloss", "translation", "english"], False),
        "asr_confidence": (["asrconfidence", "confidence", "conf"], False),
        "alternatives": (["asralternatives", "alternatives", "nbest"], False),
        "start": (["start", "starttime", "begin"], False),
        "end": (["end", "endtime"], False),
    },
    Role.FEEDBACK: {
        "native_id": (["id", "commentid"], False),
        "author": (["user", "username", "author", "name"], False),
        "date": (["date", "timestamp", "created"], False),
        "text": (["text", "comment", "feedback", "body", "message"], True),
        "likes": (["likes", "upvotes", "votes"], False),
    },
    Role.EXPERT_NOTES: {
        "native_id": (["id", "noteid"], False),
        "author": (["parentname", "name", "expert", "author", "parentexpertid"], False),
        "author_role": (["parentrole", "role", "title"], False),
        "date": (["date", "created"], False),
        "text": (["text", "note", "comment", "body"], True),
        "cited_refs": (["refs", "references", "cites", "sources"], False),
    },
    Role.CHARACTERS: {
        "name": (["name", "character"], True),
        "description": (["description", "bio", "notes"], False),
        "age": (["age"], False),
    },
}


def auto_map(role: Role, records: list[dict[str, Any]]) -> tuple[dict[str, Optional[str]], list[FieldMapping]]:
    spec = SPECS[role]
    src_keys: list[str] = []
    for r in records:  # all rows: a rare column (e.g. one row's notes) is where injected text hides
        for k in r:
            if k not in src_keys:
                src_keys.append(k)
    by_norm = {norm_key(k): k for k in src_keys}
    used: set[str] = set()
    mapping: dict[str, Optional[str]] = {f: None for f in spec}
    methods: dict[str, str] = {}
    # pass 1: exact synonyms for every field, so a fuzzy guess can never steal an exact column
    for field, (syns, _req) in spec.items():
        hit = next((by_norm[s] for s in syns if s in by_norm and by_norm[s] not in used), None)
        if hit is not None:
            used.add(hit)
            mapping[field], methods[field] = hit, "synonym"
    # pass 2: fuzzy, required fields first
    for field, (syns, _req) in sorted(spec.items(), key=lambda kv: not kv[1][1]):
        if mapping[field] is not None:
            continue
        free = [n for n, k in by_norm.items() if k not in used]
        close: list[str] = []
        for s in [norm_key(field)] + syns:
            close = difflib.get_close_matches(s, free, n=1, cutoff=0.85)
            if close:
                break
        if close:
            used.add(by_norm[close[0]])
            mapping[field], methods[field] = by_norm[close[0]], "fuzzy"
    report = [FieldMapping(canonical_field=f, source_field=mapping[f], method=methods.get(f, "missing"))
              for f in spec]
    return mapping, report


_EN_STOP = {"the", "a", "an", "is", "are", "am", "was", "were", "i", "you", "he", "she", "we", "they",
            "it", "to", "of", "and", "in", "on", "at", "my", "your", "will", "did", "do", "not", "for",
            "with", "me", "him", "her", "this", "that", "be", "have", "has", "what", "who", "where"}


def column_stats(records: list[dict[str, Any]], key: str) -> tuple[float, float]:
    """(average tokens per value, share of tokens that are common English function words)."""
    toks = [t for r in records for t in re.findall(r"[a-zA-Z']+", str(r.get(key) or "").lower())]
    n_vals = sum(1 for r in records if r.get(key))
    if not toks or not n_vals:
        return 0.0, 0.0
    return len(toks) / n_vals, sum(t in _EN_STOP for t in toks) / len(toks)


def foreign_sentence_columns(records: list[dict[str, Any]], exclude: set[str] = frozenset()) -> list[str]:
    """Columns holding multi-word text that is clearly not English. Language-agnostic by design."""
    keys = [k for k in {k for r in records for k in r} if k not in exclude]
    out = []
    for k in keys:
        avg, stop = column_stats(records, k)
        if avg >= 2 and stop < 0.1:
            out.append((stop, k))
    return [k for _, k in sorted(out)]


def content_guess(role: Role, mapping: dict[str, Optional[str]], report: list[FieldMapping],
                  records: list[dict[str, Any]]) -> None:
    """Fill a missing target-text column by looking at the values, not the header."""
    if role != Role.EXAMPLES or mapping.get("target_text") or not mapping.get("source_text"):
        return
    _, src_stop = column_stats(records, mapping["source_text"])
    if src_stop < 0.2:
        return
    cands = foreign_sentence_columns(records, exclude={v for v in mapping.values() if v})
    if cands:
        mapping["target_text"] = cands[0]
        for fm in report:
            if fm.canonical_field == "target_text":
                fm.source_field, fm.method = cands[0], "content"


def missing_required(role: Role, mapping: dict[str, Optional[str]]) -> list[str]:
    return [f for f, (_, req) in SPECS[role].items() if req and not mapping.get(f)]


def apply_llm_mapping(role: Role, mapping: dict[str, Optional[str]], report: list[FieldMapping],
                      proposal: dict[str, str], records: list[dict[str, Any]]) -> list[str]:
    """Accept a model-proposed mapping only if the proposed source field really exists."""
    problems = []
    keys = {k for r in records for k in r}
    for field, src_field in proposal.items():
        if field not in SPECS[role]:
            problems.append(f"model proposed unknown field '{field}'")
            continue
        if src_field not in keys:
            problems.append(f"model proposed non-existent column '{src_field}' for '{field}'")
            continue
        mapping[field] = src_field
        for fm in report:
            if fm.canonical_field == field:
                fm.source_field, fm.method = src_field, "llm"
    return problems


# ---------- builders ----------

def _s(v: Any) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _get(rec: dict[str, Any], mapping: dict[str, Optional[str]], f: str) -> Any:
    k = mapping.get(f)
    return rec.get(k) if k else None


def build_records(role: Role, src: RawSource, mapping: dict[str, Optional[str]]) -> tuple[list[Any], list[str]]:
    builder: Callable[..., Any] = BUILDERS[role]
    out, errors = [], []
    for i, rec in enumerate(src.records):
        nid = _s(_get(rec, mapping, "native_id") or _get(rec, mapping, "line_id")) or f"row{i + 1}"
        try:
            out.append(builder(rec, mapping, src.source_id, nid))
        except (ValueError, TypeError) as e:
            errors.append(f"{src.source_id}:{nid}: {e}")
    return out, errors


def _lex(rec, m, sid, nid) -> LexEntry:
    term, gloss = _s(_get(rec, m, "term")), _s(_get(rec, m, "gloss"))
    if not term or not gloss:
        raise ValueError("missing term or gloss")
    return LexEntry(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, term=term.lower(), gloss=gloss,
                    pos=_s(_get(rec, m, "pos")), notes=_s(_get(rec, m, "notes")), raw=rec)


def _example(rec, m, sid, nid) -> Example:
    s, t = _s(_get(rec, m, "source_text")), _s(_get(rec, m, "target_text"))
    if not s or not t:
        raise ValueError("missing source or target text")
    return Example(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, source_text=s, target_text=t,
                   speaker=_s(_get(rec, m, "speaker")), addressee=_s(_get(rec, m, "addressee")),
                   relationship=_s(_get(rec, m, "relationship")), context=_s(_get(rec, m, "context")),
                   status=_s(_get(rec, m, "status")), raw=rec)


def _line(rec, m, sid, nid) -> EpisodeLine:
    start, end, text = to_ms(_get(rec, m, "start")), to_ms(_get(rec, m, "end")), _s(_get(rec, m, "text"))
    if start is None or end is None or not text:
        raise ValueError("missing start/end/text")
    return EpisodeLine(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, line_id=nid, start_ms=start,
                       end_ms=end, text=text, speaker=_s(_get(rec, m, "speaker")),
                       addressee=_s(_get(rec, m, "addressee")), scene=_s(_get(rec, m, "scene")),
                       notes=_s(_get(rec, m, "notes")), raw=rec)


def _segment(rec, m, sid, nid) -> Segment:
    conf = _get(rec, m, "asr_confidence")
    alts = _get(rec, m, "alternatives") or []
    return Segment(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, text=_s(_get(rec, m, "text")),
                   gloss=_s(_get(rec, m, "gloss")),
                   asr_confidence=float(conf) if conf not in (None, "") else None,
                   alternatives=[str(a) for a in alts] if isinstance(alts, list) else [str(alts)],
                   start_ms=to_ms(_get(rec, m, "start")), end_ms=to_ms(_get(rec, m, "end")), raw=rec)


def _note(rec, m, sid, nid) -> Note:
    text = _s(_get(rec, m, "text"))
    if not text:
        raise ValueError("missing text")
    refs = _get(rec, m, "cited_refs") or []
    likes = _get(rec, m, "likes")
    return Note(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, author=_s(_get(rec, m, "author")),
                author_role=_s(_get(rec, m, "author_role")), date=_s(_get(rec, m, "date")), text=text,
                cited_refs=[str(r) for r in refs] if isinstance(refs, list) else [str(refs)],
                likes=int(likes) if str(likes or "").isdigit() else None, raw=rec)


def _character(rec, m, sid, nid) -> Character:
    name = _s(_get(rec, m, "name"))
    if not name:
        raise ValueError("missing name")
    age = _get(rec, m, "age")
    return Character(ref=f"{sid}:{name}", source_id=sid, native_id=name, name=name,
                     description=_s(_get(rec, m, "description")),
                     age=int(age) if str(age or "").isdigit() else None, raw=rec)


BUILDERS: dict[Role, Callable[..., Any]] = {
    Role.DICTIONARY: _lex, Role.EXAMPLES: _example, Role.EPISODE: _line, Role.INTERVIEW: _segment,
    Role.FEEDBACK: _note, Role.EXPERT_NOTES: _note, Role.CHARACTERS: _character,
}
