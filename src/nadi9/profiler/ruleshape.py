"""The exact shape of a rule, shared by every prompt, plus a deterministic normaliser for the
variants real models actually return.

Why this exists: the first real run (openai/gpt-oss-120b on Groq) returned correct, well-quoted rules
in its own vocabulary ({"order": "SOV"}, {"tense_suffixes": {...}}, "allowed_status": "higher",
"term": "you (familiar)"), and the validator rejected all 23 grammar rules. The fix is twofold:
say the exact shape in the prompt (RULE_SCHEMA), and map common variants in code (coerce).
Anything that still does not fit becomes UNTESTABLE, which keeps the quoted claim visible to
reviewers instead of throwing it away. Grounding (the quote check) is never relaxed.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..models import RuleTemplate

RULE_SCHEMA = """Rule templates and their EXACT params (use these keys and values, nothing else):
- word_order: {"verb_position": "final" | "initial" | "medial"}          (SOV = final, VSO = initial)
- affix: ONE rule per ending: {"kind": "suffix" | "prefix", "form": "en" (no hyphen),
         "feature": "tense=past" | "tense=present" | "tense=future" | "number=plural" | "case=locative" | ...,
         "attaches_to": "verb" | "noun"}
- particle_position: {"particle": "na", "relative_to": "verb" | "clause_end" | "clause_start",
         "position": "before" | "after" | "at", "feature": "negation" | "yes_no_question" | ...}
- agreement: {"if_word": "tavo", "then_affix": "sa", "on": "verb", "tenses": ["present", "past", "future"]}
- lexical_choice: {"term": <the Nadi-9 word exactly as written, never its English meaning>,
         "condition": <plain words>, and where the text says so (each a LIST of higher | equal | lower | elder_sibling):
           "allowed_status": the word is used ONLY for these addressees,
           "permitted_status": the word is fine for these (says nothing about others),
           "forbidden_status": the word must NOT be used for these,
         and "requires_tag" / "forbidden_tag": "estranged" | "reconciled" when use depends on the relationship state}
  Status is the addressee relative to the speaker: higher (parents, elders, bosses, headman, guests),
  equal (friends, peers), lower (children, younger people), elder_sibling (older brother or sister).
- gap: {"topic": <what the text says is undocumented or unclear>}
- untestable: {}   (a real rule that fits none of the above)
Example: {"section_id": "neg-1", "template": "particle_position",
  "params": {"particle": "na", "relative_to": "verb", "position": "before", "feature": "negation"},
  "quote": "The particle `na` is placed immediately **before** the verb."}"""

REQUIRED: dict[RuleTemplate, set[str]] = {
    RuleTemplate.WORD_ORDER: {"verb_position"},
    RuleTemplate.AFFIX: {"kind", "form", "feature"},
    RuleTemplate.PARTICLE_POSITION: {"particle", "relative_to", "position"},
    RuleTemplate.AGREEMENT: {"if_word", "then_affix"},
    RuleTemplate.LEXICAL_CHOICE: {"term", "condition"},
    RuleTemplate.GAP: {"topic"},
    RuleTemplate.CODE_SWITCHING: set(),
    RuleTemplate.UNTESTABLE: set(),
}
STATUS_VALUES = {"higher", "equal", "lower", "elder_sibling"}
ORDER_TO_VERB = {"sov": "final", "osv": "final", "verb-final": "final", "verb final": "final", "final": "final",
                 "vso": "initial", "vos": "initial", "verb-initial": "initial", "initial": "initial",
                 "svo": "medial", "ovs": "medial", "medial": "medial"}
POSITION_WORDS = [  # (pattern, relative_to, position)
    (r"pre-?verb|before (the )?verb|preverbal", "verb", "before"),
    (r"post-?verb|after (the )?verb|postverbal", "verb", "after"),
    (r"(clause|sentence)[- ]?(final|end)|\bend\b|\bfinal\b", "clause_end", "at"),
    (r"(clause|sentence)[- ]?(initial|start)|\bstart\b|\bbegin", "clause_start", "at"),
]
REGISTER_STATUS = {"familiar": ["equal", "lower"], "informal": ["equal", "lower"],
                   "respectful": ["higher"], "polite": ["higher"], "honorific": ["higher"], "formal": ["higher"]}
FEATURE_NAMES = {"plural": "number=plural", "locative": "case=locative", "present": "tense=present",
                 "past": "tense=past", "future": "tense=future"}

Coerced = tuple[RuleTemplate, dict[str, Any], Optional[str]]


def coerce(template: Any, params: Any) -> list[Coerced]:
    """Return one or more (template, params, note). note is None when nothing had to change."""
    params = dict(params) if isinstance(params, dict) else {}
    try:
        tpl = RuleTemplate(template)
    except ValueError:
        return [(RuleTemplate.UNTESTABLE, params, f"unknown template {template!r}; kept as untestable")]
    out = _FIX[tpl](params)
    fixed = []
    for t, p, note in out:
        p = _lists(p)
        missing = REQUIRED[t] - set(k for k, v in p.items() if v not in (None, "", []))
        if missing:
            fixed.append((RuleTemplate.UNTESTABLE, params,
                          f"did not fit {t.value} (missing {sorted(missing)}); kept as untestable"))
        else:
            fixed.append((t, p, note))
    return fixed


def _lists(p: dict[str, Any]) -> dict[str, Any]:
    for k in ("allowed_status", "forbidden_status", "permitted_status"):
        v = p.get(k)
        if isinstance(v, str):
            p[k] = [x.strip() for x in re.split(r"[,/|]| or ", v) if x.strip()]
    return p


def _word_order(p: dict[str, Any]) -> list[Coerced]:
    if p.get("verb_position") in ("final", "initial", "medial"):
        return [(RuleTemplate.WORD_ORDER, p, None)]
    for k in ("verb_position", "order", "basic_order", "default", "word_order"):
        v = str(p.get(k, "")).strip().lower()
        if v in ORDER_TO_VERB:
            return [(RuleTemplate.WORD_ORDER, {"verb_position": ORDER_TO_VERB[v]}, f"normalised {k}={v!r}")]
    return [(RuleTemplate.UNTESTABLE, p, "word-order detail that is not about the verb position; kept as untestable")]


_TRIGGER = re.compile(r"\b(?:subject|word|pronoun)\s+(?:is|=)\s+[`'\"]?([A-Za-z-]+)", re.I)


def _affix(p: dict[str, Any]) -> list[Coerced]:
    # an ending that depends on another word is agreement: {"honorific_suffix": "-sa", "condition": "subject is tavo"}
    trig = _TRIGGER.search(str(p.get("condition") or p.get("trigger") or ""))
    form = next((str(v) for k, v in p.items() if isinstance(v, str) and re.search(r"(suffix|prefix|form)$", k.lower())), None)
    if trig and form and form.strip("-"):
        return [(RuleTemplate.AGREEMENT, {"if_word": trig.group(1).lower(), "then_affix": form.strip("-"), "on": "verb"},
                 "normalised to agreement: the ending depends on another word")]
    if {"kind", "form", "feature"} <= set(p):
        return [(RuleTemplate.AFFIX, {**p, "form": str(p["form"]).strip("-")}, None)]
    out: list[Coerced] = []
    for k, v in p.items():
        kl = k.lower()
        kind = "prefix" if "prefix" in kl else "suffix"
        if isinstance(v, dict):  # {"tense_suffixes": {"present": "-a", "past": "-en"}}
            base = kl.replace("_suffixes", "").replace("_prefixes", "").replace("_suffix", "").replace("_prefix", "")
            for feat, form in v.items():
                if isinstance(form, str) and form.strip("-"):
                    out.append((RuleTemplate.AFFIX, {"kind": kind, "form": form.strip("-"),
                                                     "feature": f"{base}={feat}" if base else str(feat),
                                                     "attaches_to": "verb" if "tense" in base else p.get("attaches_to", "")},
                                f"normalised from {k}"))
        elif isinstance(v, str) and re.search(r"(suffix|prefix)$", kl) and v.strip("-"):
            name = kl.rsplit("_", 1)[0]
            out.append((RuleTemplate.AFFIX, {"kind": kind, "form": v.strip("-"),
                                             "feature": FEATURE_NAMES.get(name, f"role={name}")},
                        f"normalised from {k}"))
    return out or [(RuleTemplate.UNTESTABLE, p, "affix without a form and meaning; kept as untestable")]


def _particle(p: dict[str, Any]) -> list[Coerced]:
    if {"particle", "relative_to", "position"} <= set(p):
        return [(RuleTemplate.PARTICLE_POSITION, p, None)]
    where = str(p.get("position") or p.get("relative_to") or "").lower()
    for pat, rel, pos in POSITION_WORDS:
        if re.search(pat, where) and p.get("particle"):
            q = {"particle": p["particle"], "relative_to": rel, "position": pos}
            feat = p.get("feature") or p.get("function")
            if feat:
                q["feature"] = "yes_no_question" if "question" in str(feat) else "negation" if "negat" in str(feat) else feat
            return [(RuleTemplate.PARTICLE_POSITION, q, f"normalised position {where!r}")]
    return [(RuleTemplate.UNTESTABLE, p, "particle position not recognised; kept as untestable")]


def _lexical(p: dict[str, Any]) -> list[Coerced]:
    out: list[Coerced] = []
    for k in ("pronoun_form", "possessive_form", "form"):
        if isinstance(p.get(k), dict):  # {"pronoun_form": {"familiar": "tu", "respectful": "tavo"}}
            for register, term in p[k].items():
                q = {"term": str(term), "condition": str(register)}
                if register.lower() in REGISTER_STATUS:
                    q["allowed_status"] = REGISTER_STATUS[register.lower()]
                out.append((RuleTemplate.LEXICAL_CHOICE, q, f"normalised from {k}"))
    if out:
        return out
    q = dict(p)
    note = None
    if not q.get("condition"):
        extra = {k: v for k, v in q.items() if k != "term"}
        q["condition"] = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in extra.items()) or "as stated in the quote"
        note = "condition filled from the other params"
    return [(RuleTemplate.LEXICAL_CHOICE, q, note)]


def _gap(p: dict[str, Any]) -> list[Coerced]:
    if p.get("topic"):
        return [(RuleTemplate.GAP, p, None)]
    text = next((str(v) for v in p.values() if isinstance(v, str) and v), "")
    return [(RuleTemplate.GAP, {"topic": text}, "topic taken from the other params")] if text else \
        [(RuleTemplate.UNTESTABLE, p, "gap without a topic; kept as untestable")]


def _agreement(p: dict[str, Any]) -> list[Coerced]:
    if p.get("if_word") and p.get("then_affix"):
        return [(RuleTemplate.AGREEMENT, {**p, "then_affix": str(p["then_affix"]).strip("-")}, None)]
    return [(RuleTemplate.UNTESTABLE, p, "agreement without trigger word and ending; kept as untestable")]


_FIX = {
    RuleTemplate.WORD_ORDER: _word_order, RuleTemplate.AFFIX: _affix,
    RuleTemplate.PARTICLE_POSITION: _particle, RuleTemplate.LEXICAL_CHOICE: _lexical,
    RuleTemplate.GAP: _gap, RuleTemplate.AGREEMENT: _agreement,
    RuleTemplate.UNTESTABLE: lambda p: [(RuleTemplate.UNTESTABLE, p, None)],
    RuleTemplate.CODE_SWITCHING: lambda p: [(RuleTemplate.CODE_SWITCHING, p, None)],
}
