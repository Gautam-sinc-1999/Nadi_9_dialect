"""Stage 3: understand every episode line before translating it, then order the work by risk.

For each line: who speaks to whom (and who outranks whom), what the line needs (tense, negation,
question, command, names, numbers, humour), which Nadi-9 words exist for its English words and how
well supported they are, which rule disputes and documented gaps it touches, and timing.
Everything inferred is written down as an assumption.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Optional

from . import english as E
from .budget import Budget
from .lexicon import parse_gloss
from .models import (Candidate, ClaimStatus, EpisodeLine, EpisodePlan, HypStatus, LineAnalysis, Role,
                     RuleTemplate, ScenePlan, SocialLink)
from .pipeline import Learned
from .social import character_link, relation_tags

DEFAULT_LIMITS = {"max_chars_per_line": 42, "max_lines": 2, "max_chars_per_second": 17, "min_duration_ms": 1000}
STATUS_RANK = {"corroborated": 0, "loanword": 0, "resolved_conflict": 1, "agreed_untested": 2,
               "single_source": 3, "unresolved_conflict": 4}
WEAK = {"single_source", "agreed_untested"}
PRONOUNS = {"i", "me", "my", "you", "your", "he", "she", "him", "her", "we", "us", "our", "they", "them", "their"}
DISCOURSE = {"hey", "oh", "ah", "well", "just", "so", "and", "but", "or", "at", "all", "as", "much", "now",
             "then", "very", "too", "here", "there", "with", "us"}
WH = {"what", "who", "where", "when", "why", "how", "which"}
TITLES = {"sir", "madam", "headman", "mother", "father", "brother", "sister", "friend", "grandfather",
          "grandmother", "boss", "elder", "son", "daughter", "uncle", "aunt"}
HUMOUR = re.compile(r"joke|teas|grin|funny|laugh|sarcas|pun\b", re.I)
IMPERATIVE_STARTERS = {"come", "go", "eat", "sit", "forgive", "ask", "look", "wait", "tell", "give", "take",
                       "call", "listen", "stop", "stay", "leave", "don't", "do", "let", "be", "help", "bring"}
CONSTRUCTIONS = {
    "clock time": (re.compile(r"\bat (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|\d{1,2})\b|o'clock", re.I),
                   re.compile(r"o'clock|clock|time of day|hours?\b", re.I)),
    "comparison": (re.compile(r"\b(twice|than|as much|more|less)\b", re.I), re.compile(r"compar", re.I)),
    "conditional": (re.compile(r"\bif\b", re.I), re.compile(r"condition", re.I)),
}
RISK = {"gap": 3.0, "weak": 1.5, "unresolved": 3.0, "clash": 4.0, "disputed": 2.0, "grammar_gap": 3.0,
        "construction": 2.0, "humour": 1.5, "numbers": 1.0, "timing": 1.0, "pronoun": 0.5, "unknown_addressee": 1.0}


# ---------- line features ----------

def _clauses(text: str) -> list[str]:
    return [c.strip() for c in re.split(r"[,.!?;]+", text) if c.strip()]


AUX = {"have", "has", "had", "haven't", "hasn't", "hadn't", "'ve", "be", "being", "been", "am", "is", "are",
       "was", "were", "do", "does", "did", "don't", "doesn't", "didn't", "will", "won't", "would", "can",
       "could", "shall", "should", "'s", "'m", "'re", "'ll"}


def normalise_line(text: str) -> str:
    """'thank you' is a formula, not an address to someone."""
    return re.sub(r"\bthank you\b", "thanks", text, flags=re.I)


def features(text: str, names: set[str], limits: dict[str, Any]) -> dict[str, Any]:
    text = normalise_line(text)
    words = [w.lower() for w in E.words(text)]
    chunks = _clauses(text)
    vocative = next((c for c in chunks if len(c.split()) <= 3 and all(
        w.lower() in TITLES or w.lower() in names or w.lower() in {"elder", "younger", "my", "hey", "dear"}
        for w in E.words(c)) and any(w.lower() in TITLES | names for w in E.words(c))), None)
    first_words = [E.words(c)[0].lower() for c in chunks if E.words(c)]
    main = next((c for c in chunks if c != vocative), "")
    main_first = (E.words(main) or [""])[0].lower()
    return {
        "tenses": sorted(E.tense_cues(text)),
        "negated": E.has_negation(text),
        "question": text.strip().endswith("?"),
        "yes_no_question": E.is_yes_no_question(text),
        "wh_question": text.strip().endswith("?") and main_first in WH,
        "imperative": any(w in IMPERATIVE_STARTERS for w in first_words) and main_first not in PRONOUNS | WH,
        "negative_command": bool(re.match(r"\s*(don't|do not|never)\b", main, re.I)),
        "vocative": vocative,
        "second_person": bool({"you", "your"} & set(words)),
        "you_subject": "you" in words,  # 'your food' needs a possessive, not an honorific verb
        "names": sorted({w for w in names if re.search(rf"\b{re.escape(w)}\b", text, re.I)}),
        "numbers": [w for w in words if w in E.NUMBERS or w.isdigit()],
        "exclamation": "!" in text,
    }


# ---------- lexical coverage ----------

def reverse_index(learned: Learned) -> dict[str, list[tuple[str, str, Optional[str], set[str]]]]:
    """English lemma -> [(term, claim status, entry, full sense words)]."""
    idx: dict[str, list] = defaultdict(list)
    for c in learned.claims:
        if c.kind != "word" or c.status == ClaimStatus.UNSUPPORTED:
            continue
        glosses = [c.meaning] if c.meaning else [s.gloss for s in c.senses]
        for g in glosses:
            senses, _ = parse_gloss(g)
            for sense in senses:
                for w in sense:
                    idx[w].append((c.term, c.status.value, c.chosen_entry, sense))
    return idx


def lookup(word: str, line_lemmas: set[str], idx) -> list[tuple[str, str, Optional[str]]]:
    hits = idx.get(word) or [x for k, v in idx.items() if E.same(k, word) for x in v]
    full = [h for h in hits if E.overlap(h[3], line_lemmas) == h[3]]  # whole sense present in the line
    pool = full or hits
    seen, out = set(), []
    for term, st, entry, _ in sorted(pool, key=lambda h: STATUS_RANK.get(h[1], 9)):
        if term not in seen:
            seen.add(term)
            out.append((term, st, entry))
    return out


# ---------- main ----------

def analyse_episode(learned: Learned, budget: Budget) -> EpisodePlan:
    canon, profile = learned.canon, learned.profile
    ep_src = next((s for s in profile.sources if s.role == Role.EPISODE), None)
    meta = ep_src.metadata if ep_src else {}
    limits = {**DEFAULT_LIMITS, **(meta.get("subtitle_limits") or {})}
    scenes = {str(s.get("scene") or s.get("id")): s for s in meta.get("scenes", []) if isinstance(s, dict)}

    chars = {c.name.lower(): c for c in canon.characters}
    names = set(chars)
    idx = reverse_index(learned)
    loanwords = {t.lower() for t in profile.shape.unanalysed_tokens}

    lexical_hyps: dict[str, list] = defaultdict(list)
    for h in learned.hypotheses:
        if h.template == RuleTemplate.LEXICAL_CHOICE and h.status != HypStatus.REFUTED and h.params.get("term"):
            lexical_hyps[h.params["term"]].append(h)
    gaps = [h for h in learned.hypotheses if h.template == RuleTemplate.GAP and not h.resolved_by]
    doc_text = " ".join(s.text for s in canon.sections if "quarantined:injection" not in s.flags)
    all_terms = {c.term for c in learned.claims if c.kind == "word"}

    social_cache: dict[tuple[str, str], SocialLink] = {}
    analyses: list[LineAnalysis] = []
    prev_speaker: dict[Optional[str], Optional[str]] = {}

    for line in canon.lines:
        f = features(line.text, names, limits)
        speaker = _char_name(line.speaker, chars)
        addressee, how = _addressee(line, f, chars, prev_speaker.get(line.scene), speaker)
        prev_speaker[line.scene] = speaker
        la = LineAnalysis(line_id=line.line_id, ref=line.ref, scene=line.scene, speaker=speaker,
                          addressee=addressee, addressee_method=how, text=line.text, features=f)
        if how in ("vocative", "turn_taking"):
            la.assumptions.append(f"addressee assumed to be {addressee} ({how.replace('_', ' ')})")

        # social link
        if speaker and addressee:
            key = (speaker, addressee)
            if key not in social_cache:
                social_cache[key] = _link(speaker, addressee, chars)
            la.social = social_cache[key]
            if la.social.status != "unknown":
                la.assumptions.append(f"{addressee} is {la.social.status} relative to {speaker}: {la.social.reason}")

        # relationship state: the line's own direction note wins over the scene summary
        scene_summary = str(scenes.get(line.scene or "", {}).get("summary", ""))
        tags = relation_tags(line.notes) or relation_tags(scene_summary)
        la.tags = sorted(tags)
        if tags:
            la.assumptions.append(f"relationship state {sorted(tags)} from "
                                  f"{'line note' if relation_tags(line.notes) else 'scene summary ' + (line.scene or '')}")

        _coverage(la, line, f, idx, loanwords, names)
        _sensitivity(la, lexical_hyps, learned)
        _grammar_gaps(la, gaps, doc_text, all_terms)
        _timing(la, line, limits)
        _risk(la, line.notes)
        analyses.append(la)

    order = _order(analyses, scenes)
    return EpisodePlan(episode=ep_src.source_id if ep_src else "episode", order=order, lines=analyses,
                       social=list(social_cache.values()), limits=limits, budget=_budget(budget, order),
                       notes=[f"{sum(1 for a in analyses if a.gaps)} line(s) contain words with no Nadi-9 evidence",
                              f"{sum(1 for a in analyses if any(s.startswith('clash') for s in a.sensitive))} line(s) risk a status clash"])


def _char_name(raw: Optional[str], chars) -> Optional[str]:
    if not raw:
        return None
    low = raw.lower().strip()
    if low in chars:
        return chars[low].name
    hit = next((c.name for k, c in chars.items() if re.search(rf"\b{re.escape(k)}\b", low)), None)
    return hit or raw


def _addressee(line: EpisodeLine, f, chars, prev: Optional[str], speaker: Optional[str]) -> tuple[Optional[str], str]:
    if line.addressee:
        return _char_name(line.addressee, chars), "given"
    voc = (f.get("vocative") or "").lower()
    if voc:
        for k, c in chars.items():
            if k in voc or any(w in (c.description or "").lower() for w in E.words(voc) if len(w) > 3):
                if c.name != speaker:
                    return c.name, "vocative"
    if prev and prev != speaker:
        return prev, "turn_taking"
    return None, "none"


def _link(speaker: str, addressee: str, chars) -> SocialLink:
    if addressee.lower() in {"everyone", "all", "group"}:
        return SocialLink(speaker=speaker, addressee=addressee, relation="group", status="mixed", method="none",
                          reason="addressed to several people")
    s, a = chars.get(speaker.lower()), chars.get(addressee.lower())
    if not s or not a:
        return SocialLink(speaker=speaker, addressee=addressee, relation="unknown", status="unknown",
                          method="none", reason="character not in the character list")
    rel, st, how, why = character_link(s.name, s.description or "", s.age, a.name, a.description or "", a.age)
    return SocialLink(speaker=speaker, addressee=addressee, relation=rel, status=st, method=how, reason=why,
                      refs=[s.ref, a.ref])


def _coverage(la: LineAnalysis, line: EpisodeLine, f, idx, loanwords: set[str], names: set[str]) -> None:
    raw = [w.lower() for w in E.words(normalise_line(line.text))]
    lemmas = {E.lemma(w) for w in raw}
    wanted: list[tuple[str, str]] = []   # (word as written, lemma)
    for w in raw:
        lw = E.lemma(w)
        if w in names or lw in {x[1] for x in wanted} or w in AUX:
            continue
        if w in E.STOP or w in DISCOURSE or w in {"not", "no", "n't"}:
            if w in PRONOUNS and idx.get(lw):  # pronouns matter: tu / tavo is a real choice
                wanted.append((w, lw))
            continue
        wanted.append((w, lw))
    for w, lw in wanted:
        if w in loanwords:  # check the written form: 'meeting' must not become 'meet' first
            la.coverage.append(Candidate(english=w, term=w, status="loanword"))
            continue
        cands = lookup(lw, lemmas, idx)
        if cands:
            term, st, entry = cands[0]
            la.coverage.append(Candidate(english=w, term=term, status=st, entry=entry))
            if st in WEAK:
                la.weak_words.append(f"{w} -> {term} ({st}, {entry})")
            if st == "unresolved_conflict":
                la.conflicts.append(f"{w} -> {term}: dictionaries disagree, unresolved")
            elif st == "resolved_conflict":
                la.conflicts.append(f"{w} -> {term}: dictionaries disagreed; resolved to {entry}")
            for alt, ast, aentry in cands[1:3]:
                if w in PRONOUNS or alt != term:
                    la.coverage.append(Candidate(english=w, term=alt, status=ast, entry=aentry))
        else:
            la.coverage.append(Candidate(english=w, status="gap"))
            la.gaps.append(w)


def _fits(h, status: str) -> Optional[bool]:
    """Does lexical-choice rule h allow its word for this addressee status? None = rule does not say."""
    p = h.params
    if status in set(p.get("forbidden_status") or []):
        return False
    if p.get("allowed_status"):
        if status in p["allowed_status"]:
            return True
        return None if status in {"unknown", "elder_sibling", "mixed"} else False
    return None


def _sensitivity(la: LineAnalysis, lexical_hyps, learned: Learned) -> None:
    status = la.social.status if la.social else "unknown"
    trusted = (HypStatus.SUPPORTED, HypStatus.WEAK)
    # pronoun alternatives (tu / tavo): a choice to make, not a clash
    pron = sorted({c.term for c in la.coverage if c.english in PRONOUNS and c.term and c.term in lexical_hyps})
    if len(pron) > 1:
        ok = [t for t in pron if any(_fits(h, status) for h in lexical_hyps[t] if h.status in trusted)]
        la.sensitive.append(f"choice: {'/'.join(pron)} by addressee status ({status})"
                            + (f" -> rules point to {'/'.join(ok)}" if ok else " -> rules do not decide"))
    # the one word a line would actually use: can it be said to this addressee?
    chosen = {c.term for c in la.coverage if c.term and c.term not in pron}
    for term in sorted(chosen):
        hyps = lexical_hyps.get(term, [])
        for h in hyps:
            if h.status in trusted and _fits(h, status) is False:
                la.sensitive.append(f"clash: '{term}' {h.params.get('condition', '')} but addressee is {status} ({h.hyp_id})")
                break
    terms = chosen | set(pron)
    for term in sorted(terms):
        hyps = lexical_hyps.get(term, [])
        # each competing pair once, whichever side it was found from (A vs B, never also B vs A or A vs A)
        pairs = sorted({tuple(sorted((h.hyp_id, r))) for h in hyps for r in h.competes_with if r != h.hyp_id})
        if pairs:
            la.sensitive.append(f"disputed: '{term}' rules compete ({'; '.join(f'{a} vs {b}' for a, b in pairs[:3])})")


def _grammar_gaps(la: LineAnalysis, gaps, doc_text: str, all_terms: set[str]) -> None:
    f = la.features
    status = la.social.status if la.social else "unknown"
    terms = {c.term for c in la.coverage if c.term}
    plural_loan = any(c.status == "loanword" and re.search(rf"\b{re.escape(c.english)}s\b", la.text, re.I)
                      for c in la.coverage)
    checks = [
        (r"negat", f["negated"]), (r"future", "future" in f["tenses"]), (r"command|imperative", f["imperative"]),
        (r"question word", f["wh_question"]), (r"elder sibling", status == "elder_sibling"),
        (r"\baddress", f["second_person"] or bool(f["vocative"])),
        (r"present", "present" in f["tenses"]),
        # the honorific ending goes on the verb when 'you' is the subject (stated, or implied in a command)
        (r"honorific|respect", status == "higher" and (f["you_subject"] or f["imperative"])),
        (r"number", bool(f["numbers"]) and any(c.status == "gap" for c in la.coverage if c.english.isdigit())),
        (r"loanword", plural_loan), (r"plural", plural_loan),
    ]
    for g in gaps:
        topic_all = str(g.params.get("topic") or g.quote or "").lower()
        for topic in re.split(r";|\band\b(?= negative)", topic_all):  # "future negation; negative commands"
            matched = [ok for pat, ok in checks if re.search(pat, topic)]
            # a gap that names a word (e.g. "when dovanor is used") applies only where that word is a candidate
            matched += [t in terms for t in all_terms if len(t) > 3 and re.search(rf"\b{re.escape(t)}\b", topic)]
            if matched and all(matched):
                la.grammar_gaps.append(f"{g.hyp_id}: {topic.strip()}")
                break
    for name, (detect, documented) in CONSTRUCTIONS.items():
        if detect.search(la.text) and not documented.search(doc_text):
            la.grammar_gaps.append(f"undocumented construction: {name}")


def _timing(la: LineAnalysis, line: EpisodeLine, limits) -> None:
    dur = line.end_ms - line.start_ms
    cps = len(line.text) / max(dur / 1000, 0.001)
    issues = []
    if dur < limits["min_duration_ms"]:
        issues.append(f"duration {dur} ms < {limits['min_duration_ms']} ms")
    if cps > limits["max_chars_per_second"]:
        issues.append(f"source reads at {cps:.0f} chars/s > {limits['max_chars_per_second']}")
    la.timing = {"duration_ms": dur, "source_cps": round(cps, 1), "issues": issues}


def _risk(la: LineAnalysis, notes: Optional[str]) -> None:
    def add(kind: str, n: float, why: str) -> None:
        if n:
            la.risk += RISK[kind] * n
            la.risk_factors.append(f"+{RISK[kind] * n:g} {why}")
    add("gap", len(la.gaps), f"no evidence for {la.gaps}")
    add("weak", len(la.weak_words), f"weak words {[w.split(' ->')[0] for w in la.weak_words]}")
    add("unresolved", sum("unresolved" in c for c in la.conflicts), "unresolved dictionary conflict")
    add("clash", sum(s.startswith("clash") for s in la.sensitive), "status clash")
    add("disputed", sum(s.startswith("disputed") for s in la.sensitive), "disputed word choice")
    add("pronoun", sum(s.startswith("choice") for s in la.sensitive), "pronoun choice by status")
    add("grammar_gap", sum(not g.startswith("undocumented") for g in la.grammar_gaps), "documented grammar gap")
    add("construction", sum(g.startswith("undocumented") for g in la.grammar_gaps), "undocumented construction")
    add("humour", 1 if HUMOUR.search(la.text + " " + (notes or "")) else 0, "humour")
    add("numbers", 1 if la.features["numbers"] else 0, "numbers")
    add("timing", 1 if la.timing.get("issues") else 0, "timing")
    add("unknown_addressee", 1 if la.features["second_person"] and not (la.social and la.social.status != "unknown") else 0,
        "addressee status unknown")
    la.risk = round(la.risk, 2)


def _order(analyses: list[LineAnalysis], scenes) -> list[ScenePlan]:
    by_scene: dict[str, list[LineAnalysis]] = defaultdict(list)
    for a in analyses:
        by_scene[a.scene or "all"].append(a)
    plans = []
    for sc, ls in by_scene.items():
        top = max(ls, key=lambda a: a.risk)
        plans.append(ScenePlan(scene=sc, lines=[a.line_id for a in ls], max_risk=top.risk,
                               total_risk=round(sum(a.risk for a in ls), 2),
                               rationale=f"riskiest line {top.line_id} ({top.risk}): "
                                         f"{'; '.join(top.risk_factors[:2]) or 'no risk factors'}"))
    return sorted(plans, key=lambda p: (-p.max_risk, -p.total_risk))


def _budget(budget: Budget, order: list[ScenePlan]) -> dict[str, Any]:
    u = budget.usage
    left = u.max_model_calls - u.model_calls
    translate = len(order)
    reserve = min(4, max(0, left - translate))
    critic = len(order) if left - translate - reserve >= len(order) else 0
    return {"model_calls_used_so_far": u.model_calls, "model_calls_left": left, "translation_calls": translate,
            "critic_calls": critic, "replan_reserve": reserve,
            "note": "critic skipped: not enough budget" if not critic else "critic enabled"}
