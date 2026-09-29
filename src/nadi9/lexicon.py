"""Stage 1: check every dictionary meaning against real usage, resolve disagreements, score sources.

The idea in one sentence: a dictionary says `brem` = "yesterday", but the approved example
"brem mas manu khaniru" is translated "We will eat food tomorrow" -> that meaning is contradicted.

Evidence used, strongest first:
  usage   - the word appears in an example / interview segment whose English contains the meaning
  grammar - the grammar note states the meaning right next to the word
Expert notes and viewer feedback are attached as discussion, never as deciding evidence here.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field

from . import english as E
from .models import (CanonicalPack, ClaimStatus, EntryOutcome, LexClaim, LexEntry, PackProfile, Role,
                     SenseCheck, SourceTrust)
from .profiler.morphology import Morphology, clauses
from .profiler.profile import target_texts, usable_lexicon

PRIOR_STRENGTH = 4          # how many observations the prior is worth
PRIORS = {Role.EXAMPLES: 0.9, Role.GRAMMAR: 0.8, Role.DICTIONARY: 0.6, Role.EXPERT_NOTES: 0.7,
          Role.INTERVIEW: 0.7, Role.FEEDBACK: 0.3, Role.CHARACTERS: 0.9, Role.EPISODE: 1.0}


# ---------- reading a gloss ----------

def parse_gloss(gloss: str) -> tuple[list[set[str]], set[str]]:
    """'elder brother (everyday form)' -> senses [{elder, brother}], qualifiers {everyday}.
    'respected elder; sir. Polite address...' -> senses [{respect, elder}, {sir}], qualifiers {polite, address, ...}."""
    quals = " ".join(re.findall(r"\(([^)]*)\)", gloss))
    g = re.sub(r"\([^)]*\)", " ", gloss)
    head, _, rest = g.partition(". ")
    quals += " " + rest
    senses = []
    for part in re.split(r"[,;/]", head):
        part = re.split(r"\b(?:when|used|only|if|for)\b", part, flags=re.I)[0]
        part = part.strip(" '\"")
        cw = E.content_lemmas(part) or {E.lemma(w) for w in E.words(part)}  # pronouns are all stopwords
        if cw:
            senses.append(cw)
    return senses, E.content_lemmas(quals)


def sense_fits(senses: list[set[str]], hay: set[str]) -> bool:
    for s in senses:
        if s and len(E.overlap(s, hay)) * 2 >= len(s):
            return True
    return False


# ---------- evidence contexts ----------

@dataclass
class Context:
    ref: str
    roots: set[str]
    affixes: set[str]
    english: set[str]          # all lemmas of the English side (stopwords kept: pronouns matter)
    weight: float


@dataclass
class TermEvidence:
    used_in: list[Context] = field(default_factory=list)
    grammar: list[tuple[str, set[str]]] = field(default_factory=list)   # (section ref, window lemmas)


def build_contexts(canon: CanonicalPack, morph: Morphology,
                   example_weights: dict[str, float] | None = None) -> list[Context]:
    ew = example_weights or {}
    eng_by_ref = {e.ref: (e.source_text, ew.get(e.ref, 1.0)) for e in canon.examples}
    for s in canon.segments:
        if s.gloss:
            eng_by_ref[s.ref] = (s.gloss, s.asr_confidence if s.asr_confidence is not None else 0.7)
    out = []
    for ref, txt in target_texts(canon):
        if ref not in eng_by_ref:
            continue
        eng, w = eng_by_ref[ref]
        roots, affixes = set(), set()
        for c in clauses(txt):
            for tok in c:
                a = morph.analyse(tok)
                roots.add(a.root or a.token)
                affixes.update(a.suffixes + a.prefixes)
        out.append(Context(ref, roots, affixes, {E.lemma(x) for x in E.words(eng)}, w))
    return out


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z`*|>-])|\n")


def grammar_windows(canon: CanonicalPack, term: str) -> list[tuple[str, set[str]]]:
    """Every grammar sentence (or table row) that mentions the term. A grammar states a meaning in the
    same sentence as the word, before or after it, so the sentence is the window."""
    rx = re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])" if not term.startswith("-")
                    else rf"(?<![\w]){re.escape(term)}(?![\w])", re.I)
    out = []
    for sec in canon.sections:
        if "quarantined:injection" in sec.flags:
            continue
        for sent in _SENTENCE_SPLIT.split(sec.text):
            if rx.search(sent):
                out.append((sec.ref, {E.lemma(x) for x in E.words(sent)}))
    return out


# ---------- per-term judgement ----------

def check_sense(entry: LexEntry, ev: TermEvidence, is_affix: bool) -> SenseCheck:
    senses, quals = parse_gloss(entry.gloss)
    sc = SenseCheck(entry_ref=entry.ref, gloss=entry.gloss,
                    head_words=sorted(set().union(*senses)) if senses else [], qualifier_words=sorted(quals))
    if not is_affix:
        for ctx in ev.used_in:
            (sc.usage_hits if sense_fits(senses, ctx.english) else sc.usage_misses).append(ctx.ref)
    for ref, win in ev.grammar:
        if sense_fits(senses, win) and ref not in sc.grammar_hits:
            sc.grammar_hits.append(ref)
        if quals and E.overlap(quals, win) and ref not in sc.qualifier_hits:
            sc.qualifier_hits.append(ref)
    w = {c.ref: c.weight for c in ev.used_in}
    sc.score = round(sum(w[r] for r in sc.usage_hits) + (0.6 if sc.grammar_hits else 0.0), 2)
    return sc


def _agree(a: SenseCheck, b: SenseCheck) -> bool:
    return bool(E.overlap(set(a.head_words), set(b.head_words)))


def judge(term: str, entries: list[LexEntry], ev: TermEvidence, is_affix: bool) -> LexClaim:
    checks = [check_sense(e, ev, is_affix) for e in entries]
    used = bool(ev.used_in)
    kind = "affix" if is_affix else "word"

    def fitted(c: SenseCheck) -> bool:
        return bool(c.usage_hits or c.grammar_hits)

    # group entries whose meanings agree
    groups: list[list[SenseCheck]] = []
    for c in checks:
        for g in groups:
            if any(_agree(c, x) for x in g):
                g.append(c)
                break
        else:
            groups.append([c])

    def gscore(g):  # usage outranks grammar
        return (sum(len(c.usage_hits) for c in g), sum(len(c.grammar_hits) for c in g), max(c.score for c in g))

    groups.sort(key=gscore, reverse=True)
    best = groups[0]
    best_fits = any(fitted(c) for c in best)
    conflict = None

    if len(groups) == 1:
        status = ClaimStatus.CORROBORATED if best_fits else (
            ClaimStatus.UNSUPPORTED if used else
            ClaimStatus.SINGLE_SOURCE if len(checks) == 1 else ClaimStatus.AGREED_UNTESTED)
        if status == ClaimStatus.AGREED_UNTESTED:
            reason = f"{len(checks)} dictionaries agree; the word never appears in usable evidence"
        elif status == ClaimStatus.UNSUPPORTED:
            reason = f"used in {len(ev.used_in)} places but no listed meaning fits the English there"
        elif status == ClaimStatus.SINGLE_SOURCE:
            reason = f"only {checks[0].entry_ref} lists it; not seen in examples, interviews or grammar"
        else:
            reason = _fit_reason(best)
        # qualifier conflict: same core meaning, incompatible usage notes (e.g. "plural" vs "respectful")
        qual = [c for c in best if c.qualifier_words]
        if len(qual) >= 2 and not E.overlap(set(qual[0].qualifier_words), set(qual[1].qualifier_words)):
            q_hit = [c for c in qual if c.qualifier_hits]
            q_miss = [c for c in qual if not c.qualifier_hits]
            conflict = " vs ".join(f"{c.entry_ref} '{c.gloss}'" for c in qual)
            if len(q_hit) == 1:
                status = ClaimStatus.RESOLVED_CONFLICT
                for c in q_miss:
                    c.outcome = EntryOutcome.CONTRADICTED
                best = q_hit + [c for c in best if c not in q_miss and c not in q_hit]
                reason = f"same core meaning, conflicting usage notes; grammar {q_hit[0].qualifier_hits} supports {q_hit[0].entry_ref}"
            else:
                status = ClaimStatus.UNRESOLVED_CONFLICT
                reason = "same core meaning but conflicting usage notes, and the grammar does not decide"
    else:
        rivals = [c for g in groups[1:] for c in g]
        conflict = " vs ".join(f"{g[0].entry_ref} '{g[0].gloss}'" for g in groups)
        b_usage, r_usage = gscore(best)[0], max(gscore(g)[0] for g in groups[1:])
        b_gram, r_gram = gscore(best)[1], max(gscore(g)[1] for g in groups[1:])
        if b_usage > 0 and r_usage == 0:
            status = ClaimStatus.RESOLVED_CONFLICT
            reason = f"usage supports '{best[0].gloss}' ({', '.join(_hits(best)[:3])}); rival meaning never fits"
        elif b_usage == 0 and r_usage == 0 and b_gram > 0 and r_gram == 0:
            status = ClaimStatus.RESOLVED_CONFLICT
            reason = f"not used in examples; grammar {best[0].grammar_hits} states '{best[0].gloss}'"
        else:
            status = ClaimStatus.UNRESOLVED_CONFLICT
            reason = "dictionaries disagree and the evidence does not clearly favour one meaning"
        if status == ClaimStatus.RESOLVED_CONFLICT:
            for c in rivals:
                c.outcome = EntryOutcome.CONTRADICTED

    # outcomes for entries not already marked contradicted
    for g in groups:
        g_fits = any(fitted(c) for c in g)
        for c in g:
            if c.outcome == EntryOutcome.CONTRADICTED:
                continue
            if g is best and g_fits or (status == ClaimStatus.CORROBORATED and g_fits):
                c.outcome = EntryOutcome.SUPPORTED
            else:
                c.outcome = EntryOutcome.UNTESTED

    chosen = None
    if status in (ClaimStatus.CORROBORATED, ClaimStatus.RESOLVED_CONFLICT, ClaimStatus.SINGLE_SOURCE,
                  ClaimStatus.AGREED_UNTESTED):
        pool = [c for c in best if c.outcome != EntryOutcome.CONTRADICTED] or best
        chosen = max(pool, key=lambda c: (fitted(c), len(c.head_words), len(c.qualifier_words), c.score))
    return LexClaim(term=term, kind=kind, status=status, meaning=chosen.gloss if chosen else None,
                    chosen_entry=chosen.entry_ref if chosen else None, senses=checks, conflict=conflict,
                    reason=reason)


def _hits(g: list[SenseCheck]) -> list[str]:
    return list(dict.fromkeys(r for c in g for r in c.usage_hits))


def _fit_reason(g: list[SenseCheck]) -> str:
    u, gr = _hits(g), [r for c in g for r in c.grammar_hits]
    parts = []
    if u:
        parts.append(f"fits usage in {len(u)} place(s): {', '.join(u[:3])}")
    if gr:
        parts.append(f"stated in grammar {', '.join(sorted(set(gr))[:2])}")
    return "; ".join(parts)


# ---------- whole lexicon + trust ----------

def build_lexicon(canon: CanonicalPack, example_weights: dict[str, float] | None = None) -> list[LexClaim]:
    """`example_weights` comes from the Stage 2 audit: suspect examples count for less."""
    ew = example_weights or {}
    lex = usable_lexicon(canon)
    morph = Morphology(lex)
    morph.induce_affixes([t for _, txt in target_texts(canon) for c in clauses(txt) for t in c])
    contexts = build_contexts(canon, morph, ew)

    by_term: dict[str, list[LexEntry]] = defaultdict(list)
    for e in lex:
        by_term[e.term.strip().lower()].append(e)

    notes = [n for n in canon.notes if "quarantined:injection" not in n.flags]
    claims = []
    for term, entries in sorted(by_term.items()):
        is_affix = term.startswith("-") or term.endswith("-")
        form = term.strip("-")
        ev = TermEvidence(
            used_in=[] if is_affix else [c for c in contexts if term in c.roots],
            grammar=grammar_windows(canon, term),
        )
        claim = judge(term, entries, ev, is_affix)
        if is_affix:
            n = sum(1 for c in contexts if form in c.affixes)
            claim.reason += f"; attested on {n} word(s) in usage" if n else "; never attested in usage"
        if len(form) >= 3 or is_affix:
            rx = re.compile(rf"(?<![\w-]){re.escape(term)}(?![\w-])", re.I)
            claim.discussed_in = [n.ref for n in notes if rx.search(n.text)]
        chosen = next((s for s in claim.senses if s.entry_ref == claim.chosen_entry), None)
        if chosen and chosen.usage_hits and all(ew.get(r, 1.0) < 1.0 for r in chosen.usage_hits):
            claim.reason += f"; caution: usage evidence comes only from suspect example(s) {chosen.usage_hits}"
        claims.append(claim)
    return claims


def score_sources(profile: PackProfile, canon: CanonicalPack, claims: list[LexClaim],
                  hypotheses: list | None = None, audit: list | None = None) -> list[SourceTrust]:
    """Dictionaries are scored on their meanings (Stage 1). With Stage 2 results, examples are
    scored on how many are suspect, and grammar / expert notes on how their rules fared."""
    counts: dict[str, dict[EntryOutcome, int]] = defaultdict(lambda: defaultdict(int))
    for cl in claims:
        for s in cl.senses:
            counts[s.entry_ref.split(":", 1)[0]][s.outcome] += 1
    if audit:
        for a in audit:
            sid = a.example_ref.split(":", 1)[0]
            counts[sid][EntryOutcome.CONTRADICTED if a.suspect else EntryOutcome.SUPPORTED] += 1
    for h in hypotheses or []:
        if h.origin not in ("grammar", "expert"):
            continue
        sid = h.source_refs[0].split(":", 1)[0] if h.source_refs else None
        outcome = {"supported": EntryOutcome.SUPPORTED, "weak": EntryOutcome.SUPPORTED,
                   "refuted": EntryOutcome.CONTRADICTED}.get(h.status.value, EntryOutcome.UNTESTED)
        if sid:
            counts[sid][outcome] += 1
    units = {Role.DICTIONARY: "meanings", Role.EXAMPLES: "examples", Role.GRAMMAR: "rules",
             Role.EXPERT_NOTES: "claims"}
    coi = _conflicts_of_interest(profile, canon)
    out = []
    for sp in profile.sources:
        if sp.role not in PRIORS or sp.status.value in ("unavailable", "unclassified"):
            continue
        prior = PRIORS[sp.role]
        c = counts.get(sp.source_id)
        if sp.role in units and c and (c[EntryOutcome.SUPPORTED] or c[EntryOutcome.CONTRADICTED]):
            s, x, u = c[EntryOutcome.SUPPORTED], c[EntryOutcome.CONTRADICTED], c[EntryOutcome.UNTESTED]
            trust = (prior * PRIOR_STRENGTH + s) / (PRIOR_STRENGTH + s + x)
            reason = f"{s} {units[sp.role]} confirmed, {x} contradicted by evidence, {u} untestable"
            out.append(SourceTrust(source_id=sp.source_id, role=sp.role, prior=prior, supported=s, contradicted=x,
                                   untested=u, trust=round(trust, 3), measured=True, reason=reason,
                                   conflicts_of_interest=coi.get(sp.source_id, [])))
        else:
            out.append(SourceTrust(source_id=sp.source_id, role=sp.role, prior=prior, trust=prior, measured=False,
                                   reason="prior only; tested in a later stage" if sp.role in (Role.EXAMPLES, Role.GRAMMAR)
                                   else "prior only", conflicts_of_interest=coi.get(sp.source_id, [])))
    return out


def _conflicts_of_interest(profile: PackProfile, canon: CanonicalPack) -> dict[str, list[str]]:
    """An expert who wrote, or works for whoever wrote, a source has a stake in that source."""
    authors = {sp.source_id: sp.author for sp in profile.sources if sp.author}
    out: dict[str, list[str]] = defaultdict(list)
    seen: set[tuple[str, str, str]] = set()
    for n in canon.notes:
        if not n.author:
            continue
        for sid, auth in authors.items():
            if sid == n.source_id:
                continue
            org = re.sub(r"\([^)]*\)", "", auth).strip()
            role = n.author_role or ""
            if n.author.lower() in auth.lower():
                why = f"{n.author} is the author of {sid}"
            elif len(org.split()) >= 2 and org.lower() in role.lower():
                why = f"{n.author} works for '{org}', author of {sid}"
            else:
                continue
            if (n.source_id, n.author, sid) not in seen:
                out[n.source_id].append(why)
                seen.add((n.source_id, n.author, sid))
    return dict(out)
