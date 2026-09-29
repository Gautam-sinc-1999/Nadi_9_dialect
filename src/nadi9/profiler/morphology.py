"""Measure the language's shape from the pack. Nothing about any specific language is hard-coded.

- Affixes come from dictionary entries written as "-x" / "x-", plus endings/beginnings that
  repeatedly attach to known roots in the examples (induced).
- Verb position and particle positions are counted over examples and trusted interview segments.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Optional

from ..models import AffixStat, LanguageShape, LexEntry

MIN_INDUCED = 2          # an undeclared affix must attach to known roots at least this often
MAX_AFFIX_LEN = 4
MIN_ASR_CONF = 0.6


@dataclass
class Analysis:
    token: str
    root: Optional[str]
    prefixes: list[str] = field(default_factory=list)
    suffixes: list[str] = field(default_factory=list)


class Morphology:
    def __init__(self, lexicon: Iterable[LexEntry]):
        self.roots: dict[str, set[str]] = defaultdict(set)   # root -> pos labels
        self.suffixes: dict[str, list[str]] = defaultdict(list)
        self.prefixes: dict[str, list[str]] = defaultdict(list)
        for e in lexicon:
            t = e.term.strip().lower()
            if t.startswith("-") and len(t) > 1:
                self.suffixes[t[1:]].append(e.ref)
            elif t.endswith("-") and len(t) > 1:
                self.prefixes[t[:-1]].append(e.ref)
            elif t:
                self.roots[t].add((e.pos or "").lower())
        self.induced_suffixes: set[str] = set()
        self.induced_prefixes: set[str] = set()

    # ----- analysis -----

    def is_pos(self, root: Optional[str], word: str) -> bool:
        # whole-word match: "verb" must not match "adverb"
        rx = re.compile(rf"\b{re.escape(word)}\b")
        return bool(root) and any(rx.search(p) for p in self.roots.get(root, ()))

    def analyse(self, token: str) -> Analysis:
        tok = token.lower()
        if tok in self.roots:  # a dictionary word that itself contains a hyphen, e.g. 'tosh-guna'
            return Analysis(tok, tok)
        parts = tok.split("-")
        if len(parts) > 1:  # hyphenated: stem-affix-affix (e.g. loanword + suffix)
            stem, rest = parts[0], parts[1:]
            a = self._analyse_plain(stem)
            return Analysis(stem if a.root is None else tok, a.root, a.prefixes, a.suffixes + rest)
        return self._analyse_plain(tok)

    def _analyse_plain(self, tok: str) -> Analysis:
        if tok in self.roots:
            return Analysis(tok, tok)
        sfx = set(self.suffixes) | self.induced_suffixes
        pfx = set(self.prefixes) | self.induced_prefixes
        best: Optional[Analysis] = None
        # up to two suffixes and one prefix, prefer the longest root
        for p in [""] + sorted(pfx, key=len, reverse=True):
            if p and not tok.startswith(p):
                continue
            body = tok[len(p):]
            for cand in self._strip_suffixes(body, sfx, depth=2):
                root, sufs = cand
                if root in self.roots and (best is None or len(root) > len(best.root or "")):
                    best = Analysis(tok, root, [p] if p else [], sufs)
        return best or Analysis(tok, None)

    def _strip_suffixes(self, body: str, sfx: set[str], depth: int) -> list[tuple[str, list[str]]]:
        out = [(body, [])]
        if depth == 0:
            return out
        for s in sfx:
            if body.endswith(s) and len(body) > len(s):
                for root, rest in self._strip_suffixes(body[: -len(s)], sfx, depth - 1):
                    out.append((root, rest + [s]))
        return out

    # ----- induction -----

    def induce_affixes(self, tokens: Iterable[str]) -> None:
        sfx_count: Counter[str] = Counter()
        pfx_count: Counter[str] = Counter()
        for tok in tokens:
            t = tok.lower().split("-")[0]
            if t in self.roots:
                continue
            for root in self.roots:
                if len(root) < 3:  # 1-2 letter roots match almost anything
                    continue
                if t.startswith(root) and 0 < len(t) - len(root) <= MAX_AFFIX_LEN:
                    sfx_count[t[len(root):]] += 1
                if t.endswith(root) and 0 < len(t) - len(root) <= MAX_AFFIX_LEN:
                    pfx_count[t[: -len(root)]] += 1
        declared = set(self.suffixes)
        self.induced_suffixes = {s for s, n in sfx_count.items()
                                 if n >= MIN_INDUCED and s not in declared and not self._is_chain(s, declared)}
        self.induced_prefixes = {p for p, n in pfx_count.items() if n >= MIN_INDUCED and p not in self.prefixes}

    def _is_chain(self, s: str, declared: set[str]) -> bool:
        """'ensa' = 'en' + 'sa': a combination of declared suffixes, not a new one."""
        if not s:
            return True
        return any(s.startswith(d) and self._is_chain(s[len(d):], declared) for d in declared)


# ---------- measuring ----------

_WORD = re.compile(r"[a-zA-ZÀ-ɏऀ-ॿ]+(?:-[a-zA-ZÀ-ɏऀ-ॿ]+)*|\[[^\]]*\]")


def clauses(text: str) -> list[list[str]]:
    """Split on sentence punctuation and commas; drop bracketed ASR tags."""
    out = []
    for chunk in re.split(r"[.!?;,]+", text):
        toks = [t for t in _WORD.findall(chunk) if not t.startswith("[")]
        if toks:
            out.append(toks)
    return out


def measure_shape(morph: Morphology, texts: list[tuple[str, str]]) -> LanguageShape:
    """`texts` is [(ref, target-language text)]."""
    all_tokens = [t for _, txt in texts for c in clauses(txt) for t in c]
    morph.induce_affixes(all_tokens)

    shape = LanguageShape()
    affix_attest: dict[tuple[str, str], list[str]] = defaultdict(list)
    vpos: Counter[str] = Counter()
    vpos_refs: dict[str, list[str]] = defaultdict(list)
    ppos: dict[str, Counter[str]] = defaultdict(Counter)
    unknown: dict[str, list[str]] = defaultdict(list)

    for ref, txt in texts:
        for clause in clauses(txt):
            analyses = [morph.analyse(t) for t in clause]
            for a in analyses:
                for s in a.suffixes:
                    affix_attest[("suffix", s)].append(ref)
                for p in a.prefixes:
                    affix_attest[("prefix", p)].append(ref)
                if a.root is None and ref not in unknown[a.token]:
                    unknown[a.token].append(ref)
            # ignore particles when locating the verb, so a clause-final question particle
            # does not hide a verb-final order
            content = [i for i, a in enumerate(analyses) if not morph.is_pos(a.root, "particle")]
            verbs = [i for i in content if morph.is_pos(analyses[i].root, "verb")]
            if len(clause) >= 2 and verbs:
                v = verbs[-1]
                where = "final" if v == content[-1] else "initial" if v == content[0] else "medial"
                vpos[where] += 1
                if ref not in vpos_refs[where]:
                    vpos_refs[where].append(ref)
            # a particle can be both "clause_start" and "before_verb"; count each fact separately
            for i, a in enumerate(analyses):
                if not morph.is_pos(a.root, "particle"):
                    continue
                labels = []
                if i + 1 < len(analyses) and morph.is_pos(analyses[i + 1].root, "verb"):
                    labels.append("before_verb")
                if i > 0 and morph.is_pos(analyses[i - 1].root, "verb"):
                    labels.append("after_verb")
                if i == 0:
                    labels.append("clause_start")
                if i == len(analyses) - 1:
                    labels.append("clause_end")
                for lab in labels or ["other"]:
                    ppos[a.token][lab] += 1

    forms = [("suffix", s, morph.suffixes.get(s, [])) for s in set(morph.suffixes) | morph.induced_suffixes]
    forms += [("prefix", p, morph.prefixes.get(p, [])) for p in set(morph.prefixes) | morph.induced_prefixes]
    for kind, form, declared in sorted(forms, key=lambda x: (x[0], x[1])):
        if not declared and not affix_attest.get((kind, form)):
            continue  # induced but never needed by any analysis: noise
        shape.affixes.append(AffixStat(form=form, kind=kind, declared_by=declared,
                                       attested_in=sorted(set(affix_attest.get((kind, form), [])))))
    shape.verb_position = dict(vpos)
    shape.verb_position_refs = dict(vpos_refs)
    shape.particle_positions = {k: dict(v) for k, v in ppos.items()}
    shape.unanalysed_tokens = dict(sorted(unknown.items()))
    return shape
