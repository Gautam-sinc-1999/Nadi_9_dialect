"""Stage 5: independent checks on every proposal. No model is involved, so the verifier cannot repeat
the translator's reasoning: it re-analyses every word itself, re-applies the rules Stage 2 tested on
real evidence, and compares a lexicon back-gloss with the English line.

    V1 vocabulary      every word's status in the learned lexicon (not the translator's claimed ref)
    V2 invented words  words that exist in no usable source, gap markers, and Stage 4 structural flags
    V3 grammar         tested rules re-applied to the output; status/register clashes
    V4 names, numbers  names kept; numbers rendered with the lexicon's number words
    V5 meaning         omissions and additions, from the back-gloss
    V6 form, timing    reading speed, duration, length; morpheme boundaries the examples do not write
    V7 evidence        cited refs that are poisoned or wrong; cited rules that were refuted
"""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

from . import english as E
from .hypotheses import CHECKERS, Env, _ctx
from .lexicon import parse_gloss
from .models import (ClaimStatus, EntryOutcome, EpisodePlan, HypStatus, Issue, LineAnalysis, Proposal,
                     RuleTemplate, Verification)
from .pipeline import Learned
from .planner import PRONOUNS, lookup, normalise_line, reverse_index
from .profiler.morphology import Morphology, clauses
from .profiler.profile import target_texts, usable_lexicon
from .profiler.ruleshape import REGISTER_STATUS

ORDER = ["pass", "info", "warning", "error", "critical"]
CHECK_NAMES = {"V1": "vocabulary", "V2": "invented words", "V3": "grammar", "V4": "names and numbers",
               "V5": "omissions and additions", "V6": "form and timing", "V7": "evidence citations"}
WORD_STATUS_SEVERITY = {ClaimStatus.CORROBORATED: None, ClaimStatus.RESOLVED_CONFLICT: None,
                        ClaimStatus.AGREED_UNTESTED: "info", ClaimStatus.SINGLE_SOURCE: "warning",
                        ClaimStatus.UNRESOLVED_CONFLICT: "error", ClaimStatus.UNSUPPORTED: "error"}
_TOKEN = re.compile(r"\[\?[^\]]+\]|[A-Za-zÀ-ɏ]+(?:-[A-Za-zÀ-ɏ]+)*")
FUNCTION_POS = re.compile(r"\b(particle|pronoun|possessive|interjection|question word)\b")


class Verifier:
    def __init__(self, learned: Learned, plan: EpisodePlan):
        self.L, self.plan = learned, plan
        lex = usable_lexicon(learned.canon)
        self.morph = Morphology(lex)
        texts = target_texts(learned.canon)
        self.morph.induce_affixes([t for _, txt in texts for c in clauses(txt) for t in c])
        self.env = Env(self.morph)
        for h in learned.hypotheses:
            if h.template == RuleTemplate.AFFIX and str(h.params.get("feature", "")).startswith("tense=") \
                    and h.status in (HypStatus.SUPPORTED, HypStatus.WEAK):
                self.env.tense_forms[h.params["form"]] = h.params["feature"].split("=", 1)[1]
        self.claims = {c.term: c for c in learned.claims}
        self.pos = {e.term: (e.pos or "").lower() for e in lex}
        self.contradicted = {s.entry_ref for c in learned.claims for s in c.senses
                             if s.outcome == EntryOutcome.CONTRADICTED}
        self.all_refs = {e.ref for e in learned.canon.lexicon}
        self.hyps = {h.hyp_id: h for h in learned.hypotheses}
        self.idx = reverse_index(learned)
        self.names = {c.name.lower() for c in learned.canon.characters}
        # how the approved material writes endings: which ones are ever joined with a hyphen
        self.hyphen_ok = Counter(s for _, txt in texts for tok in re.findall(r"\w+(?:-\w+)+", txt)
                                 for s in tok.lower().split("-")[1:])
        self.affixes = set(self.morph.suffixes) | set(self.morph.prefixes) | self.morph.induced_suffixes \
            | self.morph.induced_prefixes
        self.imperative_forms = {h.params["form"] for h in learned.hypotheses if h.template == RuleTemplate.AFFIX
                                 and re.search(r"imperative|command", str(h.params.get("feature", "")))}

    # ------------------------------------------------------------------

    def verify(self, p: Proposal, la: LineAnalysis) -> Verification:
        v = Verification(line_id=p.line_id, status="checked" if p.nadi_9_text else "not_translated",
                         checks={k: "pass" for k in CHECK_NAMES})
        if la.social:
            v.depends_on.append(f"social:{la.social.speaker}->{la.social.addressee}")
        if not p.nadi_9_text:
            v.checks = {"V2": "error"}
            v.issues.append(Issue(check="V2", severity="error", message=f"no translation: {p.reason}"))
            v.depends_on += sorted({f"lex:{c.term}" for c in la.coverage if c.term})
            return v
        text = p.nadi_9_text
        toks = _TOKEN.findall(text)
        loans = {c.english for c in la.coverage if c.status == "loanword"}
        analysed = []
        for t in toks:
            low = t.lower()
            if low.startswith("[?"):
                analysed.append({"form": t, "kind": "gap", "root": None})
                continue
            if low in self.names and low in {n.lower() for n in la.features.get("names", [])}:
                # 'ama' is both the word for mother and a character's name: a name only if the English uses it
                analysed.append({"form": t, "kind": "name", "root": low})
                continue
            stem = low.split("-")[0]
            if low in loans or stem in loans:
                analysed.append({"form": t, "kind": "loanword", "root": stem, "suffixes": low.split("-")[1:]})
                continue
            a = self.morph.analyse(low)
            analysed.append({"form": t, "kind": "word" if a.root else "unknown", "root": a.root,
                             "prefixes": a.prefixes, "suffixes": a.suffixes})
        v.tokens = analysed
        v.depends_on += sorted({f"lex:{x['root']}" for x in analysed if x["kind"] == "word"})

        self._v1_vocabulary(v, analysed)
        self._v2_invented(v, analysed, p)
        self._v3_grammar(v, p, la)
        self._v4_names_numbers(v, analysed, la)
        self._v5_meaning(v, analysed, la)
        self._v6_form_timing(v, analysed, p, la)
        self._v7_evidence(v, analysed, p)
        v.depends_on = list(dict.fromkeys(v.depends_on))
        return v

    def _add(self, v: Verification, check: str, severity: str, message: str, refs: list[str] | None = None,
             **meta) -> None:
        v.issues.append(Issue(check=check, severity=severity, message=message, refs=refs or [], meta=meta))
        if ORDER.index(severity) > ORDER.index(v.checks.get(check, "pass")):
            v.checks[check] = severity

    # V1 ---------------------------------------------------------------
    def _v1_vocabulary(self, v, analysed) -> None:
        for x in analysed:
            if x["kind"] != "word":
                continue
            c = self.claims.get(x["root"])
            if c is None:
                continue
            sev = WORD_STATUS_SEVERITY.get(c.status)
            if sev:
                self._add(v, "V1", sev, f"'{x['form']}' ({x['root']}): {c.status.value} - {c.reason[:90]}",
                          [c.chosen_entry] if c.chosen_entry else [])
            bad = [s for s in x.get("suffixes", []) + x.get("prefixes", []) if s not in self.affixes]
            if bad:
                self._add(v, "V1", "error", f"'{x['form']}': ending(s) {bad} are not in any usable source")

    # V2 ---------------------------------------------------------------
    def _v2_invented(self, v, analysed, p: Proposal) -> None:
        for x in analysed:
            if x["kind"] == "gap":
                self._add(v, "V2", "error", f"{x['form']}: no Nadi-9 evidence for this word")
            elif x["kind"] == "unknown":
                self._add(v, "V2", "error", f"'{x['form']}' is not a Nadi-9 word in any usable source (invented)")
        for f in p.flags:
            if not f.startswith("invented"):  # invented words are already reported above, from our own analysis
                self._add(v, "V2", "warning" if f.startswith("outside package") else "error", f"structure: {f}")

    # V3 ---------------------------------------------------------------
    def _v3_grammar(self, v, p: Proposal, la: LineAnalysis) -> None:
        status = la.social.status if la.social else "unknown"
        ctx = _ctx("output", 1.0, p.nadi_9_text, la.text, self.morph, status, "", set(la.tags))
        stated = {self._topic(h) for h in self.L.hypotheses if h.origin not in ("induced", "model")
                  and h.status in (HypStatus.SUPPORTED, HypStatus.WEAK)}
        for h in self.L.hypotheses:
            if h.origin == "model" or h.template not in CHECKERS or h.status not in (HypStatus.SUPPORTED, HypStatus.WEAK):
                continue
            if h.origin == "induced" and self._topic(h) in stated:
                continue  # a measured restatement; the stated rule speaks for this topic
            if h.template == RuleTemplate.AFFIX and not str(h.params.get("feature", "")).startswith("tense="):
                continue  # meaning not checkable from the English
            for _, ok, feat in CHECKERS[h.template](h, [ctx], self.env):
                v.depends_on.append(f"rule:{h.hyp_id}" + (f"@{feat}" if feat else ""))
                unattested = feat and not h.by_feature.get(feat, {}).get("support")
                if ok:
                    if unattested:
                        self._add(v, "V3", "info", f"relies on {h.hyp_id} for the {feat} form, which has no "
                                                   f"evidence (extrapolated from the grammar note)", h.source_refs,
                                  hyp=h.hyp_id, feature=feat, extrapolated=True)
                    continue
                sev = self._rule_severity(h, status)
                if unattested:
                    sev = "warning"
                msg = f"breaks {h.hyp_id}: {self._describe(h)}"
                if unattested:
                    msg += f" (no evidence for the {feat} form; the rule is extrapolated there)"
                self._add(v, "V3", sev, msg, h.source_refs, hyp=h.hyp_id, feature=feat,
                          extrapolated=bool(unattested), term=h.params.get("term"))
        self._register(v, p, la, status)
        self._pronoun_choice(v, la)
        self._command_form(v, la, status)

    def _rule_severity(self, h, status: str) -> str:
        if h.template == RuleTemplate.LEXICAL_CHOICE and status in set(h.params.get("forbidden_status") or []):
            return "critical"  # the word is offensive to this addressee
        if h.origin == "induced":
            return "warning"   # a pattern in the data, not a stated rule
        return "error" if h.status == HypStatus.SUPPORTED else "warning"

    @staticmethod
    def _topic(h) -> tuple:
        p = h.params
        return (h.template.value, p.get("particle") or p.get("form") or p.get("term") or p.get("if_word")
                or ("verb_position" if h.template == RuleTemplate.WORD_ORDER else None))

    @staticmethod
    def _describe(h) -> str:
        from .translator import rule_text
        return rule_text(h)

    def _register(self, v, p: Proposal, la: LineAnalysis, status: str) -> None:
        """A word whose meaning says 'respectful' or 'familiar' must suit the addressee (from the lexicon itself)."""
        if status in ("unknown", "elder_sibling", "mixed"):
            return
        for x in v.tokens:
            c = self.claims.get(x.get("root") or "")
            if x["kind"] != "word" or not c or not c.meaning:
                continue
            _, quals = parse_gloss(c.meaning)
            for reg, allowed in REGISTER_STATUS.items():
                if E.lemma(reg) in quals and status not in allowed:
                    self._add(v, "V3", "error", f"'{x['form']}' is {reg} ({c.chosen_entry}) but the addressee is "
                                                f"{status} relative to the speaker", [c.chosen_entry or ""])
                    break

    def _pronoun_choice(self, v, la: LineAnalysis) -> None:
        for s in la.sensitive:
            m = re.match(r"choice: ([\w/]+) .* rules point to ([\w/]+)", s)
            if not m:
                continue
            options, preferred = set(m.group(1).split("/")), set(m.group(2).split("/"))
            used = {x["root"] for x in v.tokens if x.get("root") in options}
            wrong = used - preferred
            if wrong:
                self._add(v, "V3", "error", f"uses {sorted(wrong)} where the status rules point to {sorted(preferred)}")

    def _command_form(self, v, la: LineAnalysis, status: str) -> None:
        if not (self.imperative_forms and la.features.get("imperative") and status == "higher"):
            return
        verbs = [x for x in v.tokens if x["kind"] == "word" and self.morph.is_pos(x["root"], "verb")
                 and not set(x.get("suffixes", [])) & set(self.env.tense_forms)]
        plain = [x["form"] for x in verbs if not set(x.get("suffixes", [])) & self.imperative_forms]
        if plain:
            self._add(v, "V3", "warning", f"familiar command form {plain} to a higher-status addressee; "
                                          f"the grammar gives a respectful command ending {sorted(self.imperative_forms)}")

    # V4 ---------------------------------------------------------------
    def _v4_names_numbers(self, v, analysed, la: LineAnalysis) -> None:
        out = {x["form"].lower() for x in analysed}
        for n in la.features.get("names", []):
            if n.lower() not in out:
                self._add(v, "V4", "error", f"name '{n}' is missing from the subtitle")
        roots = {x.get("root") for x in analysed}
        for num in la.features.get("numbers", []):
            cands = [t for t, _, _ in lookup(E.lemma(num), {E.lemma(num)}, self.idx)]
            if cands and not set(cands) & roots:
                self._add(v, "V4", "error", f"number '{num}' is not rendered (lexicon: {cands[0]})")

    # V5 ---------------------------------------------------------------
    def _v5_meaning(self, v, analysed, la: LineAnalysis) -> None:
        roots = {x.get("root") for x in analysed if x["kind"] in ("word", "loanword")}
        by_word: dict[str, set[str]] = {}
        for c in la.coverage:
            by_word.setdefault(c.english, set())
            if c.term:
                by_word[c.english].add(c.term)
        for w, terms in by_word.items():
            if not terms or terms & roots:
                continue
            sev = "info" if w in PRONOUNS else "warning"
            self._add(v, "V5", sev, f"'{w}' is not expressed (expected {sorted(terms)[:3]})")
        # additions: words no English word of this line asks for
        text = normalise_line(la.text)
        lemmas = {E.lemma(w) for w in E.words(text)}
        allowed = {t for terms in by_word.values() for t in terms}
        for w in lemmas:
            allowed |= {t for t, _, _ in lookup(w, lemmas, self.idx)[:3]}
        for x in analysed:
            r = x.get("root")
            if x["kind"] == "word" and r not in allowed and not FUNCTION_POS.search(self.pos.get(r, "")):
                c = self.claims.get(r)
                self._add(v, "V5", "warning", f"'{x['form']}' ({c.meaning[:40] if c and c.meaning else r}) "
                                              f"adds meaning that is not in the English line")
        gloss = []
        for x in analysed:
            if x["kind"] == "word":
                c = self.claims.get(x["root"])
                gloss.append((re.split(r"[.;(]", c.meaning)[0].split(",")[0].strip() if c and c.meaning else x["root"])
                             + ("+" + "+".join(x["suffixes"]) if x.get("suffixes") else ""))
            else:
                gloss.append(x["form"] if x["kind"] != "gap" else x["form"].upper())
        v.backgloss = " ".join(gloss)

    # V6 ---------------------------------------------------------------
    def _v6_form_timing(self, v, analysed, p: Proposal, la: LineAnalysis) -> None:
        lim = self.plan.limits
        dur = la.timing.get("duration_ms") or 0
        n = len(p.nadi_9_text)
        cps = n / max(dur / 1000, 0.001)
        if dur < lim["min_duration_ms"]:
            self._add(v, "V6", "warning", f"on screen {dur} ms, minimum is {lim['min_duration_ms']} ms")
        if cps > lim["max_chars_per_second"]:
            self._add(v, "V6", "warning", f"reading speed {cps:.0f} chars/s exceeds {lim['max_chars_per_second']}")
        if n > lim["max_chars_per_line"] * lim["max_lines"]:
            self._add(v, "V6", "error", f"{n} characters do not fit {lim['max_lines']} lines of {lim['max_chars_per_line']}")
        fused, changed = [], False
        for x, raw in zip(analysed, _TOKEN.findall(p.nadi_9_text)):
            parts = raw.split("-")
            if x["kind"] == "word" and len(parts) > 1 and raw.lower() != x["root"] \
                    and any(s.lower() not in self.hyphen_ok for s in parts[1:]):
                keep = [parts[0]]
                for s in parts[1:]:
                    keep[-1:] = [keep[-1] + "-" + s] if s.lower() in self.hyphen_ok else [keep[-1] + s]
                fused.append((raw, keep[0]))
                changed = True
        if changed:
            v.fused_form = p.nadi_9_text
            for raw, new in fused:
                v.fused_form = re.sub(rf"(?<![\w-]){re.escape(raw)}(?![\w-])", new, v.fused_form)
            self._add(v, "V6", "warning", "morpheme boundaries written out "
                      f"({', '.join(r for r, _ in fused)}); the approved examples write them joined: {v.fused_form!r}")

    # V7 ---------------------------------------------------------------
    def _v7_evidence(self, v, analysed, p: Proposal) -> None:
        for t in p.tokens:
            for ref in re.split(r"[|,;\s]+", t.ref or ""):
                if not re.match(r"^[\w-]+:[\w.-]+$", ref) or ref.split(":")[0] in {"name", "names", "loanword"}:
                    continue  # not a reference (e.g. 'names', 'loanword'): nothing to check
                if ref in self.contradicted:
                    c = self.claims.get(t.term or "")
                    self._add(v, "V7", "warning", f"'{t.form}' cites {ref}, an entry contradicted by the evidence; "
                                                  f"the supported entry is {c.chosen_entry if c else '?'}", [ref])
                elif ref not in self.all_refs:
                    self._add(v, "V7", "warning", f"'{t.form}' cites {ref!r}, which does not exist")
        for r in p.rules_used:
            h = self.hyps.get(r.split("@", 1)[0])
            if h and h.status == HypStatus.REFUTED:
                self._add(v, "V7", "warning", f"relies on {h.hyp_id}, which the evidence refuted", h.source_refs)


def verify_all(proposals: list[Proposal], plan: EpisodePlan, learned: Learned) -> list[Verification]:
    """Verifies every proposal. Where V6 found written-out morpheme boundaries, the fused spelling is
    verified from scratch and adopted only if it raises nothing new; the model's text is kept in
    `original_text`. `proposals` is updated in place."""
    ver = Verifier(learned, plan)
    by_id = {la.line_id: la for la in plan.lines}
    out = []
    for i, p in enumerate(proposals):
        la = by_id[p.line_id]
        v = ver.verify(p, la)
        if v.fused_form and v.fused_form != p.nadi_9_text:
            fixed = p.model_copy(update={"nadi_9_text": v.fused_form, "original_text": p.nadi_9_text})
            v2 = ver.verify(fixed, la)
            # compare counts per (check, severity): messages quote the word as written, so they always differ
            before = Counter((x.check, x.severity) for x in v.issues if not x.message.startswith("morpheme"))
            after = Counter((x.check, x.severity) for x in v2.issues)
            if not v2.fused_form and not (after - before):
                v2.issues.append(Issue(check="V6", severity="info",
                                       message=f"spelling normalised from {p.nadi_9_text!r} to match the approved "
                                               f"examples (morpheme boundaries joined); re-verified"))
                proposals[i], v = fixed, v2
        out.append(v)
    return out
