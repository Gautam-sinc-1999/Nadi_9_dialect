"""Stage 6: turn a proposal + its verification into a decision with a computed confidence and one
precise review question. No model is asked how sure it is.

    confidence = lexical x grammar x context x verification

    lexical      weakest word:  corroborated 1.0 · loanword 0.95 · resolved conflict 0.9 ·
                                agreed but untested 0.85 · single source 0.55 · unresolved 0.3 · gap/invented 0
    grammar      weakest rule the line actually used:  well supported (>= 3) 1.0 · supported 0.9 ·
                 weak 0.75 · a form with no evidence (extrapolated) 0.6
    context      0.9 per assumption that changes a word form and is not documented
                 (addressee inferred, relation guessed from age, relationship state for a disputed word)
    verification 0.85 per V1-V6 warning; any error caps at 0.35; critical caps at 0.2
                 (V7 citation problems are reported but do not lower confidence in the subtitle itself)

APPROVE needs confidence >= 0.80 and nothing that policy sends to a human anyway.
"""
from __future__ import annotations

import re
from typing import Optional

from .models import (ClaimStatus, Decision, EpisodePlan, HypStatus, LineAnalysis, Proposal, RuleTemplate,
                     Verification)
from .pipeline import Learned

APPROVE_AT = 0.80
WORD_WEIGHT = {ClaimStatus.CORROBORATED: 1.0, ClaimStatus.RESOLVED_CONFLICT: 0.9,
               ClaimStatus.AGREED_UNTESTED: 0.85, ClaimStatus.SINGLE_SOURCE: 0.55,
               ClaimStatus.UNRESOLVED_CONFLICT: 0.3, ClaimStatus.UNSUPPORTED: 0.0}
KIND_WEIGHT = {"loanword": 0.95, "name": 1.0, "gap": 0.0, "unknown": 0.0}
WARNING_FACTOR, ERROR_CAP, CRITICAL_CAP = 0.85, 0.35, 0.2
HUMOUR = re.compile(r"humou?r|joke|teas|grin|funny|sarcas", re.I)  # "humour" is how Stage 3 labels it


class Decider:
    def __init__(self, learned: Learned, plan: EpisodePlan):
        self.L, self.plan = learned, plan
        self.claims = {c.term: c for c in learned.claims}
        self.hyps = {h.hyp_id: h for h in learned.hypotheses}
        self.coi = {h.hyp_id: h.conflict_of_interest for h in learned.hypotheses if h.conflict_of_interest}

    # ------------------------------------------------------------------
    def decide(self, p: Proposal, v: Verification, la: LineAnalysis, state_version: int = 0) -> Decision:
        d = Decision(subtitle_id=p.line_id, source_text=la.text, nadi_9_text=p.nadi_9_text, confidence=0.0,
                     confidence_reason="", decision="HUMAN_REVIEW", checks=v.checks, backgloss=v.backgloss,
                     speaker=la.speaker, addressee=la.addressee,
                     depends_on=list(dict.fromkeys(v.depends_on + self._plan_deps(la))),
                     state_version=state_version, suggested_text=v.fused_form)
        line = next((x for x in self.L.canon.lines if x.line_id == p.line_id), None)
        if line:
            d.start_ms, d.end_ms = line.start_ms, line.end_ms
        d.issues = [f"{i.check} {i.severity}: {i.message}" for i in v.issues if i.severity != "info"]
        d.assumptions = list(dict.fromkeys(p.assumptions + la.assumptions))
        d.evidence = self._evidence(v)
        d.conflicts = self._conflicts(v, la)

        if not p.nadi_9_text:
            d.decision, d.confidence = "INSUFFICIENT_EVIDENCE", 0.0
            d.confidence_reason = f"no translation: {p.reason}"
            d.priority = "high"
            d.review_question = self._gap_question(p, la) or f"No translation could be supported: {p.reason}. How should this line be rendered?"
            return d

        f = self._factors(v, la)
        d.factors = f
        d.confidence = round(f["lexical"] * f["grammar"] * f["context"] * f["verification"], 2)
        d.confidence_reason = self._reason(f, v, la)

        causes = self._causes(p, v, la)
        content = [x for x in v.tokens if x["kind"] in ("word", "gap", "unknown", "loanword")]
        gaps = [x for x in v.tokens if x["kind"] == "gap"]
        if content and len(gaps) * 2 >= len(content):
            d.decision = "INSUFFICIENT_EVIDENCE"
        elif d.confidence >= APPROVE_AT and not causes:
            d.decision = "APPROVE"
        else:
            d.decision = "HUMAN_REVIEW"
        if d.decision != "APPROVE":
            causes = causes or [(5, f"Confidence {d.confidence} is below {APPROVE_AT}: {d.confidence_reason}. Is the line acceptable?")]
            causes.sort(key=lambda c: c[0])
            d.review_question = causes[0][1]
            d.other_questions = [c[1] for c in causes[1:4]]
            d.priority = ("critical" if v.worst == "critical" else
                          "high" if v.worst == "error" or any(c[0] <= 2 for c in causes) else "normal")
        return d

    def _command_rule(self) -> bool:
        """Is there any tested rule about the form of respectful commands?"""
        return any(h.status in (HypStatus.SUPPORTED, HypStatus.WEAK) and re.search(
            r"imperative|command", str(h.params.get("feature", "")) + str(h.params.get("condition", "")))
            for h in self.L.hypotheses)

    @staticmethod
    def _plan_deps(la: LineAnalysis) -> list[str]:
        """Gaps and disputed rules from Stage 3 also shape the decision, so they are dependencies too."""
        deps = [f"gap:{g.split(': ', 1)[0]}" for g in la.grammar_gaps if not g.startswith("undocumented")]
        for s in la.sensitive:
            deps += [f"rule:{m}" for m in re.findall(r"([\w_]+:[\w.-]+#\d+)", s)]
        return deps

    # ------------------------------------------------------------------
    def _factors(self, v: Verification, la: LineAnalysis) -> dict[str, float]:
        lex = 1.0
        for x in v.tokens:
            w = KIND_WEIGHT.get(x["kind"])
            if w is None:
                c = self.claims.get(x["root"])
                w = WORD_WEIGHT.get(c.status, 0.5) if c else 0.0
            lex = min(lex, w)
        gram = 1.0
        for dep in v.depends_on:
            if not dep.startswith("rule:"):
                continue
            hid, _, feat = dep[5:].partition("@")
            h = self.hyps.get(hid)
            if not h:
                continue
            if feat and not h.by_feature.get(feat, {}).get("support"):
                w = 0.6
            elif h.status == HypStatus.SUPPORTED:
                w = 1.0 if h.support_w >= 3 else 0.9
            else:
                w = 0.75
            gram = min(gram, w)
        ctx_n = self._undocumented_assumptions(v, la)
        ver = 1.0
        for i in v.issues:
            if i.check != "V7" and i.severity == "warning":
                ver *= WARNING_FACTOR
        if v.worst == "error":
            ver = min(ver, ERROR_CAP)
        if v.worst == "critical":
            ver = min(ver, CRITICAL_CAP)
        return {"lexical": round(lex, 3), "grammar": round(gram, 3), "context": round(0.9 ** ctx_n, 3),
                "verification": round(ver, 3)}

    def _undocumented_assumptions(self, v: Verification, la: LineAnalysis) -> int:
        n = 0
        form_choice = any(s.startswith(("choice", "clash", "disputed")) for s in la.sensitive)
        if la.addressee_method in ("vocative", "turn_taking") and form_choice:
            n += 1
        if la.social and la.social.method in ("age", "none") and form_choice:
            n += 1
        if la.tags and any(s.startswith("disputed") for s in la.sensitive):
            n += 1
        return n

    def _reason(self, f: dict[str, float], v: Verification, la: LineAnalysis) -> str:
        weakest = min(f, key=f.get)
        if f[weakest] >= 1.0:
            return "every word corroborated, every rule used well supported, no check raised a warning"
        if weakest == "lexical":
            worst = min((x for x in v.tokens), key=lambda x: KIND_WEIGHT.get(x["kind"], WORD_WEIGHT.get(
                getattr(self.claims.get(x.get("root") or ""), "status", None), 0.5)))
            c = self.claims.get(worst.get("root") or "")
            what = f"{c.status.value.replace('_', ' ')} ({c.chosen_entry})" if c else worst["kind"]
            return f"weakest word '{worst['form']}': {what}"
        if weakest == "grammar":
            for dep in v.depends_on:
                hid, _, feat = dep[5:].partition("@") if dep.startswith("rule:") else ("", "", "")
                h = self.hyps.get(hid)
                if h and feat and not h.by_feature.get(feat, {}).get("support"):
                    return f"uses the {feat} form of {hid}, which no example or recording confirms"
            weak = [dep[5:] for dep in v.depends_on if dep.startswith("rule:")
                    and self.hyps.get(dep[5:].split("@")[0]) is not None
                    and not (self.hyps[dep[5:].split("@")[0]].status == HypStatus.SUPPORTED
                             and self.hyps[dep[5:].split("@")[0]].support_w >= 3)]
            return f"relies on {weak[0]}, which has limited support" if weak else "relies on a rule with limited support"
        if weakest == "context":
            return "depends on assumptions about the speakers that no source states"
        worst_issue = max(v.issues, key=lambda i: ["info", "warning", "error", "critical"].index(i.severity))
        return f"{worst_issue.check} {worst_issue.severity}: {worst_issue.message[:120]}"

    # ------------------------------------------------------------------
    def _evidence(self, v: Verification) -> list[str]:
        refs: list[str] = []
        for x in v.tokens:
            c = self.claims.get(x.get("root") or "")
            if c and c.chosen_entry:
                refs.append(c.chosen_entry)
        for dep in v.depends_on:
            h = self.hyps.get(dep[5:].split("@")[0]) if dep.startswith("rule:") else None
            if h and h.origin != "induced":
                refs += h.source_refs[:1]
                refs += h.support[:1]  # one example that shows the rule at work
        return list(dict.fromkeys(r for r in refs if r))

    def _conflicts(self, v: Verification, la: LineAnalysis) -> list[str]:
        out = list(la.conflicts)
        used = {x.get("root") for x in v.tokens}
        for s in la.sensitive:
            if s.startswith("disputed"):
                out.append(s)
        for term in sorted(t for t in used if t):  # sorted: set order changes between processes
            c = self.claims.get(term)
            if c and c.conflict and c.status == ClaimStatus.RESOLVED_CONFLICT:
                out.append(f"{term}: {c.conflict} (resolved: {c.chosen_entry})")
        return list(dict.fromkeys(out))

    # ------------------------------------------------------------------
    def _causes(self, p: Proposal, v: Verification, la: LineAnalysis) -> list[tuple[int, str]]:
        """Everything that should put this line in front of a human, most decisive first."""
        causes: list[tuple[int, str]] = []
        speaker, addressee = la.speaker or "the speaker", la.addressee or "the listener"
        status = la.social.status if la.social else "unknown"

        for i in v.issues:
            if i.severity == "critical":
                t = i.meta.get("term") or "this word"
                c = self.claims.get(t)
                disc = ", ".join(c.discussed_in[:3]) if c else ""
                meaning = " ".join(re.split(r"(?<=\.)\s+", c.meaning or "")[:2]) if c else ""
                causes.append((0, f"Should {speaker}'s line to {addressee} ({status} status) keep '{t}'? It is meant as "
                                  f"cheeky or rude, or it should be respectful. Evidence: \"{meaning}\" "
                                  f"({c.chosen_entry if c else ''}; also {disc})."))
                break
        for x in v.tokens:
            if x["kind"] == "unknown":
                causes.append((1, f"'{x['form']}' is not a Nadi-9 word in any source (the model invented it). "
                                  f"What is the right word for this part of \"{la.text}\"?"))
        gap_q = self._gap_question(p, la)
        if gap_q:
            causes.append((1, gap_q))
        if la.features.get("imperative") and status in ("higher", "elder_sibling") and not self._command_rule():
            causes.append((2, f"\"{la.text}\" is a command to {addressee} ({status} status). No tested rule says how a "
                              f"respectful command is formed, so '{p.nadi_9_text}' cannot be checked for politeness. "
                              f"Is it respectful enough?"))
        for s in la.sensitive:
            if s.startswith("disputed") and any(x.get("root") and f"'{x['root']}'" in s for x in v.tokens):
                causes.append((2, self._dispute_question(s, la)))
                break
        for g in la.grammar_gaps:
            causes.append((2, self._grammar_gap_question(g, p, la)))
        for i in v.issues:
            if i.check == "V3" and i.severity in ("error", "warning") and i.meta.get("extrapolated"):
                hid, feat = i.meta["hyp"], i.meta.get("feature")
                causes.append((2, f"'{p.nadi_9_text}' does not follow {hid} in the {feat}, but no example shows "
                                  f"the {feat} form either ({', '.join(self.hyps[hid].source_refs)}). Which form is correct?"))
            elif i.check in ("V3", "V4") and i.severity == "error":
                causes.append((3, f"The proposal {i.message[0].lower()}{i.message[1:]}. "
                                  f"Correct it, or confirm the proposal is acceptable?"))
        for i in v.issues:
            if i.check == "V3" and i.severity == "info" and i.meta.get("extrapolated"):
                hid, feat = i.meta["hyp"], i.meta.get("feature")
                h = self.hyps.get(hid)
                causes.append((3, f"'{p.nadi_9_text}' uses the {feat} form of {hid} "
                                  f"(\"{h.quote[:70] if h and h.quote else ''}\"), which no example or recording "
                                  f"confirms. Is it correct?"))
        for x in v.tokens:
            c = self.claims.get(x.get("root") or "")
            if c and c.status == ClaimStatus.SINGLE_SOURCE:
                causes.append((4, f"Only {c.chosen_entry} lists '{x['root']}' ({c.meaning}); no example or recording "
                                  f"confirms it. Is it correct here?"))
        for i in v.issues:
            if i.check == "V5" and i.severity == "warning":
                causes.append((4, f"{i.message[0].upper()}{i.message[1:]}. Is that acceptable in the subtitle?"))
            elif i.check == "V6" and i.severity == "warning" and "morpheme" not in i.message:
                causes.append((5, f"Timing: {i.message}. Shorten, merge, or retime?"))
            elif i.check == "V2" and i.severity in ("error", "warning") and i.message.startswith("structure"):
                causes.append((3 if i.severity == "error" else 5, f"The model's answer had a structural problem ({i.message[11:][:90]}). "
                                  f"The text itself: is it acceptable?"))
        if HUMOUR.search((la.text or "") + " " + " ".join(la.risk_factors)) and not any("humour" in c[1] for c in causes):
            causes.append((4, f"Humour: does '{p.nadi_9_text}' (back-gloss: {v.backgloss}) keep the joke in "
                              f"\"{la.text}\", or does it need a freer rendering?"))
        if any(c.startswith("unresolved") or "unresolved" in c for c in la.conflicts):
            causes.append((1, f"Dictionaries disagree on a word in this line and the evidence does not decide: "
                              f"{[c for c in la.conflicts if 'unresolved' in c][0]}. Which meaning is right?"))
        seen, out = set(), []
        for pr, q in causes:
            if q not in seen:
                seen.add(q)
                out.append((pr, q))
        return out

    def _gap_question(self, p: Proposal, la: LineAnalysis) -> Optional[str]:
        words = [g.strip("[]?") for g in (p.gaps or la.gaps) if " " not in g] or la.gaps
        if not words:
            return None
        have = {c.english: c.term for c in la.coverage if c.term and c.status != "gap"}
        avoided = {w: have[w] for w in words if w in have}
        missing = [w for w in words if w not in have]
        parts = []
        for w, term in avoided.items():  # a word exists, but the translator left it out: say why
            c = self.claims.get(term)
            clash = next((s for s in la.sensitive if f"'{term}'" in s and s.startswith("clash")), None)
            parts.append(f"'{w}' was left untranslated although the lexicon has '{term}' "
                         f"({(c.meaning or '')[:60] if c else ''})"
                         + (f", probably because {clash.split(': ', 1)[1].split(' (')[0]}" if clash else "")
                         + f". Use '{term}' here, or rephrase?")
        if missing:
            parts.append(f"No source gives a Nadi-9 word for {', '.join(repr(w) for w in missing)} "
                         f"(needed in \"{la.text}\"). What word should be used, or should the line be rephrased?")
        return " ".join(parts)

    def _dispute_question(self, s: str, la: LineAnalysis) -> str:
        m = re.search(r"'(\w+)' rules compete", s)
        term = m.group(1) if m else "this word"
        state = f" (relationship: {', '.join(la.tags)})" if la.tags else ""
        ids = list(dict.fromkeys(i for i in re.findall(r"[\w-]+:[\w.-]+#\d+", s) if i in self.hyps))  # each rule once
        sides = [self._side(i) for i in ids]
        return (f"Which form should {la.speaker} use to {la.addressee}{state}? "
                f"The line currently uses '{term}'. " + " vs ".join(sides[:3]) + ".")

    def _side(self, hid: str) -> str:
        h = self.hyps[hid]
        p = h.params
        claim = (f"'{p.get('term')}' " + (f"only when {p['requires_tag']}" if p.get("requires_tag") else
                                           f"not when {p['forbidden_tag']}" if p.get("forbidden_tag") else
                                           "always" if p.get("always") else (p.get("condition") or "")))
        ev = f"{h.status.value}: {h.support_w:g} for / {h.counter_w:g} against"
        if h.counter:
            ev += f" ({', '.join(h.counter[:2])})"
        coi = f"; conflict of interest: {h.conflict_of_interest}" if h.conflict_of_interest else ""
        return f"{hid} says {claim.strip()} [{ev}{coi}]"

    def _grammar_gap_question(self, g: str, p: Proposal, la: LineAnalysis) -> str:
        if g.startswith("undocumented construction"):
            what = g.split(": ", 1)[1]
            return (f"The grammar says nothing about {what}, which \"{la.text}\" needs. "
                    f"The proposal renders it as '{p.nadi_9_text}'. Is that how {what} is expressed?")
        hid, _, topic = g.partition(": ")
        h = self.hyps.get(hid)
        return (f"The sources say {topic} is undocumented ({hid}: \"{(h.quote if h else '')[:80]}\"). "
                f"The proposal assumes '{p.nadi_9_text}'. Is that correct?")


def decide_all(proposals: list[Proposal], verifications: list[Verification], plan: EpisodePlan,
               learned: Learned, state_version: int = 0) -> list[Decision]:
    d = Decider(learned, plan)
    by_id = {la.line_id: la for la in plan.lines}
    return [d.decide(p, v, by_id[p.line_id], state_version) for p, v in zip(proposals, verifications)]
