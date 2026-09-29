"""Stage 8: the deliverables. subtitles.srt, subtitles_approved.srt, review_queue.json, final_report.md.

Release rule: any open critical item -> DO NOT RELEASE; any other open item -> RELEASE AFTER REVIEW;
nothing open -> READY.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models import ClaimStatus, Decision, EntryOutcome, HypStatus, RuleTemplate, SourceStatus

if TYPE_CHECKING:
    from .run import Run

PRIORITY = ["critical", "high", "normal"]


def tc(ms: int) -> str:
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms_ = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms_:03d}"


def srt(decisions: list[Decision], approved_only: bool = False) -> str:
    """Episode order. Lines awaiting an expert carry a visible marker so they cannot pass for final text."""
    blocks = []
    for d in sorted(decisions, key=lambda d: d.start_ms):
        if approved_only and d.decision != "APPROVE":
            continue
        if d.decision == "APPROVE":
            text = d.nadi_9_text or ""
        elif d.nadi_9_text:
            text = f"[REVIEW {d.subtitle_id}] {d.nadi_9_text}"
        else:
            text = f"[NO SUBTITLE {d.subtitle_id}: insufficient evidence]"
        blocks.append(f"{len(blocks) + 1}\n{tc(d.start_ms)} --> {tc(d.end_ms)}\n{text}\n")
    return "\n".join(blocks)


def review_queue(decisions: list[Decision]) -> list[dict[str, Any]]:
    open_ = [d for d in decisions if d.decision != "APPROVE"]
    open_.sort(key=lambda d: (PRIORITY.index(d.priority or "normal"), d.start_ms))
    return [{
        "subtitle_id": d.subtitle_id, "priority": d.priority, "decision": d.decision,
        "timecode": f"{tc(d.start_ms)} --> {tc(d.end_ms)}", "speaker": d.speaker, "addressee": d.addressee,
        "source_text": d.source_text, "proposal": d.nadi_9_text, "back_gloss": d.backgloss or None,
        "suggested_text": d.suggested_text, "confidence": d.confidence, "confidence_reason": d.confidence_reason,
        "question": d.review_question, "also_check": d.other_questions, "evidence": d.evidence,
        "conflicts": d.conflicts, "assumptions": d.assumptions, "state_version": d.state_version,
    } for d in open_]


def release(decisions: list[Decision], not_applied: list[dict[str, Any]]) -> tuple[str, list[str]]:
    c = Counter(d.priority for d in decisions if d.decision != "APPROVE")
    approved = sum(d.decision == "APPROVE" for d in decisions)
    why = [f"{approved}/{len(decisions)} lines approved automatically"]
    if c:
        why.append("open review items: " + ", ".join(f"{c[p]} {p}" for p in PRIORITY if c[p]))
    if not_applied:
        why.append(f"{len(not_applied)} correction(s) could not be applied automatically")
    if c["critical"]:
        return "DO NOT RELEASE", why + ["a critical item (insult / status risk) is open; it must be resolved by a language expert first"]
    if c or not_applied:
        return "RELEASE AFTER REVIEW", why + ["every open item has a specific question; release once they are answered"]
    return "READY", why


def final_report(run: "Run") -> str:
    L, st = run.L, run.state
    decs = sorted(st.decisions.values(), key=lambda d: d.start_ms)
    verdict, why = release(decs, st.event_items)
    u = run.budget.usage
    counts = Counter(d.decision for d in decs)
    lines: list[str] = []
    w = lines.append

    w(f"# Final report: {L.profile.pack_id}\n")
    prov = getattr(run.provider, "provenance", None)
    if prov:
        w(f"> Replayed answers: {prov}\n")
    w(f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} · provider `{run.provider.name}` · "
      f"model calls {u.model_calls}/{u.max_model_calls} · tool calls {u.tool_calls}/{u.max_tool_calls} · "
      f"final state version v{st.version}\n")
    w(f"## Release recommendation: **{verdict}**\n")
    for x in why:
        w(f"- {x}")
    w("")
    w(f"| Decision | Lines |\n|---|---|")
    for k in ("APPROVE", "HUMAN_REVIEW", "INSUFFICIENT_EVIDENCE"):
        w(f"| {k} | {counts.get(k, 0)} |")
    w("")

    w("## Review queue (most urgent first)\n")
    w("| Line | Priority | Source | Proposal | Conf. | Question |\n|---|---|---|---|---|---|")
    for q in review_queue(decs):
        w(f"| {q['subtitle_id']} | {q['priority']} | {_md(q['source_text'])} | {_md(q['proposal'] or '—')} | "
          f"{q['confidence']} | {_md(q['question'] or '')} |")
    w("\nFull items, with evidence, conflicts and assumptions: `review_queue.json`.\n")

    w("## Approved lines\n")
    w("| Line | Source | Subtitle | Conf. | Why it is safe |\n|---|---|---|---|---|")
    for d in decs:
        if d.decision == "APPROVE":
            w(f"| {d.subtitle_id} | {_md(d.source_text)} | {_md(d.nadi_9_text or '')} | {d.confidence} | {_md(d.confidence_reason)} |")
    w("")

    w("## Surprises from the brief, and what the run did\n")
    for title, body in _surprises(run):
        w(f"**{title}.** {body}\n")

    w("## Corrections received during the run\n")
    if st.replan_log or st.event_items:
        w("| Event | Change | Lines re-translated | Lines re-checked | Others unchanged |\n|---|---|---|---|---|")
        for e in st.replan_log:
            if e.get("applied"):
                w(f"| {e['event']} | {_md(e['change'])} | {', '.join(e['retranslated']) or '—'} | "
                  f"{', '.join(e['rechecked']) or '—'} | {'yes' if e['unaffected_lines_unchanged'] else 'NO'} |")
            else:
                w(f"| {e['event']} | not applied: {_md(e.get('reason', ''))} | — | — | — |")
        for i in st.event_items:
            w(f"| {i['event']} | not applied: {_md(i['reason'])} | — | — | — |")
        for e in st.replan_log:
            for r in e.get("redone", []):
                w(f"\n- {e['event']} · {r['line']} ({r['how']}): `{r['before']['text']}` {r['before']['decision']} "
                  f"{r['before']['confidence']} → `{r['after']['text']}` {r['after']['decision']} {r['after']['confidence']}")
    else:
        w("None.")
    w("\nProcessing order and state versions: `replan_log.json`; what the agent knew at each version: `run_state/`.\n")

    w("## Sources: who, when, what they cover, and measured trust\n")
    w("| Source | Role | Author | Date | Scope | Status | Trust | Basis |\n|---|---|---|---|---|---|---|---|")
    trust = {t.source_id: t for t in L.trust}
    for sp in sorted(L.profile.sources, key=lambda s: -(trust[s.source_id].trust if s.source_id in trust else -1)):
        if sp.role.value in ("manifest", "documentation", "event"):
            continue
        t = trust.get(sp.source_id)
        basis = ("measured: " + _md(t.reason)) if t and t.measured else ("prior only" if t else "")
        w(f"| {sp.source_id} | {sp.role.value} | {_md(sp.author or '—')} | {sp.date or '—'} | "
          f"{_md((sp.scope or '—')[:80])} | {sp.status.value} | {f'{t.trust:.2f}' if t else '—'} | {basis} |")
    coi = [c for t in L.trust for c in t.conflicts_of_interest]
    if coi:
        w("\nConflicts of interest noted: " + "; ".join(dict.fromkeys(coi)) + ".")
    w("")

    open_gaps = [h for h in L.hypotheses if h.template == RuleTemplate.GAP and not h.resolved_by]
    w("## Still undocumented (open questions for the language team)\n")
    for h in open_gaps:
        w(f"- {h.params.get('topic')} ({h.hyp_id})")
    closed = [h for h in L.hypotheses if h.template == RuleTemplate.GAP and h.resolved_by]
    for h in closed:
        w(f"- ~~{h.params.get('topic')}~~ resolved by {h.resolved_by}")
    w("\nLimitations of the system itself: `KNOWN_LIMITATIONS.md`.")
    return "\n".join(lines) + "\n"


def _surprises(run: "Run") -> list[tuple[str, str]]:
    L, st = run.L, run.state
    out = []
    poisoned = [(c.term, s.entry_ref, s.gloss) for c in L.claims for s in c.senses if s.outcome == EntryOutcome.CONTRADICTED]
    if poisoned:
        out.append(("Poisoned dictionary entries", f"{len(poisoned)} entries contradicted by usage and never used: " +
                    "; ".join(f"`{t}` {r} ('{g[:30]}')" for t, r, g in poisoned) + "."))
    suspects = [a for a in L.audit if a.suspect]
    confirmed = [e["event"] for e in st.replan_log if e.get("applied") and "already flagged suspect" in e.get("change", "")]
    corrected = [e for e in L.canon.examples if (e.status or "").startswith("corrected")]
    flagged = [a.example_ref for a in suspects] or []
    w_text = ("Found by the example audit before any correction arrived: " +
              ", ".join(f"{e.ref.split(':')[-1]}" for e in corrected) if corrected else "")
    if corrected or suspects:
        out.append(("Wrong approved examples", (w_text or f"Suspect now: {', '.join(flagged)}") +
                    (f"; later confirmed by {', '.join(confirmed)}." if confirmed else ".")))
    clashes = [la for la in run.plan.lines if any(x.startswith("clash") for x in la.sensitive)]
    if clashes:
        parts = []
        for la in clashes:
            d = st.decisions.get(la.line_id)
            term = next(x for x in la.sensitive if x.startswith("clash")).split("'")[1]
            used = bool(d and d.nadi_9_text and re.search(rf"\b{re.escape(term)}\b", d.nadi_9_text))
            parts.append(f"{la.line_id}: '{term}' would be insulting from {la.speaker} to {la.addressee} "
                         f"({la.social.status if la.social else '?'} status); flagged before translation. "
                         + (f"The translator used it anyway, so the line is {d.priority} and asks: "
                            f"\"{(d.review_question or '').split('?')[0]}?\"" if used else
                            f"The translator avoided it; the line goes to an expert with: \"{(d.review_question or '')[:140]}\""))
        out.append(("A word whose meaning depends on status", " ".join(parts)))
    applied = [e for e in st.replan_log if e.get("applied") and e.get("retranslated")]
    if applied:
        e = applied[0]
        out.append(("A linguist's correction mid-run", f"{e['event']} ({e['change']}) arrived after "
                    f"{e['lines_already_done']} lines were decided; only {', '.join(e['affected_lines'])} was redone; "
                    f"every other decision stayed identical."))
    refuted = [h for h in L.hypotheses if h.origin == "model" and h.status == HypStatus.REFUTED]
    rejected_quotes = [r for r in L.profile.rules if r.status.value == "rejected_ungrounded"]
    if refuted or rejected_quotes:
        out.append(("The model proposing unsupported rules", f"{len(refuted)} model-proposed rule(s) refuted by the "
                    f"examples ({'; '.join(run_rule(h) for h in refuted[:4])})"
                    + (f"; {len(rejected_quotes)} rule(s) rejected because their quote is not in the source" if rejected_quotes else "")
                    + "."))
    if L.profile.quarantined:
        out.append(("Instructions hidden in retrieved text", "Quarantined, never used as evidence or shown to a model: " +
                    ", ".join(q["ref"] for q in L.profile.quarantined) + "."))
    down = [s for s in L.profile.sources if s.status == SourceStatus.UNAVAILABLE]
    fails = [w for w in st.warnings + L.warnings if "failed" in w or "circuit" in w]
    if down or fails:
        out.append(("Tool or model failures", "; ".join(f"{s.source_id}: {s.warnings[-1][:110]}" for s in down)
                    + (f"; {len(fails)} failed call(s) retried or stopped by the circuit breaker" if fails else "")
                    + ". Nothing was invented in their place."))
    return out


def run_rule(h) -> str:
    from .translator import rule_text
    return rule_text(h)


def _md(s: Any) -> str:
    return str(s).replace("|", "\\|").replace("\n", " ")


def write_reports(run: "Run", out: Path) -> str:
    decs = list(run.state.decisions.values())
    (out / "subtitles.srt").write_text(srt(decs))
    (out / "subtitles_approved.srt").write_text(srt(decs, approved_only=True))
    (out / "review_queue.json").write_text(json.dumps(review_queue(decs), indent=2, ensure_ascii=False))
    report = final_report(run)
    (out / "final_report.md").write_text(report)
    return release(decs, run.state.event_items)[0]
