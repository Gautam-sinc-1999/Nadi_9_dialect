"""Command line entry point.

    python -m nadi9 profile --pack packs/synthetic --out runs/profile
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from .budget import Budget
from .config import load_dotenv, setting
from .providers.base import LLMProvider, ProviderError, ProviderUnavailable
from .providers.mock import MockProvider
from .profiler import build_profile
from .tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures" / "replay"


def _peek_pack_id(pack: Path) -> str | None:
    for m in pack.glob("*manifest*.json"):
        try:
            return json.loads(m.read_text()).get("pack_id")
        except (json.JSONDecodeError, OSError):
            return None
    return None


def _profile(a: argparse.Namespace):
    pack, out = Path(a.pack), Path(a.out)
    provider = _provider(a, pack)
    budget = Budget(max_model_calls=a.max_model_calls, max_tool_calls=a.max_tool_calls)
    profile, canon = build_profile(pack, provider, _transcriber(a), budget)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack_profile.json").write_text(profile.model_dump_json(indent=2))
    (out / "canonical_pack.json").write_text(canon.model_dump_json(indent=2))
    return profile, canon, out


def cmd_profile(a: argparse.Namespace) -> int:
    profile, canon, out = _profile(a)
    _print_summary(profile, canon, out)
    return 0


def cmd_lexicon(a: argparse.Namespace) -> int:
    from .lexicon import build_lexicon, score_sources
    profile, canon, out = _profile(a)
    claims = build_lexicon(canon)
    trust = score_sources(profile, canon, claims)
    (out / "learned_lexicon.json").write_text(json.dumps([c.model_dump(mode="json") for c in claims], indent=2))
    (out / "source_trust.json").write_text(json.dumps([t.model_dump(mode="json") for t in trust], indent=2))

    print(f"{len(claims)} terms: " + ", ".join(f"{k} {v}" for k, v in Counter(c.status.value for c in claims).items()))
    print("\nCONFLICTS")
    for c in claims:
        if c.conflict:
            print(f"  {c.term:<9} {c.status.value:<20} -> {c.meaning!r}")
            print(f"  {'':<9} {c.conflict}")
            print(f"  {'':<9} why: {c.reason}")
    print("\nCONTRADICTED ENTRIES (poison candidates)")
    for c in claims:
        for s in c.senses:
            if s.outcome.value == "contradicted":
                print(f"  {s.entry_ref:<18} {c.term:<8} '{s.gloss[:50]}'")
    print("\nSOURCE TRUST")
    for t in trust:
        print(f"  {t.source_id:<22} {t.trust:.2f}  {'measured' if t.measured else 'prior   '}  {t.reason}")
        for w in t.conflicts_of_interest:
            print(f"  {'':<22} conflict of interest: {w}")
    print(f"\nwrote {out / 'learned_lexicon.json'} and {out / 'source_trust.json'}")
    return 0


def _print_summary(profile, canon, out: Path) -> None:
    print(f"pack {profile.pack_id}  provider={profile.provider}  "
          f"model calls {profile.budget.model_calls}/{profile.budget.max_model_calls}  "
          f"tool calls {profile.budget.tool_calls}/{profile.budget.max_tool_calls}\n")
    print(f"{'source':<28}{'role':<14}{'via':<10}{'status':<13}records")
    for s in profile.sources:
        print(f"{s.source_id[:27]:<28}{s.role.value:<14}{s.role_method:<10}{s.status.value:<13}{s.n_records}")
        for w in s.warnings[:3]:
            print(f"{'':<4}! {w}")
    print(f"\ncanonical: {len(canon.lexicon)} lexicon, {len(canon.examples)} examples, {len(canon.sections)} sections, "
          f"{len(canon.notes)} notes, {len(canon.segments)} segments, {len(canon.lines)} lines, "
          f"{len(canon.characters)} characters, {len(canon.events)} events")
    st = Counter(r.status.value for r in profile.rules)
    tp = Counter(r.template.value for r in profile.rules if r.status.value == "grounded")
    print(f"rules: {dict(st)}  grounded by template: {dict(tp)}")
    for r in profile.rules:
        if r.status.value != "grounded":
            print(f"    x {r.rule_id}: {r.reason}  quote={r.quote[:60]!r}")
    print(f"quarantined: {[q['ref'] for q in profile.quarantined]}")
    sh = profile.shape
    print(f"verb position: {sh.verb_position}")
    print(f"particles: {sh.particle_positions}")
    print("affixes: " + ", ".join(f"{a.kind[0]}:{a.form}({len(a.attested_in)}{'' if a.declared_by else ',induced'})"
                                   for a in sh.affixes))
    print(f"unanalysed tokens: {list(sh.unanalysed_tokens)}")
    for w in profile.warnings:
        print(f"! {w}")
    print(f"\nwrote {out / 'pack_profile.json'} and {out / 'canonical_pack.json'}")


def cmd_hypotheses(a: argparse.Namespace) -> int:
    from .pipeline import learn
    pack, out = Path(a.pack), Path(a.out)
    provider, budget = _provider(a, pack), Budget(a.max_model_calls, a.max_tool_calls)
    L = learn(pack, provider, _transcriber(a), budget)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pack_profile.json").write_text(L.profile.model_dump_json(indent=2))
    (out / "canonical_pack.json").write_text(L.canon.model_dump_json(indent=2))
    dump = lambda xs: json.dumps([x.model_dump(mode="json") for x in xs], indent=2)  # noqa: E731
    (out / "learned_lexicon.json").write_text(dump(L.claims))
    (out / "source_trust.json").write_text(dump(L.trust))
    (out / "learned_rules.json").write_text(json.dumps(
        {"hypotheses": [h.model_dump(mode="json") for h in L.hypotheses], "rejected": L.rejected}, indent=2))
    (out / "example_audit.json").write_text(dump(L.audit))

    print(f"model calls {budget.usage.model_calls}/{budget.usage.max_model_calls}  "
          f"tool calls {budget.usage.tool_calls}/{budget.usage.max_tool_calls}  passes {L.passes}\n")
    print(f"{'hypothesis':<34}{'origin':<13}{'status':<12}{'conf':<6}support/counter")
    for h in sorted(L.hypotheses, key=lambda h: (h.status.value, h.hyp_id)):
        if h.status.value == "untested":
            continue
        print(f"{h.hyp_id[:33]:<34}{h.origin:<13}{h.status.value:<12}{h.confidence:<6}"
              f"{h.support_w:g}/{h.counter_w:g}  {_short(h)}")
        if h.counter:
            print(f"{'':<34}counterexamples: {', '.join(h.counter[:4])}")
        if h.note:
            print(f"{'':<34}note: {h.note}")
        if h.competes_with:
            print(f"{'':<34}competes with: {', '.join(h.competes_with)}")
        if h.conflict_of_interest:
            print(f"{'':<34}conflict of interest: {h.conflict_of_interest}")
    unt = [h for h in L.hypotheses if h.status.value == "untested"]
    print(f"\nknown unknowns ({len(unt)}): " + "; ".join(
        str(h.params.get("topic") or h.quote)[:45] for h in unt if h.template.value == "gap"))
    for r in L.rejected:
        print(f"REJECTED {r.get('ref', r.get('template'))}: {r['reason']}")
    print("\nEXAMPLE AUDIT")
    for a_ in L.audit:
        if a_.suspicion > 0:
            print(f"  {a_.example_ref:<14} suspicion {a_.suspicion:<5} {'SUSPECT' if a_.suspect else 'ok':<8} {a_.reason}")
    print("\nSOURCE TRUST")
    for t in L.trust:
        if t.measured:
            print(f"  {t.source_id:<16} {t.trust:.2f}  {t.reason}")
    for w in L.warnings:
        print(f"! {w}")
    return 0


def cmd_plan(a: argparse.Namespace) -> int:
    from .pipeline import learn
    from .planner import analyse_episode
    pack, out = Path(a.pack), Path(a.out)
    budget = Budget(a.max_model_calls, a.max_tool_calls)
    L = learn(pack, _provider(a, pack), _transcriber(a), budget)
    plan = analyse_episode(L, budget)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))

    print("SOCIAL GRAPH (assumptions)")
    for s in plan.social:
        print(f"  {s.speaker:>9} -> {s.addressee:<9} {s.relation:<13}{s.status:<14}{s.method:<9}{s.reason[:60]}")
    print("\nLINES")
    for la in plan.lines:
        print(f"  {la.line_id} risk {la.risk:<5} {la.text[:44]!r}")
        for x in la.risk_factors:
            print(f"        {x}")
        for x in la.sensitive + la.grammar_gaps:
            print(f"        · {x[:110]}")
    print("\nPLAN")
    for i, p in enumerate(plan.order, 1):
        print(f"  {i}. {p.scene}  max {p.max_risk:<5} total {p.total_risk:<6} {p.rationale[:90]}")
    print(f"\nbudget: {plan.budget}")
    return 0


def cmd_translate(a: argparse.Namespace) -> int:
    from .pipeline import learn
    from .planner import analyse_episode
    from .translator import translate_episode
    pack, out = Path(a.pack), Path(a.out)
    budget = Budget(a.max_model_calls, a.max_tool_calls)
    provider = _provider(a, pack)
    L = learn(pack, provider, _transcriber(a), budget)
    plan = analyse_episode(L, budget)
    log: list[str] = []
    props = translate_episode(plan, L, provider, budget, log)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan.model_dump_json(indent=2))
    (out / "proposals.json").write_text(json.dumps([p.model_dump(mode="json") for p in props], indent=2))
    print(f"provider {provider.name}  model calls {budget.usage.model_calls}/{budget.usage.max_model_calls}  "
          f"tool calls {budget.usage.tool_calls}/{budget.usage.max_tool_calls}\n")
    for p in props:
        print(f"  {p.line_id} [{p.scene}] {p.nadi_9_text or '— ' + (p.reason or '')}")
        for f in p.flags:
            print(f"        ! {f}")
    for w in log:
        print(f"! {w}")
    return 0


def cmd_verify(a: argparse.Namespace) -> int:
    from .pipeline import learn
    from .planner import analyse_episode
    from .translator import translate_episode
    from .verifier import verify_all
    pack, out = Path(a.pack), Path(a.out)
    budget = Budget(a.max_model_calls, a.max_tool_calls)
    provider = _provider(a, pack)
    L = learn(pack, provider, _transcriber(a), budget)
    plan = analyse_episode(L, budget)
    props = translate_episode(plan, L, provider, budget)
    ver = verify_all(props, plan, L)
    out.mkdir(parents=True, exist_ok=True)
    (out / "proposals.json").write_text(json.dumps([p.model_dump(mode="json") for p in props], indent=2))
    (out / "verification.json").write_text(json.dumps([v.model_dump(mode="json") for v in ver], indent=2))
    print(f"provider {provider.name}  model calls {budget.usage.model_calls}/{budget.usage.max_model_calls}\n")
    for p_, v in sorted(zip(props, ver), key=lambda pv: pv[0].line_id):
        print(f"{p_.line_id} {v.worst.upper():<9} {p_.nadi_9_text or '-'}")
        if v.backgloss:
            print(f"          gloss: {v.backgloss}")
        for i in v.issues:
            if i.severity != "info":
                print(f"          {i.check} {i.severity:<8} {i.message[:115]}")
    print("\n" + ", ".join(f"{k} {n}" for k, n in Counter(v.worst for v in ver).most_common()))
    return 0


def cmd_decide(a: argparse.Namespace) -> int:
    from .decider import decide_all
    from .pipeline import learn
    from .planner import analyse_episode
    from .translator import translate_episode
    from .verifier import verify_all
    pack, out = Path(a.pack), Path(a.out)
    budget = Budget(a.max_model_calls, a.max_tool_calls)
    provider = _provider(a, pack)
    L = learn(pack, provider, _transcriber(a), budget)
    plan = analyse_episode(L, budget)
    props = translate_episode(plan, L, provider, budget)
    ver = verify_all(props, plan, L)
    decs = sorted(decide_all(props, ver, plan, L), key=lambda d: d.subtitle_id)
    out.mkdir(parents=True, exist_ok=True)
    (out / "subtitle_decisions.jsonl").write_text("".join(d.model_dump_json() + "\n" for d in decs))
    queue = sorted([d for d in decs if d.decision != "APPROVE"],
                   key=lambda d: (["critical", "high", "normal"].index(d.priority or "normal"), d.subtitle_id))
    (out / "review_queue.json").write_text(json.dumps([{
        "subtitle_id": d.subtitle_id, "priority": d.priority, "decision": d.decision, "source_text": d.source_text,
        "proposal": d.nadi_9_text, "confidence": d.confidence, "question": d.review_question,
        "also_check": d.other_questions, "evidence": d.evidence} for d in queue], indent=2))
    key = {}
    key_file = pack.parent / f"{pack.name}_answer_key.json"
    if key_file.exists():
        key = {k: v["decision"] for k, v in json.loads(key_file.read_text()).get("expected_decisions", {}).items()}
    for d in decs:
        exp = key.get(d.subtitle_id, "")
        mark = "" if not exp else ("  = key" if exp.startswith(d.decision) or (exp == "APPROVE_AFTER_EV01" and d.decision == "HUMAN_REVIEW") else f"  (key: {exp})")
        print(f"{d.subtitle_id} {d.decision:<21} {d.confidence:<5} {(d.priority or ''):<8} {(d.nadi_9_text or '-')[:34]:<35}{mark}")
        print(f"      why: {d.confidence_reason[:120]}")
        if d.review_question:
            print(f"      Q: {d.review_question[:230]}")
    print("\n" + ", ".join(f"{k} {n}" for k, n in Counter(d.decision for d in decs).most_common()))
    if key:
        agree = sum(1 for d in decs if key.get(d.subtitle_id, "").startswith(d.decision)
                    or (key.get(d.subtitle_id) == "APPROVE_AFTER_EV01" and d.decision == "HUMAN_REVIEW"))
        print(f"agreement with answer key: {agree}/{len(decs)}")
    return 0


def cmd_run(a: argparse.Namespace) -> int:
    from .operate import RunConfig, rel, save_config
    from .run import Run
    pack, out = Path(a.pack), Path(a.out)
    if a.provider == "groq" and not a.record:
        a.record = str(out / "answers")  # so the run can be reopened in the UI without new model calls
    budget = Budget(a.max_model_calls, a.max_tool_calls)
    r = Run(pack, _provider(a, pack), _transcriber(a), budget)
    st = r.execute()
    verdict = r.write(out)
    fixtures = None if a.provider == "mock" else rel(a.record if a.provider == "groq" else (a.fixtures or FIXTURES))
    save_config(out, RunConfig(pack=rel(pack), provider="replay" if a.provider == "groq" else a.provider,
                               fixtures=fixtures, max_model_calls=a.max_model_calls, max_tool_calls=a.max_tool_calls))
    print(f"provider {r.provider.name}  model calls {budget.usage.model_calls}/{budget.usage.max_model_calls}  "
          f"tool calls {budget.usage.tool_calls}/{budget.usage.max_tool_calls}  final state v{st.version}\n")
    for line in st.order_log:
        print(f"  {line}")
    for e in st.replan_log:
        print(f"\n{e['event']}: {'applied' if e.get('applied') else 'NOT applied: ' + e.get('reason', '')}")
        if e.get("applied"):
            print(f"  change: {e['change']}")
            print(f"  changed keys: {e['changed_keys'][:8]}{' ...' if len(e['changed_keys']) > 8 else ''}")
            print(f"  lines already decided: {e['lines_already_done']}, affected: {e['affected_lines'] or 'none'}, "
                  f"unaffected unchanged: {e['unaffected_lines_unchanged']}")
            for x in e["redone"]:
                print(f"    {x['line']} {x['how']}: {x['before']['text']!r} {x['before']['decision']} {x['before']['confidence']}"
                      f"  ->  {x['after']['text']!r} {x['after']['decision']} {x['after']['confidence']} (v{x['after']['state_version']})")
    for i in st.event_items:
        print(f"\n{i['event']}: {i['status']} - {i['reason']}")
    c = Counter(d.decision for d in st.decisions.values())
    print("\n" + ", ".join(f"{k} {n}" for k, n in c.most_common()) + f"   -> {out}")
    print(f"release recommendation: {verdict}")
    return 0


def _short(h) -> str:
    p = {k: v for k, v in h.params.items() if k not in ("condition",)}
    return str(p)[:70]


def _provider(a: argparse.Namespace, pack: Path) -> LLMProvider:
    """mock: never uses fixtures · replay: fixtures for this pack · groq: real model (key from .env)."""
    pack_id = _peek_pack_id(pack) or pack.name
    if a.provider == "groq":
        from .providers.groq import DEFAULT_MODEL, GroqProvider
        return GroqProvider(setting("GROQ_API_KEY") or "", model=setting("GROQ_MODEL", DEFAULT_MODEL),
                            base_url=setting("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                            record_dir=Path(a.record) if a.record else None, pack_id=pack_id,
                            max_tokens=int(setting("GROQ_MAX_TOKENS", "6000")))
    scenario = {"hallucinate": a.hallucinate, "fail_first": {}}
    if a.fail_model:
        scenario["fail_first"] = {t: a.fail_model for t in
                                  ("classify_role", "map_fields", "extract_rules", "extract_claims", "propose_rules",
                                   "translate_scene")}
    fixtures = Path(a.fixtures) if a.fixtures else FIXTURES
    return MockProvider(fixtures if a.provider == "replay" else None, pack_id, scenario)


def _transcriber(a: argparse.Namespace):
    """Real audio needs a real engine: with --provider groq, Whisper on Groq; otherwise the mock (never invents)."""
    choice = a.transcriber or ("groq" if a.provider == "groq" else "mock")
    if choice == "groq":
        from .tools.transcribe import GroqTranscriber
        return GroqTranscriber(setting("GROQ_API_KEY") or "", model=setting("GROQ_ASR_MODEL", "whisper-large-v3"),
                               base_url=setting("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
    return MockTranscriber(fail_times=a.fail_asr)


def cmd_check_provider(a: argparse.Namespace) -> int:
    """Verify the Groq key and model with one tiny request. Sends no pack content."""
    from .providers.groq import DEFAULT_MODEL, GroqProvider
    try:
        g = GroqProvider(setting("GROQ_API_KEY") or "", model=setting("GROQ_MODEL", DEFAULT_MODEL),
                         base_url=setting("GROQ_BASE_URL", "https://api.groq.com/openai/v1"))
        models = g.list_models()
        print(f"key ok. {len(models)} models available:")
        for m in models:
            print(f"  {'*' if m == g.model else ' '} {m}")
        if g.model not in models:
            print(f"\n! GROQ_MODEL={g.model} is not in the list; set GROQ_MODEL in .env")
            return 1
        ans = g.complete_json("ping", "Reply with the JSON object {\"ok\": true}.", {"ping": 1})
        print(f"\nmodel {g.model} answered: {ans}")
        return 0
    except (ProviderUnavailable, ProviderError) as e:
        print(f"provider check failed: {e}")
        return 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nadi9")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [("profile", cmd_profile, "stage 0: study a pack, write pack_profile.json + canonical_pack.json"),
                            ("lexicon", cmd_lexicon, "stages 0-1: also check meanings, resolve conflicts, score sources"),
                            ("hypotheses", cmd_hypotheses, "stages 0-2: also test rules and audit the examples"),
                            ("plan", cmd_plan, "stages 0-3: also analyse every episode line and order the work by risk"),
                            ("translate", cmd_translate, "stages 0-4: also get checked translation proposals per scene"),
                            ("verify", cmd_verify, "stages 0-5: also run the independent checks on every proposal"),
                            ("decide", cmd_decide, "stages 0-6: also compute confidence, decide, and write review questions"),
                            ("run", cmd_run, "the full run: stages 0-7 with mid-run events and targeted replanning")]:
        pp = sub.add_parser(name, help=help_)
        pp.add_argument("--pack", required=True)
        pp.add_argument("--out", required=True)
        pp.add_argument("--provider", choices=["mock", "replay", "groq"], default="replay",
                        help="replay: fixtures for this pack; mock: never uses fixtures; groq: real model (GROQ_API_KEY in .env)")
        pp.add_argument("--fixtures", help="replay from this directory instead of fixtures/replay (e.g. a recorded run)")
        pp.add_argument("--record", help="with --provider groq: save every real answer here as replay fixtures")
        pp.add_argument("--max-model-calls", type=int, default=25, help="one budget for the whole run")
        pp.add_argument("--max-tool-calls", type=int, default=50)
        pp.add_argument("--fail-asr", type=int, default=0, help="simulate N transcription failures")
        pp.add_argument("--transcriber", choices=["mock", "groq"],
                        help="audio engine (default: groq Whisper with --provider groq, else mock)")
        pp.add_argument("--fail-model", type=int, default=0, help="simulate N failures per model task")
        pp.add_argument("--hallucinate", action="store_true", help="model proposes one unsupported rule")
        pp.set_defaults(fn=fn)
    cp = sub.add_parser("check-provider", help="verify GROQ_API_KEY / GROQ_MODEL from .env (sends no pack data)")
    cp.set_defaults(fn=cmd_check_provider)
    load_dotenv(ROOT / ".env")
    load_dotenv(Path.cwd() / ".env")
    a = p.parse_args(argv)
    try:
        return a.fn(a)
    except ProviderUnavailable as e:
        print(f"cannot start: {e}")
        return 2
