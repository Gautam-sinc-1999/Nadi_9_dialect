"""Nadi-9 subtitle studio: operate the product.

    streamlit run ui/app.py

New run -> Review -> Corrections -> Export. Every action goes through nadi9.operate, the same code the CLI and
tests use; the UI only collects input and displays results.
"""
from __future__ import annotations

import io
import json
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nadi9 import operate as op  # noqa: E402
from nadi9.config import load_dotenv, setting  # noqa: E402

st.set_page_config(page_title="Nadi-9 subtitle studio", page_icon=":material/subtitles:", layout="wide")
load_dotenv(ROOT / ".env")

DECISION_COLOR = {"APPROVE": "green", "HUMAN_REVIEW": "orange", "INSUFFICIENT_EVIDENCE": "red"}
FINAL_COLOR = {"APPROVED": "green", "OPEN": "orange", "REJECTED": "red"}
PRIORITY_COLOR = {"critical": "violet", "high": "red", "normal": "orange"}
SEVERITY_COLOR = {"pass": "green", "info": "blue", "warning": "orange", "error": "red", "critical": "violet"}
VERDICT_COLOR = {"READY": "green", "RELEASE AFTER REVIEW": "orange", "DO NOT RELEASE": "red"}
CHECKS = {"V1": "vocabulary", "V2": "invented / gaps", "V3": "grammar & status", "V4": "names, numbers",
          "V5": "omissions", "V6": "form, timing", "V7": "citations"}


# ---------------------------------------------------------------- helpers

def badge(text: str, c: str) -> str:
    return f":{c}-background[{text}]"


def all_runs() -> list[Path]:
    found = [ROOT / "sample_run"] if (ROOT / "sample_run" / "subtitle_decisions.jsonl").exists() else []
    found += sorted((d for d in (ROOT / "runs").glob("*") if (d / "subtitle_decisions.jsonl").exists()),
                    key=lambda d: -d.stat().st_mtime)
    return found


def current() -> Optional[Path]:
    p = st.session_state.get("run_dir")
    return Path(p) if p and (Path(p) / "subtitle_decisions.jsonl").exists() else None


def label(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


@st.cache_data(show_spinner=False)
def run_files(run: str, stamp: float) -> dict[str, Any]:
    d = Path(run)
    j = lambda n, default=None: json.loads((d / n).read_text()) if (d / n).exists() else default  # noqa: E731
    canon = j("canonical_pack.json", {})
    records: dict[str, str] = {}
    for e in canon.get("lexicon", []):
        records[e["ref"]] = f"`{e['term']}` = {e['gloss']}"
    for e in canon.get("examples", []):
        records[e["ref"]] = f"`{e['target_text']}` = \"{e['source_text']}\""
    for s in canon.get("segments", []):
        records[s["ref"]] = f"`{s.get('text')}` = \"{s.get('gloss') or ''}\""
    for s in canon.get("sections", []):
        records[s["ref"]] = f"{s.get('heading') or ''}: {s['text'][:240]}"
    for n in canon.get("notes", []):
        records[n["ref"]] = f"{n.get('author') or ''}: \"{n['text'][:240]}\""
    return {"records": records, "examples": canon.get("examples", []),
            "verification": {v["line_id"]: v for v in j("verification.json", [])},
            "rules": j("learned_rules.json", {"hypotheses": []})["hypotheses"],
            "replan": j("replan_log.json", {})}


def files(out: Path) -> dict[str, Any]:
    stamp = max((out / n).stat().st_mtime for n in ("subtitle_decisions.jsonl", "learned_rules.json", "replan_log.json")
                if (out / n).exists())
    return run_files(str(out), stamp)


def groq_key() -> Optional[str]:
    return st.session_state.get("groq_key") or setting("GROQ_API_KEY")


def live_run(out: Path, allow_live: bool):
    """The run in memory, for corrections. Reopened from its recorded answers when needed (no new model calls)."""
    cache = st.session_state.setdefault("live", {})
    k = (str(out), allow_live)
    if k not in cache:
        with st.spinner("Reopening the run from its recorded answers…"):
            cache[k] = op.reopen(out, key=groq_key(), allow_live=allow_live)
    return cache[k]


def link(page, text: str, icon: str) -> None:
    try:
        st.page_link(page, label=text, icon=icon)
    except Exception:  # noqa: BLE001 - only when a page is rendered outside navigation (tests)
        st.caption(text)


def need_run() -> Optional[Path]:
    out = current()
    if out is None:
        st.info("No run selected. Start one, or pick an existing run in the sidebar.")
        link(PAGE_RUN, "New run", ":material/play_arrow:")
    return out


# ---------------------------------------------------------------- New run

def page_run() -> None:
    st.title("New run")

    st.subheader("1 · Pack")
    source = st.radio("Where is the pack?", ["A pack in the project", "Upload files"], horizontal=True, key="pack_source")
    pack: Optional[Path] = None
    if source == "A pack in the project":
        candidates = [p for p in sorted((ROOT / "packs").iterdir()) if p.is_dir() and p.name != "uploads"]
        candidates += sorted(p for p in (ROOT / "packs" / "uploads").glob("*") if p.is_dir())
        names = [label(p) for p in candidates]
        chosen = st.session_state.get("uploaded_pack")
        idx = names.index(chosen) if chosen in names else (names.index("packs/synthetic") if "packs/synthetic" in names else 0)
        pack = ROOT / st.selectbox("Pack", names, index=idx, key="pack")
    else:
        ups = st.file_uploader("Pack files, or one .zip of the pack folder", accept_multiple_files=True, key="upload")
        name = st.text_input("Name for this pack", value="my_pack", key="upload_name")
        if st.button("Save pack", key="save_pack", disabled=not ups):
            items: list[tuple[str, bytes]] = []
            for f in ups:
                if f.name.lower().endswith(".zip"):
                    with zipfile.ZipFile(io.BytesIO(f.getvalue())) as z:
                        items += [(n, z.read(n)) for n in z.namelist() if not n.endswith("/") and "__MACOSX" not in n]
                else:
                    items.append((f.name, f.getvalue()))
            dest = op.save_upload(items, name)
            st.session_state["uploaded_pack"] = label(dest)
            st.session_state["pack_source"] = "A pack in the project"
            st.success(f"Saved {len(items)} file(s) to `{label(dest)}`.")
            st.rerun()
    if pack:
        found = sorted(p.relative_to(pack).as_posix() for p in pack.rglob("*") if p.is_file() and not p.name.startswith("."))
        st.caption(f"{len(found)} files: " + ", ".join(found[:14]) + (" …" if len(found) > 14 else ""))

    st.subheader("2 · Model")
    mode = st.radio("Model", ["Recorded answers (no key)", "Groq (live)", "No model"], horizontal=True, key="model_mode",
                    help="Recorded answers: replay a recorded run, free and repeatable. Groq: a real model. "
                         "No model: everything except translation (the product never invents text).")
    fixtures = model = None
    if mode == "Recorded answers (no key)":
        sets = [p for p in sorted((ROOT / "fixtures").iterdir()) if p.is_dir()]
        sets += [p / "answers" for p in (ROOT / "runs").glob("*") if (p / "answers").is_dir()]
        pid = op.pack_id_of(pack) if pack else ""
        usable = [p for p in sets if (p / pid).is_dir()]
        if not usable:
            st.warning("No recorded answers exist for this pack; use Groq or No model.")
        else:
            names = [label(p) for p in usable]
            default = names.index("fixtures/recorded_full") if "fixtures/recorded_full" in names else 0
            fixtures = st.selectbox("Recorded answers", names, index=default, key="fixtures",
                                    help="fixtures/replay: hand-written answers with planted mistakes. "
                                         "fixtures/recorded_full: a real openai/gpt-oss-120b run.")
    elif mode == "Groq (live)":
        st.text_input("Groq API key (kept for this session only; leave empty to use .env)", type="password", key="groq_key")
        c1, c2 = st.columns([1, 2])
        if c1.button("Check key", key="check_key"):
            from nadi9.providers.groq import GroqProvider
            try:
                st.session_state["groq_models"] = [m for m in GroqProvider(groq_key() or "").list_models()
                                                   if not m.startswith(("whisper", "canopylabs", "meta-llama/llama-prompt"))]
            except Exception as e:  # noqa: BLE001 - shown to the operator
                st.error(f"Key check failed: {e}")
        models = st.session_state.get("groq_models") or [setting("GROQ_MODEL", "openai/gpt-oss-120b")]
        model = c2.selectbox("Model", models, key="groq_model")
        st.caption(("Key found. " if groq_key() else ":red[No key yet.] ") + "A run takes about 5–6 minutes on Groq's "
                   "free tier. The pack's text and audio are sent to Groq: check its data terms before using real material.")

    with st.expander("Budget"):
        b1, b2 = st.columns(2)
        max_model = b1.slider("Max model calls", 1, 60, 25, key="max_model")
        max_tool = b2.slider("Max tool calls", 1, 120, 50, key="max_tool")

    st.subheader("3 · Start")
    name = st.text_input("Run name", value=f"run-{time.strftime('%Y%m%d-%H%M%S')}", key="run_name")
    out = ROOT / "runs" / name
    blocked = out.exists() or not pack or (mode == "Recorded answers (no key)" and not fixtures) \
        or (mode == "Groq (live)" and not groq_key())
    if out.exists():
        st.caption(":red[A run with this name exists.]")
    if not st.button("Start run", type="primary", key="start", disabled=blocked):
        return
    provider = {"Recorded answers (no key)": "replay", "Groq (live)": "groq", "No model": "mock"}[mode]
    cfg = op.RunConfig(pack=str(pack), provider=provider, fixtures=str(ROOT / fixtures) if fixtures else None,
                       model=model, max_model_calls=max_model, max_tool_calls=max_tool)
    with st.status("Running…", expanded=True) as status:
        r = op.start_run(cfg, out, key=groq_key(), progress=lambda m: st.write(m))
        s = op.status(out)
        status.update(label=f"Done · {s['verdict']} · model calls {r.budget.usage.model_calls}/{max_model}",
                      state="complete")
    st.session_state["run_dir"] = str(out)
    st.session_state.setdefault("live", {})[(str(out), False)] = r
    st.success(f"Run `{name}` finished: {s['approved']} approved, {s['open']} to review.")
    link(PAGE_REVIEW, "Go to Review", ":material/rate_review:")


# ---------------------------------------------------------------- Review

def page_review() -> None:
    out = need_run()
    if out is None:
        return
    st.title("Review")
    s = op.status(out)
    m = st.columns(6)
    m[0].metric("Release", s["verdict"])
    m[1].metric("Open", s["open"])
    m[2].metric("Critical open", s["critical_open"])
    m[3].metric("Approved", s["approved"])
    m[4].metric("Rejected", s["rejected"])
    m[5].metric("Need re-review", s["stale"])

    rows = op.merged(out)
    f1, f2, f3 = st.columns([1, 2, 2])
    show = f1.selectbox("Show", ["Open", "All", "Approved", "Rejected", "Need re-review"], key="filter_status")
    prio = f2.multiselect("Priority", ["critical", "high", "normal"], default=["critical", "high", "normal"], key="filter_prio")
    q = f3.text_input("Search", key="filter_text", placeholder="words in English or Nadi-9")

    def keep(r: dict[str, Any]) -> bool:
        stale = (r.get("review") or {}).get("action") == "stale"
        ok = {"All": True, "Open": r["final"] == "OPEN" or stale, "Approved": r["final"] == "APPROVED",
              "Rejected": r["final"] == "REJECTED", "Need re-review": stale}[show]
        if r["final"] == "OPEN" and r.get("priority") not in prio:
            ok = False
        text = f"{r['source_text']} {r.get('final_text') or ''} {r['subtitle_id']}".lower()
        return ok and (not q or q.lower() in text)

    view = [r for r in rows if keep(r)]
    st.dataframe(pd.DataFrame([{"line": r["subtitle_id"], "time": f"{r['start_ms'] / 1000:.1f}s", "speaker": r.get("speaker"),
                                "English": r["source_text"], "Nadi-9": r.get("final_text") or "—", "status": r["final"],
                                "by": r["decided_by"], "priority": r.get("priority") or "", "confidence": r["confidence"]}
                               for r in view], columns=["line", "time", "speaker", "English", "Nadi-9", "status", "by",
                                                        "priority", "confidence"]),
                 hide_index=True, width="stretch", height=min(35 * (len(view) + 1) + 3, 320),
                 column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1,
                                                                              format="%.2f")})
    if not view:
        st.success("Nothing to show here. Change the filter, or go to Export.")
        return
    ids = [r["subtitle_id"] for r in view]
    if "line_next" in st.session_state:  # a jump staged by the last action (a drawn widget cannot be changed)
        st.session_state["line_pick"] = st.session_state.pop("line_next")
    if st.session_state.get("line_pick") not in ids:
        st.session_state["line_pick"] = ids[0]
    c1, c2, c3 = st.columns([1, 6, 1])
    if c1.button("◀ Previous", key="prev", width="stretch"):
        st.session_state["line_pick"] = ids[max(0, ids.index(st.session_state["line_pick"]) - 1)]
    if c3.button("Next ▶", key="next", width="stretch"):
        st.session_state["line_pick"] = ids[min(len(ids) - 1, ids.index(st.session_state["line_pick"]) + 1)]
    lid = c2.selectbox("Line", ids, key="line_pick", label_visibility="collapsed",
                       format_func=lambda i: f"{i} · {next(r for r in view if r['subtitle_id'] == i)['source_text']}")
    line_panel(out, next(r for r in view if r["subtitle_id"] == lid), ids)


def line_panel(out: Path, r: dict[str, Any], ids: list[str]) -> None:
    lid = r["subtitle_id"]
    F = files(out)
    v = F["verification"].get(lid, {})
    h = r.get("review")
    with st.container(border=True):
        st.markdown(f"#### {lid} · {r.get('speaker')} → {r.get('addressee')} · "
                    f"{r['start_ms'] / 1000:.1f}s–{r['end_ms'] / 1000:.1f}s")
        st.markdown(f"**{r['source_text']}**")
        a, b = st.columns(2)
        a.markdown(f"Agent: {badge(r['decision'], DECISION_COLOR.get(r['decision'], 'gray'))} "
                   + (f"{badge(r['priority'], PRIORITY_COLOR[r['priority']])} " if r.get("priority") else "")
                   + f"confidence **{r['confidence']}**  \n`{r.get('nadi_9_text') or '— no translation —'}`")
        a.caption(r["confidence_reason"])
        b.markdown(f"Now: {badge(r['final'], FINAL_COLOR[r['final']])} by {r['decided_by']}"
                   + (f"  \n`{r['final_text']}`" if r.get("final_text") and r["final_text"] != r.get("nadi_9_text") else ""))
        if h and h.get("note") and h["action"] != "stale":
            b.caption(f"Note: {h['note']}")
        if h and h["action"] == "stale":
            st.warning(h["note"])
        if r.get("review_question"):
            st.info(f"**Question:** {r['review_question']}")
            for x in r.get("other_questions", []):
                st.caption(f"also: {x}")
        if r.get("suggested_text"):
            st.caption(f"Suggested spelling: `{r['suggested_text']}`")

        with st.expander("Evidence and checks"):
            cols = st.columns(7)
            for i, (k, n) in enumerate(CHECKS.items()):
                sev = (v.get("checks") or {}).get(k, "—")
                cols[i].markdown(f"**{k}** {n}  \n:{SEVERITY_COLOR.get(sev, 'gray')}[{sev}]")
            if v.get("backgloss"):
                st.markdown(f"Back-translation from the lexicon: *{v['backgloss']}*")
            for i in v.get("issues", []):
                st.markdown(f"- :{SEVERITY_COLOR[i['severity']]}[{i['check']} {i['severity']}] {i['message']}")
            st.markdown("**Evidence**")
            for ref in r.get("evidence", []):
                txt = F["records"].get(ref)
                st.markdown(f"- `{ref}` — {txt}" if txt else f"- `{ref}`")
            for x in r.get("assumptions", []):
                st.markdown(f"- *assumption:* {x}")
            for x in r.get("conflicts", []):
                st.markdown(f"- *conflict:* {x}")

        with st.form(f"review_{lid}"):
            start = (h or {}).get("text") or r.get("suggested_text") or r.get("nadi_9_text") or ""
            text = st.text_area("Subtitle text", value=start, key=f"text_{lid}")
            note = st.text_input("Note (optional)", value=(h or {}).get("note", "") if h and h["action"] != "stale" else "",
                                 key=f"note_{lid}")
            b1, b2, b3, b4 = st.columns(4)
            approve = b1.form_submit_button("Approve as proposed", type="primary", disabled=not r.get("nadi_9_text"))
            edit = b2.form_submit_button("Save my text and approve")
            reject = b3.form_submit_button("Reject (no subtitle)")
            reset = b4.form_submit_button("Back to agent's decision", disabled=not h)
        action = "approve" if approve else "edit" if edit else "reject" if reject else "reset" if reset else None
        if action:
            try:
                op.review(out, lid, action, text=text if action == "edit" else None, note=note)
            except ValueError as e:
                st.error(str(e))
                return
            nxt = ids[ids.index(lid) + 1] if ids.index(lid) + 1 < len(ids) else lid
            st.session_state["line_next"] = nxt if action != "reset" else lid
            st.toast(f"{lid}: {action}")
            st.rerun()


# ---------------------------------------------------------------- Corrections

def page_corrections() -> None:
    out = need_run()
    if out is None:
        return
    st.title("Corrections")
    if not op.can_reopen(out):
        st.warning("This run has no run_config.json, so it cannot be reopened to apply corrections.")
        return
    st.caption("The agent validates the correction, applies it as a new version, and redoes only the lines it affects. "
               "Your reviews of changed lines are flagged for re-review.")
    F = files(out)
    kind = st.radio("What changed?", ["A rule's ending", "An approved example", "A message from a linguist"],
                    horizontal=True, key="corr_kind")
    sender = st.text_input("From", value="", key="corr_sender", placeholder="e.g. Dr. Leela Varkey")
    event: Optional[dict[str, Any]] = None
    if kind == "A rule's ending":
        rules = [h for h in F["rules"] if h["template"] == "agreement"]
        if not rules:
            st.info("No agreement rule was learned in this run.")
        else:
            rid = st.selectbox("Rule", [h["hyp_id"] for h in rules], key="corr_rule")
            h = next(x for x in rules if x["hyp_id"] == rid)
            by_tense = h["params"].get("affix_by_tense") or {}
            tenses = sorted((set(h.get("by_feature", {})) | set(h["params"].get("tenses") or [])) - {"untensed"})
            st.caption(f"'{h['params']['if_word']}' as subject → verb ending: " + ", ".join(
                f"{t} -{by_tense.get(t, h['params']['then_affix'])}" for t in tenses)
                + f"  ·  “{(h.get('quote') or '')[:120]}”")
            c1, c2 = st.columns(2)
            tense = c1.selectbox("Tense", tenses, key="corr_tense")
            new = c2.text_input("Correct ending", placeholder="-si", key="corr_new")
            if new.strip("- "):
                event = op.correction_from_form("rule", rule_source=h["source_refs"][0], tense=tense,
                                                old_suffix=by_tense.get(tense, h["params"]["then_affix"]),
                                                new_suffix=new.strip(), sender=sender)
    elif kind == "An approved example":
        exs = F["examples"]
        eid = st.selectbox("Example", [e["native_id"] for e in exs], key="corr_example",
                           format_func=lambda i: f"{i} · {next(e for e in exs if e['native_id'] == i)['source_text']}")
        ex = next(e for e in exs if e["native_id"] == eid)
        st.caption(f"Currently: `{ex['target_text']}`" + (f" ({ex['status']})" if ex.get("status") else ""))
        new = st.text_input("Correct Nadi-9", key="corr_extext")
        if new.strip():
            event = op.correction_from_form("example", example_id=eid, old_text=ex["target_text"], new_text=new.strip(),
                                            sender=sender)
    else:
        msg = st.text_area("Message", key="corr_msg", placeholder="Paste the linguist's message. A model reads it "
                                                                   "into a specific change, which is then validated.")
        if not groq_key():
            st.caption(":orange[Reading a free-text message needs the live model (Groq key).]")
        if msg.strip():
            event = op.correction_from_form("message", message=msg.strip(), sender=sender)
    allow_live = st.toggle("Use the live model for lines that must be re-translated", value=False, key="corr_live",
                           disabled=not groq_key(),
                           help="Off: affected lines keep their proposal and are re-checked under the new rules.")
    if st.button("Apply correction", type="primary", key="apply",
                 disabled=event is None or (kind == "A message from a linguist" and not groq_key())):
        r = live_run(out, allow_live or kind == "A message from a linguist")
        res = op.apply_correction(r, out, event)
        st.session_state["last_correction"] = res
        st.rerun()

    res = st.session_state.get("last_correction")
    if res:
        if res.get("applied"):
            st.success(f"{res['event']} applied → state v{res['version']}: {res['change']}")
            redone = res.get("redone", [])
            st.markdown(f"Changed: {', '.join(f'`{k}`' for k in res['changed_keys']) or 'nothing decision-relevant'}  \n"
                        f"Lines redone: **{len(redone)}** · every other line unchanged: "
                        f"**{'yes' if res.get('unaffected_lines_unchanged') else 'NO'}**")
            if redone:
                st.dataframe(pd.DataFrame([{"line": x["line"], "how": x["how"], "before": x["before"]["text"],
                                            "before decision": x["before"]["decision"], "after": x["after"]["text"],
                                            "after decision": x["after"]["decision"]} for x in redone]),
                             hide_index=True, width="stretch")
        else:
            st.error(f"Not applied: {res.get('reason')}")

    st.subheader("History")
    hist = json.loads((out / "corrections.json").read_text()) if (out / "corrections.json").exists() else []
    rows = [{"event": e["event"], "applied": e.get("applied"), "change": e.get("change") or e.get("reason"),
             "lines redone": ", ".join(x["line"] for x in e.get("redone", [])) or "—"}
            for e in F["replan"].get("events", [])]
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(f"{len(hist)} correction(s) submitted here; they are re-applied whenever this run is reopened.")


# ---------------------------------------------------------------- Export

def page_export() -> None:
    out = need_run()
    if out is None:
        return
    st.title("Export")
    s = op.status(out)
    st.markdown(f"### {badge(s['verdict'], VERDICT_COLOR[s['verdict']])}")
    st.markdown(f"{s['approved']} approved ({s['by_reviewer']} by a reviewer) · {s['rejected']} rejected · "
                f"{s['open']} open" + (" (" + ", ".join(f"{n} {p}" for p, n in s["open_by_priority"].items() if n) + ")"
                                       if s["open"] else ""))
    if s["verdict"] != "READY":
        st.warning("Open lines are left out of the final subtitles. Review them first for a complete file.")
    final = op.final_srt(out, approved_only=True)
    working = op.final_srt(out, approved_only=False)
    c = st.columns(3)
    c[0].download_button("Final subtitles (.srt)", final, file_name="subtitles_final.srt", width="stretch", type="primary")
    c[1].download_button("Working subtitles, open lines marked (.srt)", working,
                         file_name="subtitles_working.srt", width="stretch")
    c[2].download_button("Decisions with reviews (.jsonl)", op.final_decisions_jsonl(out),
                         file_name="subtitle_decisions_reviewed.jsonl", width="stretch")
    c = st.columns(3)
    rv = out / "human_reviews.json"
    c[0].download_button("Review log (.json)", rv.read_bytes() if rv.exists() else b"{}", file_name="human_reviews.json",
                         width="stretch")
    for i, n in enumerate(("final_report.md", "review_queue.json")):
        p = out / n
        if p.exists():
            c[i + 1].download_button(f"Agent {n}", p.read_bytes(), file_name=n, width="stretch")
    st.subheader("Final subtitles")
    st.code(final or "(no approved lines yet)", language=None)


# ---------------------------------------------------------------- shell

PAGES = {"run": page_run, "review": page_review, "corrections": page_corrections, "export": page_export}


def sidebar() -> None:
    st.sidebar.markdown("### Nadi-9 subtitle studio")
    runs = all_runs()
    if not runs:
        return
    labels = [label(p) for p in runs]
    cur = current()
    idx = labels.index(label(cur)) if cur and label(cur) in labels else 0
    choice = st.sidebar.selectbox("Run", labels, index=idx, key="run_select")
    st.session_state["run_dir"] = str(ROOT / choice)
    s = op.status(ROOT / choice)
    st.sidebar.markdown(f"{badge(s['verdict'], VERDICT_COLOR[s['verdict']])}  \n{s['open']} open · {s['approved']} approved")
    if not op.can_reopen(ROOT / choice):
        st.sidebar.caption("Review only: this run has no run config, so corrections cannot be applied.")


sidebar()
has_run = current() is not None
PAGE_RUN = st.Page(page_run, title="New run", icon=":material/play_arrow:", url_path="run", default=not has_run)
PAGE_REVIEW = st.Page(page_review, title="Review", icon=":material/rate_review:", url_path="review", default=has_run)
PAGE_CORR = st.Page(page_corrections, title="Corrections", icon=":material/published_with_changes:", url_path="corrections")
PAGE_EXPORT = st.Page(page_export, title="Export", icon=":material/download:", url_path="export")
only = st.session_state.get("_render_only")  # tests render one page directly
nav = st.navigation([PAGE_RUN, PAGE_REVIEW, PAGE_CORR, PAGE_EXPORT], position="hidden" if only else "sidebar")
if only:
    PAGES[only]()
else:
    nav.run()
