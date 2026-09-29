"""Is the operator UI a faithful window onto the backend? Every screen, every field, every action.

Two runs: the recorded real Groq run (sample) and the hand-written replay (which has a critical line), each as a
private copy. Expected values come from the backend (nadi9.operate and the run's files); where the UI applies a
rule (e.g. which rows a filter shows), the test states that rule independently rather than reusing the UI's code.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from streamlit.delta_generator import DeltaGenerator  # noqa: E402

from nadi9 import operate as op  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = __import__("os").environ.get("NADI9_UI_APP") or str(ROOT / "ui" / "app.py")  # override: mutation checks


# ---------------------------------------------------------------- fixtures

def _copy(src: Path, name: str) -> Path:
    out = ROOT / "runs" / name
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out)
    return out


@pytest.fixture(scope="module")
def HAND_SRC():
    out = ROOT / "runs" / "_map_hand_src"
    shutil.rmtree(out, ignore_errors=True)
    op.start_run(op.RunConfig(pack=str(ROOT / "packs" / "synthetic"), provider="replay",
                              fixtures=str(ROOT / "fixtures" / "replay")), out)
    yield out
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture(params=["real", "hand"])
def RUN(request, HAND_SRC):
    src = ROOT / "sample_run" if request.param == "real" else HAND_SRC
    out = _copy(src, f"_map_{request.param}")
    yield out
    shutil.rmtree(out, ignore_errors=True)


@pytest.fixture()
def downloads(monkeypatch):
    """Capture the bytes every download button would deliver."""
    got: list[dict] = []
    original = DeltaGenerator.download_button

    def spy(self, label, data, file_name=None, *a, **k):
        got.append({"label": label, "data": data if isinstance(data, (bytes, str)) else bytes(data), "file": file_name})
        return original(self, label, data, file_name, *a, **k)
    monkeypatch.setattr(DeltaGenerator, "download_button", spy)
    return got


def render(page: str, out: Path, **state):
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.session_state["run_dir"] = str(out)
    at.session_state["_render_only"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def btn(at, label):
    return next(b for b in at.button if b.label == label)


def md(at) -> str:
    return "\n".join(m.value for m in at.main.markdown)


def expected_rows(out: Path, show: str, prio: list[str], q: str = "") -> list[str]:
    """The filter rules, stated independently of the UI."""
    ids = []
    for r in op.merged(out):
        stale = (r.get("review") or {}).get("action") == "stale"
        wanted = {"All": True, "Open": r["final"] == "OPEN" or stale, "Approved": r["final"] == "APPROVED",
                  "Rejected": r["final"] == "REJECTED", "Need re-review": stale}[show]
        if r["final"] == "OPEN" and r.get("priority") not in prio:
            wanted = False
        hay = " ".join([r["source_text"], r.get("final_text") or "", r["subtitle_id"]]).lower()
        if wanted and (not q or q.lower() in hay):
            ids.append(r["subtitle_id"])
    return ids


def record_text_ok(canon: dict, ref: str, shown: str) -> bool:
    """Does the line for `ref` show that record's actual content?"""
    for e in canon["lexicon"]:
        if e["ref"] == ref:
            return e["term"] in shown and e["gloss"][:30] in shown
    for e in canon["examples"]:
        if e["ref"] == ref:
            return e["target_text"] in shown and e["source_text"] in shown
    for s in canon["segments"]:
        if s["ref"] == ref:
            return (s.get("text") or "") in shown
    for s in canon["sections"]:
        if s["ref"] == ref:
            return s["text"][:40] in shown
    for n in canon["notes"]:
        if n["ref"] == ref:
            return n["text"][:40] in shown
    return False


# ---------------------------------------------------------------- sidebar

def test_sidebar_lists_every_run_and_its_status(RUN):
    at = render("review", RUN)
    expected = ["sample_run"] + [str(p.relative_to(ROOT)) for p in sorted(
        (d for d in (ROOT / "runs").glob("*") if (d / "subtitle_decisions.jsonl").exists()), key=lambda d: -d.stat().st_mtime)]
    sb = at.sidebar.selectbox[0]
    assert sorted(sb.options) == sorted(expected) and sb.value == str(RUN.relative_to(ROOT))
    s = op.status(RUN)
    assert f"[{s['verdict']}]" in at.sidebar.markdown[1].value
    assert f"{s['open']} open · {s['approved']} approved" in at.sidebar.markdown[1].value


def test_run_without_config_is_review_only():
    out = _copy(ROOT / "sample_run", "_map_noconfig")
    try:
        (out / "run_config.json").unlink()
        at = render("corrections", out)
        assert any("cannot be reopened" in w.value for w in at.main.warning)
        assert any("Review only" in c.value for c in at.sidebar.caption)
    finally:
        shutil.rmtree(out)


# ---------------------------------------------------------------- review: numbers, filters, every line

def test_review_metrics_equal_backend_status(RUN):
    at = render("review", RUN)
    s = op.status(RUN)
    m = {x.label: x.value for x in at.main.metric}
    assert m == {"Release": s["verdict"], "Open": str(s["open"]), "Critical open": str(s["critical_open"]),
                 "Approved": str(s["approved"]), "Rejected": str(s["rejected"]), "Need re-review": str(s["stale"])}


@pytest.mark.parametrize("show", ["Open", "All", "Approved", "Rejected", "Need re-review"])
def test_every_filter_shows_exactly_the_right_rows(RUN, show):
    op.review(RUN, "S004", "reject")  # make every filter non-trivial
    # a line the agent approved whose review was invalidated by a correction: must show under Open and re-review
    reviews = op.load_reviews(RUN)
    reviews["S002"] = {"action": "stale", "previous_action": "approve", "note": "UI-9 changed this line", "text": None}
    (RUN / "human_reviews.json").write_text(json.dumps(reviews))
    assert next(r for r in op.merged(RUN) if r["subtitle_id"] == "S002")["final"] == "APPROVED"
    at = render("review", RUN, filter_status=show)
    rows = list(at.main.dataframe[0].value["line"])
    assert rows == expected_rows(RUN, show, ["critical", "high", "normal"])
    for prio in (["high"], ["normal"], []):
        at.main.multiselect[0].set_value(prio).run()
        assert list(at.main.dataframe[0].value["line"]) == expected_rows(RUN, show, prio)


def test_search(RUN):
    at = render("review", RUN, filter_status="All")
    for q in ("elder brother", "S01", "HULMAR", "no-such-words"):
        at.main.text_input[0].input(q).run()
        assert list(at.main.dataframe[0].value["line"]) == expected_rows(RUN, "All", ["critical", "high", "normal"], q)


def test_table_columns_equal_backend(RUN):
    at = render("review", RUN, filter_status="All")
    df = at.main.dataframe[0].value
    rows = op.merged(RUN)
    assert list(df["status"]) == [r["final"] for r in rows]
    assert list(df["Nadi-9"]) == [r.get("final_text") or "—" for r in rows]
    assert list(df["confidence"]) == [r["confidence"] for r in rows]
    assert list(df["by"]) == [r["decided_by"] for r in rows]


def test_every_line_panel_shows_the_backend_record(RUN):
    op.review(RUN, "S009", "edit", text="reviewer's own text")  # so the agent's and the final text differ
    op.review(RUN, "S003", "reject")
    canon = json.loads((RUN / "canonical_pack.json").read_text())
    ver = {v["line_id"]: v for v in json.loads((RUN / "verification.json").read_text())}
    at = render("review", RUN, filter_status="All")
    for r in op.merged(RUN):
        lid = r["subtitle_id"]
        at.main.selectbox(key="line_pick").set_value(lid).run()
        assert not at.exception
        text = md(at)
        assert f"#### {lid} · {r.get('speaker')} → {r.get('addressee')} · " \
               f"{r['start_ms'] / 1000:.1f}s–{r['end_ms'] / 1000:.1f}s" in text
        assert f"**{r['source_text']}**" in text
        agent = next(m.value for m in at.main.markdown if m.value.startswith("Agent:"))
        assert f"[{r['decision']}]" in agent and f"confidence **{r['confidence']}**" in agent
        assert f"`{r.get('nadi_9_text') or '— no translation —'}`" in agent
        assert (f"[{r['priority']}]" in agent) == bool(r.get("priority"))
        assert r["confidence_reason"] in [c.value for c in at.main.caption]
        now = next(m.value for m in at.main.markdown if m.value.startswith("Now:"))
        assert f"[{r['final']}] by {r['decided_by']}" in now
        if r.get("final_text") and r["final_text"] != r.get("nadi_9_text"):
            assert f"`{r['final_text']}`" in now and f"`{r['final_text']}`" not in agent
        infos = [i.value for i in at.main.info]
        assert (f"**Question:** {r['review_question']}" in infos) == bool(r.get("review_question"))
        for x in r.get("other_questions", []):
            assert f"also: {x}" in [c.value for c in at.main.caption]
        v = ver[lid]
        for k, sev in v["checks"].items():
            assert any(m.value.startswith(f"**{k}**") and f"[{sev}]" in m.value for m in at.main.markdown)
        for i in v["issues"]:
            assert i["message"] in text
        for ref in r.get("evidence", []):
            line = next(l for l in text.splitlines() if l.startswith(f"- `{ref}`"))
            assert record_text_ok(canon, ref, line), f"{lid}: {ref} shown as {line!r}"
        for a in r.get("assumptions", []):
            assert f"- *assumption:* {a}" in text
        for c in r.get("conflicts", []):
            assert f"- *conflict:* {c}" in text
        h = r.get("review")
        assert at.text_area(key=f"text_{lid}").value == ((h or {}).get("text") or r.get("suggested_text")
                                                        or r.get("nadi_9_text") or "")
        assert btn(at, "Approve as proposed").disabled == (not r.get("nadi_9_text"))
        assert btn(at, "Back to agent's decision").disabled == (not h)


# ---------------------------------------------------------------- review: actions

def test_actions_reach_the_backend_and_the_screen_follows(RUN):
    at = render("review", RUN, filter_status="All", line_pick="S002")
    btn(at, "Approve as proposed").click().run()
    assert op.load_reviews(RUN)["S002"]["action"] == "approve"
    assert at.main.selectbox(key="line_pick").value == "S003"             # moved on to the next line
    at.main.selectbox(key="line_pick").set_value("S009").run()
    at.text_area(key="text_S009").input("salo, sorai! vessa roven!")
    at.text_input(key="note_S009").input("drop the gap marker")
    btn(at, "Save my text and approve").click().run()
    rv = op.load_reviews(RUN)["S009"]
    assert (rv["action"], rv["text"], rv["note"]) == ("edit", "salo, sorai! vessa roven!", "drop the gap marker")
    at.main.selectbox(key="line_pick").set_value("S009").run()
    assert at.text_area(key="text_S009").value == "salo, sorai! vessa roven!"  # the box shows your text now
    assert not btn(at, "Back to agent's decision").disabled
    btn(at, "Back to agent's decision").click().run()
    assert "S009" not in op.load_reviews(RUN)
    at.main.selectbox(key="line_pick").set_value("S017").run()
    btn(at, "Reject (no subtitle)").click().run()
    assert op.load_reviews(RUN)["S017"]["action"] == "reject"
    m = {x.label: x.value for x in at.main.metric}
    s = op.status(RUN)
    assert (m["Approved"], m["Rejected"], m["Open"]) == (str(s["approved"]), str(s["rejected"]), str(s["open"]))


def test_an_empty_edit_is_refused_and_nothing_is_saved(RUN):
    at = render("review", RUN, filter_status="All", line_pick="S003")
    at.text_area(key="text_S003").input("   ")
    btn(at, "Save my text and approve").click().run()
    assert any("needs the corrected text" in e.value for e in at.main.error)
    assert "S003" not in op.load_reviews(RUN)


def test_a_correction_flags_the_reviewed_line_for_re_review(RUN):
    op.review(RUN, "S022", "approve")
    r = op.reopen(RUN)
    res = op.apply_correction(r, RUN, op.correction_from_form(
        "rule", rule_source="grammar:respect-2", tense="present",
        old_suffix=next(h for h in r.L.hypotheses if h.hyp_id == "grammar:respect-2#1").params.get("affix_by_tense", {})
        .get("present", "sa"), new_suffix="-se"))
    assert res["applied"] and "S022" in [x["line"] for x in res["redone"]]
    at = render("review", RUN, filter_status="Need re-review")
    assert list(at.main.dataframe[0].value["line"]) == ["S022"]
    assert {x.label: x.value for x in at.main.metric}["Need re-review"] == "1"
    assert any("changed this line after your review" in w.value for w in at.main.warning)


# ---------------------------------------------------------------- corrections

def test_correction_form_lists_come_from_the_run(RUN):
    at = render("corrections", RUN)
    hyps = json.loads((RUN / "learned_rules.json").read_text())["hypotheses"]
    agreement = [h for h in hyps if h["template"] == "agreement"]
    assert at.main.selectbox(key="corr_rule").options == [h["hyp_id"] for h in agreement]
    h = agreement[0]
    by = h["params"].get("affix_by_tense") or {}
    tenses = sorted((set(h.get("by_feature", {})) | set(h["params"].get("tenses") or [])) - {"untensed"})
    assert at.main.selectbox(key="corr_tense").options == tenses
    for t in tenses:
        assert f"{t} -{by.get(t, h['params']['then_affix'])}" in " ".join(c.value for c in at.main.caption)
    at.radio(key="corr_kind").set_value("An approved example").run()
    canon = json.loads((RUN / "canonical_pack.json").read_text())
    # the harness reports options as displayed: "E01 · I am going home."
    assert at.main.selectbox(key="corr_example").options == [f"{e['native_id']} · {e['source_text']}"
                                                              for e in canon["examples"]]
    for e in canon["examples"][:20]:
        at.main.selectbox(key="corr_example").set_value(e["native_id"]).run()
        assert f"Currently: `{e['target_text']}`" in " ".join(c.value for c in at.main.caption)


def test_applied_correction_result_equals_the_backend_record(RUN):
    at = render("corrections", RUN)
    at.selectbox(key="corr_tense").set_value("future").run()
    at.text_input(key="corr_new").input("-su").run()
    at.button(key="apply").click().run()
    log = json.loads((RUN / "replan_log.json").read_text())["events"]
    e = log[-1]
    assert e["event"] == "UI-1" and e["applied"]
    assert any(s.value == f"UI-1 applied → state v{e['version']}: {e['change']}" for s in at.main.success)
    for k in e["changed_keys"]:
        assert f"`{k}`" in md(at)
    assert f"Lines redone: **{len(e['redone'])}**" in md(at)
    tables = at.main.dataframe
    if e["redone"]:
        assert list(tables[0].value["line"]) == [x["line"] for x in e["redone"]]
        assert list(tables[0].value["after"]) == [x["after"]["text"] for x in e["redone"]]
    assert list(tables[-1].value["event"]) == [x["event"] for x in log]            # history = the backend's log
    saved = json.loads((RUN / "corrections.json").read_text())
    assert f"{len(saved)} correction(s) submitted here" in " ".join(c.value for c in at.main.caption)


def test_refused_correction_shows_the_backends_reason(RUN):
    at = render("corrections", RUN)
    at.radio(key="corr_kind").set_value("An approved example").run()
    at.selectbox(key="corr_example").set_value("E09").run()
    current = next(e for e in json.loads((RUN / "canonical_pack.json").read_text())["examples"] if e["native_id"] == "E09")
    at.text_input(key="corr_extext").input(current["target_text"]).run()
    at.button(key="apply").click().run()
    assert [e.value for e in at.main.error] == [f"Not applied: nothing to change: E09 already reads {current['target_text']!r}"]


def test_free_text_needs_a_key(RUN, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    at = render("corrections", RUN)
    at.radio(key="corr_kind").set_value("A message from a linguist").run()
    at.text_area(key="corr_msg").input("the present ending is -si").run()
    assert at.button(key="apply").disabled
    assert any("needs the live model" in c.value for c in at.main.caption)


# ---------------------------------------------------------------- export

def test_every_download_delivers_the_backends_bytes(RUN, downloads):
    op.review(RUN, "S012", "edit", text="salo, hulmar!")
    at = render("export", RUN)
    got = {d["file"]: d["data"] for d in downloads}
    as_bytes = lambda x: x.encode() if isinstance(x, str) else x  # noqa: E731
    assert as_bytes(got["subtitles_final.srt"]) == op.final_srt(RUN, True).encode()
    assert as_bytes(got["subtitles_working.srt"]) == op.final_srt(RUN, False).encode()
    assert as_bytes(got["subtitle_decisions_reviewed.jsonl"]) == op.final_decisions_jsonl(RUN).encode()
    assert as_bytes(got["human_reviews.json"]) == (RUN / "human_reviews.json").read_bytes()
    assert as_bytes(got["final_report.md"]) == (RUN / "final_report.md").read_bytes()
    assert as_bytes(got["review_queue.json"]) == (RUN / "review_queue.json").read_bytes()
    assert at.main.code[0].value == op.final_srt(RUN, True).rstrip("\n")  # st.code trims the last newline
    s = op.status(RUN)
    assert f"[{s['verdict']}]" in at.main.markdown[0].value
    assert f"{s['approved']} approved ({s['by_reviewer']} by a reviewer) · {s['rejected']} rejected · {s['open']} open" \
           in at.main.markdown[1].value


# ---------------------------------------------------------------- new run

def test_new_run_choices_come_from_the_project(monkeypatch):
    at = render("run", ROOT / "sample_run")
    packs = [p for p in sorted((ROOT / "packs").iterdir()) if p.is_dir() and p.name != "uploads"]
    packs += sorted(p for p in (ROOT / "packs" / "uploads").glob("*") if p.is_dir())
    assert at.main.selectbox(key="pack").options == [str(p.relative_to(ROOT)) for p in packs]
    files = [p for p in (ROOT / "packs" / "synthetic").rglob("*") if p.is_file() and not p.name.startswith(".")]
    assert at.main.caption[0].value.startswith(f"{len(files)} files:")
    sets = [p for p in sorted((ROOT / "fixtures").iterdir()) if p.is_dir()]
    sets += [p / "answers" for p in (ROOT / "runs").glob("*") if (p / "answers").is_dir()]
    usable = sorted(str(p.relative_to(ROOT)) for p in sets if (p / "nadi9-synthetic-v1").is_dir())
    assert sorted(at.main.selectbox(key="fixtures").options) == usable
    at.main.selectbox(key="pack").set_value("packs/variant_veshti").run()     # nothing was recorded for this pack
    assert any("No recorded answers" in w.value for w in at.main.warning) and at.button(key="start").disabled
    at.main.radio(key="model_mode").set_value("No model").run()
    assert not at.button(key="start").disabled
    at.main.text_input(key="run_name").input("../sample_run").run()
    monkeypatch.setenv("GROQ_API_KEY", "")
    at2 = render("run", ROOT / "sample_run")
    at2.main.radio(key="model_mode").set_value("Groq (live)").run()
    assert at2.button(key="start").disabled and any("No key yet" in c.value for c in at2.main.caption)


def test_existing_run_name_blocks_start():
    at = render("run", ROOT / "sample_run")
    existing = next(p.name for p in (ROOT / "runs").iterdir() if p.is_dir())
    at.main.text_input(key="run_name").input(existing).run()
    assert at.button(key="start").disabled and any("exists" in c.value for c in at.main.caption)


def test_no_model_run_from_the_screen_matches_the_backend():
    name = "_map_mockrun"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    at = render("run", ROOT / "sample_run")
    at.main.radio(key="model_mode").set_value("No model").run()
    at.main.text_input(key="run_name").input(name).run()
    at.button(key="start").click().run(timeout=120)
    ref = ROOT / "runs" / "_map_mockref"
    try:
        assert not at.exception
        op.start_run(op.RunConfig(pack=str(ROOT / "packs" / "synthetic"), provider="mock"), ref)
        ui = (ROOT / "runs" / name / "subtitle_decisions.jsonl").read_text()
        assert ui == (ref / "subtitle_decisions.jsonl").read_text()
        s = op.status(ROOT / "runs" / name)
        assert any(f"{s['approved']} approved, {s['open']} to review" in x.value for x in at.main.success)
        assert json.loads((ROOT / "runs" / name / "run_config.json").read_text())["provider"] == "mock"
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
        shutil.rmtree(ref, ignore_errors=True)


# ---------------------------------------------------------------- edge case: a pack with no translations

def test_every_screen_on_a_run_without_translations():
    out = ROOT / "runs" / "_map_pack2"
    shutil.rmtree(out, ignore_errors=True)
    op.start_run(op.RunConfig(pack=str(ROOT / "packs" / "variant_veshti"), provider="mock"), out)
    try:
        for page in ("review", "corrections", "export"):
            render(page, out, filter_status="All")
        at = render("review", out, filter_status="All")
        assert btn(at, "Approve as proposed").disabled                      # nothing to approve
        assert "no approved lines yet" in render("export", out).main.code[0].value
    finally:
        shutil.rmtree(out)
