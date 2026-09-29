"""The operator UI, headless (Streamlit AppTest): every screen renders, shows the backend's state, and each
action (review, correction, export, new run) goes through to the backend."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

st_testing = pytest.importorskip("streamlit.testing.v1")
from nadi9 import operate as op  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = str(ROOT / "ui" / "app.py")


@pytest.fixture()
def COPY():
    """A private copy of the sample run, so reviews made here never touch the committed sample."""
    out = ROOT / "runs" / "_ui_test"
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(ROOT / "sample_run", out)
    yield out
    shutil.rmtree(out, ignore_errors=True)


def render(page: str, out: Path, **state):
    at = st_testing.AppTest.from_file(APP, default_timeout=120)
    at.session_state["run_dir"] = str(out)
    at.session_state["_render_only"] = page
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def button(at, text):
    return next(b for b in at.button if b.label == text)


@pytest.mark.parametrize("page", ["run", "review", "corrections", "export"])
def test_every_screen_renders(COPY, page):
    render(page, COPY)


def test_review_screen_shows_the_backend_state(COPY):
    at = render("review", COPY)
    s = op.status(COPY)
    m = {x.label: x.value for x in at.main.metric}
    assert (m["Release"], m["Open"], m["Critical open"], m["Approved"]) == \
           (s["verdict"], str(s["open"]), str(s["critical_open"]), str(s["approved"]))
    table = at.main.dataframe[0].value
    open_ids = [r["subtitle_id"] for r in op.merged(COPY) if r["final"] == "OPEN"]
    assert list(table["line"]) == open_ids                       # default filter: open lines
    first = next(r for r in op.merged(COPY) if r["subtitle_id"] == open_ids[0])
    assert any(first["review_question"] in i.value for i in at.main.info)


def test_approve_edit_and_reject_from_the_screen(COPY):
    at = render("review", COPY, line_pick="S012")
    button(at, "Approve as proposed").click().run()
    assert op.load_reviews(COPY)["S012"]["action"] == "approve"
    at = render("review", COPY, line_pick="S009")
    at.text_area(key="text_S009").input("salo, sorai! vessa roven!")
    button(at, "Save my text and approve").click().run()
    rv = op.load_reviews(COPY)["S009"]
    assert rv["action"] == "edit" and rv["text"] == "salo, sorai! vessa roven!"
    at = render("review", COPY, line_pick="S017")
    button(at, "Reject (no subtitle)").click().run()
    assert op.load_reviews(COPY)["S017"]["action"] == "reject"
    assert {x.label: x.value for x in at.main.metric}["Rejected"] == "1"
    assert "salo, sorai! vessa roven!" in op.final_srt(COPY)


def test_correction_from_the_screen_is_validated_and_applied(COPY):
    at = render("corrections", COPY)
    at.radio(key="corr_kind").set_value("An approved example").run()
    at.selectbox(key="corr_example").set_value("E09").run()
    at.text_input(key="corr_extext").input("surat, brem tavo tamirusa ko?").run()
    at.button(key="apply").click().run()
    assert not at.exception
    assert any("Not applied" in e.value and "already reads" in e.value for e in at.main.error)  # E09 was fixed by EV03
    assert json.loads((COPY / "corrections.json").read_text()) == []
    # a real rule change: the present ending of respect-2 (already -si after EV01) becomes -se
    at = render("corrections", COPY)
    at.selectbox(key="corr_tense").set_value("present").run()
    at.text_input(key="corr_new").input("-se").run()
    at.button(key="apply").click().run()
    assert not at.exception
    assert any("UI-" in s.value and "applied" in s.value for s in at.main.success)
    saved = json.loads((COPY / "corrections.json").read_text())
    assert saved[-1]["change"]["new_suffix"] == "-se" and saved[-1]["change"]["old_suffix"] == "si"


def test_export_offers_every_file_and_the_final_subtitles(COPY):
    op.review(COPY, "S012", "edit", text="salo, hulmar!")
    at = render("export", COPY)
    assert len(at.main.get("download_button")) == 6
    assert "salo, hulmar!" in at.main.code[0].value


def test_new_run_from_the_screen():
    name = "_ui_newrun"
    shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
    at = render("run", ROOT / "sample_run")
    at.text_input(key="run_name").input(name).run()
    at.button(key="start").click().run(timeout=120)
    try:
        assert not at.exception
        out = ROOT / "runs" / name
        assert (out / "run_config.json").exists() and (out / "subtitles.srt").exists()
        assert any("finished" in s.value for s in at.main.success)
        assert (out / "subtitle_decisions.jsonl").read_text() == (ROOT / "sample_run" / "subtitle_decisions.jsonl").read_text()
    finally:
        shutil.rmtree(ROOT / "runs" / name, ignore_errors=True)
