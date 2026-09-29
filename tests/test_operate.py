"""The operations layer behind the UI: runs, reopening, corrections, human review, export."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from nadi9 import operate as op

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def RUN():
    out = ROOT / "runs" / "_op_test"
    shutil.rmtree(out, ignore_errors=True)
    cfg = op.RunConfig(pack=str(ROOT / "packs" / "synthetic"), provider="replay", fixtures=str(ROOT / "fixtures" / "replay"))
    steps: list[str] = []
    r = op.start_run(cfg, out, progress=steps.append)
    yield r, out, steps
    shutil.rmtree(out, ignore_errors=True)


def test_a_run_is_self_contained_and_portable(RUN):
    r, out, steps = RUN
    cfg = json.loads((out / "run_config.json").read_text())
    assert cfg["pack"] == "packs/synthetic" and cfg["fixtures"] == "fixtures/replay"  # relative: survives a rename
    assert json.loads((out / "corrections.json").read_text()) == []
    assert steps[0].startswith("reading the pack") and any(s.startswith("scene SC3") for s in steps)
    assert any(s.startswith("correction EV01") for s in steps)


def test_human_review_changes_the_release_status(RUN):
    _, out, _ = RUN
    assert op.status(out)["verdict"] == "DO NOT RELEASE"  # S012 is critical
    op.review(out, "S012", "edit", text="salo, hulmar! tamsa.", note="respectful greeting")
    s = op.status(out)
    assert s["verdict"] == "RELEASE AFTER REVIEW" and s["critical_open"] == 0 and s["by_reviewer"] == 1
    op.review(out, "S017", "reject")
    op.review(out, "S001", "approve")
    rows = {r["subtitle_id"]: r for r in op.merged(out)}
    assert rows["S012"]["final_text"] == "salo, hulmar! tamsa." and rows["S012"]["decided_by"] == "reviewer (edited)"
    assert rows["S017"]["final"] == "REJECTED" and rows["S001"]["final"] == "APPROVED"
    srt = op.final_srt(out)
    assert "salo, hulmar! tamsa." in srt and "S017" not in srt and "[OPEN" not in srt
    assert "[REJECTED S017]" in op.final_srt(out, approved_only=False)
    op.review(out, "S012", "reset")
    assert op.status(out)["verdict"] == "DO NOT RELEASE"


def test_review_input_is_validated(RUN):
    _, out, _ = RUN
    with pytest.raises(ValueError):
        op.review(out, "S001", "delete")
    with pytest.raises(ValueError):
        op.review(out, "S001", "edit", text="  ")


def test_everything_approved_is_ready(RUN):
    _, out, _ = RUN
    for r in op.merged(out):
        if r["final"] == "OPEN":
            op.review(out, r["subtitle_id"], "approve" if r.get("nadi_9_text") else "reject")
    assert op.status(out)["verdict"] == "READY"


def test_reopening_reproduces_the_run_without_model_calls(RUN):
    r, out, _ = RUN
    r2 = op.reopen(out)
    assert {k: d.model_dump_json() for k, d in r2.state.decisions.items()} == \
           {k: d.model_dump_json() for k, d in r.state.decisions.items()}


def test_corrections_are_validated_persisted_and_replayed(RUN):
    r, out, _ = RUN
    stale = op.apply_correction(r, out, op.correction_from_form(
        "example", example_id="E09", old_text="surat, brem tu tamiru ko?", new_text="x"))
    assert not stale["applied"] and "stale" in stale["reason"]
    assert json.loads((out / "corrections.json").read_text()) == []  # refused corrections are not kept
    op.review(out, "S022", "approve")
    ok = op.apply_correction(r, out, op.correction_from_form(
        "rule", rule_source="grammar:respect-2", tense="past", old_suffix="-sa", new_suffix="-so", sender="Dr. Varkey"))
    assert ok["applied"] and "rule:grammar:respect-2#1@past" in ok["changed_keys"]
    assert [x["line"] for x in ok["redone"]] == ["S022"]
    assert op.load_reviews(out)["S022"]["action"] == "stale"  # the reviewer must look again
    assert json.loads((out / "corrections.json").read_text())[0]["event_id"] == "UI-1"
    r2 = op.reopen(out)  # the correction is re-applied on reopen
    h = next(x for x in r2.L.hypotheses if x.hyp_id == "grammar:respect-2#1")
    assert h.params["affix_by_tense"] == {"present": "si", "past": "so"}


def test_uploaded_pack_keeps_folders_and_refuses_escapes():
    dest = op.save_upload([("events/e1.json", b"{}"), ("../../evil.txt", b"x"), ("lexicon.csv", b"a,b")], "_upl_test")
    try:
        assert (dest / "events" / "e1.json").exists() and (dest / "lexicon.csv").exists()
        assert (dest / "evil.txt").exists() and not (ROOT.parent / "evil.txt").exists()
    finally:
        shutil.rmtree(dest)


def test_the_cli_run_is_operable_in_the_ui():
    assert op.can_reopen(ROOT / "sample_run")
    assert json.loads((ROOT / "sample_run" / "run_config.json").read_text())["fixtures"] == "fixtures/recorded_full"
