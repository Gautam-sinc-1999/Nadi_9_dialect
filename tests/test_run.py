"""Stage 7 tests: corrections mid-run update only what they affect (the brief's correction-event test)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.providers.mock import MockProvider
from nadi9.run import Run
from nadi9.tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[1]
KEY = json.loads((ROOT / "packs" / "synthetic_answer_key.json").read_text())


def make(fixtures="replay", pack="synthetic", scenario=None):
    b = Budget(25, 50)
    r = Run(ROOT / "packs" / pack, MockProvider(ROOT / "fixtures" / fixtures, "nadi9-synthetic-v1", scenario),
            MockTranscriber(), b)
    return r, b


@pytest.fixture(scope="module")
def R():
    r, b = make()
    st = r.execute()
    return r, st, {e["event"]: e for e in st.replan_log}, b


@pytest.fixture(scope="module")
def BASELINE():
    """The same run with no events, to prove untouched lines really are untouched."""
    r, _ = make()
    r.events = []
    return r.execute()


def test_correction_updates_exactly_the_affected_line(R):
    _, st, E, _ = R
    ev = E["EV01"]
    assert ev["applied"] and ev["affected_lines"] == ["S022"] == ev["retranslated"]
    assert ev["unaffected_lines_unchanged"]
    redo = ev["redone"][0]
    assert redo["before"]["text"] == "hulmar, tavo khanasa ko?" and redo["before"]["decision"] == "HUMAN_REVIEW"
    assert redo["after"]["text"] == "hulmar, tavo khanasi ko?" and redo["after"]["decision"] == "APPROVE"
    assert set(KEY["event_impact"]["EV01"]["must_update"]) == set(ev["affected_lines"])


def test_lines_the_answer_key_says_must_not_change_are_identical_to_a_run_without_events(R, BASELINE):
    _, st, _, _ = R
    for lid in KEY["event_impact"]["EV01"]["must_not_touch"] + ["S012", "S001", "S019"]:
        a, b = st.decisions[lid], BASELINE.decisions[lid]
        assert (a.nadi_9_text, a.decision, a.confidence, a.review_question) == \
               (b.nadi_9_text, b.decision, b.confidence, b.review_question), lid
    changed = [lid for lid in st.decisions
               if st.decisions[lid].nadi_9_text != BASELINE.decisions[lid].nadi_9_text
               or st.decisions[lid].decision != BASELINE.decisions[lid].decision]
    assert changed == ["S022"]


def test_example_corrections_confirm_what_the_audit_already_suspected(R):
    _, _, E, _ = R
    for eid in ("EV02", "EV03"):
        assert E[eid]["applied"] and "already flagged suspect" in E[eid]["change"]
        assert E[eid]["retranslated"] == []
    assert E["EV03"]["rechecked"] == ["S022"]  # 'tavo' rule got stronger: re-decide, no model call
    assert E["EV02"]["affected_lines"] == []


def test_only_one_extra_model_call_for_all_three_events(R):
    *_, b = R
    assert b.usage.model_calls == 3 + 5 + 1


def test_rule_is_versioned_and_the_gap_it_answers_is_closed(R):
    r, *_ = R
    h = next(x for x in r.L.hypotheses if x.hyp_id == "grammar:respect-2#1")
    assert h.version == 2 and h.params["affix_by_tense"] == {"present": "si"}
    assert h.history[0]["changed_by"] == "EV01" and "affix_by_tense" not in h.history[0]["params"]
    assert h.by_feature["present"]["support"] > 0  # the linguist's correction is now evidence
    gap = next(x for x in r.L.hypotheses if x.hyp_id == "expert_notes:VAR-2#1")
    assert gap.resolved_by == "EV01"


def test_every_state_is_snapshotted(R, tmp_path):
    r, st, _, _ = R
    r.write(tmp_path)
    snaps = sorted(p.name for p in (tmp_path / "run_state").glob("v*.json"))
    assert snaps == ["v01.json", "v02.json", "v03.json", "v04.json"]
    v2 = json.loads((tmp_path / "run_state" / "v02.json").read_text())
    assert v2["decisions"]["S022"]["text"] == "hulmar, tavo khanasa ko?"  # what the agent knew before EV01
    assert st.decisions["S022"].state_version == 3


def test_correction_before_the_line_is_reached_just_changes_the_state():
    r, _ = make()
    next(e for e in r.events if e.event_id == "EV01").trigger_after = "S012"  # fires during the first scene
    st = r.execute()
    ev = next(e for e in st.replan_log if e["event"] == "EV01")
    assert ev["affected_lines"] == []  # S022 had not been decided yet
    d = st.decisions["S022"]  # translated later; the stale answer 'khanasa' now breaks the corrected rule
    assert d.decision == "HUMAN_REVIEW" and "respect-2" in d.review_question


def test_stale_or_impossible_corrections_are_not_applied():
    r, _ = make()
    ev = next(e for e in r.events if e.event_id == "EV01")
    r._apply(ev, set())
    r._apply(ev, set())  # the same correction again: -sa is no longer the present ending
    log = [e for e in r.state.replan_log if e["event"] == "EV01"]
    assert log[0]["applied"] and not log[1]["applied"] and "stale" in log[1]["reason"]
    ev.target, ev.change = "grammar:no-such-rule", {"tense": "present", "new_suffix": "-xx"}
    r._apply(ev, set())
    assert any("no agreement rule" in i["reason"] for i in r.state.event_items)


def test_real_output_without_a_recorded_retranslation_keeps_and_rechecks_the_old_proposal():
    r, _ = make("recorded")
    st = r.execute()
    ev = next(e for e in st.replan_log if e["event"] == "EV01")
    assert ev["redone"][0]["how"] == "re-translation unavailable; previous proposal re-checked"
    d = st.decisions["S022"]
    assert d.nadi_9_text == "hulmar tavo khana ko" and d.decision == "HUMAN_REVIEW"
    assert "(present: -si)" in d.review_question


def test_free_text_correction_without_a_model_becomes_a_review_item():
    r, _ = make(pack="variant_veshti")
    st = r.execute()
    assert st.replan_log == [] and st.event_items[0]["status"] == "not applied"
    assert "apply this correction by hand" in st.event_items[0]["question"]
