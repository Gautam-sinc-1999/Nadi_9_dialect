"""Stage 5 tests: the verifier finds planted mistakes AND the mistakes a real model actually made.

Two fixture sets:
  fixtures/replay/    hand-written answers with planted mistakes
  fixtures/recorded/  real openai/gpt-oss-120b output from Groq (run 2), mistakes not planted by anyone
"""
from __future__ import annotations

from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.models import Proposal
from nadi9.pipeline import learn
from nadi9.planner import analyse_episode
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import MockTranscriber
from nadi9.translator import translate_episode
from nadi9.verifier import Verifier, verify_all

ROOT = Path(__file__).resolve().parents[1]


def run(fixtures: str):
    b = Budget(25, 50)
    prov = MockProvider(ROOT / "fixtures" / fixtures, "nadi9-synthetic-v1")
    L = learn(ROOT / "packs" / "synthetic", prov, MockTranscriber(), b)
    plan = analyse_episode(L, b)
    props = translate_episode(plan, L, prov, b)
    ver = verify_all(props, plan, L)  # may replace a proposal's text with a re-verified spelling fix
    return L, plan, {v.line_id: v for v in ver}, {p.line_id: p for p in props}


@pytest.fixture(scope="module")
def HAND():
    return run("replay")


@pytest.fixture(scope="module")
def REAL():
    return run("recorded")


def msgs(v, check=None, severity=None):
    return [i.message for i in v.issues if (check is None or i.check == check) and (severity is None or i.severity == severity)]


# ---------- planted mistakes (hand-written answers) ----------

def test_status_insult_is_critical(HAND):
    _, _, V, _ = HAND
    v = V["S012"]
    assert v.worst == "critical"
    assert any("'sorai'" in m for m in msgs(v, "V3", "critical"))
    assert any("familiar command" in m for m in msgs(v, "V3"))


def test_omission_and_addition(HAND):
    _, _, V, _ = HAND
    assert any("'first' is not expressed" in m for m in msgs(V["S008"], "V5"))
    assert any("'manu'" in m and "adds meaning" in m for m in msgs(V["S008"], "V5"))


def test_poisoned_citation_is_noticed_even_when_the_word_is_right(HAND):
    _, _, V, _ = HAND
    m = msgs(V["S016"], "V7")
    assert m and "dictionary_a:A-066" in m[0] and "dictionary_b:B-032" in m[0]


def test_invented_word_and_fake_rule(HAND):
    _, _, V, _ = HAND
    assert any("'hapo'" in m and "invented" in m for m in msgs(V["S010"], "V2", "error"))
    assert any("does not exist" in m for m in msgs(V["S013"], "V2"))


def test_timing(HAND):
    _, _, V, _ = HAND
    assert V["S011"].checks["V6"] == "warning" and any("800 ms" in m for m in msgs(V["S011"], "V6"))


def test_declined_line_is_an_error_not_a_blank(HAND):
    _, _, V, _ = HAND
    assert V["S017"].status == "not_translated" and V["S017"].worst == "error"


def test_word_that_is_also_a_name(HAND):
    _, _, V, _ = HAND
    assert not msgs(V["S007"], "V5")  # 'ama' = mother here, not the character Ama
    assert V["S007"].backgloss.startswith("mother")


def test_correct_lines_pass(HAND):
    _, _, V, _ = HAND
    for lid in ("S019", "S004", "S025"):
        assert V[lid].worst in ("pass", "info"), (lid, V[lid].issues)


def test_dependencies_separate_s022_from_s021(HAND):
    """What EV01 (present honorific -sa -> -si) will rely on: only S022 used respect-2 in the present."""
    _, _, V, _ = HAND
    assert "rule:grammar:respect-2#1@present" in V["S022"].depends_on
    assert not any(d.startswith("rule:grammar:respect-2") for d in V["S021"].depends_on)
    assert any("extrapolated" in m for m in msgs(V["S022"], "V3", "info"))


# ---------- real model mistakes (recorded Groq output) ----------

@pytest.mark.parametrize("lid", ["S001", "S019", "S024", "S012"])
def test_real_verb_order_errors(REAL, lid):
    _, _, V, _ = REAL
    assert any("verb comes final" in m for m in msgs(V[lid], "V3", "error"))


def test_real_register_error(REAL):
    _, _, V, _ = REAL
    assert any("'tavora' is respectful" in m and "lower" in m for m in msgs(V["S023"], "V3", "error"))


def test_real_missing_honorific_is_flagged_as_extrapolated(REAL):
    _, _, V, _ = REAL
    m = msgs(V["S022"], "V3", "warning")
    assert m and "tavo" in m[0] and "extrapolated" in m[0]


def test_real_silent_omission_to_dodge_the_insult(REAL):
    _, _, V, _ = REAL
    assert any("'friend' is not expressed" in m for m in msgs(V["S012"], "V5"))


def test_real_gap_marker_written_as_question_mark(REAL):
    """gpt-oss wrote '?and' for a gap. It is read as the gap it is ([?and]), not as an invented word."""
    _, _, V, P = REAL
    assert "[?and]" in P["S024"].nadi_9_text and "normalised" in P["S024"].model_notes
    assert any("[?and]" in m and "no Nadi-9 evidence" in m for m in msgs(V["S024"], "V2", "error"))


def test_real_hyphenated_endings_are_fixed_only_after_reverifying(REAL):
    _, _, V, P = REAL
    assert P["S014"].nadi_9_text == "hulmar, mi na jana" and P["S014"].original_text == "hulmar, mi na jan-a"
    assert P["S016"].nadi_9_text == "surat mi office-ma brem tamiru"  # '-ma' keeps its hyphen, as in the examples
    assert any("spelling normalised" in m for m in msgs(V["S014"], "V6", "info"))
    # a line with other problems is still fixed in spelling, but those problems remain
    assert P["S019"].nadi_9_text == "tavo nasirusa brem dovan ko"
    assert any("verb comes final" in m for m in msgs(V["S019"], "V3", "error"))


def test_real_run_has_no_citation_noise_or_duplicates(REAL):
    _, _, V, _ = REAL
    all_msgs = [m for v in V.values() for m in msgs(v)]
    assert not any("cites 'names'" in m or "cites 'loanword'" in m for m in all_msgs)
    assert not any("induced:" in m for m in all_msgs)  # measured rules never duplicate stated ones


# ---------- independence ----------

def test_verifier_ignores_what_the_translator_claims(HAND):
    L, plan, _, _ = HAND
    la = next(x for x in plan.lines if x.line_id == "S022")
    lying = Proposal(line_id="S022", scene="SC5", nadi_9_text="hulmar, tu khanasa ko?",
                     tokens=[], rules_used=["grammar:respect-2#1@present"], provider="test")
    v = Verifier(L, plan).verify(lying, la)
    assert any("uses ['tu']" in m for m in msgs(v, "V3", "error"))  # own analysis, not the claimed evidence


def test_loanword_with_an_unknown_ending_breaks_the_code_switching_rule(HAND):
    L, plan, _, _ = HAND
    la = next(x for x in plan.lines if x.line_id == "S016")
    p = Proposal(line_id="S016", scene="SC4", nadi_9_text="surat, brem mi office-zu tamiru.", tokens=[], provider="t")
    v = Verifier(L, plan).verify(p, la)
    assert any("code_switching" in m for m in msgs(v, "V3"))


def test_dispute_question_names_each_rule_once(REAL):
    import re
    _, plan, _, _ = REAL
    for la in plan.lines:
        for s in la.sensitive:
            if s.startswith("disputed"):
                pairs = re.findall(r"([\w-]+:[\w.-]+#\d+) vs ([\w-]+:[\w.-]+#\d+)", s)
                assert all(a != b for a, b in pairs) and len(pairs) == len(set(pairs)), s
