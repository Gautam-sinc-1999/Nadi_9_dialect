"""Stage 3 tests: every line is understood, and risky lines are found, before any translation."""
from __future__ import annotations

from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.pipeline import learn
from nadi9.planner import analyse_episode
from nadi9.providers.mock import MockProvider
from nadi9.social import character_link
from nadi9.tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "replay"


def plan_for(pack: str):
    b = Budget(25, 50)
    L = learn(ROOT / "packs" / pack, MockProvider(FIXTURES, "nadi9-synthetic-v1"), MockTranscriber(), b)
    return analyse_episode(L, b)


@pytest.fixture(scope="module")
def P():
    p = plan_for("synthetic")
    return p, {la.line_id: la for la in p.lines}


@pytest.mark.parametrize("spk,adr,status", [
    ("Riya", "Arun", "elder_sibling"), ("Arun", "Riya", "lower"), ("Ama", "Arun", "lower"),
    ("Arun", "Ama", "higher"), ("Kabir", "Arun", "equal"), ("Kabir", "Bhola", "higher"),
    ("Arun", "Mr. Dass", "higher"), ("Mr. Dass", "Arun", "lower"),
])
def test_social_graph(P, spk, adr, status):
    plan, _ = P
    link = next(s for s in plan.social if s.speaker == spk and s.addressee == adr)
    assert link.status == status and link.reason  # every inference says why


def test_status_clash_is_the_top_risk(P):
    plan, L = P
    assert any(s.startswith("clash: 'sorai'") for s in L["S012"].sensitive)
    assert max(plan.lines, key=lambda a: a.risk).line_id == "S012"
    assert plan.order[0].scene == "SC3"
    assert not any(s.startswith("clash") for s in L["S009"].sensitive)  # same word, said to an equal


def test_present_honorific_only_where_you_is_the_subject(P):
    _, L = P
    assert any("present-tense honorific" in g for g in L["S022"].grammar_gaps)  # "are you eating?"
    assert not L["S021"].grammar_gaps  # "your food is good": no honorific verb, so EV01 must not touch it


def test_documented_gaps_hit_the_right_lines(P):
    _, L = P
    assert any("negative commands" in g for g in L["S017"].grammar_gaps)
    assert any("future negation" in g for g in L["S020"].grammar_gaps)
    assert any("question words" in g for g in L["S005"].grammar_gaps)
    assert any("clock time" in g for g in L["S015"].grammar_gaps)


def test_kinship_dispute_on_every_elder_brother_line(P):
    _, L = P
    for lid in ("S001", "S019", "S025"):
        assert any(s.startswith("disputed: 'dovan") for s in L[lid].sensitive), lid
    assert L["S001"].tags == ["estranged"] and L["S019"].tags == ["reconciled"]


@pytest.mark.parametrize("lid,expect", [("S022", "tavo"), ("S013", "tu"), ("S006", "tu"), ("S001", "do not decide")])
def test_pronoun_choice_follows_status(P, lid, expect):
    _, L = P
    choice = next(s for s in L[lid].sensitive if s.startswith("choice"))
    assert choice.endswith(expect)


def test_thank_you_is_not_an_address(P):
    _, L = P
    assert not L["S018"].features["second_person"]
    assert not any(s.startswith("choice") for s in L["S018"].sensitive)


def test_coverage_gaps_and_loanwords(P):
    _, L = P
    assert L["S009"].gaps == ["boy"] and L["S017"].gaps == ["late"]
    assert any(c.english == "meeting" and c.status == "loanword" for c in L["S015"].coverage)
    assert any(w.startswith("years -> yoru (single_source") for w in L["S023"].weak_words)


def test_timing_problem_found_before_translation(P):
    _, L = P
    assert L["S011"].timing["duration_ms"] == 800 and L["S011"].timing["issues"]


def test_low_risk_lines_stay_low(P):
    _, L = P
    for lid in ("S004", "S016", "S021"):
        assert L[lid].risk == 0, lid


def test_addressee_inferred_when_missing():
    plan = plan_for("variant_veshti")
    L = {la.line_id: la for la in plan.lines}
    assert (L["4"].addressee, L["4"].addressee_method) == ("Oran", "vocative")
    assert (L["2"].addressee, L["2"].addressee_method) == ("Mat", "turn_taking")
    assert L["1"].addressee is None and any("addressee status unknown" in f for f in L["1"].risk_factors)


def test_character_link_units():
    assert character_link("A", "Younger sister of B", 20, "B", "", 30)[1] == "elder_sibling"
    assert character_link("A", "", 30, "B", "A's boss", 50)[1] == "higher"
    assert character_link("A", "", 30, "B", "Village headman", 60)[:2] == ("authority", "higher")
    assert character_link("A", "Elder brother of C", 30, "B", "Farmer", 32)[1] == "equal"  # 'elder' is kinship, not rank
