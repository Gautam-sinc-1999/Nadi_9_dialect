"""Stage 6 tests: computed confidence, decisions that match the answer key, precise questions."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.decider import APPROVE_AT, decide_all
from nadi9.pipeline import learn
from nadi9.planner import analyse_episode
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import MockTranscriber
from nadi9.translator import translate_episode
from nadi9.verifier import verify_all

ROOT = Path(__file__).resolve().parents[1]
KEY = {k: v["decision"] for k, v in
       json.loads((ROOT / "packs" / "synthetic_answer_key.json").read_text())["expected_decisions"].items()}


def run(fixtures: str):
    b = Budget(25, 50)
    prov = MockProvider(ROOT / "fixtures" / fixtures, "nadi9-synthetic-v1")
    L = learn(ROOT / "packs" / "synthetic", prov, MockTranscriber(), b)
    plan = analyse_episode(L, b)
    props = translate_episode(plan, L, prov, b)
    ver = verify_all(props, plan, L)
    return L, {d.subtitle_id: d for d in decide_all(props, ver, plan, L)}


@pytest.fixture(scope="module")
def HAND():
    return run("replay")


@pytest.fixture(scope="module")
def REAL():
    return run("recorded")


def matches_key(d) -> bool:
    exp = KEY[d.subtitle_id]
    return exp.startswith(d.decision) or (exp == "APPROVE_AFTER_EV01" and d.decision == "HUMAN_REVIEW")


def test_hand_answers_match_the_key_except_the_planted_citation(HAND):
    _, D = HAND
    differ = [lid for lid, d in D.items() if not matches_key(d)]
    assert differ == ["S013"]  # the hand-written answer cites a rule that does not exist
    assert "does not exist" in D["S013"].review_question


def test_real_model_answers_match_the_key_except_a_real_omission(REAL):
    _, D = REAL
    differ = [lid for lid, d in D.items() if not matches_key(d)]
    assert differ == ["S013"]  # gpt-oss rendered "go back" as "go"
    assert any("'back' is not expressed" in q for q in [D["S013"].review_question] + D["S013"].other_questions)


def test_brief_fields_are_present(HAND):
    _, D = HAND
    rec = json.loads(D["S012"].model_dump_json())
    for f in ("subtitle_id", "source_text", "nadi_9_text", "confidence", "decision", "evidence", "assumptions",
              "conflicts", "review_question"):
        assert f in rec


def test_confidence_is_computed_from_named_factors(HAND):
    _, D = HAND
    for d in D.values():
        if d.nadi_9_text:
            f = d.factors
            assert d.confidence == round(f["lexical"] * f["grammar"] * f["context"] * f["verification"], 2)
            assert d.confidence_reason
    assert D["S004"].confidence == 1.0
    assert "single source" in D["S006"].confidence_reason and "putra" in D["S006"].confidence_reason


def test_approve_means_confident_and_nothing_to_ask(HAND, REAL):
    for _, D in (HAND, REAL):
        for d in D.values():
            if d.decision == "APPROVE":
                assert d.confidence >= APPROVE_AT and d.review_question is None
            else:
                assert d.review_question and d.priority


def test_insult_risk_is_critical_and_asks_the_right_question(HAND):
    _, D = HAND
    d = D["S012"]
    assert d.priority == "critical" and d.confidence < 0.2
    assert d.review_question.startswith("Should Kabir's line to Bhola (higher status) keep 'sorai'?")
    assert "dictionary_b:B-028" in d.review_question


def test_kinship_question_shows_both_sides_with_evidence_and_interests(HAND):
    _, D = HAND
    q = D["S001"].review_question
    assert "expert_notes:VAR-1#1" in q and "expert_notes:REH-1#1" in q
    assert "refuted" in q and "examples:E07" in q
    assert "Tomas Rehn" in q and "Leela Varkey" in q


def test_extrapolated_form_is_the_question_for_s022(HAND):
    _, D = HAND
    assert D["S022"].decision == "HUMAN_REVIEW"
    assert "present-tense honorific" in D["S022"].review_question
    assert "extrapolated" in " ".join(D["S022"].issues + [D["S022"].confidence_reason]) or \
           "no example" in D["S022"].confidence_reason


def test_declined_line_is_insufficient_evidence(HAND):
    _, D = HAND
    d = D["S017"]
    assert d.decision == "INSUFFICIENT_EVIDENCE" and d.nadi_9_text is None
    assert "'late'" in d.review_question


def test_every_evidence_reference_can_be_looked_up(HAND):
    L, D = HAND
    c = L.canon
    records = {r.ref for coll in (c.lexicon, c.examples, c.segments, c.sections, c.notes) for r in coll}
    for d in D.values():
        for ref in d.evidence:
            assert ref in records, (d.subtitle_id, ref)


def test_review_items_carry_the_spelling_fix_when_there_is_one(REAL):
    _, D = REAL
    assert D["S014"].decision == "APPROVE" and D["S014"].nadi_9_text == "hulmar, mi na jana"


def test_humour_is_never_approved_without_a_human(REAL):
    """Regression: the humour check once never fired, because Stage 3 labels it 'humour' and the pattern did not."""
    _, D = REAL
    humour = ["S009", "S011", "S012", "S024"]  # direction notes say joke / teasing
    assert all(D[lid].decision != "APPROVE" for lid in humour)
    assert any(q.startswith("Humour:") for lid in humour for q in [D[lid].review_question] + D[lid].other_questions)


def test_dispute_question_lists_each_side_once(HAND, REAL):
    import re
    for _, D in (HAND, REAL):
        for lid in ("S001", "S019", "S025"):
            q = D[lid].review_question or ""
            if "Which form" in q:
                ids = re.findall(r"([\w-]+:[\w.-]+#\d+) says", q)
                assert ids and len(ids) == len(set(ids)), q
