"""Stage 2 tests: rules are tested on evidence; wrong examples are found before anyone reports them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nadi9 import english as E
from nadi9.budget import Budget
from nadi9.hypotheses import validate_claims
from nadi9.models import HypStatus
from nadi9.pipeline import learn
from nadi9.providers.mock import MockProvider
from nadi9.social import addressee_status, relation_tags
from nadi9.tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "replay"
PACK1, PACK2 = ROOT / "packs" / "synthetic", ROOT / "packs" / "variant_veshti"
KEY = json.loads((ROOT / "packs" / "synthetic_answer_key.json").read_text())


def run(pack=PACK1, scenario=None, budget=None):
    budget = budget or Budget(25, 50)
    return learn(pack, MockProvider(FIXTURES, "nadi9-synthetic-v1", scenario), MockTranscriber(), budget), budget


@pytest.fixture(scope="module")
def L():
    return run(scenario={"hallucinate": True})[0]


def hyp(L, hid):
    return next(h for h in L.hypotheses if h.hyp_id == hid)


def test_wrong_examples_are_flagged_before_any_correction(L):
    expected = {r["ref"].replace("example:", "examples:") for r in KEY["planted_traps"]["wrong_approved_examples"]}
    assert {a.example_ref for a in L.audit if a.suspect} == expected  # exactly E09 and E14
    e14 = next(a for a in L.audit if a.example_ref == "examples:E14")
    assert {v["hypothesis"] for v in e14.violations} == {"grammar:tense-1#2", "grammar:tense-1#3"}
    e09 = next(a for a in L.audit if a.example_ref == "examples:E09")
    assert e09.challenges == ["expert_notes:KIR-1"]


def test_suspect_examples_count_less_afterwards(L):
    assert L.example_weights == {"examples:E09": 0.3, "examples:E14": 0.3}
    hulm = next(c for c in L.claims if c.term == "hulm")
    assert "suspect example" in hulm.reason  # its only usage evidence is E14


def test_model_invented_rule_is_refuted_by_the_examples(L):
    h = next(h for h in L.hypotheses if h.origin == "model")
    assert h.status == HypStatus.REFUTED and h.support_w == 0
    assert {"examples:E06", "examples:E11", "examples:E19"} <= set(h.counter)


def test_present_honorific_is_marked_extrapolated(L):
    h = hyp(L, "grammar:respect-2#1")
    assert h.by_feature["present"] == {"support": 0.0, "counter": 0.0}
    assert h.by_feature["past"]["support"] > 0 and h.by_feature["future"]["support"] > 0
    assert "present" in h.note


def test_expert_dispute_stays_visible_and_is_judged_on_evidence(L):
    reh = hyp(L, "expert_notes:REH-1#1")
    assert reh.status == HypStatus.REFUTED
    assert set(reh.counter) == {"examples:E07", "examples:E18"}  # approved subtitles use 'dovan'
    assert "Meridian" in reh.conflict_of_interest
    var = hyp(L, "expert_notes:VAR-1#2")
    assert var.status == HypStatus.SUPPORTED and "expert_notes:REH-1#1" in var.competes_with


def test_status_word_dispute_is_linked_not_merged(L):
    reh3 = hyp(L, "expert_notes:REH-3#1")
    assert reh3.status == HypStatus.UNSUPPORTED  # no evidence of 'sorai' used upward
    assert {"expert_notes:KIR-4#1", "dictionary_b:B-028#1"} <= set(reh3.competes_with)


def test_poisoned_entries_never_become_rules(L):
    poisoned = {"dictionary_a:" + p["ref"].split(":")[-1] for p in KEY["planted_traps"]["poisoned_dictionary_entries"]}
    assert not any(set(h.source_refs) & poisoned for h in L.hypotheses)


def test_meaning_of_an_ending_is_not_claimed_from_its_presence(L):
    assert hyp(L, "grammar:imp-1#1").status == HypStatus.WEAK
    assert "cannot be checked" in hyp(L, "grammar:imp-1#1").note


def test_trust_now_measures_examples_grammar_and_experts(L):
    t = {x.source_id: x for x in L.trust}
    assert t["examples"].measured and t["examples"].contradicted == 2
    assert t["grammar"].measured and t["grammar"].contradicted == 0
    assert t["expert_notes"].measured and t["expert_notes"].contradicted == 1  # REH-1


def test_one_budget_for_the_run():
    L, b = run()
    assert b.usage.model_calls == 3  # extract_rules, extract_claims, propose_rules
    assert b.usage.max_model_calls == 25


def test_stage2_model_outage_degrades_safely():
    L, _ = run(scenario={"fail_first": {"extract_claims": 9, "propose_rules": 9}})
    assert any("not extracted" in w for w in L.warnings)
    assert not any(h.origin in ("expert", "lexicon_note", "model") for h in L.hypotheses)
    assert {a.example_ref for a in L.audit if a.suspect} == {"examples:E09", "examples:E14"}  # grammar alone suffices


def test_pack2_has_no_false_suspects():
    L, _ = run(PACK2)
    assert not any(a.suspect for a in L.audit)
    vp = next(h for h in L.hypotheses if h.hyp_id == "induced:verb_position")
    assert vp.params == {"verb_position": "initial"} and vp.status == HypStatus.SUPPORTED


# ---------- units ----------

def test_claim_with_invented_quote_is_rejected():
    items = [{"ref": "n:1", "text": "Use tavo for elders.", "author": "X", "kind": "expert"}]
    ok, bad = validate_claims(items, [
        {"ref": "n:1", "template": "lexical_choice",
         "params": {"term": "tavo", "allowed_status": ["higher"], "condition": "guests"},
         "quote": "tavo is required for all guests"},
        {"ref": "n:9", "template": "gap", "params": {"topic": "x"}, "quote": "Use tavo"},
    ], {})
    assert not ok and [b["reason"] for b in bad] == ["quote not found in the cited item",
                                                     "cites 'n:9', which was not in the material given"]


@pytest.mark.parametrize("spk,adr,rel,expected", [
    ("worker", "boss", "employee to employer", "higher"),
    ("woman", "village headman", "villager to headman", "higher"),
    ("mother", "son", "parent to child", "lower"),
    ("young man", "friend", "equals, close friends", "equal"),
    ("younger sister", "elder brother", "siblings, close", "elder_sibling"),
    ("someone", "someone else", None, "unknown"),
])
def test_addressee_status(spk, adr, rel, expected):
    assert addressee_status(spk, adr, rel)[0] == expected


def test_relation_tags():
    assert relation_tags("My brother did not come. [We had quarrelled.]") == {"estranged"}
    assert "estranged" not in relation_tags("They quarrelled, but he said sorry and now he is my dovan again.")


@pytest.mark.parametrize("text,cues", [
    ("They left the village.", {"past"}), ("We will eat food tomorrow.", {"future"}),
    ("Mother is cooking food.", {"present"}), ("Did you eat?", {"past"}),
    ("He said sorry. Now he is my brother again.", {"past", "present"}), ("I want water.", set()),
])
def test_tense_cues(text, cues):
    assert E.tense_cues(text) == cues


def test_code_switching_is_a_tested_rule(L):
    h = hyp(L, "induced:code_switching")
    assert h.status == HypStatus.SUPPORTED and {"office", "meeting"} <= set(h.params["seen"])
    assert "examples:E17" in h.support  # 'osa mi office-ma meeting kora'
    assert not any(x.hyp_id == "induced:code_switching" for x in run(PACK2)[0].hypotheses)  # no English words there
