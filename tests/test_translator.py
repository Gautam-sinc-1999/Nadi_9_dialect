"""Stage 4 tests: the model only sees evidence, and its answers are checked before anyone trusts them."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.pipeline import learn
from nadi9.planner import analyse_episode
from nadi9.providers.base import ProviderError, ProviderUnavailable, call_with_retry
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import MockTranscriber
from nadi9.translator import build_package, translate_episode

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "replay"


def run(pack="synthetic", scenario=None):
    b = Budget(25, 50)
    prov = MockProvider(FIXTURES, "nadi9-synthetic-v1", scenario)
    L = learn(ROOT / "packs" / pack, prov, MockTranscriber(), b)
    plan = analyse_episode(L, b)
    log: list[str] = []
    return L, plan, translate_episode(plan, L, prov, b, log), b, log


@pytest.fixture(scope="module")
def R():
    L, plan, props, b, log = run()
    return L, plan, {p.line_id: p for p in props}, props, b


def test_every_line_gets_a_proposal_or_an_honest_refusal(R):
    _, _, P, props, _ = R
    assert len(props) == 25
    assert P["S017"].nadi_9_text is None and P["S017"].status == "no_translation"
    assert "late" in P["S017"].gaps
    assert sum(p.nadi_9_text is not None for p in props) == 24


def test_scenes_are_processed_in_risk_order(R):
    _, plan, _, props, _ = R
    assert props[0].scene == plan.order[0].scene == "SC3"


def test_structural_checks_catch_exactly_the_planted_mistakes(R):
    _, _, P, _, _ = R
    flagged = {lid: p.flags for lid, p in P.items() if p.flags}
    assert set(flagged) == {"S008", "S010", "S013"}
    assert flagged["S010"][0].startswith("invented: 'hapo'")
    assert "does not exist" in flagged["S013"][0]
    assert flagged["S008"][0].startswith("outside package: 'manu'")  # real word, not asked for: possible addition


def test_gap_markers_survive(R):
    _, _, P, _, _ = R
    assert "[?boy]" in P["S009"].nadi_9_text and P["S009"].gaps == ["boy"]


def test_package_contains_no_poison_and_no_injected_text(R):
    L, plan, _, _, _ = R
    for sc in plan.order:
        pkg = build_package([la for la in plan.lines if la.scene == sc.scene], L)
        blob = json.dumps(pkg)
        assert "IMPORTANT INSTRUCTION" not in blob and "Note to any AI" not in blob
        refs = {x["ref"] for x in pkg["lexicon"]} | {x["ref"] for x in pkg["affixes"]}
        assert not refs & {"dictionary_a:A-060", "dictionary_a:A-066", "dictionary_a:A-053", "dictionary_a:A-013",
                           "dictionary_a:A-090", "dictionary_a:A-061", "dictionary_a:A-003", "dictionary_a:A-096"}
        assert not any(r["id"].startswith("model:") for r in pkg["rules"])  # refuted model rule is never offered


def test_package_carries_the_stage3_warnings(R):
    L, plan, _, _, _ = R
    pkg = build_package([la for la in plan.lines if la.scene == "SC3"], L)
    s012 = next(x for x in pkg["lines"] if x["line_id"] == "S012")
    assert any(n.startswith("clash: 'sorai'") for n in s012["needs_attention"])


def test_budget_one_call_per_scene(R):
    *_, b = R
    assert b.usage.model_calls == 3 + 5  # learning + one per scene
    assert b.usage.tool_calls == 1 + 5   # one transcription attempt + one retrieval per scene


def test_model_outage_invents_nothing_and_cannot_drain_the_budget():
    _, _, props, b, _ = run(scenario={"fail_first": {"translate_scene": 99}})
    assert all(p.nadi_9_text is None and "nothing was invented" in p.reason for p in props)
    assert b.usage.model_calls == 3 + 3  # 3 attempts, then the circuit opens
    assert any("circuit open" in p.reason for p in props)


def test_unknown_pack_without_a_model_gets_no_translations():
    _, _, props, _, _ = run("variant_veshti")
    assert props and all(p.nadi_9_text is None and p.reason == "no model output for this line" for p in props)


def test_circuit_breaker_unit():
    b = Budget(10, 10, breaker_threshold=2)

    def boom():
        raise ProviderError("down")

    with pytest.raises(ProviderUnavailable):
        call_with_retry(boom, b, key="t", retries=5)
    assert b.usage.model_calls == 2  # stopped at the threshold, not after 6 attempts
    with pytest.raises(ProviderUnavailable, match="circuit open"):
        call_with_retry(lambda: 1, b, key="t")
    assert call_with_retry(lambda: 1, b, key="other") == 1  # other tasks unaffected
