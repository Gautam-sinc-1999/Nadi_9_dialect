"""Stage 1 tests: meanings are checked against evidence, conflicts are resolved or left visible."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nadi9.lexicon import build_lexicon, score_sources
from nadi9.models import (CanonicalPack, ClaimStatus, EntryOutcome, Example, LexEntry, TextSection)
from nadi9.profiler import build_profile
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "replay"
KEY = json.loads((ROOT / "packs" / "synthetic_answer_key.json").read_text())


@pytest.fixture(scope="module")
def pack1():
    profile, canon = build_profile(ROOT / "packs" / "synthetic", MockProvider(FIXTURES, "nadi9-synthetic-v1"),
                                   MockTranscriber())
    claims = build_lexicon(canon)
    return profile, canon, {c.term: c for c in claims}, score_sources(profile, canon, claims)


def _key_ref(r: str) -> str:  # "dictionary:A:A-013" -> "dictionary_a:A-013"
    _, d, entry = r.split(":")
    return f"dictionary_{d.lower()}:{entry}"


def contradicted(claims) -> set[str]:
    return {s.entry_ref for c in claims.values() for s in c.senses if s.outcome == EntryOutcome.CONTRADICTED}


def test_every_poisoned_entry_is_caught(pack1):
    _, _, claims, _ = pack1
    poisoned = {_key_ref(p["ref"]) for p in KEY["planted_traps"]["poisoned_dictionary_entries"]}
    assert poisoned <= contradicted(claims)


def test_no_correct_entry_is_called_poisoned(pack1):
    _, _, claims, _ = pack1
    poisoned = {_key_ref(p["ref"]) for p in KEY["planted_traps"]["poisoned_dictionary_entries"]}
    assert contradicted(claims) <= poisoned


@pytest.mark.parametrize("term,meaning", [
    ("brem", "tomorrow"), ("vel", "go"), ("kelim", "thank you"), ("dem", "ten"), ("hulm", "village")])
def test_resolved_meanings(pack1, term, meaning):
    _, _, claims, _ = pack1
    assert claims[term].status == ClaimStatus.RESOLVED_CONFLICT
    assert claims[term].meaning == meaning
    assert claims[term].reason  # every decision explains itself


def test_status_word_keeps_its_warning(pack1):
    _, _, claims, _ = pack1
    assert claims["sorai"].chosen_entry == "dictionary_b:B-028"
    assert "insulting" in claims["sorai"].meaning
    assert "expert_notes:REH-3" in claims["sorai"].discussed_in  # the dissenting expert stays visible


def test_dictionary_a_only_words_are_single_source(pack1):
    _, _, claims, _ = pack1
    for t in ["putra", "yoru", "dovar", "meethu", "agra"]:
        assert claims[t].status == ClaimStatus.SINGLE_SOURCE, t


def test_quarantined_entry_never_becomes_a_claim(pack1):
    _, _, claims, _ = pack1
    assert "ashru" not in claims


def test_trust_is_earned_not_assumed(pack1):
    _, _, _, trust = pack1
    t = {x.source_id: x for x in trust}
    assert t["dictionary_a"].measured and t["dictionary_b"].measured
    assert t["dictionary_a"].contradicted == 8 and t["dictionary_b"].contradicted == 0
    assert t["dictionary_b"].trust > t["dictionary_a"].trust  # the smaller dictionary wins on evidence


def test_conflicts_of_interest_flagged_for_both_dictionaries(pack1):
    _, _, _, trust = pack1
    coi = " ".join(next(x for x in trust if x.source_id == "expert_notes").conflicts_of_interest)
    assert "Tomas Rehn" in coi and "dictionary_a" in coi
    assert "Asha Kiran" in coi and "dictionary_b" in coi


def test_pack2_function_word_confirmed_by_grammar_sentence():
    _, canon = build_profile(ROOT / "packs" / "variant_veshti", MockProvider(), MockTranscriber())
    claims = {c.term: c for c in build_lexicon(canon)}
    assert claims["ne"].status == ClaimStatus.CORROBORATED
    assert all(c.status == ClaimStatus.CORROBORATED for c in claims.values())


# ---------- small hand-built packs ----------

def _lex(sid, nid, term, gloss, pos="noun"):
    return LexEntry(ref=f"{sid}:{nid}", source_id=sid, native_id=nid, term=term, gloss=gloss, pos=pos)


def _ex(nid, src, tgt):
    return Example(ref=f"ex:{nid}", source_id="ex", native_id=nid, source_text=src, target_text=tgt)


def test_conflict_without_evidence_stays_unresolved():
    canon = CanonicalPack(lexicon=[_lex("da", "1", "zub", "river"), _lex("db", "1", "zub", "mountain")])
    c = build_lexicon(canon)[0]
    assert c.status == ClaimStatus.UNRESOLVED_CONFLICT
    assert c.meaning is None  # no guess when evidence cannot decide
    assert "da:1" in c.conflict and "db:1" in c.conflict


def test_unsupported_meaning_is_flagged():
    canon = CanonicalPack(lexicon=[_lex("da", "1", "zub", "river")],
                          examples=[_ex("1", "The mountain is tall.", "zub tall."),
                                    _ex("2", "I climbed the mountain.", "mi zub.")])
    c = build_lexicon(canon)[0]
    assert c.status == ClaimStatus.UNSUPPORTED and c.meaning is None


def test_grammar_sentence_can_settle_a_conflict():
    canon = CanonicalPack(lexicon=[_lex("da", "1", "zub", "river"), _lex("db", "1", "zub", "mountain")],
                          sections=[TextSection(ref="g:1", source_id="g", native_id="1",
                                                text="The word zub means mountain. It is common.")])
    c = build_lexicon(canon)[0]
    assert c.status == ClaimStatus.RESOLVED_CONFLICT and c.meaning == "mountain"
