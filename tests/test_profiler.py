"""Stage 0 tests: the profiler must adapt to any pack and fail safely."""
from __future__ import annotations

from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.models import Role, RuleStatus, RuleTemplate, SourceStatus
from nadi9.profiler import build_profile
from nadi9.profiler.mapping import auto_map
from nadi9.profiler.morphology import Morphology
from nadi9.profiler.rules import validate_rules
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import MockTranscriber
from nadi9.models import LexEntry

ROOT = Path(__file__).resolve().parents[1]
PACK1 = ROOT / "packs" / "synthetic"
PACK2 = ROOT / "packs" / "variant_veshti"
FIXTURES = ROOT / "fixtures" / "replay"


def run(pack: Path, scenario=None, budget=None, fail_asr=0, replay=True):
    provider = MockProvider(FIXTURES if replay else None, "nadi9-synthetic-v1", scenario)
    return build_profile(pack, provider, MockTranscriber(fail_times=fail_asr), budget)


def src(profile, sid):
    return next(s for s in profile.sources if s.source_id == sid)


# ---------- pack 1 ----------

def test_pack1_roles_and_records():
    profile, canon = run(PACK1)
    roles = {s.source_id: s.role for s in profile.sources}
    assert roles["dictionary_a"] == roles["dictionary_b"] == Role.DICTIONARY
    assert roles["examples"] == Role.EXAMPLES
    assert roles["grammar"] == Role.GRAMMAR
    assert roles["episode"] == Role.EPISODE
    assert roles["event_01_rule_correction"] == Role.EVENT
    assert (len(canon.lexicon), len(canon.examples), len(canon.lines), len(canon.characters)) == (150, 20, 25, 6)
    assert canon.lines[0].start_ms == 4000 and canon.lines[0].end_ms == 6200


def test_injections_are_quarantined_in_both_packs():
    p1, c1 = run(PACK1)
    assert {q["ref"] for q in p1.quarantined} == {"dictionary_a:A-060", "viewer_feedback:VF-08"}
    a060 = next(e for e in c1.lexicon if e.ref == "dictionary_a:A-060")
    assert "quarantined:injection" in a060.flags
    p2, _ = run(PACK2)
    assert [q["ref"] for q in p2.quarantined] == ["viewer_comments:row2"]


def test_hallucinated_rule_is_rejected_and_logged():
    profile, _ = run(PACK1, scenario={"hallucinate": True})
    bad = [r for r in profile.rules if r.status == RuleStatus.REJECTED_UNGROUNDED]
    assert len(bad) == 1 and "front of the sentence" in bad[0].quote
    good = [r for r in profile.rules if r.status == RuleStatus.GROUNDED]
    assert any(r.template == RuleTemplate.AGREEMENT and r.source_ref == "grammar:respect-2" for r in good)


def test_all_replay_rules_are_grounded():
    profile, _ = run(PACK1)
    assert profile.rules and all(r.status == RuleStatus.GROUNDED for r in profile.rules)


# ---------- failures ----------

def test_transcription_failure_is_contained():
    profile, canon = run(PACK1, fail_asr=5)
    i5 = src(profile, "interview_05")
    assert i5.status == SourceStatus.UNAVAILABLE
    assert any("transcription failed" in w for w in i5.warnings)
    assert src(profile, "interview_01").status == SourceStatus.OK
    assert len(canon.lines) == 25  # the rest of the pack is unaffected


def test_model_outage_falls_back_to_untestable_rules():
    profile, _ = run(PACK1, scenario={"fail_first": {"extract_rules": 10}})
    assert profile.rules and all(r.template == RuleTemplate.UNTESTABLE for r in profile.rules)
    assert all(r.method == "fallback" for r in profile.rules)
    assert any("rule extraction unavailable" in w for w in src(profile, "grammar").warnings)


def test_budget_is_a_hard_cap():
    b = Budget(max_model_calls=0, max_tool_calls=0)
    profile, canon = run(PACK1, budget=b)
    assert b.usage.model_calls == 0 and b.usage.tool_calls == 0
    assert all(r.method == "fallback" for r in profile.rules)
    assert len(canon.examples) == 20  # deterministic ingestion still works


def test_unknown_file_is_not_forced_into_a_role(tmp_path):
    (tmp_path / "mystery.json").write_text('[{"alpha": 1, "beta": 2}]')
    profile, _ = build_profile(tmp_path, MockProvider(), MockTranscriber())
    assert src(profile, "mystery").status == SourceStatus.UNCLASSIFIED


# ---------- adaptation ----------

def test_adapts_to_a_different_language_and_format():
    profile, canon = run(PACK2)
    roles = {s.source_id: s.role for s in profile.sources}
    assert roles == {
        "lexicon": Role.DICTIONARY, "parallel_sentences": Role.EXAMPLES, "grammar": Role.GRAMMAR,
        "episode_02": Role.EPISODE, "characters": Role.CHARACTERS, "viewer_comments": Role.FEEDBACK,
        "expert_notes": Role.EXPERT_NOTES, "interview_a": Role.INTERVIEW, "interview_b": Role.INTERVIEW,
        "correction_email": Role.EVENT,
    }
    shape = profile.shape
    assert shape.verb_position == {"initial": 14}
    affixes = {(a.kind, a.form): a for a in shape.affixes}
    assert ("prefix", "ka") in affixes and affixes[("prefix", "ka")].declared_by
    assert ("prefix", "zu") in affixes and not affixes[("prefix", "zu")].declared_by  # induced from data
    assert shape.particle_positions["mo"] == {"after_verb": 2}
    assert shape.particle_positions["ne"]["clause_start"] == 2


def test_target_column_found_by_content_not_header():
    profile, canon = run(PACK2)
    fm = {m.canonical_field: m for m in src(profile, "parallel_sentences").field_mapping}
    assert fm["target_text"].source_field == "Veshti" and fm["target_text"].method == "content"
    assert canon.examples[0].target_text == "kasim ek brak."


def test_srt_speakers_parsed():
    _, canon = run(PACK2)
    assert [l.speaker for l in canon.lines] == ["MAT", "JONAS", "MAT", "JONAS", "ORAN"]
    assert canon.lines[4].end_ms - canon.lines[4].start_ms == 800


# ---------- units ----------

def test_exact_column_beats_fuzzy_match():
    m, _ = auto_map(Role.FEEDBACK, [{"username": "a", "comment": "hi", "upvotes": 1}])
    assert m["text"] == "comment" and m["native_id"] is None


def test_verb_does_not_match_adverb():
    lex = [LexEntry(ref="x:1", source_id="x", native_id="1", term="sura", gloss="tomorrow", pos="adverb")]
    assert not Morphology(lex).is_pos("sura", "verb")


@pytest.mark.parametrize("quote,ok", [
    ("The particle `na` is placed immediately **before** the verb.", True),
    ("the particle na is placed   immediately before the verb.", True),
    ("The particle na always follows the verb.", False),
])
def test_quote_grounding(quote, ok):
    sections = [{"section_id": "neg-1", "heading": "",
                 "text": "The particle `na` is placed immediately **before** the verb."}]
    r = validate_rules("g", sections, [{"section_id": "neg-1", "template": "particle_position",
                                        "params": {"particle": "na", "relative_to": "verb", "position": "before"},
                                        "quote": quote}], "llm")[0]
    assert (r.status == RuleStatus.GROUNDED) is ok


def test_source_scope_author_and_date_are_captured():
    profile, _ = run(PACK1)
    a = src(profile, "dictionary_a")
    assert a.author.startswith("Meridian") and a.date == "2022-08-30" and a.scope.startswith("Broad coverage")
    assert src(profile, "grammar").scope.startswith("Word order, tense")  # from the note's own "Scope:" line


def test_pdf_without_its_library_is_reported_not_silently_lost(tmp_path):
    pytest.importorskip  # noqa: B018 (keeps the import used)
    try:
        import pypdf  # noqa: F401
        pytest.skip("pypdf is installed; the missing-library path cannot be exercised")
    except ImportError:
        pass
    (tmp_path / "grammar_note.pdf").write_bytes(b"%PDF-1.4 not really")
    profile, _ = build_profile(tmp_path, MockProvider(), MockTranscriber())
    s = src(profile, "grammar_note")
    assert s.status == SourceStatus.UNAVAILABLE and "pip install pypdf" in s.warnings[0]
