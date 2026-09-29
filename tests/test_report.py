"""Stage 8 tests: the deliverables the brief asks for, and the release rule."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.models import Decision
from nadi9.providers.mock import MockProvider
from nadi9.report import release, srt, tc
from nadi9.run import Run
from nadi9.tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[1]
BRIEF_FILES = ["subtitles.srt", "subtitle_decisions.jsonl", "learned_rules.json", "review_queue.json", "final_report.md"]


def full_run(fixtures: str, out: Path) -> tuple[Run, str]:
    r = Run(ROOT / "packs" / "synthetic", MockProvider(ROOT / "fixtures" / fixtures, "nadi9-synthetic-v1"),
            MockTranscriber(), Budget(25, 50))
    r.execute()
    return r, r.write(out)


@pytest.fixture(scope="module")
def HAND(tmp_path_factory):
    out = tmp_path_factory.mktemp("hand")
    return full_run("replay", out) + (out,)


@pytest.fixture(scope="module")
def REAL(tmp_path_factory):
    out = tmp_path_factory.mktemp("real")
    return full_run("recorded_full", out) + (out,)


def test_all_brief_deliverables_are_written(REAL):
    *_, out = REAL
    for f in BRIEF_FILES:
        assert (out / f).stat().st_size > 0, f


def test_committed_sample_run_is_complete_and_current(REAL):
    """sample_run/ must be exactly what replaying the recorded real run produces today."""
    *_, out = REAL
    sample = ROOT / "sample_run"
    for f in BRIEF_FILES:
        assert (sample / f).exists(), f
    fresh = [json.loads(x) for x in (out / "subtitle_decisions.jsonl").read_text().splitlines()]
    kept = [json.loads(x) for x in (sample / "subtitle_decisions.jsonl").read_text().splitlines()]
    strip = lambda d: {k: v for k, v in d.items() if k != "state_version"}  # noqa: E731
    assert [strip(d) for d in fresh] == [strip(d) for d in kept]


def test_srt_is_in_episode_order_and_marks_unfinished_lines(HAND):
    _, _, out = HAND
    text = (out / "subtitles.srt").read_text()
    blocks = [b for b in text.strip().split("\n\n")]
    assert len(blocks) == 25
    starts = [re.search(r"(\d\d:\d\d:\d\d,\d{3}) -->", b).group(1) for b in blocks]
    assert starts == sorted(starts)
    assert "[REVIEW S012] salo, sorai!" in text and "[NO SUBTITLE S017" in text
    assert "\nhulmar, tavo khanasi ko?\n" in text + "\n"  # approved after the linguist's correction
    approved = (out / "subtitles_approved.srt").read_text()
    assert "[REVIEW" not in approved and "[NO SUBTITLE" not in approved and approved.count("-->") == 7


def test_timecode_format():
    assert tc(3_723_004) == "01:02:03,004"


def test_release_rule(HAND, REAL):
    assert HAND[1] == "DO NOT RELEASE"          # S012 uses 'sorai' for the headman: critical
    assert REAL[1] == "RELEASE AFTER REVIEW"    # the real model avoided it; nothing critical is open
    ok = Decision(subtitle_id="S1", source_text="x", nadi_9_text="y", confidence=0.9, confidence_reason="r",
                  decision="APPROVE")
    assert release([ok], [])[0] == "READY"
    assert release([ok], [{"event": "E"}])[0] == "RELEASE AFTER REVIEW"  # an unapplied correction blocks READY


def test_review_queue_is_ordered_and_self_contained(HAND):
    _, _, out = HAND
    q = json.loads((out / "review_queue.json").read_text())
    assert q[0]["subtitle_id"] == "S012" and q[0]["priority"] == "critical"
    order = ["critical", "high", "normal"]
    assert [order.index(x["priority"]) for x in q] == sorted(order.index(x["priority"]) for x in q)
    for item in q:
        assert item["question"] and item["timecode"] and "evidence" in item


def test_report_explains_every_surprise_from_the_brief(REAL):
    _, _, out = REAL
    rep = (out / "final_report.md").read_text()
    for title in ("Poisoned dictionary entries", "Wrong approved examples", "A word whose meaning depends on status",
                  "A linguist's correction mid-run", "The model proposing unsupported rules",
                  "Instructions hidden in retrieved text", "Tool or model failures"):
        assert f"**{title}.**" in rep, title
    assert "Recorded from groq:openai/gpt-oss-120b" in rep
    assert "| EV01 | grammar:respect-2: present ending -sa -> -si | S022 |" in rep
    assert "`hulmar tavo khanasi ko` APPROVE" in rep  # the real model's own re-translation after the correction
