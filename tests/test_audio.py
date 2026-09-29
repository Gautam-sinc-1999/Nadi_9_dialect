"""Audio: real transcription through Groq Whisper (fake transport here), and how transcripts are used."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from nadi9.budget import Budget
from nadi9.profiler import build_profile
from nadi9.profiler.profile import target_texts
from nadi9.providers.base import ProviderUnavailable
from nadi9.providers.mock import MockProvider
from nadi9.tools.transcribe import GroqTranscriber, Transcriber, segments_from_whisper

WHISPER = {"language": "english", "segments": [
    {"start": 0.0, "end": 2.0, "text": " mi gedo vela.", "avg_logprob": -0.1, "no_speech_prob": 0.05},
    {"start": 2.0, "end": 3.0, "text": " ", "avg_logprob": -0.1, "no_speech_prob": 0.9},
    {"start": 3.0, "end": 5.0, "text": " nesh mi tolv velen.", "avg_logprob": -1.2, "no_speech_prob": 0.1}]}


def test_whisper_segments_and_confidence():
    segs = segments_from_whisper(WHISPER)
    assert [s["text"] for s in segs] == ["mi gedo vela.", "nesh mi tolv velen."]  # empty segment dropped
    assert segs[0]["asr_confidence"] == round(math.exp(-0.1) * 0.95, 3)
    assert segs[1]["asr_confidence"] < 0.6  # Whisper itself was unsure: flagged downstream


class Fake:
    def __init__(self, status=200, body=WHISPER):
        self.status, self.body, self.calls = status, body, []

    def __call__(self, method, url, headers, body, timeout):
        self.calls.append((url, headers, body))
        return self.status, {}, json.dumps(self.body).encode()


def test_groq_transcriber_request(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....WAVE")
    t = GroqTranscriber("sk-secret", transport=(fake := Fake()))
    t.vocabulary = ["gedo", "vela", "nesh"]
    assert len(t.transcribe(wav)) == 2
    url, headers, body = fake.calls[0]
    assert url.endswith("/audio/transcriptions") and headers["Content-Type"].startswith("multipart/form-data")
    assert b'name="model"' in body and b"whisper-large-v3" in body and b'filename="a.wav"' in body
    assert b"gedo vela nesh" in body  # the pack's own words as a spelling hint
    assert "sk-secret" not in t.name


def test_groq_transcriber_failures(tmp_path):
    with pytest.raises(FileNotFoundError):
        GroqTranscriber("k", transport=Fake()).transcribe(tmp_path / "missing.wav")
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"x")
    with pytest.raises(ProviderUnavailable, match="GROQ_API_KEY"):
        GroqTranscriber("k", transport=Fake(401, {})).transcribe(wav)


class FakeASR(Transcriber):
    name = "fake-asr"

    def __init__(self):
        self.calls, self.vocabulary = 0, []

    def transcribe(self, audio_path):
        self.calls += 1
        return segments_from_whisper(WHISPER)


def audio_pack(tmp_path: Path) -> Path:
    (tmp_path / "interview_x.wav").write_bytes(b"RIFF....WAVE")
    (tmp_path / "interview_x.json").write_text(json.dumps(
        {"topic": "family", "recorded_by": "someone", "audio_file": "interview_x.wav", "segments": []}))
    (tmp_path / "dictionary.csv").write_text("term,gloss\ngedo,home\nvela,go\n")
    return tmp_path


def test_transcribed_segments_become_evidence_for_reviewers_not_for_learning(tmp_path):
    asr = FakeASR()
    profile, canon = build_profile(audio_pack(tmp_path), MockProvider(), asr, Budget(25, 50))
    ids = [s.source_id for s in profile.sources]
    assert "interview_x" in ids and "interview_x_wav" in ids  # same name, different files: distinct ids
    assert asr.calls == 1  # the referenced .wav is transcribed once, through its interview record
    assert asr.vocabulary[:2] == ["gedo", "vela"]  # vocabulary known before transcription
    assert len(canon.segments) == 2 and all("machine_transcript" in s.flags for s in canon.segments)
    assert not [t for t in target_texts(canon) if t[0].startswith("interview_x")]  # not learned from


def test_mock_asr_never_invents_speech(tmp_path):
    from nadi9.tools.transcribe import MockTranscriber
    profile, canon = build_profile(audio_pack(tmp_path), MockProvider(), MockTranscriber(), Budget(25, 50))
    sp = next(s for s in profile.sources if s.source_id == "interview_x")
    assert sp.status.value == "unavailable" and not canon.segments
