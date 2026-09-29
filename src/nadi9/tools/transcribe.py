"""Transcription tools. The mock never invents a transcript; it either fails or reports no audio.

GroqTranscriber sends the audio file (and nothing else from the pack) to Groq's Whisper endpoint and returns
segments in the interview format. Nadi-9 is not in Whisper's training data, so its output is treated like any
other low-trust evidence: each segment's confidence comes from Whisper's own scores (average log-probability and
no-speech probability), and segments below the ASR threshold are flagged and not learned from.
"""
from __future__ import annotations

import json
import math
import uuid
from pathlib import Path
from typing import Any, Optional

from ..providers.base import ProviderError, ProviderUnavailable

AUDIO_TYPES = {".wav": "audio/wav", ".mp3": "audio/mpeg", ".m4a": "audio/mp4", ".flac": "audio/flac",
               ".ogg": "audio/ogg"}


class Transcriber:
    name = "base"

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        raise NotImplementedError


class MockTranscriber(Transcriber):
    """`fail_times`: raise a transient error that many times first (simulates an outage)."""
    name = "mock-asr"

    def __init__(self, fail_times: int = 0):
        self.fail_times = fail_times
        self.calls = 0

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise ProviderError(f"ASR service timeout on {audio_path.name}")
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not in pack: {audio_path.name}")
        # The mock refuses to fabricate speech; use GroqTranscriber (or another engine) for real audio.
        raise ProviderError("mock ASR cannot decode audio; run with --provider groq to transcribe with Whisper")


class GroqTranscriber(Transcriber):
    def __init__(self, api_key: str, model: str = "whisper-large-v3",
                 base_url: str = "https://api.groq.com/openai/v1", transport=None, timeout: float = 120.0,
                 language: Optional[str] = None):
        from ..providers.groq import USER_AGENT, urllib_transport
        if not api_key:
            raise ProviderUnavailable("GROQ_API_KEY is not set (add it to .env)")
        self._key, self.model, self.base_url = api_key, model, base_url.rstrip("/")
        self.transport, self.timeout, self.language = transport or urllib_transport, timeout, language
        self.user_agent = USER_AGENT
        self.name = f"groq:{model}"
        self.vocabulary: list[str] = []  # filled by the profiler: the pack's own words, as a spelling hint

    def transcribe(self, audio_path: Path) -> list[dict[str, Any]]:
        if not audio_path.exists():
            raise FileNotFoundError(f"audio file not in pack: {audio_path.name}")
        fields = {"model": self.model, "response_format": "verbose_json", "temperature": "0"}
        if self.language:
            fields["language"] = self.language
        if self.vocabulary:  # Whisper's prompt is short: the first ~120 words are plenty for spelling
            fields["prompt"] = " ".join(self.vocabulary[:120])
        body, ctype = _multipart(fields, audio_path)
        headers = {"Authorization": f"Bearer {self._key}", "Content-Type": ctype, "Accept": "application/json",
                   "User-Agent": self.user_agent}
        status, _, raw = self.transport("POST", f"{self.base_url}/audio/transcriptions", headers, body, self.timeout)
        if status == 429 or status >= 500:
            raise ProviderError(f"groq ASR {status}")
        if status in (401, 403):
            raise ProviderUnavailable(f"groq ASR {status}: check GROQ_API_KEY")
        if status != 200:
            raise ProviderError(f"groq ASR {status}: {raw[:200].decode(errors='replace')}")
        try:
            data = json.loads(raw)
        except ValueError as e:
            raise ProviderError(f"groq ASR answer was not JSON: {e}") from e
        return segments_from_whisper(data)


def segments_from_whisper(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Whisper verbose_json -> interview segments. Confidence = exp(avg_logprob) x (1 - no_speech_prob)."""
    out = []
    for i, s in enumerate(data.get("segments") or [], 1):
        text = str(s.get("text", "")).strip()
        if not text:
            continue
        conf = math.exp(float(s.get("avg_logprob", -5.0))) * (1.0 - float(s.get("no_speech_prob", 0.0)))
        out.append({"seg": f"seg{i}", "start": s.get("start"), "end": s.get("end"), "text": text,
                    "asr_confidence": round(max(0.0, min(1.0, conf)), 3),
                    "asr_note": f"whisper, detected language {data.get('language', '?')}"})
    return out


def _multipart(fields: dict[str, str], file_path: Path) -> tuple[bytes, str]:
    boundary = f"nadi9-{uuid.uuid4().hex}"
    parts = []
    for k, v in fields.items():
        parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n{v}\r\n".encode())
    ctype = AUDIO_TYPES.get(file_path.suffix.lower(), "application/octet-stream")
    parts.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"{file_path.name}\"\r\n"
                 f"Content-Type: {ctype}\r\n\r\n".encode() + file_path.read_bytes() + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"
