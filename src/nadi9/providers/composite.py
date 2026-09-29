"""Recorded answers first, a live model only for what was never asked before.

Used when a run is reopened: everything the model already answered is replayed (free and identical), and only new
requests, such as a re-translation after a correction, go to the live model, if one is allowed.
"""
from __future__ import annotations

from typing import Any, Optional

from .base import LLMProvider
from .mock import MockProvider


class ReplayThenLive(LLMProvider):
    def __init__(self, replay: MockProvider, live: Optional[LLMProvider] = None):
        self.replay, self.live = replay, live
        self.name = f"replay+{live.name}" if live else replay.name
        self.provenance = getattr(replay, "provenance", None)

    def complete_json(self, task: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.replay.has(task, payload) or self.live is None:
            return self.replay.complete_json(task, system, payload)
        return self.live.complete_json(task, system, payload)
