"""Mock / replay provider. Lets evaluators run everything without an API key.

Behaviour per task:
  1. If a replay fixture exists for (pack_id, task, key), return it.
  2. Otherwise return a deliberately conservative answer (e.g. "untestable",
     "unknown role") so an unfamiliar pack degrades to human review, never to invention.

Scenario knobs simulate the brief's surprise events:
  fail_first:   {"task": n}  -> first n calls of that task raise ProviderError
  hallucinate:  True         -> extract_rules adds one confident rule with no support in the text
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import LLMProvider, ProviderError

HALLUCINATED_RULE = {
    "template": "word_order",
    "params": {"verb_position": "initial"},
    "quote": "Questions always move the verb to the front of the sentence.",
    "note": "confidently proposed by the model",
}


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(self, fixtures_dir: Path | None = None, pack_id: str | None = None,
                 scenario: dict[str, Any] | None = None):
        self.fixtures: dict[str, Any] = {}
        if fixtures_dir and pack_id:
            d = Path(fixtures_dir) / pack_id
            if d.is_dir():
                for f in d.glob("*.json"):
                    self.fixtures[f.stem] = json.loads(f.read_text())
        self.scenario = scenario or {}
        # where the replayed answers came from (hand-written stand-ins, or a recorded real run)
        self.provenance = next((f.get("_provenance") for f in self.fixtures.values()
                                if isinstance(f, dict) and f.get("_provenance")), None)
        self._calls: dict[str, int] = {}
        self.name = "replay" if self.fixtures else "mock"

    def complete_json(self, task: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        n = self._calls.get(task, 0) + 1
        self._calls[task] = n
        if n <= self.scenario.get("fail_first", {}).get(task, 0):
            raise ProviderError(f"simulated outage on {task} (call {n})")

        handler = getattr(self, f"_task_{task}", None)
        if handler is None:
            raise ProviderError(f"mock has no handler for task '{task}'")
        return handler(payload)

    # --- tasks ---

    def _task_classify_role(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("classify_role", {}).get(payload["source_id"])
        return fx or {"role": "unclassified", "confidence": 0.0, "reason": "mock: no fixture"}

    def _task_map_fields(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("map_fields", {}).get(payload["source_id"])
        return fx or {"mapping": {}, "reason": "mock: no fixture"}

    def _task_extract_claims(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("extract_claims")
        return {"claims": json.loads(json.dumps(fx["claims"]))} if fx else {"claims": []}

    def _task_propose_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("propose_rules")
        rules = json.loads(json.dumps(fx["rules"])) if fx else []
        if self.scenario.get("hallucinate"):
            rules.append({"template": "word_order", "params": {"verb_position": "initial", "scope": "question"},
                          "note": "confidently proposed by the model: questions move the verb to the front"})
        return {"rules": rules}

    def _task_translate_scene(self, payload: dict[str, Any]) -> dict[str, Any]:
        """No fixture -> no lines. The mock never makes up a translation."""
        fx = self.fixtures.get("translate_scene", {}).get(payload["scene"])
        return {"lines": json.loads(json.dumps(fx["lines"]))} if fx else {"lines": []}

    def _task_parse_event(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("parse_event", {}).get(payload.get("event_id") or payload.get("source_id"))
        return json.loads(json.dumps(fx)) if fx else {}

    def _task_extract_rules(self, payload: dict[str, Any]) -> dict[str, Any]:
        fx = self.fixtures.get("extract_rules", {}).get(payload["source_id"])
        if fx is not None:
            rules = json.loads(json.dumps(fx["rules"]))  # deep copy
        else:
            # Conservative fallback: every section becomes an untestable rule quoting itself.
            rules = []
            for s in payload["sections"]:
                first = s["text"].strip().split("\n")[0][:200]
                rules.append({"section_id": s["section_id"], "template": "untestable",
                              "params": {}, "quote": first, "note": "mock: no fixture"})
        if self.scenario.get("hallucinate") and payload["sections"]:
            rules.append({"section_id": payload["sections"][0]["section_id"], **HALLUCINATED_RULE})
        return {"rules": rules}
