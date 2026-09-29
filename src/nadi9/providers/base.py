"""Provider abstraction. Callers never talk to a model SDK directly."""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from ..budget import Budget, BudgetExceeded


class ProviderError(RuntimeError):
    """A transient failure (timeout, 5xx, rate limit). Safe to retry."""


class ProviderUnavailable(RuntimeError):
    """Retries exhausted or budget spent. Caller must fall back, not crash."""


class LLMProvider(ABC):
    name: str = "base"

    @abstractmethod
    def complete_json(self, task: str, system: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Run one task and return parsed JSON.

        `system` holds our instructions. `payload` holds pack content and is always
        presented to the model as data, never as instructions.
        """


def call_with_retry(
    fn: Callable[[], Any],
    budget: Budget,
    kind: str = "model",
    retries: int = 2,
    backoff_s: float = 0.0,
    log: list[str] | None = None,
    key: str | None = None,
) -> Any:
    """Spend one budget unit per attempt; retry transient errors; never raise past the caller's fallback.

    `key` names the task for the circuit breaker: once a task has failed `threshold` times in a row,
    further calls fail immediately and spend nothing, so an outage cannot drain the budget.
    """
    key = key or kind
    last: Exception | None = None
    for attempt in range(retries + 1):
        if budget.breaker.is_open(key):
            raise ProviderUnavailable(f"circuit open for '{key}' after repeated failures{f': {last}' if last else ''}")
        try:
            budget.spend(kind)
        except BudgetExceeded as e:
            raise ProviderUnavailable(str(e)) from e
        try:
            result = fn()
            budget.breaker.ok(key)
            return result
        except ProviderError as e:
            last = e
            budget.breaker.fail(key)
            if log is not None:
                log.append(f"{key} call failed (attempt {attempt + 1}/{retries + 1}): {e}")
            if backoff_s:
                time.sleep(backoff_s * (2 ** attempt))
    raise ProviderUnavailable(f"failed after {retries + 1} attempts: {last}")
