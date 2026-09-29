"""Hard caps on model and tool calls. Enforced in code, not by prompt."""
from __future__ import annotations

from .models import BudgetUsage


class BudgetExceeded(RuntimeError):
    pass


class CircuitBreaker:
    """After `threshold` consecutive failures of the same task, stop calling it (fail fast, spend nothing)."""

    def __init__(self, threshold: int = 3):
        self.threshold = threshold
        self.failures: dict[str, int] = {}

    def is_open(self, key: str) -> bool:
        return self.failures.get(key, 0) >= self.threshold

    def fail(self, key: str) -> None:
        self.failures[key] = self.failures.get(key, 0) + 1

    def ok(self, key: str) -> None:
        self.failures[key] = 0


class Budget:
    def __init__(self, max_model_calls: int = 25, max_tool_calls: int = 50, breaker_threshold: int = 3):
        self.usage = BudgetUsage(max_model_calls=max_model_calls, max_tool_calls=max_tool_calls)
        self.breaker = CircuitBreaker(breaker_threshold)

    def can(self, kind: str) -> bool:
        if kind == "model":
            return self.usage.model_calls < self.usage.max_model_calls
        return self.usage.tool_calls < self.usage.max_tool_calls

    def spend(self, kind: str) -> None:
        if not self.can(kind):
            raise BudgetExceeded(f"{kind} call budget exhausted ({self._limit(kind)})")
        if kind == "model":
            self.usage.model_calls += 1
        else:
            self.usage.tool_calls += 1

    def _limit(self, kind: str) -> int:
        return self.usage.max_model_calls if kind == "model" else self.usage.max_tool_calls
