"""Runs the learning stages in order with one shared budget.

    S0 profile -> S1 lexicon -> S2 collect hypotheses (model calls, once)
                            -> [S2 test + audit -> S1 again with suspect examples down-weighted] x <= 2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .budget import Budget
from .hypotheses import collect, test_and_audit
from .lexicon import build_lexicon, score_sources
from .models import CanonicalPack, ExampleAudit, Hypothesis, LexClaim, PackProfile, SourceTrust
from .profiler import build_profile
from .providers.base import LLMProvider
from .tools.transcribe import Transcriber

MAX_PASSES = 2


@dataclass
class Learned:
    profile: PackProfile
    canon: CanonicalPack
    claims: list[LexClaim]
    trust: list[SourceTrust]
    hypotheses: list[Hypothesis]
    audit: list[ExampleAudit]
    rejected: list[dict[str, Any]] = field(default_factory=list)
    example_weights: dict[str, float] = field(default_factory=dict)
    passes: int = 0
    warnings: list[str] = field(default_factory=list)


def _coi_by_author(trust: list[SourceTrust]) -> dict[str, str]:
    out: dict[str, str] = {}
    for t in trust:
        for line in t.conflicts_of_interest:
            name = line.split(" is the author", 1)[0].split(" works for", 1)[0]
            out.setdefault(name, line)
    return out


def learn(pack_dir: Path, provider: LLMProvider, transcriber: Transcriber, budget: Budget,
          propose: bool = True) -> Learned:
    profile, canon = build_profile(pack_dir, provider, transcriber, budget)
    claims = build_lexicon(canon)
    trust = score_sources(profile, canon, claims)
    collected = collect(profile, canon, claims, provider, budget, _coi_by_author(trust), propose=propose)

    weights: dict[str, float] = {}
    passes = 0
    while True:
        passes += 1
        hyps, audit = test_and_audit(collected, profile, canon, weights)
        new = {a.example_ref: a.weight for a in audit if a.suspect}
        if new == weights or passes >= MAX_PASSES:
            break
        weights = new
        claims = build_lexicon(canon, weights)  # re-judge meanings with suspect examples down-weighted

    trust = score_sources(profile, canon, claims, hyps, audit)
    return Learned(profile, canon, claims, trust, hyps, audit, collected.rejected, weights, passes,
                   collected.warnings)
