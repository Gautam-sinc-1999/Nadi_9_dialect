"""Operating the product: start and reopen runs, apply corrections, record human reviews, export.

A run directory is self-contained:
    run_config.json      how the run was made (pack, model, budget)
    answers/<pack_id>/   every model answer, recorded, so the run can be reopened without new model calls
    corrections.json     corrections applied after the run, in order (re-applied when the run is reopened)
    human_reviews.json   the reviewer's decisions per line
    + every output the pipeline writes
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from .budget import Budget
from .config import load_dotenv, setting
from .providers.composite import ReplayThenLive
from .providers.mock import MockProvider
from .report import PRIORITY, tc
from .run import Run
from .tools.transcribe import MockTranscriber

ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ("approve", "edit", "reject", "reset")


@dataclass
class RunConfig:
    pack: str                                  # path to the pack folder
    provider: str = "replay"                   # mock | replay | groq
    fixtures: Optional[str] = None             # for replay: a folder of recorded answers
    model: Optional[str] = None                # for groq
    max_model_calls: int = 25
    max_tool_calls: int = 50
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))


def rel(p: Path | str) -> str:
    """Store paths relative to the project, so a run still opens after the folder is moved or renamed."""
    p = Path(p).resolve()
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def absolute(p: Optional[str]) -> Optional[Path]:
    if not p:
        return None
    q = Path(p)
    return q if q.is_absolute() else ROOT / q


def pack_id_of(pack: Path) -> str:
    for m in pack.glob("*manifest*.json"):
        try:
            return json.loads(m.read_text()).get("pack_id") or pack.name
        except (ValueError, OSError):
            break
    return pack.name


def _live(cfg: RunConfig, out: Path, key: Optional[str]):
    from .providers.groq import DEFAULT_MODEL, GroqProvider
    from .tools.transcribe import GroqTranscriber
    load_dotenv(ROOT / ".env")
    k = key or setting("GROQ_API_KEY") or ""
    prov = GroqProvider(k, model=cfg.model or setting("GROQ_MODEL", DEFAULT_MODEL),
                        max_tokens=int(setting("GROQ_MAX_TOKENS", "6000")),
                        record_dir=out / "answers", pack_id=pack_id_of(absolute(cfg.pack)))
    return prov, GroqTranscriber(k)


def _provider(cfg: RunConfig, out: Path, key: Optional[str], allow_live: bool, reopening: bool):
    pack = absolute(cfg.pack)
    pid = pack_id_of(pack)
    if cfg.provider == "groq":
        if reopening:  # replay what was recorded; the live model only for new requests, if allowed
            live, asr = _live(cfg, out, key) if allow_live else (None, MockTranscriber())
            return ReplayThenLive(MockProvider(out / "answers", pid), live), asr
        return _live(cfg, out, key)
    fixtures = absolute(cfg.fixtures) if (cfg.provider == "replay" and cfg.fixtures) else None
    replay = MockProvider(fixtures, pid)
    if allow_live and reopening:
        live, _ = _live(cfg, out, key)
        return ReplayThenLive(replay, live), MockTranscriber()
    return replay, MockTranscriber()


def start_run(cfg: RunConfig, out: Path, key: Optional[str] = None,
              progress: Optional[Callable[[str], None]] = None) -> Run:
    out.mkdir(parents=True, exist_ok=True)
    cfg.pack, cfg.fixtures = rel(cfg.pack), (rel(cfg.fixtures) if cfg.fixtures else None)
    save_config(out, cfg)
    (out / "corrections.json").write_text("[]")  # a fresh run starts with no corrections
    prov, asr = _provider(cfg, out, key, allow_live=False, reopening=False)
    say = progress or (lambda _m: None)
    say("reading the pack, checking meanings and rules, planning the episode")
    r = Run(absolute(cfg.pack), prov, asr, Budget(cfg.max_model_calls, cfg.max_tool_calls))
    r.execute(progress=say)
    say("writing outputs")
    r.write(out)
    return r


def save_config(out: Path, cfg: RunConfig) -> None:
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    if not (out / "corrections.json").exists():
        (out / "corrections.json").write_text("[]")


def can_reopen(out: Path) -> bool:
    return (out / "run_config.json").exists()


def reopen(out: Path, key: Optional[str] = None, allow_live: bool = False) -> Run:
    """Rebuild a finished run in memory, from its recorded answers, and re-apply its corrections."""
    cfg = RunConfig(**json.loads((out / "run_config.json").read_text()))
    prov, asr = _provider(cfg, out, key, allow_live=allow_live, reopening=True)
    r = Run(absolute(cfg.pack), prov, asr, Budget(cfg.max_model_calls + 25, cfg.max_tool_calls + 25))
    r.execute()
    for c in json.loads((out / "corrections.json").read_text()):
        r.apply_event(c)
    return r


# ---------------------------------------------------------------- corrections

def correction_from_form(kind: str, **f: Any) -> dict[str, Any]:
    if kind == "rule":
        return {"type": "rule_correction", "target": f["rule_source"],
                "change": {"tense": f["tense"], "old_suffix": f.get("old_suffix", ""), "new_suffix": f["new_suffix"]},
                "from": {"name": f.get("sender") or "reviewer"}, "message": f.get("message", "")}
    if kind == "example":
        return {"type": "example_challenge", "target": f"example:{f['example_id']}",
                "change": {"example_id": f["example_id"], "old_nadi_9": f.get("old_text", ""),
                           "new_nadi_9": f["new_text"]},
                "from": {"name": f.get("sender") or "reviewer"}, "message": f.get("message", "")}
    return {"message": f["message"], "from": {"name": f.get("sender") or "reviewer"}}  # free text: the model reads it


def apply_correction(r: Run, out: Path, event: dict[str, Any]) -> dict[str, Any]:
    """Apply, persist (so a reopened run repeats it), rewrite outputs, and flag reviews of changed lines."""
    corrections = json.loads((out / "corrections.json").read_text()) if (out / "corrections.json").exists() else []
    event = {**event, "event_id": event.get("event_id") or f"UI-{len(corrections) + 1}"}
    result = r.apply_event(event)
    if result.get("applied"):
        corrections.append(event)
        (out / "corrections.json").write_text(json.dumps(corrections, indent=2))
        r.write(out)
        reviews = load_reviews(out)
        for x in result.get("redone", []):
            if x["line"] in reviews and reviews[x["line"]]["action"] != "stale":
                reviews[x["line"]] = {**reviews[x["line"]], "action": "stale", "previous_action": reviews[x["line"]]["action"],
                                      "note": f"{event['event_id']} changed this line after your review; review again"}
        _save(out, reviews)
    return result


# ---------------------------------------------------------------- human review

def load_reviews(out: Path) -> dict[str, dict[str, Any]]:
    f = out / "human_reviews.json"
    return json.loads(f.read_text()) if f.exists() else {}


def _save(out: Path, reviews: dict[str, dict[str, Any]]) -> None:
    (out / "human_reviews.json").write_text(json.dumps(reviews, indent=2, ensure_ascii=False))


def review(out: Path, line: str, action: str, text: Optional[str] = None, note: str = "",
           reviewer: str = "reviewer") -> None:
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    reviews = load_reviews(out)
    if action == "reset":
        reviews.pop(line, None)
    else:
        if action == "edit" and not (text or "").strip():
            raise ValueError("an edit needs the corrected text")
        reviews[line] = {"action": action, "text": (text or "").strip() or None, "note": note, "reviewer": reviewer,
                         "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    _save(out, reviews)


def load_decisions(out: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in (out / "subtitle_decisions.jsonl").read_text().splitlines() if x.strip()]


def merged(out: Path) -> list[dict[str, Any]]:
    """Each line's final state: the reviewer's decision where there is one, else the agent's."""
    reviews = load_reviews(out)
    rows = []
    for d in sorted(load_decisions(out), key=lambda x: x["start_ms"]):
        h = reviews.get(d["subtitle_id"])
        action = h["action"] if h else None
        if action == "approve":
            final, text, by = "APPROVED", d.get("nadi_9_text"), "reviewer"
        elif action == "edit":
            final, text, by = "APPROVED", h["text"], "reviewer (edited)"
        elif action == "reject":
            final, text, by = "REJECTED", None, "reviewer"
        elif d["decision"] == "APPROVE":
            final, text, by = "APPROVED", d.get("nadi_9_text"), "agent"
        else:
            final, text, by = "OPEN", d.get("nadi_9_text"), "—"
        rows.append({**d, "final": final, "final_text": text, "decided_by": by, "review": h})
    return rows


def status(out: Path) -> dict[str, Any]:
    rows = merged(out)
    open_ = [r for r in rows if r["final"] == "OPEN"]
    crit = [r for r in open_ if r.get("priority") == "critical"]
    stale = [r for r in rows if (r.get("review") or {}).get("action") == "stale"]
    if crit:
        verdict = "DO NOT RELEASE"
    elif open_:
        verdict = "RELEASE AFTER REVIEW"
    else:
        verdict = "READY"
    return {"verdict": verdict, "total": len(rows), "open": len(open_), "critical_open": len(crit), "stale": len(stale),
            "approved": sum(r["final"] == "APPROVED" for r in rows), "rejected": sum(r["final"] == "REJECTED" for r in rows),
            "by_reviewer": sum(r["decided_by"].startswith("reviewer") for r in rows),
            "open_by_priority": {p: sum(r.get("priority") == p for r in open_) for p in PRIORITY}}


def final_srt(out: Path, approved_only: bool = True) -> str:
    blocks = []
    for r in merged(out):
        if r["final"] == "APPROVED":
            text = r["final_text"] or ""
        elif approved_only:
            continue
        elif r["final"] == "REJECTED":
            text = f"[REJECTED {r['subtitle_id']}]"
        else:
            text = f"[OPEN {r['subtitle_id']}] {r['final_text'] or ''}".strip()
        blocks.append(f"{len(blocks) + 1}\n{tc(r['start_ms'])} --> {tc(r['end_ms'])}\n{text}\n")
    return "\n".join(blocks)


def final_decisions_jsonl(out: Path) -> str:
    keep = ("subtitle_id", "source_text", "nadi_9_text", "confidence", "confidence_reason", "decision", "evidence",
            "assumptions", "conflicts", "review_question", "priority")
    return "".join(json.dumps({**{k: r.get(k) for k in keep}, "final": r["final"], "final_text": r["final_text"],
                               "decided_by": r["decided_by"], "human_review": r["review"]}, ensure_ascii=False) + "\n"
                   for r in merged(out))


def save_upload(files: list[tuple[str, bytes]], name: str) -> Path:
    """Store uploaded pack files (keeping sub-folders in their names, e.g. 'events/x.json') as a new pack."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name) or "upload"
    dest = ROOT / "packs" / "uploads" / safe
    if dest.exists():
        shutil.rmtree(dest)
    for fname, data in files:
        rel = Path(*[p for p in Path(fname).parts if p not in ("..", "/", "\\")])
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        (dest / rel).write_bytes(data)
    return dest
