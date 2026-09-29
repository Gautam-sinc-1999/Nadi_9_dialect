"""Find every file in a pack and read it into a format-neutral RawSource.

No knowledge of Nadi-9 or of any particular pack lives here.
"""
from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}
TEXT_EXT = {".md", ".txt", ".markdown", ".eml"}
SKIP_NAMES = {".DS_Store"}


@dataclass
class RawSource:
    source_id: str
    path: Path
    rel_path: str
    format: str
    data: Any = None                              # parsed JSON
    text: Optional[str] = None                    # full text for text-like formats
    file_meta: dict[str, Any] = field(default_factory=dict)
    records: list[dict[str, Any]] = field(default_factory=list)
    records_key: Optional[str] = None
    side_lists: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    sections: list[dict[str, str]] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def keys(self) -> set[str]:
        ks: set[str] = set()
        for r in self.records:
            ks.update(norm_key(k) for k in r)
        return ks


def norm_key(k: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


def discover(pack_dir: Path) -> list[RawSource]:
    out: list[RawSource] = []
    seen: set[str] = set()
    for p in sorted(pack_dir.rglob("*")):
        if not p.is_file() or p.name in SKIP_NAMES or p.name.startswith("."):
            continue
        rel = p.relative_to(pack_dir).as_posix()
        sid = p.stem
        if sid in seen:  # same name, different file: keep the extension so ids and refs never collide
            sid = re.sub(r"[^A-Za-z0-9_]", "_", rel)
        n = 2
        while sid in seen:
            sid, n = f"{sid}_{n}", n + 1
        seen.add(sid)
        out.append(read_source(p, rel, sid))
    return out


def read_source(p: Path, rel: str, sid: str) -> RawSource:
    ext = p.suffix.lower()
    src = RawSource(source_id=sid, path=p, rel_path=rel, format=ext.lstrip(".") or "none")
    try:
        if ext == ".json":
            src.data = json.loads(p.read_text(encoding="utf-8"))
            _load_json(src)
        elif ext in {".csv", ".tsv"}:
            _load_csv(src, p.read_text(encoding="utf-8-sig"))
        elif ext in {".srt", ".vtt"}:
            src.text = p.read_text(encoding="utf-8-sig")
            src.records = parse_srt(src.text)
            src.records_key = "cues"
        elif ext in TEXT_EXT:
            src.text = p.read_text(encoding="utf-8")
            src.sections = split_sections(src.text)
        elif ext == ".pdf":
            src.text = _read_pdf(p)
            src.sections = split_sections(src.text)
        elif ext == ".docx":
            src.text = _read_docx(p)
            src.sections = split_sections(src.text)
        elif ext in AUDIO_EXT:
            src.format = "audio"
        else:
            src.error = f"unsupported file type '{ext}'"
    except Exception as e:  # a broken file must not stop the whole pack
        src.error = f"could not read: {type(e).__name__}: {e}"
    return src


# ---------- JSON ----------

def _load_json(src: RawSource) -> None:
    d = src.data
    if isinstance(d, list):
        src.records = [r for r in d if isinstance(r, dict)]
        src.records_key = "$"
        return
    if not isinstance(d, dict):
        return
    src.file_meta = {k: v for k, v in d.items() if not isinstance(v, list)}
    lists = {k: v for k, v in d.items() if isinstance(v, list) and all(isinstance(x, dict) for x in v)}
    if not lists:
        return
    key = max(lists, key=lambda k: len(_flatten(lists[k])))
    src.records_key = key
    src.records = _flatten(lists[key])
    src.side_lists = {k: v for k, v in lists.items() if k != key}


def _flatten(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """[{name, notes:[{..},{..}]}] -> [{parent_name, ..}, ..]. One level is enough in practice."""
    out: list[dict[str, Any]] = []
    for it in items:
        child_key = next((k for k, v in it.items()
                          if isinstance(v, list) and v and all(isinstance(x, dict) for x in v)), None)
        if child_key is None:
            out.append(it)
            continue
        parent = {f"parent_{k}": v for k, v in it.items() if not isinstance(v, (list, dict))}
        for c in it[child_key]:
            out.append({**parent, **c})
    return out


# ---------- CSV ----------

def _load_csv(src: RawSource, text: str) -> None:
    try:
        dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    rows = list(csv.DictReader(io.StringIO(text), dialect=dialect))
    src.records = [{k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k} for r in rows]
    src.records_key = "rows"


# ---------- SRT / VTT ----------

_TC = r"(\d{1,2}:\d{2}:\d{2}[,.]\d{1,3})"
_SPEAKER = re.compile(r"^\s*(?:\[(?P<a>[^\]]+)\]|(?P<b>[A-Z][\w .'-]{0,30}?))\s*:\s*(?P<text>.+)$", re.S)


def parse_srt(text: str) -> list[dict[str, Any]]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = [l for l in block.strip().splitlines() if l.strip() and l.strip() != "WEBVTT"]
        m_i = next((i for i, l in enumerate(lines) if re.search(_TC + r"\s*-->\s*" + _TC, l)), None)
        if m_i is None:
            continue
        m = re.search(_TC + r"\s*-->\s*" + _TC, lines[m_i])
        idx = lines[m_i - 1].strip() if m_i > 0 else str(len(cues) + 1)
        body = " ".join(lines[m_i + 1:]).strip()
        cue: dict[str, Any] = {"id": idx, "start": m.group(1), "end": m.group(2), "text": body}
        sm = _SPEAKER.match(body)
        if sm:
            cue["speaker"] = (sm.group("a") or sm.group("b")).strip()
            cue["text"] = sm.group("text").strip()
        cues.append(cue)
    return cues


# ---------- free text ----------

_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_ID_IN_HEADING = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


def split_sections(text: str) -> list[dict[str, str]]:
    """Markdown headings become sections. Without headings, paragraphs do."""
    lines = text.splitlines()
    if any(_HEADING.match(l) for l in lines):
        sections: list[dict[str, str]] = []
        cur = {"section_id": "preamble", "heading": "", "text": ""}
        for l in lines:
            m = _HEADING.match(l)
            if m:
                if cur["text"].strip() or cur["heading"]:
                    sections.append(cur)
                heading = m.group(2).strip()
                mid = _ID_IN_HEADING.search(heading)
                sid = mid.group(1) if mid else _slug(heading)
                cur = {"section_id": sid, "heading": heading, "text": ""}
            else:
                cur["text"] += l + "\n"
        if cur["text"].strip() or cur["heading"]:
            sections.append(cur)
        return _dedupe_ids(sections)
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return [{"section_id": f"p{i + 1}", "heading": "", "text": p} for i, p in enumerate(paras)]


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:40] or "section"


def _dedupe_ids(sections: list[dict[str, str]]) -> list[dict[str, str]]:
    seen: dict[str, int] = {}
    for s in sections:
        n = seen.get(s["section_id"], 0)
        seen[s["section_id"]] = n + 1
        if n:
            s["section_id"] = f"{s['section_id']}-{n + 1}"
    return sections


def _read_pdf(p: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as e:
        raise RuntimeError("PDF support needs the pypdf package: pip install pypdf") from e
    return "\n\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)


def _read_docx(p: Path) -> str:
    try:
        import docx
    except ImportError as e:
        raise RuntimeError("DOCX support needs the python-docx package: pip install python-docx") from e
    return "\n".join(par.text for par in docx.Document(str(p)).paragraphs)


# ---------- time ----------

def to_ms(v: Any) -> Optional[int]:
    """'00:01:02,500' | '00:01:02.5' | '01:02' | 62.5 -> milliseconds."""
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return int(float(v) * 1000)
    s = str(v).strip().replace(",", ".")
    parts = s.split(":")
    try:
        nums = [float(x) for x in parts]
    except ValueError:
        return None
    secs = 0.0
    for n in nums:
        secs = secs * 60 + n
    return int(round(secs * 1000))
