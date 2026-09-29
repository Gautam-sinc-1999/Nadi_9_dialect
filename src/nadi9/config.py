"""Settings from the environment, with a minimal .env reader (no dependency).

Values already in the environment win over .env, so CI or a shell export can override the file.
"""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.is_file():
        return loaded
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip().removeprefix("export ").strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v
            loaded[k] = v
    return loaded


def setting(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name) or default
