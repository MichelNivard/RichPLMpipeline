from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def source_revision(start: str | Path) -> dict[str, Any]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(start), check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=Path(start), check=True, capture_output=True, text=True).stdout)
        return {"git_revision": revision, "git_dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"git_revision": "unknown", "git_dirty": None}


def runtime_metadata(project_root: str | Path) -> dict[str, Any]:
    return {
        **source_revision(project_root),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pid": os.getpid(),
        "started_unix": time.time(),
    }


def atomic_json(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
