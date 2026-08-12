from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path
from typing import Any, Iterator

from .config import read_yaml, resolve_path
from .metadata import sha256_file


def load_registry(path: str | Path) -> dict[str, Any]:
    registry = read_yaml(path)
    if int(registry.get("version", 0)) != 1:
        raise ValueError("unsupported source registry version")
    return registry


def _unexpanded(value: str) -> bool:
    return "${" in value


def source_path(entry: dict[str, Any], registry_path: str | Path) -> Path:
    value = str(entry.get("path", ""))
    return resolve_path(value, base=Path(registry_path).resolve().parent)


def check_entry(name: str, entry: dict[str, Any], registry_path: str | Path) -> list[str]:
    raw = str(entry.get("path", ""))
    if not raw or _unexpanded(os.path.expandvars(raw)):
        return [f"{name}: path is unset ({raw or 'empty'})"]
    path = source_path(entry, registry_path)
    kind = entry.get("kind", "file")
    missing: list[str] = []
    if kind == "file" and not path.is_file():
        missing.append(f"{name}: missing file {path}")
    elif kind == "directory" and not path.is_dir():
        missing.append(f"{name}: missing directory {path}")
    elif kind == "sqlite" and not path.is_file():
        missing.append(f"{name}: missing SQLite index {path}")
    elif kind == "mmseqs_db" and not Path(str(path) + ".dbtype").is_file():
        missing.append(f"{name}: missing MMseqs DB marker {path}.dbtype")
    elif kind == "foldseek_prefix":
        for suffix in (".lookup", ".index", "_ca", "_ca.index", "_ss", "_ss.index"):
            candidate = Path(str(path) + suffix)
            if not candidate.is_file():
                missing.append(f"{name}: missing Foldseek component {candidate}")
    expected_sha256 = str(entry.get("sha256", "")).lower()
    if expected_sha256 and path.is_file():
        observed = sha256_file(path)
        if observed.lower() != expected_sha256:
            missing.append(f"{name}: SHA-256 mismatch for {path}; expected {expected_sha256}, got {observed}")
    return missing


def preflight(
    registry_path: str | Path,
    *,
    capabilities: set[str],
    include_optional: bool = False,
) -> dict[str, Any]:
    registry = load_registry(registry_path)
    missing: list[str] = []
    checked: list[dict[str, Any]] = []
    for section in ("sources", "tools"):
        for name, entry in registry.get(section, {}).items():
            required_for = set(entry.get("required_for", []))
            if not include_optional and not (required_for & capabilities):
                continue
            problems = check_entry(name, entry, registry_path)
            missing.extend(problems)
            item = {"name": name, "kind": entry.get("kind", "tool" if section == "tools" else "file"), "problems": problems}
            raw = str(entry.get("path", ""))
            if raw and not _unexpanded(os.path.expandvars(raw)):
                path = source_path(entry, registry_path)
                item["path"] = str(path)
                if path.is_file() and path.stat().st_size < 2 * 1024**3:
                    item["sha256"] = sha256_file(path)
                if section == "tools" and path.is_file() and not problems:
                    try:
                        command = [str(path), *entry.get("version_command", ["version"])]
                        item["version_output"] = subprocess.run(command, capture_output=True, text=True, timeout=30).stdout.strip()[:1000]
                    except (OSError, subprocess.SubprocessError) as error:
                        item["version_error"] = repr(error)
            checked.append(item)
    return {"ok": not missing, "capabilities": sorted(capabilities), "checked": checked, "missing": missing}


def _connect_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    return connection


def build_uniclust_membership_index(mapping_path: str | Path, output: str | Path) -> dict[str, Any]:
    """Stream a representative/member TSV into forward and inverse SQLite indexes."""
    source, destination = Path(mapping_path), Path(output)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    connection = _connect_index(temporary)
    connection.executescript(
        """
        CREATE TABLE members(rep TEXT PRIMARY KEY, members TEXT NOT NULL, n INTEGER NOT NULL) WITHOUT ROWID;
        CREATE TABLE accession_group(accession TEXT PRIMARY KEY, rep TEXT NOT NULL) WITHOUT ROWID;
        CREATE INDEX group_rep ON accession_group(rep);
        CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
        """
    )
    import gzip

    opener = gzip.open if source.suffix == ".gz" else source.open
    current_rep: str | None = None
    members: list[str] = []
    rows = reps = 0

    def flush() -> None:
        nonlocal reps, members, current_rep
        if current_rep is None:
            return
        connection.execute("INSERT INTO members VALUES (?,?,?)", (current_rep, "\t".join(members), len(members)))
        reps += 1

    with opener("rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            rep, member = parts[0].split()[0], parts[1].split()[0]
            if current_rep is None:
                current_rep = rep
            elif rep != current_rep:
                flush()
                current_rep, members = rep, []
            members.append(member)
            connection.execute("INSERT OR IGNORE INTO accession_group VALUES (?,?)", (member, rep))
            rows += 1
            if rows % 100_000 == 0:
                connection.commit()
    flush()
    connection.execute("INSERT INTO metadata VALUES (?,?)", ("source_sha256", sha256_file(source)))
    connection.execute("INSERT INTO metadata VALUES (?,?)", ("rows", str(rows)))
    connection.commit()
    connection.close()
    temporary.replace(destination)
    return {"source": str(source), "output": str(destination), "rows": rows, "representatives": reps, "sha256": sha256_file(destination)}


def build_lookup_index(lookup_path: str | Path, output: str | Path, *, foldseek: bool = False) -> dict[str, Any]:
    source, destination = Path(lookup_path), Path(output)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    connection = _connect_index(temporary)
    connection.executescript(
        "CREATE TABLE lookup(accession TEXT PRIMARY KEY, numeric_key INTEGER NOT NULL, source_id TEXT NOT NULL) WITHOUT ROWID;"
        "CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;"
    )
    rows = 0
    with source.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) < 2:
                parts = raw.split()
            if len(parts) < 2:
                continue
            key, source_id = int(parts[0]), parts[1]
            accession = source_id
            if foldseek and source_id.startswith("AF-"):
                accession = source_id.split("-", 2)[1]
            elif "|" in source_id:
                fields = source_id.split("|")
                accession = fields[1] if len(fields) > 1 else fields[0]
            connection.execute("INSERT OR IGNORE INTO lookup VALUES (?,?,?)", (accession, key, source_id))
            rows += 1
            if rows % 100_000 == 0:
                connection.commit()
    connection.execute("INSERT INTO metadata VALUES (?,?)", ("source_sha256", sha256_file(source)))
    connection.execute("INSERT INTO metadata VALUES (?,?)", ("rows", str(rows)))
    connection.commit()
    connection.close()
    temporary.replace(destination)
    return {"source": str(source), "output": str(destination), "rows": rows, "sha256": sha256_file(destination)}
