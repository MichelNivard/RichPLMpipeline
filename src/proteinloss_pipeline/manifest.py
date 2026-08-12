from __future__ import annotations

import json
import sqlite3
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

from .shards import RecordRef
from .splits import REQUIRED_SPLITS, validate_split_counts


SCHEMA = """
CREATE TABLE IF NOT EXISTS examples (
  example_index INTEGER PRIMARY KEY,
  example_id TEXT NOT NULL UNIQUE,
  group_id TEXT NOT NULL,
  split TEXT NOT NULL CHECK(split IN ('train','valid','test')),
  split_rank INTEGER NOT NULL,
  shard_path TEXT NOT NULL,
  byte_offset INTEGER NOT NULL,
  byte_length INTEGER NOT NULL,
  record_sha256 TEXT NOT NULL,
  length INTEGER NOT NULL,
  has_pssm INTEGER NOT NULL,
  has_mi INTEGER NOT NULL,
  has_structure INTEGER NOT NULL,
  has_3di INTEGER NOT NULL,
  provenance TEXT NOT NULL,
  UNIQUE(split, split_rank)
);
CREATE INDEX IF NOT EXISTS examples_split ON examples(split, split_rank);
CREATE INDEX IF NOT EXISTS examples_group ON examples(group_id);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
"""


@dataclass(frozen=True, slots=True)
class ManifestRow:
    example_id: str
    group_id: str
    split: str
    shard_path: str
    byte_offset: int
    byte_length: int
    record_sha256: str
    length: int
    has_pssm: bool
    has_mi: bool
    has_structure: bool
    has_3di: bool
    provenance: dict[str, Any]

    @property
    def ref(self) -> RecordRef:
        return RecordRef(self.shard_path, self.byte_offset, self.byte_length, self.record_sha256)


class ManifestWriter:
    def __init__(self, path: str | Path, overwrite: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and not overwrite:
            raise FileExistsError(f"manifest exists: {self.path}")
        temporary = self.path.with_suffix(self.path.suffix + ".part")
        temporary.unlink(missing_ok=True)
        self.temporary = temporary
        self.connection = sqlite3.connect(temporary)
        self.connection.executescript(SCHEMA)
        self.count = 0
        self.split_ranks: Counter[str] = Counter()

    def add(self, row: ManifestRow) -> None:
        if Path(row.shard_path).is_absolute():
            raise ValueError(f"manifest shard path must be relative: {row.shard_path}")
        rank = self.split_ranks[row.split]
        self.connection.execute(
            "INSERT INTO examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.count,
                row.example_id,
                row.group_id,
                row.split,
                rank,
                row.shard_path,
                row.byte_offset,
                row.byte_length,
                row.record_sha256,
                row.length,
                int(row.has_pssm),
                int(row.has_mi),
                int(row.has_structure),
                int(row.has_3di),
                json.dumps(row.provenance, sort_keys=True),
            ),
        )
        self.count += 1
        self.split_ranks[row.split] += 1
        if self.count % 10_000 == 0:
            self.connection.commit()

    def set_metadata(self, key: str, value: Any) -> None:
        self.connection.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)", (key, json.dumps(value, sort_keys=True)))

    def close(self, require_splits: bool = True) -> dict[str, int]:
        counts = {split: int(self.split_ranks.get(split, 0)) for split in REQUIRED_SPLITS}
        if require_splits:
            validate_split_counts(counts)
        self.set_metadata("rows", self.count)
        self.set_metadata("split_counts", counts)
        self.connection.commit()
        result = self.connection.execute("PRAGMA integrity_check").fetchone()[0]
        if result != "ok":
            raise IOError(f"manifest integrity check failed: {result}")
        self.connection.close()
        self.temporary.replace(self.path)
        return counts


def _row_from_sql(row: sqlite3.Row) -> ManifestRow:
    return ManifestRow(
        example_id=row["example_id"],
        group_id=row["group_id"],
        split=row["split"],
        shard_path=row["shard_path"],
        byte_offset=row["byte_offset"],
        byte_length=row["byte_length"],
        record_sha256=row["record_sha256"],
        length=row["length"],
        has_pssm=bool(row["has_pssm"]),
        has_mi=bool(row["has_mi"]),
        has_structure=bool(row["has_structure"]),
        has_3di=bool(row["has_3di"]),
        provenance=json.loads(row["provenance"]),
    )


class ManifestReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.connection = sqlite3.connect(f"file:{self.path.resolve()}?mode=ro", uri=True)
        self.connection.row_factory = sqlite3.Row

    def counts(self) -> dict[str, int]:
        counts = {row[0]: int(row[1]) for row in self.connection.execute("SELECT split, count(*) FROM examples GROUP BY split")}
        return {split: counts.get(split, 0) for split in REQUIRED_SPLITS}

    def validate(self) -> dict[str, int]:
        counts = self.counts()
        validate_split_counts(counts)
        absolute = self.connection.execute("SELECT shard_path FROM examples WHERE substr(shard_path,1,1)='/' LIMIT 1").fetchone()
        if absolute:
            raise ValueError(f"manifest contains absolute path: {absolute[0]}")
        return counts

    def get(self, split: str, split_rank: int) -> ManifestRow:
        row = self.connection.execute(
            "SELECT * FROM examples WHERE split=? AND split_rank=?", (split, int(split_rank))
        ).fetchone()
        if row is None:
            raise IndexError((split, split_rank))
        return _row_from_sql(row)

    def iter_rows(self, split: str | None = None) -> Iterator[ManifestRow]:
        query, args = ("SELECT * FROM examples ORDER BY example_index", ()) if split is None else (
            "SELECT * FROM examples WHERE split=? ORDER BY split_rank", (split,)
        )
        for row in self.connection.execute(query, args):
            yield _row_from_sql(row)

    def metadata(self) -> dict[str, Any]:
        return {key: json.loads(value) for key, value in self.connection.execute("SELECT key,value FROM metadata")}

    def close(self) -> None:
        self.connection.close()


def export_jsonl(manifest_path: str | Path, output: str | Path) -> Path:
    reader = ManifestReader(manifest_path)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for row in reader.iter_rows():
            payload = asdict(row)
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
    reader.close()
    return destination
