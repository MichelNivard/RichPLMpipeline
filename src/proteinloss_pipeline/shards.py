from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Any

from .metadata import atomic_json, sha256_file


def relative_posix(path: str | Path, data_root: str | Path) -> str:
    absolute = Path(path).resolve()
    root = Path(data_root).resolve()
    try:
        relative = absolute.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact {absolute} is outside data root {root}") from exc
    return PurePosixPath(relative).as_posix()


def resolve_relative(path: str, data_root: str | Path) -> Path:
    if Path(path).is_absolute():
        raise ValueError(f"manifest path must be relative, got {path}")
    resolved = (Path(data_root).resolve() / Path(PurePosixPath(path))).resolve()
    try:
        resolved.relative_to(Path(data_root).resolve())
    except ValueError as exc:
        raise ValueError(f"manifest path escapes data root: {path}") from exc
    return resolved


@dataclass(frozen=True, slots=True)
class RecordRef:
    shard_path: str
    byte_offset: int
    byte_length: int
    record_sha256: str


class PackedShardWriter:
    """Atomic concatenated-record writer with JSONL offsets and shard inventory."""

    def __init__(self, data_root: str | Path, shard_dir: str | Path, records_per_shard: int = 10_000):
        self.data_root = Path(data_root).resolve()
        self.shard_dir = Path(shard_dir).resolve()
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        self.records_per_shard = max(1, int(records_per_shard))
        self._id = -1
        self._count = 0
        self._offset = 0
        self._handle: BinaryIO | None = None
        self._index_handle = None
        self._part_path: Path | None = None
        self._final_path: Path | None = None
        self._index_part: Path | None = None
        self._index_final: Path | None = None
        self.inventory: list[dict[str, Any]] = []

    def _open_next(self) -> None:
        self.close_shard()
        self._id += 1
        self._count = 0
        self._offset = 0
        self._final_path = self.shard_dir / f"targets-{self._id:06d}.bin"
        self._part_path = self._final_path.with_suffix(".bin.part")
        self._index_final = self.shard_dir / f"targets-{self._id:06d}.index.jsonl"
        self._index_part = self._index_final.with_suffix(".jsonl.part")
        if self._final_path.exists() or self._index_final.exists():
            raise FileExistsError(f"refusing to overwrite completed shard {self._final_path}")
        self._handle = self._part_path.open("wb")
        self._index_handle = self._index_part.open("w", encoding="utf-8")

    def write(self, example_id: str, payload: bytes, metadata: dict[str, Any] | None = None) -> RecordRef:
        if self._handle is None or self._count >= self.records_per_shard:
            self._open_next()
        assert self._handle is not None and self._index_handle is not None and self._final_path is not None
        digest = hashlib.sha256(payload).hexdigest()
        offset = self._offset
        self._handle.write(payload)
        self._handle.flush()
        ref = RecordRef(relative_posix(self._final_path, self.data_root), offset, len(payload), digest)
        row = {"example_id": example_id, **asdict(ref), **(metadata or {})}
        self._index_handle.write(json.dumps(row, sort_keys=True) + "\n")
        self._count += 1
        self._offset += len(payload)
        return ref

    def close_shard(self) -> None:
        if self._handle is None:
            return
        assert self._part_path and self._final_path and self._index_part and self._index_final and self._index_handle
        self._handle.flush()
        os.fsync(self._handle.fileno())
        self._handle.close()
        self._index_handle.flush()
        self._index_handle.close()
        self._part_path.replace(self._final_path)
        self._index_part.replace(self._index_final)
        self.inventory.append(
            {
                "shard_path": relative_posix(self._final_path, self.data_root),
                "index_path": relative_posix(self._index_final, self.data_root),
                "records": self._count,
                "bytes": self._final_path.stat().st_size,
                "sha256": sha256_file(self._final_path),
                "index_sha256": sha256_file(self._index_final),
            }
        )
        self._handle = self._index_handle = None

    def close(self) -> list[dict[str, Any]]:
        self.close_shard()
        atomic_json(self.shard_dir / "inventory.json", {"shards": self.inventory})
        return list(self.inventory)

    def __enter__(self) -> "PackedShardWriter":
        return self

    def __exit__(self, *_args) -> None:
        self.close()


def read_record(data_root: str | Path, ref: RecordRef, verify: bool = False) -> bytes:
    path = resolve_relative(ref.shard_path, data_root)
    with path.open("rb") as handle:
        handle.seek(ref.byte_offset)
        payload = handle.read(ref.byte_length)
    if len(payload) != ref.byte_length:
        raise IOError(f"short read from {path}: expected {ref.byte_length}, got {len(payload)}")
    if verify and hashlib.sha256(payload).hexdigest() != ref.record_sha256:
        raise IOError(f"record checksum mismatch in {path} at {ref.byte_offset}")
    return payload
