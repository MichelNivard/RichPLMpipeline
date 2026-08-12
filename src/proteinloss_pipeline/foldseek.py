from __future__ import annotations

import struct
from pathlib import Path

import numpy as np


def read_index(path: str | Path, keys: set[int]) -> dict[int, tuple[int, int]]:
    remaining = set(keys)
    found: dict[int, tuple[int, int]] = {}
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) < 3:
                continue
            key = int(parts[0])
            if key in remaining:
                found[key] = (int(parts[1]), int(parts[2]))
                remaining.remove(key)
                if not remaining:
                    break
    return found


def sequence_length(index_nbytes: int) -> int:
    return max(int(index_nbytes), 2) - 2


def decode_ca_record(path: str | Path, offset: int, nbytes: int, length: int) -> np.ndarray:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(nbytes)
    if len(raw) >= length * 3 * 4:
        values = np.frombuffer(raw[: length * 3 * 4], dtype="<f4").copy()
        return np.stack((values[:length], values[length : 2 * length], values[2 * length :]), axis=1)
    expected = 12 + max(length - 1, 0) * 6
    if len(raw) < expected:
        raise ValueError(f"Foldseek CA record has {len(raw)} bytes, expected at least {expected}")
    coordinates = np.empty((length, 3), dtype=np.float32)
    cursor = 0
    for dimension in range(3):
        current = struct.unpack_from("<i", raw, cursor)[0]
        cursor += 4
        coordinates[0, dimension] = current / 1000.0
        for residue in range(1, length):
            current += struct.unpack_from("<h", raw, cursor)[0]
            cursor += 2
            coordinates[residue, dimension] = current / 1000.0
    return coordinates


def read_text_record(path: str | Path, offset: int, nbytes: int) -> str:
    with Path(path).open("rb") as handle:
        handle.seek(offset)
        raw = handle.read(nbytes)
    return raw.decode("utf-8", errors="ignore").replace("\x00", "").strip()
