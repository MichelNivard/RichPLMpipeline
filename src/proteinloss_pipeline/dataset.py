from __future__ import annotations

import os
import sqlite3
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .codec import arrays_from_npz_bytes, decode_mi_target, distance_log_target, target_from_arrays
from .manifest import ManifestRow, _row_from_sql
from .shards import read_record


class PackedProteinDataset(Dataset[dict[str, Any]]):
    def __init__(self, data_root: str | Path, manifest: str | Path, split: str, *, verify_records: bool = False):
        self.data_root = Path(data_root).resolve()
        self.manifest = Path(manifest).resolve()
        self.split = split
        self.verify_records = verify_records
        connection = sqlite3.connect(f"file:{self.manifest}?mode=ro", uri=True)
        row = connection.execute("SELECT count(*) FROM examples WHERE split=?", (split,)).fetchone()
        self.length = int(row[0])
        connection.close()
        self._connection: sqlite3.Connection | None = None
        self._pid = -1

    def __len__(self) -> int:
        return self.length

    def _connect(self) -> sqlite3.Connection:
        if self._connection is None or self._pid != os.getpid():
            if self._connection is not None:
                self._connection.close()
            self._connection = sqlite3.connect(f"file:{self.manifest}?mode=ro", uri=True)
            self._connection.row_factory = sqlite3.Row
            self._pid = os.getpid()
        return self._connection

    def _row(self, rank: int) -> ManifestRow:
        row = self._connect().execute(
            "SELECT * FROM examples WHERE split=? AND split_rank=?", (self.split, int(rank))
        ).fetchone()
        if row is None:
            raise IndexError(rank)
        return _row_from_sql(row)

    def __getitem__(self, rank: int) -> dict[str, Any]:
        row = self._row(rank)
        arrays = arrays_from_npz_bytes(read_record(self.data_root, row.ref, verify=self.verify_records))
        max_len = int(arrays["input_ids"].shape[0])
        valid_length = int(arrays.get("valid_length", row.length))
        dense_mi = np.zeros((max_len, max_len), dtype=np.float32)
        if row.has_mi:
            decoded = decode_mi_target(target_from_arrays(arrays))
            dense_mi[:valid_length, :valid_length] = decoded[:valid_length, :valid_length]
        distance = np.zeros((max_len, max_len), dtype=np.float32)
        coordinates = np.zeros((max_len, 3), dtype=np.float32)
        if row.has_structure:
            decoded_distance = distance_log_target(arrays["ca_coords_fp16"])
            distance[:valid_length, :valid_length] = decoded_distance[:valid_length, :valid_length]
            coordinates[:valid_length] = arrays["ca_coords_fp16"][:valid_length].astype(np.float32)
        three_di = arrays.get("three_di_ids", np.full(max_len, 20, dtype=np.uint8))
        return {
            "example_id": row.example_id,
            "input_ids": torch.from_numpy(arrays["input_ids"].astype(np.int64)),
            "attention_mask": torch.from_numpy(arrays["attention_mask"].astype(np.bool_)),
            "pssm_log_probs": torch.from_numpy(arrays["pssm_log_probs"].astype(np.float32)),
            "dense_mi_target": torch.from_numpy(dense_mi),
            "distance_target": torch.from_numpy(distance),
            "coords_ca": torch.from_numpy(coordinates),
            "three_di_ids": torch.from_numpy(three_di.astype(np.int64)),
            "has_pssm": torch.tensor(row.has_pssm),
            "has_mi": torch.tensor(row.has_mi),
            "has_structure": torch.tensor(row.has_structure),
            "has_3di": torch.tensor(row.has_3di),
        }


def collate_records(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [key for key in batch[0] if key != "example_id"]
    return {"example_id": [row["example_id"] for row in batch], **{key: torch.stack([row[key] for row in batch]) for key in keys}}
