from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator
from urllib.request import Request, urlopen

import numpy as np
import torch
from torch.utils.data import Dataset

from .alphabet import AA, query_to_ids
from .codec import npz_bytes
from .metadata import atomic_json, sha256_file
from .shards import PackedShardWriter
from .splits import stable_split

PDB_SEQRES_URL = "https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
RCSB_MMCIF_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"
CASP15_TARGETLIST_URL = "https://predictioncenter.org/casp15/targetlist.cgi?type=csv"
PDB_ID_RE = re.compile(r"\b[0-9][A-Za-z0-9]{3}\b")

AA3_TO_1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "MSE": "M", "SEC": "C", "PYL": "K",
}


@dataclass(slots=True)
class CoordinateChain:
    pdb_id: str
    chain_id: str
    segment_id: int
    sequence: str
    coords_ca: np.ndarray
    method: str
    resolution: float

    @property
    def example_id(self) -> str:
        return f"{self.pdb_id}_{self.chain_id}_seg{self.segment_id}"


def download_url(url: str, path: str | Path, *, timeout: int = 180) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "ProteinLoss-portable/0.1"})
    with urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
    temporary.replace(destination)
    return destination


def _read_text(path: Path) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            return handle.read()
    return path.read_text(encoding="utf-8", errors="replace")


def _tokens(text: str) -> Iterator[str]:
    """Small mmCIF tokenizer sufficient for atom/method/resolution fields."""
    index, size, line_start = 0, len(text), True
    while index < size:
        value = text[index]
        if value in " \t\r\n":
            line_start = value == "\n"
            index += 1
            continue
        if value == "#":
            while index < size and text[index] != "\n":
                index += 1
            line_start = True
            continue
        if value == ";" and line_start:
            index += 1
            stop = text.find("\n;", index)
            if stop < 0:
                yield text[index:].rstrip("\n")
                return
            yield text[index:stop]
            index = stop + 2
            line_start = False
            continue
        if value in {"'", '"'}:
            quote, start = value, index + 1
            index = start
            while index < size and not (
                text[index] == quote and (index + 1 == size or text[index + 1] in " \t\r\n#")
            ):
                index += 1
            yield text[start:index]
            index += 1
            line_start = False
            continue
        start = index
        while index < size and text[index] not in " \t\r\n":
            index += 1
        yield text[start:index]
        line_start = False


def _clean(value: str | None) -> str:
    return "" if value is None or value in {".", "?"} else value


def _number(value: str | None, default: float = float("nan")) -> float:
    try:
        return float(_clean(value))
    except ValueError:
        return default


def parse_mmcif_chains(
    path: str | Path,
    *,
    pdb_id: str | None = None,
    min_len: int = 40,
    max_len: int = 512,
    methods: set[str] | None = None,
    max_resolution: float = 3.0,
    require_resolution: bool = True,
) -> list[CoordinateChain]:
    source = Path(path)
    pdb = (pdb_id or source.name.split(".", 1)[0]).lower()
    values = list(_tokens(_read_text(source)))
    atoms: list[dict[str, str]] = []
    method_values: list[str] = []
    resolutions: list[float] = []
    index = 0
    while index < len(values):
        token = values[index]
        if token == "loop_":
            index += 1
            tags: list[str] = []
            while index < len(values) and values[index].startswith("_"):
                tags.append(values[index])
                index += 1
            width = len(tags)
            if not width:
                continue
            category = tags[0].split(".", 1)[0]
            fields = [tag.split(".", 1)[-1] for tag in tags]
            while index + width <= len(values):
                if values[index] in {"loop_", "stop_"} or values[index].startswith(("_", "data_")):
                    break
                row = dict(zip(fields, values[index : index + width]))
                index += width
                if category == "_atom_site":
                    atoms.append(row)
                elif category == "_exptl" and row.get("method"):
                    method_values.append(row["method"])
                elif category == "_refine":
                    resolutions.append(_number(row.get("ls_d_res_high")))
                elif category == "_em_3d_reconstruction":
                    resolutions.append(_number(row.get("resolution")))
            continue
        if token == "_exptl.method" and index + 1 < len(values):
            method_values.append(values[index + 1])
        elif token in {"_refine.ls_d_res_high", "_em_3d_reconstruction.resolution"} and index + 1 < len(values):
            resolutions.append(_number(values[index + 1]))
        index += 2 if token.startswith("_") and index + 1 < len(values) else 1
    method = ";".join(dict.fromkeys(method_values)).upper()
    resolution_values = [value for value in resolutions if math.isfinite(value) and value > 0]
    resolution = min(resolution_values) if resolution_values else float("nan")
    if methods and not any(allowed.upper() in method for allowed in methods):
        return []
    if require_resolution and not math.isfinite(resolution):
        return []
    if math.isfinite(resolution) and resolution > max_resolution:
        return []

    residues: dict[str, dict[int, tuple[float, str, np.ndarray]]] = {}
    for row in atoms:
        if _clean(row.get("group_PDB")).upper() != "ATOM":
            continue
        if (_clean(row.get("label_atom_id")) or _clean(row.get("auth_atom_id"))).upper() != "CA":
            continue
        if _clean(row.get("pdbx_PDB_model_num")) not in {"", "1", "1.0"}:
            continue
        if _clean(row.get("label_alt_id")) not in {"", "A", "1"}:
            continue
        amino_acid = AA3_TO_1.get((_clean(row.get("label_comp_id")) or _clean(row.get("auth_comp_id"))).upper())
        try:
            residue = int(float(_clean(row.get("label_seq_id")) or _clean(row.get("auth_seq_id"))))
        except ValueError:
            continue
        chain = _clean(row.get("auth_asym_id")) or _clean(row.get("label_asym_id"))
        coordinate = np.asarray([_number(row.get(name)) for name in ("Cartn_x", "Cartn_y", "Cartn_z")], dtype=np.float32)
        if amino_acid is None or not chain or not np.isfinite(coordinate).all():
            continue
        occupancy = _number(row.get("occupancy"), 1.0)
        old = residues.setdefault(chain, {}).get(residue)
        if old is None or occupancy > old[0]:
            residues[chain][residue] = occupancy, amino_acid, coordinate

    chains: list[CoordinateChain] = []
    for chain_id, chain_residues in residues.items():
        segments: list[list[tuple[int, str, np.ndarray]]] = [[]]
        previous: int | None = None
        for residue, (_occupancy, amino_acid, coordinate) in sorted(chain_residues.items()):
            if previous is not None and residue > previous + 1:
                segments.append([])
            segments[-1].append((residue, amino_acid, coordinate))
            previous = residue
        for segment_id, segment in enumerate(segments):
            if not min_len <= len(segment) <= max_len:
                continue
            sequence = "".join(item[1] for item in segment)
            if set(sequence) - set(AA):
                continue
            chains.append(
                CoordinateChain(
                    pdb, chain_id, segment_id, sequence,
                    np.stack([item[2] for item in segment]).astype(np.float32), method, resolution,
                )
            )
    return chains


def _iter_seqres(path: Path, min_len: int, max_len: int) -> Iterator[str]:
    opener = gzip.open if path.suffix == ".gz" else path.open
    header, sequence = "", []
    with opener("rt", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith(">"):
                if header and sequence:
                    yield from _eligible_seqres(header, "".join(sequence), min_len, max_len)
                header, sequence = line[1:], []
            elif line:
                sequence.append(line)
    if header and sequence:
        yield from _eligible_seqres(header, "".join(sequence), min_len, max_len)


def _eligible_seqres(header: str, sequence: str, min_len: int, max_len: int) -> Iterator[str]:
    first = header.split()[0]
    if "_" not in first or "mol:protein" not in header:
        return
    pdb = first.split("_", 1)[0].lower()
    if min_len <= len(sequence) <= max_len and not (set(sequence.upper()) - set(AA)):
        yield pdb


def _casp_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8", errors="replace", newline="") as handle:
        rows = csv.DictReader(handle, delimiter=";")
        return {
            match.lower()
            for row in rows
            for match in PDB_ID_RE.findall(row.get("Description", ""))
        }


def _coordinate_schema(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE examples(
          example_id TEXT PRIMARY KEY, split TEXT NOT NULL, group_id TEXT NOT NULL,
          pdb_id TEXT NOT NULL, chain_id TEXT NOT NULL, sequence TEXT NOT NULL,
          length INTEGER NOT NULL, method TEXT NOT NULL, resolution REAL,
          shard_path TEXT NOT NULL, byte_offset INTEGER NOT NULL, byte_length INTEGER NOT NULL,
          record_sha256 TEXT NOT NULL
        ) WITHOUT ROWID;
        CREATE INDEX split_rows ON examples(split, example_id);
        """
    )
    return connection


def prepare_coordinate_dataset(
    *,
    output_dir: str | Path,
    cache_dir: str | Path,
    seqres_path: str | Path | None,
    casp15_table: str | Path | None,
    mode: str,
    download_missing: bool,
    min_len: int = 40,
    max_len: int = 512,
    max_resolution: float = 3.0,
    max_chains: int = 250_000,
    entry_limit: int = 0,
    records_per_shard: int = 1000,
    seed: int = 7,
) -> dict[str, Any]:
    """Build packed experimental coordinates from cached/downloaded RCSB mmCIF."""
    if mode not in {"pdb", "casp15"}:
        raise ValueError("mode must be pdb or casp15")
    output, cache = Path(output_dir).resolve(), Path(cache_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    completion = output / "COMPLETED"
    if completion.is_file():
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    seqres = Path(seqres_path).resolve() if seqres_path else cache / "pdb_seqres.txt.gz"
    casp_table = Path(casp15_table).resolve() if casp15_table else cache / "casp15_targetlist.csv"
    if download_missing:
        if mode == "pdb":
            download_url(PDB_SEQRES_URL, seqres)
        download_url(CASP15_TARGETLIST_URL, casp_table)
    if not casp_table.is_file():
        raise FileNotFoundError(f"CASP15 table required to keep the split separate: {casp_table}")
    casp_ids = _casp_ids(casp_table)
    if mode == "pdb":
        if not seqres.is_file():
            raise FileNotFoundError(seqres)
        ids = sorted(set(_iter_seqres(seqres, min_len, max_len)) - casp_ids)
    else:
        ids = sorted(casp_ids)
    if entry_limit:
        ids = ids[:entry_limit]
    manifest_part = output / "manifest.sqlite.part"
    manifest_part.unlink(missing_ok=True)
    connection = _coordinate_schema(manifest_part)
    attempted = written = failed = 0
    split_counts: dict[str, int] = {}
    failures = output / "failure_rows.jsonl"
    started = time.perf_counter()
    with failures.open("w", encoding="utf-8") as failure_handle, PackedShardWriter(
        output, output / "shards", records_per_shard=records_per_shard
    ) as writer:
        for pdb in ids:
            if max_chains and written >= max_chains:
                break
            attempted += 1
            cif = cache / "mmcif" / pdb[1:3] / f"{pdb}.cif.gz"
            try:
                if download_missing:
                    download_url(RCSB_MMCIF_URL.format(pdb_id=pdb), cif, timeout=60)
                if not cif.is_file():
                    raise FileNotFoundError(cif)
                chains = parse_mmcif_chains(
                    cif,
                    pdb_id=pdb,
                    min_len=min_len,
                    max_len=max_len,
                    methods={"X-RAY DIFFRACTION"} if mode == "pdb" else None,
                    max_resolution=max_resolution,
                    require_resolution=mode == "pdb",
                )
                if not chains:
                    raise ValueError("no eligible observed CA chain")
                for chain in chains:
                    if max_chains and written >= max_chains:
                        break
                    split = "casp15" if mode == "casp15" else stable_split(
                        chain.sequence, group_id=hashlib.sha256(chain.sequence.encode()).hexdigest(),
                        seed=seed, valid_fraction=0.05, test_fraction=0.05,
                    )
                    max_length = 512
                    arrays = {
                        "input_ids": np.pad(np.asarray(query_to_ids(chain.sequence), dtype=np.int16), (0, max_length - len(chain.sequence)), constant_values=20),
                        "attention_mask": np.arange(max_length) < len(chain.sequence),
                        "coords_ca": chain.coords_ca.astype(np.float32),
                    }
                    ref = writer.write(chain.example_id, npz_bytes(arrays), {"pdb_id": pdb, "split": split})
                    connection.execute(
                        "INSERT INTO examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (chain.example_id, split, hashlib.sha256(chain.sequence.encode()).hexdigest(), pdb,
                         chain.chain_id, chain.sequence, len(chain.sequence), chain.method,
                         chain.resolution if math.isfinite(chain.resolution) else None, ref.shard_path,
                         ref.byte_offset, ref.byte_length, ref.record_sha256),
                    )
                    written += 1
                    split_counts[split] = split_counts.get(split, 0) + 1
                    if written % 1000 == 0:
                        connection.commit()
            except Exception as error:
                failed += 1
                failure_handle.write(json.dumps({"pdb_id": pdb, "error": repr(error)}, sort_keys=True) + "\n")
    connection.commit()
    connection.close()
    manifest_part.replace(output / "manifest.sqlite")
    if not written:
        raise RuntimeError("coordinate build wrote no chains")
    if mode == "pdb" and not {"train", "valid", "test"}.issubset(split_counts):
        raise RuntimeError(f"coordinate build requires non-empty train/valid/test, got {split_counts}")
    summary = {
        "status": "complete", "mode": mode, "attempted_entries": attempted,
        "written_chains": written, "failed_entries": failed, "split_counts": split_counts,
        "seconds": time.perf_counter() - started,
        "manifest_sha256": sha256_file(output / "manifest.sqlite"),
        "source_versions": {"seqres_sha256": sha256_file(seqres) if seqres.is_file() else "", "casp15_sha256": sha256_file(casp_table)},
    }
    atomic_json(output / "summary.json", summary)
    completion.write_text("ok\n", encoding="utf-8")
    return summary


class CoordinateDataset(Dataset):
    def __init__(self, root: str | Path, split: str):
        self.root = Path(root)
        self.split = split
        connection = sqlite3.connect(f"file:{(self.root / 'manifest.sqlite').resolve()}?mode=ro", uri=True)
        self.rows = list(connection.execute(
            "SELECT example_id,shard_path,byte_offset,byte_length,record_sha256 FROM examples WHERE split=? ORDER BY example_id",
            (split,),
        ))
        connection.close()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example_id, shard, offset, length, checksum = self.rows[index]
        with (self.root / shard).open("rb") as handle:
            handle.seek(int(offset))
            payload = handle.read(int(length))
        if hashlib.sha256(payload).hexdigest() != checksum:
            raise ValueError(f"coordinate record checksum mismatch: {example_id}")
        from io import BytesIO
        with np.load(BytesIO(payload), allow_pickle=False) as arrays:
            input_ids = arrays["input_ids"].astype(np.int64)
            mask = arrays["attention_mask"].astype(np.bool_)
            coordinates = np.zeros((len(input_ids), 3), dtype=np.float32)
            raw = arrays["coords_ca"].astype(np.float32)
            coordinates[: len(raw)] = raw
        size = len(input_ids)
        return {
            "example_id": example_id,
            "input_ids": torch.from_numpy(input_ids),
            "attention_mask": torch.from_numpy(mask),
            "coords_ca": torch.from_numpy(coordinates),
            "pssm_log_probs": torch.zeros((size, 20)),
            "dense_mi_target": torch.zeros((size, size)),
            "distance_target": torch.zeros((size, size)),
            "three_di_ids": torch.full((size,), 20),
            "has_pssm": torch.tensor(False), "has_mi": torch.tensor(False),
            "has_structure": torch.tensor(True), "has_3di": torch.tensor(False),
        }


def write_coordinate_fixture(output_dir: str | Path, records: list[dict[str, Any]], max_len: int) -> dict[str, Any]:
    """Write deterministic test-only coordinates in the production benchmark format."""
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = output / "manifest.sqlite"
    manifest.unlink(missing_ok=True)
    connection = _coordinate_schema(manifest)
    counts: dict[str, int] = {}
    with PackedShardWriter(output, output / "shards", records_per_shard=4) as writer:
        for record in records:
            sequence, coordinates = str(record["sequence"]), np.asarray(record["coords_ca"], dtype=np.float32)
            arrays = {
                "input_ids": np.pad(np.asarray(query_to_ids(sequence), dtype=np.int16), (0, max_len - len(sequence)), constant_values=20),
                "attention_mask": np.arange(max_len) < len(sequence),
                "coords_ca": coordinates,
            }
            ref = writer.write(str(record["example_id"]), npz_bytes(arrays), {"source": "experimental_pdb_fixture"})
            split = str(record["split"])
            connection.execute(
                "INSERT INTO examples VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (record["example_id"], split, record.get("group_id", record["example_id"]),
                 record.get("pdb_id", str(record["example_id"])[:4]), record.get("chain_id", "A"),
                 sequence, len(sequence), "X-RAY DIFFRACTION (SMOKE FIXTURE)", 2.0,
                 ref.shard_path, ref.byte_offset, ref.byte_length, ref.record_sha256),
            )
            counts[split] = counts.get(split, 0) + 1
    connection.commit()
    connection.close()
    if not {"train", "valid", "test"}.issubset(counts):
        raise ValueError(f"coordinate fixture missing split: {counts}")
    summary = {"status": "complete", "source": "deterministic_experimental_coordinate_fixture", "split_counts": counts}
    atomic_json(output / "summary.json", summary)
    (output / "COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary
