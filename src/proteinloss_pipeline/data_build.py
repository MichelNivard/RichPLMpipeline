from __future__ import annotations

import csv
import json
import math
import random
import shutil
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .alphabet import AA, normalize_msa_rows, query_to_ids, read_a3m
from .codec import encode_ca_coords, encode_mi_target, npz_bytes, target_to_arrays
from .config import write_snapshot
from .coordinates import write_coordinate_fixture
from .manifest import ManifestReader, ManifestRow, ManifestWriter, export_jsonl
from .metadata import atomic_json, runtime_metadata, sha256_file
from .shards import PackedShardWriter
from .splits import stable_split
from .targets import msa_to_mi_apc, msa_to_pssm, pad_core, validate_pair_target
from .teacher import create_smoke_teacher


def _sequence(index: int, length: int, rng: random.Random) -> str:
    # Deterministic but non-periodic enough for target and mutation smokes.
    return "".join(AA[(rng.randrange(20) + index + position * 7) % 20] for position in range(length))


def _msa(sequence: str, depth: int, rng: random.Random) -> list[str]:
    rows = [sequence]
    for row_index in range(1, depth):
        values = list(sequence)
        for position in range(len(values)):
            draw = rng.random()
            if draw < 0.08:
                values[position] = AA[(AA.index(values[position]) + row_index + position) % 20]
            elif draw < 0.10:
                values[position] = "-"
        rows.append("".join(values))
    return rows


def _coordinates(length: int, index: int) -> np.ndarray:
    residue = np.arange(length, dtype=np.float32)
    angle = residue * (2.0 * np.pi / 10.0) + index * 0.07
    # A compact multi-turn CA trace: adjacent pseudo-bonds are about 3.8 A and
    # matching phases on successive turns create genuine nonlocal contacts.
    return np.stack((6.1 * np.cos(angle), 6.1 * np.sin(angle), np.floor(residue / 10.0) * 2.5), axis=1).astype(np.float32)


def _three_di(length: int, index: int, max_len: int) -> np.ndarray:
    values = np.full(max_len, 20, dtype=np.uint8)
    values[:length] = (np.arange(length, dtype=np.uint16) * 7 + index * 3) % 20
    return values


def _write_a3m(path: Path, example_id: str, msa: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, sequence in enumerate(msa):
            handle.write(f">{example_id}_{index}\n{sequence}\n")


def _source_rows(config: dict[str, Any]) -> list[dict[str, Any]]:
    data = config["data"]
    count, seed = int(data["count"]), int(config["seed"])
    minimum, maximum = int(data["min_len"]), int(data["max_len"])
    valid_fraction, test_fraction = float(data["valid_fraction"]), float(data["test_fraction"])
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    # Try candidate accessions until all required splits exist. No records are
    # duplicated and stable accession/group hashing remains the sole splitter.
    candidate = 0
    while len(rows) < count or {row["split"] for row in rows} != {"train", "valid", "test"}:
        if candidate > max(10_000, count * 100):
            raise RuntimeError("could not construct non-empty stable smoke splits")
        example_id = f"SMOKE{candidate:08d}"
        group_id = f"family-{candidate // 2:08d}"
        split = stable_split(
            example_id,
            group_id=group_id,
            seed=seed,
            valid_fraction=valid_fraction,
            test_fraction=test_fraction,
        )
        if len(rows) < count:
            length = rng.randint(minimum, maximum)
            rows.append(
                {
                    "example_id": example_id,
                    "group_id": group_id,
                    "split": split,
                    "sequence": _sequence(candidate, length, rng),
                    "length": length,
                    "source_index": candidate,
                }
            )
        elif split not in {row["split"] for row in rows}:
            replace_index = next(i for i, row in enumerate(rows) if Counter(r["split"] for r in rows)[row["split"]] > 1)
            length = rng.randint(minimum, maximum)
            rows[replace_index] = {
                "example_id": example_id,
                "group_id": group_id,
                "split": split,
                "sequence": _sequence(candidate, length, rng),
                "length": length,
                "source_index": candidate,
            }
        candidate += 1
    return rows


def _write_sources(rows: list[dict[str, Any]], root: Path) -> dict[str, Any]:
    source_dir = root / "source"
    source_dir.mkdir(parents=True, exist_ok=True)
    fasta = source_dir / "selected.fasta"
    jsonl = source_dir / "selected.jsonl"
    with fasta.open("w", encoding="utf-8") as fasta_handle, jsonl.open("w", encoding="utf-8") as row_handle:
        for row in rows:
            fasta_handle.write(f">{row['example_id']} group={row['group_id']} split={row['split']}\n{row['sequence']}\n")
            row_handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "attempted": len(rows),
        "written": len(rows),
        "skipped": 0,
        "split_counts": dict(Counter(row["split"] for row in rows)),
        "fasta": "source/selected.fasta",
        "fasta_sha256": sha256_file(fasta),
        "jsonl": "source/selected.jsonl",
        "jsonl_sha256": sha256_file(jsonl),
    }
    atomic_json(source_dir / "summary.json", summary)
    (source_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary


def build_smoke_dataset(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(config["paths"]["data_root"]).resolve()
    scratch_root = Path(config["paths"]["scratch_root"]).resolve()
    completion = data_root / "COMPLETED"
    summary_path = data_root / "build_summary.json"
    if completion.is_file() and summary_path.is_file():
        reader = ManifestReader(data_root / "manifest.sqlite")
        reader.validate()
        reader.close()
        return json.loads(summary_path.read_text(encoding="utf-8"))

    data_root.mkdir(parents=True, exist_ok=True)
    write_snapshot(config, data_root / "metadata")
    atomic_json(data_root / "metadata" / "runtime.json", runtime_metadata(project_root))
    atomic_json(
        data_root / "metadata" / "source_versions.json",
        {"backend": "deterministic_smoke_fixture", "version": 1, "external_downloads": False},
    )
    rows = _source_rows(config)
    source_summary = _write_sources(rows, data_root)
    teacher_path = create_smoke_teacher(data_root / "metadata" / "smoke_structencoder_teacher.pt", int(config["seed"]))
    validation_dir = data_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    casp_records: dict[str, np.ndarray] = {}
    for casp_index in range(2):
        length = min(int(config["data"]["max_len"]), int(config["data"]["min_len"]) + casp_index + 2)
        sequence = _sequence(90_000 + casp_index, length, random.Random(int(config["seed"]) + 15 + casp_index))
        core = pad_core(
            np.asarray(query_to_ids(sequence), dtype=np.int16),
            np.full((length, 20), -math.log(20.0), dtype=np.float32),
            np.zeros(length, dtype=np.float32),
            int(config["data"]["max_len"]),
        )
        casp_records[f"input_ids_{casp_index}"] = core["input_ids"]
        casp_records[f"attention_mask_{casp_index}"] = core["attention_mask"]
        casp_records[f"coords_ca_{casp_index}"] = _coordinates(length, 90_000 + casp_index)
        casp_records[f"example_id_{casp_index}"] = np.asarray(f"CASP15_SMOKE_{casp_index}")
    with (validation_dir / "casp15_smoke.npz").open("wb") as handle:
        np.savez_compressed(handle, **casp_records)
    coordinate_records = [
        {
            "example_id": f"PDB_SMOKE_{index:04d}",
            "pdb_id": f"s{index:03d}",
            "group_id": row["group_id"],
            "split": row["split"],
            "sequence": row["sequence"],
            "coords_ca": _coordinates(int(row["length"]), int(row["source_index"]) + 50_000),
        }
        for index, row in enumerate(rows)
    ]
    write_coordinate_fixture(validation_dir / "pdb_coordinates", coordinate_records, int(config["data"]["max_len"]))

    batch_size = int(config["data"]["records_per_shard"])
    manifests_dir = data_root / "worker_manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    scratch_root.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    batch_summaries: list[dict[str, Any]] = []
    sequence_only_fraction = float(config["data"].get("mixed_sequence_fraction", 0.0))
    for batch_id, start in enumerate(range(0, len(rows), batch_size)):
        batch_rows = rows[start : start + batch_size]
        batch_dir = data_root / "shards" / f"batch-{batch_id:06d}"
        batch_marker = batch_dir / "COMPLETED"
        worker_jsonl = manifests_dir / f"batch-{batch_id:06d}.jsonl"
        if batch_marker.is_file() and worker_jsonl.is_file():
            batch_summaries.append(json.loads((batch_dir / "summary.json").read_text(encoding="utf-8")))
            continue
        part_dir = batch_dir.with_name(batch_dir.name + ".part")
        if part_dir.exists():
            shutil.rmtree(part_dir)
        part_dir.mkdir(parents=True)
        batch_scratch = Path(tempfile.mkdtemp(prefix=f"proteinloss-batch-{batch_id:06d}-", dir=scratch_root))
        a3m_dir = batch_scratch / "a3m"
        mmseqs_dir = batch_scratch / "mmseqs_batch"
        a3m_dir.mkdir()
        mmseqs_dir.mkdir()
        written_rows: list[dict[str, Any]] = []
        batch_start = time.perf_counter()
        try:
            with PackedShardWriter(data_root, part_dir, records_per_shard=batch_size) as writer:
                for row in batch_rows:
                    try:
                        rng = random.Random(int(config["seed"]) * 1_000_003 + int(row["source_index"]))
                        msa = _msa(row["sequence"], int(config["data"]["msa_depth"]), rng)
                        a3m_path = a3m_dir / f"{row['example_id']}.a3m"
                        _write_a3m(a3m_path, row["example_id"], msa)
                        reduced = normalize_msa_rows(read_a3m(a3m_path))
                        pssm, gaps = msa_to_pssm(reduced)
                        mi = msa_to_mi_apc(reduced, max_depth=int(config["data"]["msa_depth"]), device="cpu")
                        validate_pair_target(mi)
                        length, max_len = int(row["length"]), int(config["data"]["max_len"])
                        arrays = pad_core(np.asarray(query_to_ids(row["sequence"]), dtype=np.int16), pssm, gaps, max_len)
                        arrays.update(target_to_arrays(encode_mi_target(mi)))
                        arrays["valid_length"] = np.asarray(length, dtype=np.int32)
                        arrays["msa_depth"] = np.asarray(len(reduced), dtype=np.int32)
                        has_structure = (int(row["source_index"]) + 1) / max(len(rows), 1) > sequence_only_fraction
                        if has_structure:
                            arrays["ca_coords_fp16"] = encode_ca_coords(_coordinates(length, int(row["source_index"])))
                            arrays["three_di_ids"] = _three_di(length, int(row["source_index"]), max_len)
                        payload = npz_bytes(arrays)
                        ref = writer.write(
                            row["example_id"],
                            payload,
                            {"length": length, "has_structure": has_structure},
                        )
                        written_rows.append(
                            {
                                **row,
                                "shard_path": ref.shard_path,
                                "byte_offset": ref.byte_offset,
                                "byte_length": ref.byte_length,
                                "record_sha256": ref.record_sha256,
                                "has_pssm": True,
                                "has_mi": True,
                                "has_structure": has_structure,
                                "has_3di": has_structure,
                            }
                        )
                        a3m_path.unlink(missing_ok=True)
                    except Exception as error:  # a failed example must not poison the batch report
                        failures.append({"example_id": row["example_id"], "batch_id": batch_id, "error": repr(error)})
            if len(written_rows) != len(batch_rows):
                raise RuntimeError(f"batch {batch_id} wrote {len(written_rows)}/{len(batch_rows)} records")
            summary = {
                "batch_id": batch_id,
                "attempted": len(batch_rows),
                "written": len(written_rows),
                "skipped": 0,
                "failed": 0,
                "seconds": time.perf_counter() - batch_start,
            }
            atomic_json(part_dir / "summary.json", summary)
            (part_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
            if batch_dir.exists():
                raise FileExistsError(batch_dir)
            part_dir.replace(batch_dir)
            # The part directory rename changes relative record paths.
            for record in written_rows:
                record["shard_path"] = record["shard_path"].replace(f"batch-{batch_id:06d}.part/", f"batch-{batch_id:06d}/")
            inventory_path = batch_dir / "inventory.json"
            inventory_payload = json.loads(inventory_path.read_text(encoding="utf-8"))
            for shard in inventory_payload["shards"]:
                shard["shard_path"] = shard["shard_path"].replace(
                    f"batch-{batch_id:06d}.part/", f"batch-{batch_id:06d}/"
                )
                shard["index_path"] = shard["index_path"].replace(
                    f"batch-{batch_id:06d}.part/", f"batch-{batch_id:06d}/"
                )
            atomic_json(inventory_path, inventory_payload)
            for index_path in batch_dir.glob("*.index.jsonl"):
                corrected = index_path.with_suffix(index_path.suffix + ".part")
                with index_path.open("r", encoding="utf-8") as source, corrected.open("w", encoding="utf-8") as destination:
                    for line in source:
                        payload = json.loads(line)
                        payload["shard_path"] = payload["shard_path"].replace(
                            f"batch-{batch_id:06d}.part/", f"batch-{batch_id:06d}/"
                        )
                        destination.write(json.dumps(payload, sort_keys=True) + "\n")
                corrected.replace(index_path)
            with worker_jsonl.open("w", encoding="utf-8") as handle:
                for record in written_rows:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            batch_summaries.append(summary)
        finally:
            if bool(config["data"].get("cleanup_scratch", True)):
                shutil.rmtree(batch_scratch, ignore_errors=True)

    failure_path = data_root / "failure_rows.csv"
    with failure_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["example_id", "batch_id", "error"])
        writer.writeheader()
        writer.writerows(failures)
    if failures:
        raise RuntimeError(f"smoke target build has {len(failures)} failure rows: {failure_path}")

    manifest_path = data_root / "manifest.sqlite"
    manifest_writer = ManifestWriter(manifest_path, overwrite=manifest_path.exists())
    for worker_path in sorted(manifests_dir.glob("batch-*.jsonl")):
        with worker_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                manifest_writer.add(
                    ManifestRow(
                        example_id=record["example_id"],
                        group_id=record["group_id"],
                        split=record["split"],
                        shard_path=record["shard_path"],
                        byte_offset=int(record["byte_offset"]),
                        byte_length=int(record["byte_length"]),
                        record_sha256=record["record_sha256"],
                        length=int(record["length"]),
                        has_pssm=bool(record["has_pssm"]),
                        has_mi=bool(record["has_mi"]),
                        has_structure=bool(record["has_structure"]),
                        has_3di=bool(record["has_3di"]),
                        provenance={"source": "smoke", "msa": "synthetic_homologs", "structure": "synthetic_ca" if record["has_structure"] else ""},
                    )
                )
    manifest_writer.set_metadata("source_summary", source_summary)
    split_counts = manifest_writer.close(require_splits=True)
    export_jsonl(manifest_path, data_root / "manifest.jsonl")
    reader = ManifestReader(manifest_path)
    reader.validate()
    reader.close()
    remaining_scratch = [path.name for path in scratch_root.iterdir()] if scratch_root.exists() else []
    summary = {
        "status": "complete",
        "backend": "smoke",
        "attempted": len(rows),
        "written": sum(item["written"] for item in batch_summaries),
        "skipped": 0,
        "failed": 0,
        "seconds": time.perf_counter() - started,
        "examples_per_second": len(rows) / max(time.perf_counter() - started, 1e-9),
        "split_counts": split_counts,
        "mixed_supervision": {
            "structure_backed": sum(1 for path in manifests_dir.glob("batch-*.jsonl") for line in path.open() if json.loads(line)["has_structure"]),
            "sequence_only": sum(1 for path in manifests_dir.glob("batch-*.jsonl") for line in path.open() if not json.loads(line)["has_structure"]),
        },
        "manifest": "manifest.sqlite",
        "manifest_sha256": sha256_file(manifest_path),
        "teacher_checkpoint": str(teacher_path.relative_to(data_root)),
        "casp15_replication": "validation/casp15_smoke.npz",
        "experimental_coordinate_fixture": "validation/pdb_coordinates",
        "failure_rows": "failure_rows.csv",
        "scratch_remaining": remaining_scratch,
        "transient_a3m_removed": not remaining_scratch,
        "batch_summaries": batch_summaries,
    }
    atomic_json(summary_path, summary)
    completion.write_text("ok\n", encoding="utf-8")
    return summary


def build_dataset(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    backend = config.get("data", {}).get("backend", "production")
    if backend == "smoke":
        return build_smoke_dataset(config, project_root=project_root)
    from .production import plan_production_build

    return plan_production_build(config, project_root=project_root)
