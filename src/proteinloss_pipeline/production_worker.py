from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .alphabet import normalize_msa_rows, query_to_ids, read_a3m
from .codec import encode_ca_coords, encode_mi_target, npz_bytes, target_to_arrays
from .foldseek import decode_ca_record, read_index, read_text_record, sequence_length
from .metadata import atomic_json, sha256_file
from .config import write_snapshot
from .registry import load_registry, source_path
from .shards import PackedShardWriter
from .targets import msa_to_mi_apc, msa_to_pssm, pad_core


def _run(command: list[str], log: Path) -> dict[str, Any]:
    started = time.perf_counter()
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, text=True)
    row = {"command": command, "returncode": result.returncode, "seconds": time.perf_counter() - started, "log": str(log)}
    if result.returncode:
        raise RuntimeError(f"external command failed ({result.returncode}): {' '.join(command)}; see {log}")
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_fasta(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(f">{row['example_id']}\n{row['sequence']}\n")


def _lookup_numeric_keys(index_path: Path, accessions: set[str]) -> dict[str, int]:
    connection = sqlite3.connect(f"file:{index_path.resolve()}?mode=ro", uri=True)
    found: dict[str, int] = {}
    requested = sorted(accessions)
    for start in range(0, len(requested), 900):
        chunk = requested[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        for accession, key in connection.execute(
            f"SELECT accession,numeric_key FROM lookup WHERE accession IN ({placeholders})", chunk
        ):
            found[accession] = int(key)
    connection.close()
    return found


def _expand_representatives(
    hits_tsv: Path,
    query_lookup: Path,
    membership: Path,
    uniprot_index: Path,
    output: Path,
    target_keys: Path,
    max_members: int,
    max_pairs: int,
) -> dict[str, Any]:
    query_keys: dict[str, int] = {}
    with query_lookup.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.rstrip("\n").split("\t")
            if len(parts) >= 2:
                query_keys[parts[1].split()[0]] = int(parts[0])
    hits: dict[str, list[str]] = {}
    with hits_tsv.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            parts = raw.split()
            if len(parts) >= 2:
                hits.setdefault(parts[0], []).append(parts[1])
    representatives = sorted({rep for values in hits.values() for rep in values})
    membership_db = sqlite3.connect(f"file:{membership.resolve()}?mode=ro", uri=True)
    rep_members: dict[str, list[str]] = {}
    for start in range(0, len(representatives), 900):
        chunk = representatives[start : start + 900]
        placeholders = ",".join("?" for _ in chunk)
        for rep, member_blob in membership_db.execute(
            f"SELECT rep,members FROM members WHERE rep IN ({placeholders})", chunk
        ):
            rep_members[rep] = member_blob.split("\t")[:max_members]
    membership_db.close()
    accessions = {member for members in rep_members.values() for member in members}
    numeric = _lookup_numeric_keys(uniprot_index, accessions)
    keys: set[int] = set()
    written = 0
    with output.open("w", encoding="utf-8") as handle:
        for query, reps in hits.items():
            query_key = query_keys.get(query)
            if query_key is None:
                continue
            per_query = 0
            seen: set[int] = set()
            for rep in reps:
                for member in rep_members.get(rep, []):
                    key = numeric.get(member)
                    if key is None or key in seen:
                        continue
                    handle.write(f"{query_key}\t{key}\n")
                    seen.add(key)
                    keys.add(key)
                    written += 1
                    per_query += 1
                    if per_query >= max_pairs:
                        break
                if per_query >= max_pairs:
                    break
    with target_keys.open("w", encoding="utf-8") as handle:
        for key in sorted(keys):
            handle.write(f"{key}\n")
    return {"expanded_pairs": written, "target_keys": len(keys), "queries_with_hits": len(hits)}


def run_production_worker(config: dict[str, Any], worker_id: int) -> dict[str, Any]:
    data_root = Path(config["paths"]["data_root"]).resolve()
    registry_path = Path(config["sources_config"])
    registry = load_registry(registry_path)
    sources, tools = registry["sources"], registry["tools"]
    mmseqs = source_path(tools["mmseqs"], registry_path)
    rep_db = source_path(sources["uniclust30_seed_db"], registry_path)
    membership = source_path(sources["uniclust30_membership"], registry_path)
    uniprot_db = source_path(sources["uniprot_mmseqs_db"], registry_path)
    uniprot_index = source_path(sources["uniprot_accession_index"], registry_path)
    afdb = source_path(sources["afdb_foldseek"], registry_path)
    teacher_path = source_path(sources["structencoder_teacher"], registry_path)
    source_partition = data_root / "source_partitions" / f"part-{worker_id:08d}.jsonl"
    if not source_partition.is_file():
        raise FileNotFoundError(source_partition)
    output = data_root / "workers" / f"worker-{worker_id:08d}"
    completed = output / "COMPLETED"
    if completed.is_file():
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    part_output = output.with_name(output.name + ".part")
    if part_output.exists():
        shutil.rmtree(part_output)
    part_output.mkdir(parents=True)
    write_snapshot(config, part_output / "metadata")
    atomic_json(part_output / "metadata" / "source_registry.json", registry)
    scratch_root = Path(config["paths"]["scratch_root"]).resolve()
    scratch_root.mkdir(parents=True, exist_ok=True)
    scratch = Path(tempfile.mkdtemp(prefix=f"proteinloss-worker-{worker_id:08d}-", dir=scratch_root))
    logs, mm = part_output / "logs", scratch / "mmseqs"
    mm.mkdir()
    rows = _read_jsonl(source_partition)
    timings = []
    started = time.perf_counter()
    try:
        queries = scratch / "queries.fasta"
        _write_fasta(rows, queries)
        query_db, hits, temporary = mm / "query", mm / "rep_hits", mm / "tmp"
        timings.append(_run([str(mmseqs), "createdb", str(queries), str(query_db)], logs / "01_createdb.log"))
        timings.append(
            _run(
                [
                    str(mmseqs), "search", str(query_db), str(rep_db), str(hits), str(temporary),
                    "--threads", str(config["data"].get("threads", 8)),
                    "--num-iterations", str(config["data"].get("search_num_iterations", 1)),
                    "-s", str(config["data"].get("search_sensitivity", 3.0)),
                    "-e", str(config["data"].get("search_evalue", "1e-3")),
                    "--max-seqs", str(config["data"].get("search_max_seqs", 200)),
                ],
                logs / "02_search.log",
            )
        )
        hits_tsv = mm / "hits.tsv"
        timings.append(_run([str(mmseqs), "createtsv", str(query_db), str(rep_db), str(hits), str(hits_tsv)], logs / "03_hits_tsv.log"))
        expanded, keys_file = mm / "expanded.numeric.tsv", mm / "target_keys.txt"
        expansion = _expand_representatives(
            hits_tsv,
            Path(str(query_db) + ".lookup"),
            membership,
            uniprot_index,
            expanded,
            keys_file,
            int(config["data"].get("max_members_per_rep", 16)),
            int(config["data"].get("max_expanded_pairs_per_query", 1024)),
        )
        if not expansion["expanded_pairs"]:
            raise RuntimeError("worker produced no expanded UniClust/UniProt pairs")
        member_db, pair_db, aligned, msa_db = mm / "members", mm / "pairs", mm / "aligned", mm / "msa"
        timings.append(_run([str(mmseqs), "createsubdb", str(keys_file), str(uniprot_db), str(member_db)], logs / "04_subdb.log"))
        timings.append(_run([str(mmseqs), "tsv2db", str(expanded), str(pair_db), "--output-dbtype", "5"], logs / "05_pairs.log"))
        timings.append(
            _run(
                [str(mmseqs), "align", str(query_db), str(member_db), str(pair_db), str(aligned), "--threads", str(config["data"].get("threads", 8)), "-a", "1"],
                logs / "06_align.log",
            )
        )
        timings.append(
            _run(
                [str(mmseqs), "result2msa", str(query_db), str(member_db), str(aligned), str(msa_db), "--msa-format-mode", "6", "--filter-msa", "1", "--diff", str(config["data"].get("msa_depth", 2048))],
                logs / "07_msa.log",
            )
        )
        a3m = scratch / "a3m"
        a3m.mkdir()
        timings.append(_run([str(mmseqs), "unpackdb", str(msa_db), str(a3m), "--unpack-suffix", ".a3m"], logs / "08_unpack.log"))
        key_to_id: dict[str, str] = {}
        with Path(str(query_db) + ".lookup").open("r", encoding="utf-8", errors="replace") as handle:
            for raw in handle:
                parts = raw.rstrip("\n").split("\t")
                if len(parts) >= 2:
                    key_to_id[parts[0]] = parts[1].split()[0]
        row_by_id = {row["example_id"]: row for row in rows}
        structure_keys = {int(row["structure_key"]) for row in rows if row.get("structure_key") is not None}
        base_index = read_index(Path(str(afdb) + ".index"), structure_keys)
        ca_index = read_index(Path(str(afdb) + "_ca.index"), structure_keys)
        ss_index = read_index(Path(str(afdb) + "_ss.index"), structure_keys)
        teacher_payload = torch.load(teacher_path, map_location="cpu", weights_only=False)
        token_to_id = {str(token): index for index, token in enumerate(teacher_payload["tokens"])}
        pad_id, max_len = int(teacher_payload["pad_id"]), int(config["data"]["max_len"])
        manifest_rows, failures = [], []
        with PackedShardWriter(data_root, part_output / "targets", records_per_shard=len(rows)) as writer:
            for a3m_path in sorted(a3m.glob("*.a3m"), key=lambda path: int(path.stem)):
                example_id = key_to_id.get(a3m_path.stem, "")
                row = row_by_id.get(example_id)
                if row is None:
                    continue
                try:
                    msa = normalize_msa_rows(read_a3m(a3m_path))[: int(config["data"].get("msa_depth", 2048))]
                    if len(msa) < 2:
                        raise ValueError("MSA has fewer than two normalized rows")
                    length = min(len(row["sequence"]), len(msa[0]), max_len)
                    msa = [sequence[:length] for sequence in msa]
                    pssm, gaps = msa_to_pssm(msa)
                    mi = msa_to_mi_apc(msa, max_depth=len(msa), device=str(config["data"].get("mi_device", "auto")))
                    arrays = pad_core(np.asarray(query_to_ids(row["sequence"][:length]), dtype=np.int16), pssm, gaps, max_len)
                    arrays.update(target_to_arrays(encode_mi_target(mi)))
                    arrays["valid_length"] = np.asarray(length, dtype=np.int32)
                    arrays["msa_depth"] = np.asarray(len(msa), dtype=np.int32)
                    has_structure = False
                    key = row.get("structure_key")
                    if key is not None and int(key) in base_index and int(key) in ca_index and int(key) in ss_index:
                        key = int(key)
                        structure_length = sequence_length(base_index[key][1])
                        ca_offset, ca_bytes = ca_index[key]
                        coordinates = decode_ca_record(Path(str(afdb) + "_ca"), ca_offset, ca_bytes, structure_length)
                        ss_offset, ss_bytes = ss_index[key]
                        ss = read_text_record(Path(str(afdb) + "_ss"), ss_offset, ss_bytes)
                        usable = min(length, len(coordinates), len(ss))
                        arrays["valid_length"] = np.asarray(usable, dtype=np.int32)
                        arrays["ca_coords_fp16"] = encode_ca_coords(coordinates[:usable])
                        three_di = np.full(max_len, pad_id, dtype=np.uint8)
                        three_di[:usable] = [token_to_id.get(token, pad_id + 2) for token in ss[:usable]]
                        arrays["three_di_ids"] = three_di
                        has_structure = True
                        length = usable
                    ref = writer.write(example_id, npz_bytes(arrays), {"length": length, "has_structure": has_structure})
                    manifest_rows.append(
                        {
                            **row,
                            "length": length,
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
                except Exception as error:
                    failures.append({"example_id": example_id, "error": repr(error)})
        manifest_path = part_output / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8") as handle:
            for row in manifest_rows:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        with (part_output / "failures.jsonl").open("w", encoding="utf-8") as handle:
            produced_ids = {row["example_id"] for row in manifest_rows}
            for missing_id in sorted(set(row_by_id) - produced_ids - {row["example_id"] for row in failures}):
                failures.append({"example_id": missing_id, "error": "missing_or_unusable_a3m"})
            for row in failures:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        summary = {
            "worker_id": worker_id,
            "attempted": len(rows),
            "written": len(manifest_rows),
            "failed": len(failures),
            "skipped": len(rows) - len(manifest_rows) - len(failures),
            "seconds": time.perf_counter() - started,
            "expansion": expansion,
            "timings": timings,
        }
        summary["examples_per_second"] = len(manifest_rows) / max(float(summary["seconds"]), 1e-9)
        atomic_json(part_output / "summary.json", summary)
        (part_output / "COMPLETED").write_text("ok\n", encoding="utf-8")
        part_output.replace(output)
        # Rewrite paths created under the atomic .part directory.
        manifest_text = (output / "manifest.jsonl").read_text(encoding="utf-8").replace(
            f"worker-{worker_id:08d}.part/", f"worker-{worker_id:08d}/"
        )
        (output / "manifest.jsonl").write_text(manifest_text, encoding="utf-8")
        for metadata_path in [*output.rglob("*.index.jsonl"), *output.rglob("inventory.json")]:
            corrected = metadata_path.read_text(encoding="utf-8").replace(
                f"worker-{worker_id:08d}.part/", f"worker-{worker_id:08d}/"
            )
            metadata_path.write_text(corrected, encoding="utf-8")
        return summary
    finally:
        if bool(config["data"].get("cleanup_scratch", True)):
            shutil.rmtree(scratch, ignore_errors=True)
