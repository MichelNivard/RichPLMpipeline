from __future__ import annotations

import json
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .alphabet import AA, iter_fasta
from .config import write_snapshot
from .metadata import atomic_json, runtime_metadata, sha256_file
from .registry import load_registry, preflight, source_path
from .splits import stable_split
from .manifest import ManifestRow, ManifestWriter, ManifestReader, export_jsonl


def _accession(header: str) -> str:
    token = header.split()[0]
    fields = token.split("|")
    return fields[1] if len(fields) >= 3 and fields[0] in {"sp", "tr"} else token


def plan_production_build(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    """Stream source selection into bounded worker partitions.

    This stage intentionally performs no global sort and retains no full source
    table in memory. `pipeline data worker` consumes each partition. The same
    work plan is usable by a local loop or a Slurm array.
    """
    started = time.perf_counter()
    registry_path = Path(config.get("sources_config", "configs/sources.example.yaml"))
    check = preflight(
        registry_path,
        capabilities={"production_data", "structure_targets", "struct_latent"},
    )
    if not check["ok"]:
        raise RuntimeError("production preflight failed:\n" + "\n".join(check["missing"]))
    registry = load_registry(registry_path)
    sources = registry["sources"]
    uniprot = source_path(sources["uniprot_fasta"], registry_path)
    membership = source_path(sources["uniclust30_membership"], registry_path)
    afdb_prefix = source_path(sources["afdb_foldseek"], registry_path)
    afdb_index = Path(str(afdb_prefix) + ".portable.sqlite")
    if not afdb_index.is_file():
        raise RuntimeError(
            f"missing portable AFDB lookup index {afdb_index}; run `pipeline bootstrap index-lookups --sources {registry_path}`"
        )
    data_root = Path(config["paths"]["data_root"]).resolve()
    data_root.mkdir(parents=True, exist_ok=True)
    write_snapshot(config, data_root / "metadata")
    atomic_json(data_root / "metadata" / "runtime.json", runtime_metadata(project_root))
    atomic_json(data_root / "metadata" / "source_preflight.json", check)
    atomic_json(data_root / "metadata" / "source_registry.json", registry)
    source_dir = data_root / "source_partitions"
    source_dir.mkdir(parents=True, exist_ok=True)
    desired = int(config["data"]["count"])
    part_size = int(config["data"].get("worker_batch_size", 2000))
    min_len, max_len = int(config["data"].get("min_len", 8)), int(config["data"]["max_len"])
    seed = int(config["seed"])
    valid_fraction = float(config["data"]["valid_fraction"])
    test_fraction = float(config["data"]["test_fraction"])
    group_db = sqlite3.connect(f"file:{membership.resolve()}?mode=ro", uri=True)
    structure_db = sqlite3.connect(f"file:{afdb_index.resolve()}?mode=ro", uri=True)
    selected = scanned = skipped = 0
    splits: Counter[str] = Counter()
    supervision: Counter[str] = Counter()
    part_handle = None
    work_rows: list[dict[str, Any]] = []
    try:
        for header, sequence_raw in iter_fasta(uniprot):
            scanned += 1
            sequence = sequence_raw.upper()
            if len(sequence) < min_len or len(sequence) > max_len or any(aa not in AA for aa in sequence):
                skipped += 1
                continue
            accession = _accession(header)
            group_row = group_db.execute("SELECT rep FROM accession_group WHERE accession=?", (accession,)).fetchone()
            group_id = group_row[0] if group_row else accession
            structure_row = structure_db.execute(
                "SELECT numeric_key,source_id FROM lookup WHERE accession=?", (accession,)
            ).fetchone()
            split = stable_split(
                accession,
                group_id=group_id,
                seed=seed,
                valid_fraction=valid_fraction,
                test_fraction=test_fraction,
            )
            if selected % part_size == 0:
                if part_handle is not None:
                    part_handle.close()
                part_id = selected // part_size
                part_path = source_dir / f"part-{part_id:08d}.jsonl"
                part_handle = part_path.open("w", encoding="utf-8")
                work_rows.append({"worker_id": part_id, "source_partition": str(part_path.relative_to(data_root))})
            record = {
                "example_id": accession,
                "group_id": group_id,
                "split": split,
                "sequence": sequence,
                "length": len(sequence),
                "structure_key": int(structure_row[0]) if structure_row else None,
                "structure_id": structure_row[1] if structure_row else "",
            }
            part_handle.write(json.dumps(record, sort_keys=True) + "\n")
            selected += 1
            splits[split] += 1
            supervision["structure_backed" if structure_row else "sequence_only"] += 1
            if selected >= desired:
                break
    finally:
        if part_handle is not None:
            part_handle.close()
        group_db.close()
        structure_db.close()
    work_plan = data_root / "work_plan.jsonl"
    with work_plan.open("w", encoding="utf-8") as handle:
        for row in work_rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    summary = {
        "status": "planned" if selected == desired else "population_ceiling_reached",
        "desired_unique_sequences": desired,
        "selected_unique_sequences": selected,
        "scanned": scanned,
        "skipped": skipped,
        "worker_partitions": len(work_rows),
        "split_counts": dict(splits),
        "supervision_counts": dict(supervision),
        "population_contract": "records are never duplicated; structure objectives apply only to AFDB-backed rows",
        "work_plan": "work_plan.jsonl",
        "work_plan_sha256": sha256_file(work_plan),
        "seconds": time.perf_counter() - started,
    }
    summary["selected_per_second"] = selected / max(float(summary["seconds"]), 1e-9)
    atomic_json(data_root / "plan_summary.json", summary)
    if selected < desired:
        raise RuntimeError(
            f"requested {desired:,} unique sequences but the configured source supplied only {selected:,}; "
            "the pipeline did not duplicate records"
        )
    (data_root / "PLAN_COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary


def finalize_production_build(config: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    data_root = Path(config["paths"]["data_root"]).resolve()
    write_snapshot(config, data_root / "finalize_metadata")
    plan_path = data_root / "work_plan.jsonl"
    if not plan_path.is_file():
        raise FileNotFoundError(f"work plan not found: {plan_path}")
    with plan_path.open("r", encoding="utf-8") as handle:
        plan = [json.loads(line) for line in handle if line.strip()]
    missing_workers, worker_manifests = [], []
    for row in plan:
        worker = data_root / "workers" / f"worker-{int(row['worker_id']):08d}"
        if not (worker / "COMPLETED").is_file():
            missing_workers.append(int(row["worker_id"]))
        else:
            worker_manifests.append(worker / "manifest.jsonl")
    if missing_workers:
        raise RuntimeError(f"{len(missing_workers)} workers incomplete; first IDs: {missing_workers[:20]}")
    manifest_path = data_root / "manifest.sqlite"
    writer = ManifestWriter(manifest_path, overwrite=manifest_path.exists())
    attempted = written = failed = 0
    failure_output = data_root / "failure_rows.jsonl"
    with failure_output.open("w", encoding="utf-8") as failure_handle:
        for worker_manifest in worker_manifests:
            worker_dir = worker_manifest.parent
            worker_summary = json.loads((worker_dir / "summary.json").read_text(encoding="utf-8"))
            attempted += int(worker_summary["attempted"])
            failed += int(worker_summary["failed"])
            failure_path = worker_dir / "failures.jsonl"
            if failure_path.exists():
                with failure_path.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        failure_handle.write(line)
            with worker_manifest.open("r", encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    writer.add(
                        ManifestRow(
                            example_id=row["example_id"],
                            group_id=row["group_id"],
                            split=row["split"],
                            shard_path=row["shard_path"],
                            byte_offset=int(row["byte_offset"]),
                            byte_length=int(row["byte_length"]),
                            record_sha256=row["record_sha256"],
                            length=int(row["length"]),
                            has_pssm=bool(row["has_pssm"]),
                            has_mi=bool(row["has_mi"]),
                            has_structure=bool(row["has_structure"]),
                            has_3di=bool(row["has_3di"]),
                            provenance={
                                "msa": "uniclust30_expand_uniprotkb",
                                "structure": "afdb_foldseek" if row["has_structure"] else "",
                                "structure_id": row.get("structure_id", ""),
                            },
                        )
                    )
                    written += 1
    writer.set_metadata("attempted", attempted)
    writer.set_metadata("failed", failed)
    split_counts = writer.close(require_splits=True)
    desired = int(config["data"]["count"])
    if written < desired and not bool(config["data"].get("allow_shortfall", False)):
        raise RuntimeError(
            f"target shortfall: wrote {written:,}/{desired:,} unique examples; failures are in {failure_output}. "
            "No completion marker was written. Rerun failed workers or explicitly set allow_shortfall."
        )
    export_jsonl(manifest_path, data_root / "manifest.jsonl")
    reader = ManifestReader(manifest_path)
    reader.validate()
    reader.close()
    summary = {
        "status": "complete",
        "attempted": attempted,
        "written": written,
        "failed": failed,
        "skipped": attempted - written - failed,
        "split_counts": split_counts,
        "workers": len(plan),
        "manifest_sha256": sha256_file(manifest_path),
        "seconds": time.perf_counter() - started,
    }
    summary["examples_per_second"] = written / max(float(summary["seconds"]), 1e-9)
    atomic_json(data_root / "build_summary.json", summary)
    (data_root / "COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary
