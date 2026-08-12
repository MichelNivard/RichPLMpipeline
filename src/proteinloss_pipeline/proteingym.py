from __future__ import annotations

import csv
import json
import re
import shutil
import time
import zipfile
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .alphabet import AA, AA_TO_ID, MASK_TOKEN_ID, query_to_ids
from .metadata import atomic_json, sha256_file
from .model import load_checkpoint_model
from .training import resolve_amp, resolve_device

MUTATION = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def _int(row: dict[str, str], key: str) -> int:
    try:
        return int(float(row.get(key, "") or 0))
    except ValueError:
        return 0


def prepare_assays(
    reference_csv: str | Path,
    bundle_zip: str | Path,
    output_dir: str | Path,
    *,
    max_len: int = 512,
    limit: int = 0,
) -> dict[str, Any]:
    output = Path(output_dir)
    completion = output / "COMPLETED"
    if completion.is_file() and (output / "summary.json").is_file():
        return json.loads((output / "summary.json").read_text(encoding="utf-8"))
    started = time.perf_counter()
    assays = output / "assays"
    assays.mkdir(parents=True, exist_ok=True)
    with Path(reference_csv).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        selected = [
            dict(row)
            for row in reader
            if row.get("includes_multiple_mutants") == "FALSE"
            and 0 < _int(row, "seq_len") <= max_len
            and row.get("target_seq")
            and not (set(row["target_seq"].strip().upper()) - set(AA))
        ]
    selected.sort(key=lambda row: row.get("DMS_id", ""))
    if limit:
        selected = selected[:limit]
    with zipfile.ZipFile(bundle_zip) as archive:
        members = {Path(name).name: name for name in archive.namelist() if name.endswith(".csv")}
        kept = []
        for row in selected:
            filename = row["DMS_filename"]
            if filename not in members:
                continue
            destination = assays / filename
            with archive.open(members[filename]) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            if _assay_variants(destination, row["target_seq"].strip().upper()):
                kept.append(row)
            else:
                destination.unlink(missing_ok=True)
    metadata = output / "selected_assays.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(kept)
    summary = {
        "status": "complete",
        "attempted_compatible_assays": len(selected),
        "selected_assays": len(kept),
        "skipped": len(selected) - len(kept),
        "max_len": max_len,
        "metadata": str(metadata),
        "assay_dir": str(assays),
        "seconds": time.perf_counter() - started,
        "source_reference_sha256": sha256_file(reference_csv),
        "source_bundle_sha256": sha256_file(bundle_zip),
        "selected_metadata_sha256": sha256_file(metadata),
    }
    atomic_json(output / "summary.json", summary)
    completion.write_text("ok\n", encoding="utf-8")
    return summary


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = x.astype(np.float64) - np.mean(x), y.astype(np.float64) - np.mean(y)
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    return float(np.dot(x, y) / denominator) if denominator > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    return pearson(_rank(x), _rank(y))


def _assay_variants(path: Path, sequence: str) -> list[dict[str, Any]]:
    variants: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"mutant", "DMS_score"}.issubset(reader.fieldnames or []):
            return []
        for row in reader:
            match = MUTATION.match((row.get("mutant") or "").strip())
            if not match:
                continue
            wildtype, position_text, alternate = match.groups()
            position = int(position_text)
            if not (1 <= position <= len(sequence)) or sequence[position - 1] != wildtype:
                continue
            if wildtype not in AA_TO_ID or alternate not in AA_TO_ID:
                continue
            try:
                score = float(row["DMS_score"])
            except (TypeError, ValueError):
                continue
            variants.append({"mutant": row["mutant"], "position": position, "wt": wildtype, "alt": alternate, "score": score})
    return variants


@torch.inference_mode()
def masked_log_probabilities(model, sequence: str, device: torch.device, batch_size: int, amp_dtype) -> np.ndarray:
    tokens = torch.tensor(query_to_ids(sequence), dtype=torch.long, device=device)
    length = len(sequence)
    positions = torch.arange(length, device=device)
    inputs = tokens.unsqueeze(0).repeat(length, 1)
    inputs[positions, positions] = MASK_TOKEN_ID
    mask = torch.ones_like(inputs, dtype=torch.bool)
    output = torch.empty((length, 20), dtype=torch.float32)
    for start in range(0, length, batch_size):
        stop = min(length, start + batch_size)
        context = torch.autocast(device_type=device.type, dtype=amp_dtype) if amp_dtype is not None else nullcontext()
        with context:
            logits = model(inputs[start:stop], mask[start:stop], objectives=set(), return_hidden=False)["pssm_logits"]
        row = torch.arange(stop - start, device=device)
        output[start:stop] = F.log_softmax(logits.float()[row, positions[start:stop]], -1).cpu()
    return output.numpy()


def evaluate_proteingym(
    checkpoint: str | Path,
    assay_metadata: str | Path,
    assay_dir: str | Path,
    output_dir: str | Path,
    *,
    device_name: str = "auto",
    amp_name: str = "bfloat16",
    mask_batch_size: int = 64,
    limit: int = 0,
) -> dict[str, Any]:
    device = resolve_device(device_name)
    amp_dtype = resolve_amp(amp_name, device)
    model, _payload = load_checkpoint_model(str(checkpoint), device)
    with Path(assay_metadata).open("r", encoding="utf-8", newline="") as handle:
        assays = [dict(row) for row in csv.DictReader(handle)]
    if limit:
        assays = assays[:limit]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    variant_rows: list[dict[str, Any]] = []
    for metadata in assays:
        sequence = metadata["target_seq"].strip().upper()
        variants = _assay_variants(Path(assay_dir) / metadata["DMS_filename"], sequence)
        if not variants:
            continue
        probabilities = masked_log_probabilities(model, sequence, device, mask_batch_size, amp_dtype)
        predictions, observations = [], []
        for variant in variants:
            index = int(variant["position"]) - 1
            wt_logp = float(probabilities[index, AA_TO_ID[variant["wt"]]])
            alt_logp = float(probabilities[index, AA_TO_ID[variant["alt"]]])
            prediction = alt_logp - wt_logp
            predictions.append(prediction)
            observations.append(float(variant["score"]))
            variant_rows.append({"DMS_id": metadata["DMS_id"], **variant, "model_score": prediction})
        prediction_array, observation_array = np.asarray(predictions), np.asarray(observations)
        summary_rows.append(
            {
                "DMS_id": metadata["DMS_id"],
                "seq_len": len(sequence),
                "num_variants": len(variants),
                "spearman": spearman(prediction_array, observation_array),
                "pearson": pearson(prediction_array, observation_array),
            }
        )
    for path, rows in ((output / "summary.csv", summary_rows), (output / "variants.csv", variant_rows)):
        fields = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            if fields:
                writer.writeheader()
                writer.writerows(rows)
    result = {
        "assays": len(summary_rows),
        "variants": sum(int(row["num_variants"]) for row in summary_rows),
        "mean_spearman": float(np.mean([row["spearman"] for row in summary_rows])) if summary_rows else 0.0,
        "mean_pearson": float(np.mean([row["pearson"] for row in summary_rows])) if summary_rows else 0.0,
    }
    atomic_json(output / "summary.json", result)
    return result
