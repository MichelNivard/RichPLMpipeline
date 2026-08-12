from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import PackedProteinDataset, collate_records
from .losses import OBJECTIVES
from .config import write_snapshot
from .metadata import atomic_json, runtime_metadata, sha256_file
from .minifold import train_minifold_probe
from .model import load_checkpoint_model
from .proteingym import evaluate_proteingym
from .teacher import load_teacher
from .training import evaluate_loader, resolve_amp, resolve_device


def validate_checkpoint(
    config: dict[str, Any],
    checkpoint: str | Path,
    output_dir: str | Path,
    *,
    proteingym_metadata: str | Path | None = None,
    proteingym_assay_dir: str | Path | None = None,
    proteingym_limit: int = 0,
    run_minifold: bool = False,
    minifold_data: str | Path | None = None,
    casp15_data: str | Path | None = None,
    pair_diagnostics: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    seed = int(config.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    write_snapshot(config, output)
    atomic_json(
        output / "metadata.json",
        {**runtime_metadata(Path(__file__).resolve().parents[2]), "seed": seed, "torch_version": torch.__version__},
    )
    validation_config = config.get("validation", {})
    training_config = config.get("training", {})
    device = resolve_device(str(validation_config.get("device", training_config.get("device", "auto"))))
    amp_name = str(validation_config.get("amp_dtype", training_config.get("amp_dtype", "bfloat16")))
    amp_dtype = resolve_amp(amp_name, device)
    model, payload = load_checkpoint_model(str(checkpoint), device)
    weights = {name: float(config.get("objectives", {}).get(name, 0.0)) for name in OBJECTIVES}
    # Always expose MLM/PSSM readouts. Pair heads are included when trained or
    # explicitly requested as diagnostics.
    weights["mlm"] = weights.get("mlm", 0.0) or 1.0
    if pair_diagnostics:
        for name in ("dense_mi", "distance", "contact"):
            weights[name] = weights.get(name, 0.0) or 1.0
    data_root = Path(config["paths"]["data_root"]).resolve()
    manifest = data_root / str(training_config.get("manifest", "manifest.sqlite"))
    test_dataset = PackedProteinDataset(data_root, manifest, "test", verify_records=bool(validation_config.get("verify_records", False)))
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(validation_config.get("batch_size", 1)),
        shuffle=False,
        num_workers=int(validation_config.get("num_workers", 0)),
        collate_fn=collate_records,
    )
    teacher = None
    if weights.get("struct_latent", 0.0):
        teacher_path = str(training_config.get("teacher_checkpoint", "")) or str(
            data_root / "metadata" / "smoke_structencoder_teacher.pt"
        )
        teacher, _info = load_teacher(teacher_path, device)
    heldout = evaluate_loader(
        model,
        test_loader,
        device,
        weights,
        config.get("loss", {}),
        teacher,
        amp_dtype,
        int(validation_config.get("max_heldout_examples", 0)),
    )
    components: dict[str, Any] = {"heldout_targets": heldout}
    flat_rows = [{"component": "heldout_targets", **heldout}]
    if proteingym_metadata and proteingym_assay_dir:
        result = evaluate_proteingym(
            checkpoint,
            proteingym_metadata,
            proteingym_assay_dir,
            output / "proteingym",
            device_name=str(device),
            amp_name=amp_name,
            mask_batch_size=int(validation_config.get("mask_batch_size", 64)),
            limit=proteingym_limit,
        )
        components["proteingym"] = result
        flat_rows.append({"component": "proteingym", **result})
    else:
        components["proteingym"] = {"status": "not_requested"}
    if run_minifold:
        if not minifold_data:
            raise ValueError("--minifold requires --minifold-data with an experimental PDB coordinate dataset")
        result = train_minifold_probe(
            checkpoint,
            minifold_data,
            output / "minifold",
            casp15_data=casp15_data,
            device_name=str(device),
            steps=int(validation_config.get("minifold_steps", 2)),
            batch_size=int(validation_config.get("minifold_batch_size", 1)),
            single_dim=int(validation_config.get("minifold_single_dim", 64)),
            pair_rank=int(validation_config.get("minifold_pair_rank", 32)),
            blocks=int(validation_config.get("minifold_blocks", 1)),
        )
        components["minifold"] = result
        for evaluation in result["evaluations"]:
            flat_rows.append({"component": f"minifold_{evaluation['dataset']}", **evaluation})
    else:
        components["minifold"] = {"status": "not_requested"}
    fields = sorted({key for row in flat_rows for key in row})
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)
    summary = {
        "status": "complete",
        "checkpoint": str(Path(checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "seed": seed,
        "elapsed_seconds": time.perf_counter() - started,
        "components": components,
    }
    atomic_json(output / "summary.json", summary)
    lines = [
        "# ProteinLoss validation report",
        "",
        f"Checkpoint: `{Path(checkpoint).name}` (step {summary['checkpoint_step']}).",
        "",
        "## Held-out targets",
        "",
        f"Evaluated {int(heldout.get('examples', 0))} test examples; weighted total loss {heldout.get('weighted_total', 0):.4f}.",
    ]
    if components["proteingym"].get("status") != "not_requested":
        pg = components["proteingym"]
        lines += ["", "## ProteinGym", "", f"{pg['assays']} assays, {pg['variants']} variants; mean Spearman {pg['mean_spearman']:.4f}."]
    if components["minifold"].get("status") != "not_requested":
        lines += ["", "## MiniFold frozen-embedding probe", ""]
        for item in components["minifold"]["evaluations"]:
            lines.append(
                f"- {item['dataset']}: {item['examples']} examples, RMSD {item.get('rmsd', 0):.3f}, "
                f"distance MAE {item.get('distance_mae', 0):.3f}, contact precision@L {item.get('contact_precision_top_l', 0):.3f}."
            )
    (output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary
