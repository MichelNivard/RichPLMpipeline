from __future__ import annotations

import csv
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import write_snapshot
from .dataset import PackedProteinDataset, collate_records
from .losses import OBJECTIVES, compute_objectives
from .manifest import ManifestReader
from .metadata import atomic_json, runtime_metadata, sha256_file
from .model import EXPECTED_PARAMETER_COUNTS, build_model, parameter_count
from .teacher import StructEncoder, load_teacher


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def resolve_amp(name: str, device: torch.device):
    if name == "none" or device.type != "cuda":
        return None
    return torch.float16 if name == "float16" else torch.bfloat16


def amp_context(device: torch.device, dtype):
    return torch.autocast(device_type=device.type, dtype=dtype) if dtype is not None else nullcontext()


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value for key, value in batch.items()}


def _objective_preflight(manifest_path: Path, weights: dict[str, float]) -> dict[str, Any]:
    reader = ManifestReader(manifest_path)
    counts = reader.validate()
    connection = reader.connection
    coverage: dict[str, dict[str, int]] = {}
    column = {
        "pssm": "has_pssm",
        "dense_mi": "has_mi",
        "distance": "has_structure",
        "contact": "has_structure",
        "struct_latent": "has_3di",
        "3di": "has_3di",
    }
    for objective, field in column.items():
        if float(weights.get(objective, 0.0)) == 0:
            continue
        by_split = {
            split: int(
                connection.execute(f"SELECT count(*) FROM examples WHERE split=? AND {field}=1", (split,)).fetchone()[0]
            )
            for split in ("train", "valid", "test")
        }
        missing = [split for split, count in by_split.items() if count == 0]
        if missing:
            raise ValueError(f"objective {objective} has no eligible examples in: {', '.join(missing)}")
        coverage[objective] = by_split
    reader.close()
    return {"split_counts": counts, "objective_coverage": coverage}


def _save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    scaler,
    config: dict[str, Any],
    step: int,
    examples: int,
    epoch: int,
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "resolved_config": config,
            "step": step,
            "examples": examples,
            "epoch": epoch,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "metadata": metadata,
            "parameters": parameter_count(model),
        },
        temporary,
    )
    temporary.replace(path)


def _save_final(
    path: Path,
    *,
    model,
    config: dict[str, Any],
    step: int,
    examples: int,
    metadata: dict[str, Any],
) -> None:
    temporary = path.with_suffix(path.suffix + ".part")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "resolved_config": config,
            "step": step,
            "examples": examples,
            "metadata": metadata,
            "parameters": parameter_count(model),
        },
        temporary,
    )
    temporary.replace(path)


def _restore_rng(payload: dict[str, Any]) -> None:
    rng = payload.get("rng") or {}
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch.random.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng.get("cuda"):
        torch.cuda.set_rng_state_all(rng["cuda"])


def evaluate_loader(
    model,
    loader: DataLoader,
    device: torch.device,
    weights: dict[str, float],
    loss_config: dict[str, Any],
    teacher: StructEncoder | None,
    amp_dtype,
    max_examples: int = 0,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    examples = 0
    with torch.no_grad():
        for batch in loader:
            batch = _move(batch, device)
            batch_size = int(batch["input_ids"].shape[0])
            with amp_context(device, amp_dtype):
                _loss, metrics = compute_objectives(
                    model,
                    batch,
                    weights,
                    teacher=teacher,
                    mlm_probability=float(loss_config.get("mlm_probability", 0.15)),
                    min_pair_separation=int(loss_config.get("min_pair_separation", 6)),
                    distance_correlation_weight=float(loss_config.get("distance_correlation_weight", 1.0)),
                    contact_cutoff=float(loss_config.get("contact_cutoff_angstrom", 8.0)),
                )
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            examples += batch_size
            if max_examples and examples >= max_examples:
                break
    model.train()
    return {"examples": float(examples), **{key: value / max(examples, 1) for key, value in totals.items()}}


def train_model(config: dict[str, Any], *, project_root: Path) -> dict[str, Any]:
    seed = int(config.get("seed", 7))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    training = config.setdefault("training", {})
    data_root = Path(config["paths"]["data_root"]).resolve()
    manifest_path = data_root / str(training.get("manifest", "manifest.sqlite"))
    run_root = Path(config["paths"]["run_root"]).resolve()
    run_name = str(training.get("run_name", f"{config['model']['preset']}-run"))
    run_dir = run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    weights = {name: float(config.get("objectives", {}).get(name, 0.0)) for name in OBJECTIVES}
    if not any(weights.values()):
        raise ValueError("at least one objective weight must be non-zero")
    preflight = _objective_preflight(manifest_path, weights)
    teacher: StructEncoder | None = None
    teacher_path = str(training.get("teacher_checkpoint", ""))
    device = resolve_device(str(training.get("device", "auto")))
    amp_dtype = resolve_amp(str(training.get("amp_dtype", "bfloat16")), device)
    if weights.get("struct_latent", 0.0):
        if not teacher_path:
            teacher_path = str(data_root / "metadata" / "smoke_structencoder_teacher.pt")
        teacher, teacher_info = load_teacher(teacher_path, device)
        if teacher_info.latent_dim != 128:
            raise ValueError(f"StructEncoder latent dimension must be 128, got {teacher_info.latent_dim}")

    preset = str(config["model"]["preset"])
    model = build_model(preset, device=device)
    exact_parameters = parameter_count(model)
    if exact_parameters != EXPECTED_PARAMETER_COUNTS[preset]:
        raise RuntimeError(f"preset {preset} parameter drift: {exact_parameters} != {EXPECTED_PARAMETER_COUNTS[preset]}")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training.get("learning_rate", 3e-4)),
        weight_decay=float(training.get("weight_decay", 1e-2)),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    metadata = {
        **runtime_metadata(project_root),
        "manifest_sha256": sha256_file(manifest_path),
        "teacher_checkpoint": teacher_path,
        "teacher_sha256": sha256_file(teacher_path) if teacher_path else "",
        "seed": seed,
        "device": str(device),
        "torch_version": torch.__version__,
    }
    write_snapshot(config, run_dir)
    atomic_json(run_dir / "preflight.json", preflight)
    atomic_json(run_dir / "metadata.json", metadata)

    batch_size = int(training.get("batch_size", 1))
    workers = int(training.get("num_workers", 0))
    generator = torch.Generator().manual_seed(seed)
    train_dataset = PackedProteinDataset(data_root, manifest_path, "train")
    valid_dataset = PackedProteinDataset(data_root, manifest_path, "valid")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=workers,
        collate_fn=collate_records,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        collate_fn=collate_records,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    step = examples = start_epoch = 0
    resume = str(training.get("resume", ""))
    if resume:
        payload = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(payload["model_state_dict"])
        optimizer.load_state_dict(payload["optimizer_state_dict"])
        scaler.load_state_dict(payload.get("scaler_state_dict", {}))
        step, examples, start_epoch = int(payload.get("step", 0)), int(payload.get("examples", 0)), int(payload.get("epoch", 0))
        _restore_rng(payload)

    metrics_path = run_dir / "steps.csv"
    metrics_jsonl = run_dir / "steps.jsonl"
    max_steps = int(training.get("max_steps", 0))
    epochs = int(training.get("epochs", 1))
    accumulation = max(1, int(training.get("gradient_accumulation", 1)))
    checkpoint_every = max(1, int(training.get("checkpoint_every_steps", 1000)))
    log_every = max(1, int(training.get("log_every_steps", 10)))
    loss_config = config.get("loss", {})
    started = time.perf_counter()
    fieldnames: list[str] | None = None
    optimizer.zero_grad(set_to_none=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    stop = False
    for epoch in range(start_epoch, epochs):
        for micro_step, batch in enumerate(train_loader, start=1):
            batch = _move(batch, device)
            batch_examples = int(batch["input_ids"].shape[0])
            with amp_context(device, amp_dtype):
                total, metrics = compute_objectives(
                    model,
                    batch,
                    weights,
                    teacher=teacher,
                    mlm_probability=float(loss_config.get("mlm_probability", 0.15)),
                    min_pair_separation=int(loss_config.get("min_pair_separation", 6)),
                    distance_correlation_weight=float(loss_config.get("distance_correlation_weight", 1.0)),
                    contact_cutoff=float(loss_config.get("contact_cutoff_angstrom", 8.0)),
                )
                loss = total / accumulation
            scaler.scale(loss).backward()
            examples += batch_examples
            if micro_step % accumulation != 0:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(training.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            elapsed = time.perf_counter() - started
            row: dict[str, Any] = {
                "epoch": epoch,
                "optimizer_step": step,
                "examples": examples,
                "elapsed_seconds": elapsed,
                "examples_per_second": examples / max(elapsed, 1e-9),
                "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0,
                "peak_gpu_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2 if device.type == "cuda" else 0.0,
                **metrics,
            }
            if fieldnames is None:
                fieldnames = list(row)
                with metrics_path.open("w", encoding="utf-8", newline="") as handle:
                    csv.DictWriter(handle, fieldnames=fieldnames).writeheader()
            with metrics_path.open("a", encoding="utf-8", newline="") as handle:
                csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore").writerow(row)
            with metrics_jsonl.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")
            if step % checkpoint_every == 0:
                _save_checkpoint(
                    run_dir / "latest.pt",
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    config=config,
                    step=step,
                    examples=examples,
                    epoch=epoch,
                    metadata=metadata,
                )
            if step % log_every == 0:
                print(json.dumps(row, sort_keys=True), flush=True)
            if max_steps and step >= max_steps:
                stop = True
                break
        if stop:
            break
    if step == 0:
        raise RuntimeError("training completed without an optimizer step")
    validation = evaluate_loader(
        model,
        valid_loader,
        device,
        weights,
        loss_config,
        teacher,
        amp_dtype,
        int(training.get("max_validation_examples", 0)),
    )
    # Always leave one rolling, optimizer-resumable checkpoint even for a run
    # shorter than the configured checkpoint interval.
    _save_checkpoint(
        run_dir / "latest.pt",
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        config=config,
        step=step,
        examples=examples,
        epoch=epoch,
        metadata=metadata,
    )
    _save_final(
        run_dir / "final.pt",
        model=model,
        config=config,
        step=step,
        examples=examples,
        metadata=metadata,
    )
    summary = {
        "status": "complete",
        "run_name": run_name,
        "model_preset": preset,
        "exact_parameters": exact_parameters,
        "active_objectives": {key: value for key, value in weights.items() if value},
        "optimizer_steps": step,
        "examples": examples,
        "elapsed_seconds": time.perf_counter() - started,
        "validation": validation,
        "checkpoint": "final.pt",
        "checkpoint_sha256": sha256_file(run_dir / "final.pt"),
        "rolling_checkpoint": "latest.pt",
        "rolling_checkpoint_sha256": sha256_file(run_dir / "latest.pt"),
    }
    atomic_json(run_dir / "summary.json", summary)
    (run_dir / "COMPLETED").write_text("ok\n", encoding="utf-8")
    return summary
