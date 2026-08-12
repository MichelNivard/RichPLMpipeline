from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import deep_merge, resolve_config
from .coordinates import prepare_coordinate_dataset
from .data_build import build_dataset
from .metadata import atomic_json
from .model import EXPECTED_PARAMETER_COUNTS, preset_parameter_count
from .proteingym import prepare_assays
from .registry import build_lookup_index, build_uniclust_membership_index, load_registry, preflight, source_path
from .training import train_model
from .validation import validate_checkpoint


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _config(args, *, require_model: bool = False):
    paths = args.config or [package_root() / "configs" / "data" / "smoke.yaml"]
    config = resolve_config(paths)
    if getattr(args, "sources", None):
        config["sources_config"] = str(Path(args.sources).resolve())
    if getattr(args, "count", None):
        config.setdefault("data", {})["count"] = int(args.count)
    model = getattr(args, "model", None)
    if model:
        model_config = resolve_config([package_root() / "configs" / "models" / f"{model}.yaml"])
        config = deep_merge(config, {"model": model_config["model"]})
    elif require_model and "model" not in config:
        raise SystemExit("--model is required unless the config declares model.preset")
    objectives = getattr(args, "objectives", None)
    if objectives:
        names = [name.strip() for name in objectives.split(",") if name.strip()]
        config["objectives"] = {name: 1.0 for name in names}
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pipeline", description="Portable ProteinLoss production pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    check = sub.add_parser("preflight", help="List every missing external dependency")
    check.add_argument("--sources", required=True)
    check.add_argument("--capability", action="append", default=[])
    check.add_argument("--all", action="store_true")

    data = sub.add_parser("data", help="Build or inspect training data")
    data_sub = data.add_subparsers(dest="data_command", required=True)
    build = data_sub.add_parser("build")
    build.add_argument("--config", action="append")
    build.add_argument("--sources")
    build.add_argument("--count", type=int)
    worker = data_sub.add_parser("worker", help="Run one bounded production target worker")
    worker.add_argument("--config", action="append")
    worker.add_argument("--sources", required=True)
    worker.add_argument("--worker-id", required=True, type=int)
    finalize = data_sub.add_parser("finalize", help="Merge completed production workers into the indexed manifest")
    finalize.add_argument("--config", action="append")
    finalize.add_argument("--sources", required=True)

    bootstrap = sub.add_parser("bootstrap", help="Create portable source indexes")
    boot_sub = bootstrap.add_subparsers(dest="bootstrap_command", required=True)
    membership = boot_sub.add_parser("index-membership")
    membership.add_argument("--mapping", required=True)
    membership.add_argument("--output", required=True)
    lookups = boot_sub.add_parser("index-lookups")
    lookups.add_argument("--sources", required=True)

    train = sub.add_parser("train", help="Train a model preset")
    train.add_argument("--config", action="append")
    train.add_argument("--sources")
    train.add_argument("--model", choices=["100m", "300m"])
    train.add_argument("--objectives")
    train.add_argument("--run-name")
    train.add_argument("--max-steps", type=int)
    train.add_argument("--device")
    train.add_argument("--resume")

    validate = sub.add_parser("validate", help="Run held-out, ProteinGym, and MiniFold validation")
    validate.add_argument("--config", action="append")
    validate.add_argument("--model", choices=["100m", "300m"])
    validate.add_argument("--objectives")
    validate.add_argument("--checkpoint", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--proteingym-metadata")
    validate.add_argument("--proteingym-assay-dir")
    validate.add_argument("--proteingym-limit", type=int, default=0)
    validate.add_argument("--minifold", action="store_true")
    validate.add_argument("--minifold-data")
    validate.add_argument("--casp15-data")
    validate.add_argument("--pair-diagnostics", action="store_true")

    protein_gym = sub.add_parser("proteingym", help="Prepare compatible ProteinGym assays")
    pg_sub = protein_gym.add_subparsers(dest="proteingym_command", required=True)
    prepare = pg_sub.add_parser("prepare")
    prepare.add_argument("--sources")
    prepare.add_argument("--reference")
    prepare.add_argument("--bundle")
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--limit", type=int, default=0)

    model = sub.add_parser("model", help="Inspect tested model presets")
    model_sub = model.add_subparsers(dest="model_command", required=True)
    model_info = model_sub.add_parser("info")
    model_info.add_argument("preset", choices=["100m", "300m", "all"])

    structure = sub.add_parser("structure", help="Prepare experimental PDB/CASP15 coordinate data")
    structure_sub = structure.add_subparsers(dest="structure_command", required=True)
    coordinate = structure_sub.add_parser("prepare")
    coordinate.add_argument("--sources")
    coordinate.add_argument("--mode", choices=["pdb", "casp15"], required=True)
    coordinate.add_argument("--output", required=True)
    coordinate.add_argument("--cache")
    coordinate.add_argument("--seqres")
    coordinate.add_argument("--casp15-table")
    coordinate.add_argument("--download-missing", action="store_true")
    coordinate.add_argument("--min-len", type=int, default=40)
    coordinate.add_argument("--max-len", type=int, default=512)
    coordinate.add_argument("--max-resolution", type=float, default=3.0)
    coordinate.add_argument("--max-chains", type=int, default=250000)
    coordinate.add_argument("--entry-limit", type=int, default=0)
    coordinate.add_argument("--records-per-shard", type=int, default=1000)
    coordinate.add_argument("--seed", type=int, default=7)

    args = parser.parse_args(argv)
    root = package_root()
    if args.command == "preflight":
        result = preflight(args.sources, capabilities=set(args.capability or ["production_data"]), include_optional=args.all)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["ok"]:
            raise SystemExit(2)
    elif args.command == "data":
        config = _config(args)
        if args.data_command == "build":
            result = build_dataset(config, project_root=root)
        elif args.data_command == "worker":
            from .production_worker import run_production_worker
            result = run_production_worker(config, args.worker_id)
        else:
            from .production import finalize_production_build
            result = finalize_production_build(config)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "bootstrap":
        if args.bootstrap_command == "index-membership":
            result = build_uniclust_membership_index(args.mapping, args.output)
        else:
            registry = load_registry(args.sources)
            prefix = source_path(registry["sources"]["afdb_foldseek"], args.sources)
            uniprot = source_path(registry["sources"]["uniprot_mmseqs_db"], args.sources)
            result = {
                "afdb": build_lookup_index(Path(str(prefix) + ".lookup"), Path(str(prefix) + ".portable.sqlite"), foldseek=True),
                "uniprot": build_lookup_index(Path(str(uniprot) + ".lookup"), Path(str(uniprot) + ".portable.sqlite")),
            }
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "train":
        config = _config(args, require_model=True)
        overrides = {key: value for key, value in {
            "run_name": args.run_name,
            "max_steps": args.max_steps,
            "device": args.device,
            "resume": args.resume,
        }.items() if value is not None}
        config.setdefault("training", {}).update(overrides)
        result = train_model(config, project_root=root)
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "validate":
        config = _config(args, require_model=True)
        result = validate_checkpoint(
            config,
            args.checkpoint,
            args.output,
            proteingym_metadata=args.proteingym_metadata,
            proteingym_assay_dir=args.proteingym_assay_dir,
            proteingym_limit=args.proteingym_limit,
            run_minifold=args.minifold,
            minifold_data=args.minifold_data,
            casp15_data=args.casp15_data,
            pair_diagnostics=args.pair_diagnostics,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "proteingym":
        reference, bundle = args.reference, args.bundle
        if args.sources:
            registry = load_registry(args.sources)
            reference = reference or source_path(registry["sources"]["proteingym_metadata"], args.sources)
            bundle = bundle or source_path(registry["sources"]["proteingym_bundle"], args.sources)
        if not reference or not bundle:
            raise SystemExit("provide --sources or both --reference and --bundle")
        print(json.dumps(prepare_assays(reference, bundle, args.output, limit=args.limit), indent=2, sort_keys=True))
    elif args.command == "model":
        names = ["100m", "300m"] if args.preset == "all" else [args.preset]
        result = {name: {"exact_parameters": preset_parameter_count(name), "expected": EXPECTED_PARAMETER_COUNTS[name]} for name in names}
        print(json.dumps(result, indent=2, sort_keys=True))
    elif args.command == "structure":
        seqres, casp15_table, cache = args.seqres, args.casp15_table, args.cache
        if args.sources:
            registry = load_registry(args.sources)
            sources = registry["sources"]
            seqres = seqres or source_path(sources["rcsb_seqres"], args.sources)
            casp15_table = casp15_table or source_path(sources["casp15_target_table"], args.sources)
            cache_key = "rcsb_mmcif_cache" if args.mode == "pdb" else "casp15_mmcif_cache"
            cache = cache or source_path(sources[cache_key], args.sources)
        if not cache:
            raise SystemExit("provide --cache or --sources")
        print(json.dumps(prepare_coordinate_dataset(
            output_dir=args.output,
            cache_dir=cache,
            seqres_path=seqres,
            casp15_table=casp15_table,
            mode=args.mode,
            download_missing=args.download_missing,
            min_len=args.min_len,
            max_len=args.max_len,
            max_resolution=args.max_resolution,
            max_chains=args.max_chains,
            entry_limit=args.entry_limit,
            records_per_shard=args.records_per_shard,
            seed=args.seed,
        ), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
