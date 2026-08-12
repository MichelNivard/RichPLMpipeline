# Validation

`pipeline validate` produces `summary.json`, `summary.csv`, component outputs,
`REPORT.md`, and `COMPLETED`. Its components are independently optional except
held-out targets, which always run.

## Held-out targets

The checkpoint is evaluated against the `test` split. MLM and PSSM readouts are
always reported; active objectives add MI regression/correlation, distance
regression/correlation/Å MAE, contacts, latent metrics, and 3Di accuracy.
`--pair-diagnostics` additionally activates MI/distance/contact readouts for a
checkpoint whose current objective config does not request them.

## ProteinGym

`pipeline proteingym prepare` selects all compatible ProteinGym assays that:

- contain single substitutions rather than multi-mutants;
- have a canonical target sequence no longer than 512;
- contain valid wild-type positions and numeric `DMS_score` rows.

Evaluation uses masked-marginal log odds, reports each variant, and computes
Pearson and tie-aware Spearman by assay plus macro means. `--proteingym-limit 1`
is appropriate only for a smoke test; an actual report should omit the limit.

## MiniFold and CASP15

The portable MiniFold diagnostic freezes every PLM parameter. Its trainable
probe adapts residue embeddings, uses a small Transformer trunk, low-rank
distance/contact heads, and a centered CA coordinate head. Probe losses combine
log distance, contact, and coordinate-distance geometry. Reports include RMSD
after Kabsch alignment, lDDT/TM proxies, distance MAE, and contact precision@L.

Ordinary structure-backed `train` rows fit the probe and ordinary `test` rows
form `pdb_test`. A supplied CASP15 NPZ is loaded independently and reported as
`casp15_replication`; it never enters probe fitting. Production data preparation
must exclude CASP15 IDs from ordinary PDB splits.

## Acceptance commands and results

The final acceptance run uses a fresh `uv` environment, local deterministic
targets, an existing ProteinGym v1.3-compatible archive, and local AMD ROCm.
It does not download UniProt/UniClust/AFDB. The exact recorded commands are:

```bash
ROCM_WHEEL_TAG=rocm6.4 TORCH_VERSION=2.9.1 scripts/setup_rocm_env.sh
.venv/bin/pytest
.venv/bin/pipeline data build --config configs/data/smoke.yaml
.venv/bin/pipeline train --config configs/data/smoke.yaml --model 100m \
  --objectives mlm --run-name verify-100m-mlm --max-steps 2 --device cuda
.venv/bin/pipeline train --config configs/data/smoke.yaml \
  --config configs/objectives/all.yaml --model 100m \
  --run-name verify-100m-all --max-steps 2 --device cuda
.venv/bin/pipeline train --config configs/data/smoke.yaml --model 300m \
  --objectives mlm --run-name verify-300m-mlm --max-steps 2 --device cuda
.venv/bin/pipeline train --config configs/data/smoke.yaml --model 100m \
  --objectives mlm --run-name verify-100m-resume --max-steps 3 --device cuda \
  --resume smoke_artifacts/runs/verify-100m-mlm/latest.pt
.venv/bin/pipeline proteingym prepare --reference <EXISTING_METADATA> \
  --bundle <EXISTING_BUNDLE> --output smoke_artifacts/proteingym --limit 1
.venv/bin/pipeline validate --config configs/data/smoke.yaml \
  --config configs/objectives/all.yaml --model 100m \
  --checkpoint smoke_artifacts/runs/verify-100m-all/final.pt \
  --output smoke_artifacts/validation/all \
  --proteingym-metadata smoke_artifacts/proteingym/selected_assays.csv \
  --proteingym-assay-dir smoke_artifacts/proteingym/assays \
  --proteingym-limit 1 --minifold \
  --minifold-data smoke_artifacts/data/validation/pdb_coordinates \
  --casp15-data smoke_artifacts/data/validation/casp15_smoke.npz \
  --pair-diagnostics
scripts/check_portability.sh
```

The checked result values are filled from the final run below; paths use
placeholders intentionally so this document remains portable.

<!-- VERIFICATION_RESULTS_START -->
- Fresh environment: Python 3.12.3, PyTorch 2.9.1+rocm6.4; AMD Radeon
  accelerator available. Final documentation/source audit `pytest`: 13 passed
  in 1.48 seconds.
- Build: 12 attempted/written, 0 failed/skipped; train 6, valid 2, test 4;
  structure-backed 11, sequence-only 1; 3 packed batches; scratch empty and
  transient A3Ms removed.
- 100M MLM: 115,669,163 parameters, 2 optimizer steps/2 examples, 2.18 seconds,
  peak allocated/reserved 2,327.6/2,648.0 MiB.
- 100M all objectives: 115,669,163 parameters, 2 steps/2 examples, 2.30 seconds,
  peak allocated/reserved 2,344.0/2,654.0 MiB. MLM, PSSM, MI, distance,
  contact, latent, and 3Di losses/metrics were all present. The rolling
  checkpoint resumed from step 2 and completed step 3 in a separate run.
- 300M MLM: 305,588,395 parameters, 2 steps/2 examples, 3.83 seconds, peak
  allocated/reserved 5,927.6/6,214.0 MiB. Its final checkpoint reloaded on CPU
  with exactly 305,588,395 parameters at step 2.
- Unified validation of the seven-loss 100M checkpoint: 4 held-out test rows;
  one real ProteinGym assay with 922 variants (Spearman 0.071405, Pearson
  0.068715); frozen PLM MiniFold with 113,539 trainable probe parameters and 2
  steps; 4 experimental-PDB-schema test rows (RMSD 6.321 Å, distance MAE 8.339
  Å) and 2 separately loaded CASP15 replication rows (RMSD 6.198 Å, distance
  MAE 8.096 Å).
<!-- VERIFICATION_RESULTS_END -->
