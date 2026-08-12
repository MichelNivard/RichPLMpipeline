# Migration from historical ProteinLoss

The historical tree remains evidence; this package does not move or delete it.
The extraction uses the 5M dense-pair route as the primary reference and retains
only coherent production concepts.

## Retained

- transient batch A3Ms and MMseqs scratch;
- PSSM plus compressed dense APC-MI;
- packed binary targets with offsets rather than loose files;
- fp16 CA primitives decoded to log distance on load;
- packed 3Di and online frozen StructEncoder targets;
- the established 16×768 100M transformer and compatible current heads;
- ProteinGym masked-marginal scoring;
- frozen-embedding MiniFold structure probing.

## Replaced or excluded

- Absolute manifest paths become data-root-relative paths.
- Full CSV manifests become indexed SQLite plus streaming JSONL.
- The all-train 5M split becomes mandatory stable family/accession hashing with
  non-empty train/valid/test validation.
- Repeated model/target interfaces become tested YAML presets and one CLI.
- Machine-specific shell/scheduler/systemd launchers become plain shell and
  portable Slurm examples.
- Engram, pair-bias experiments, obsolete sparse targets, 15k/30k launchers,
  plotting experiments, archives, logs, and historical outputs are excluded.

## Existing packed data

Do not point training at the historical 5M manifest unchanged: its 3,955,208
rows are marked train and it has no validation examples. A safe migration tool
must stream old rows, resolve every target path relative to a selected new data
root, verify byte records/checksums, assign a stable family-aware split, and
write the new schema. Because old manifests contain workstation-specific paths
and may not expose a family ID, such conversion is site-specific and is not
silently automated.

Prefer rebuilding the manifest around already packed immutable shards rather
than copying loose targets. When no family map exists, accession hashing is the
documented fallback and has more homolog leakage risk.

## Checkpoint compatibility

The 100M preset matches the mature dimensions and compatibility tensors.
`load_checkpoint_model` strips a compilation prefix and tolerates only the 3Di
head missing from older checkpoints. Any other missing/unexpected state is an
error. Always validate a migrated checkpoint on a newly split held-out set;
historical zero-validation reports are not acceptable evidence.
