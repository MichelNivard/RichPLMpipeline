# Training

## Objective configuration

YAML maps each objective to an independent floating-point weight. Zero disables
its loss and avoids its expensive head/teacher computation.

- `mlm`: online 15% masking, 80% mask / 10% random / 10% unchanged.
- `pssm`: soft cross-entropy against MSA amino-acid frequencies.
- `dense_mi`: MSE plus Pearson/RMSE readouts for compressed APC-MI.
- `distance`: log-distance MSE plus configurable correlation penalty.
- `contact`: weighted BCE; labels are derived from CA distance.
- `struct_latent`: MSE to online frozen StructEncoder latents, plus cosine and
  decoded-3Di metrics.
- `3di`: token cross-entropy and accuracy.

`configs/objectives/mlm.yaml`, `msa.yaml`, and `all.yaml` are explicit presets.
The CLI `--objectives a,b,c` is convenient but assigns weight 1.0 to each; use
YAML for tuned weights.

## Correctness preflight

Before model allocation, training validates the indexed manifest, requires
non-empty train/valid/test splits, and counts eligible rows per active target in
every split. Missing PSSM/MI/structure/3Di coverage is a hard error. This
specifically prevents recurrence of the historical 3,955,208-row, all-train 5M
manifest and its zero-example validation report.

## Optimizer and precision

The default implementation is single-process AdamW with gradient clipping,
configurable accumulation, fp16/bfloat16 autocast, and scaler support. AMD ROCm
uses the normal PyTorch `cuda` device API. Cluster launchers provide a clean
place to wrap the same command in `torchrun` if distributed support is added;
this release does not claim multi-node DDP.

At one 512-residue example per GPU, the 100M preset is the practical
single-device baseline. The 300M preset generally needs a higher-memory GPU,
shorter sequences, activation checkpointing (future), or CPU construction-only
verification. Always measure peak memory on the target PyTorch/driver version.

## Logging contract

`steps.csv` and `steps.jsonl` contain epoch, optimizer step, cumulative
examples, elapsed time, examples/second, peak GPU allocated and reserved MB,
every raw loss, every weighted contribution, weighted total, and applicable
accuracy/correlation/regression/contact/latent metrics. The run also records:

- resolved model, data, objective, optimizer, and seed configuration;
- Git revision/dirty state when available, Python/platform/Torch/device data;
- manifest and teacher SHA-256;
- objective coverage preflight and final validation metrics.

## Checkpoints and resume

`latest.pt` contains model, optimizer, AMP scaler, Python/NumPy/Torch/CUDA RNG,
step/example/epoch counters, resolved config, and metadata. It is rewritten
atomically and is always emitted even for a short run. `final.pt` is the compact
model/config/metadata checkpoint. `COMPLETED` appears only after held-out
validation and both hashes are recorded.

Pass `--resume /path/to/latest.pt`. Keep the original data manifest and resolved
configuration. A resumed job restores RNG and optimizer state; changing batch
topology or worker count can still change future sample order, so record such a
change as a new run lineage.
