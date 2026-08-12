# Inventory and extraction plan

This document preserves the inventory and extraction decisions made before the
portable implementation was built. It records which historical components are
evidence for the production design and which parts were deliberately excluded.

## Production evidence retained

| Concern | Mature source | Extraction decision |
|---|---|---|
| Single-sequence model and heads | `src/model.py`, `scripts/train_dense_pair_multiloss.py` | Keep the checkpoint-compatible Transformer, residue head, low-rank dense MI/distance/contact heads, StructEncoder latent head, direct 3Di head, and online MLM masking. Remove model-kind switches for engram and pair-bias experiments. |
| Dense MI codec | `src/compressed_target_codec.py`, `src/dense_pair_targets.py` | Vendor the tested FFT4 + block32 residual + diagonal residual + eig3 residual codec. Enforce symmetric, zero-diagonal decode. |
| Structure target | `src/foldseek_ca.py`, 5M target builder | Keep Foldseek CA decoding and fp16 CA storage; reconstruct `log1p(clamp(cdist, 64))` while loading. Derive contacts from the same distance target. |
| MSA targets | `src/a3m.py`, `src/targets.py`, `scripts/run_uniclust30_family_batched_msa.py` | Keep transient A3M reduction into sequence, PSSM, and compressed dense APC-MI. Preserve the UniClust30 representative search, membership expansion, UniProtKB realignment, and cleanup contract behind a batch worker interface. |
| Packed storage | 5M `*.bin` shards and CSV byte offsets | Keep concatenated NPZ records with byte offsets, but write paths relative to a configurable data root and inventory every shard with SHA-256. |
| StructEncoder targets | 5M packed 3Di builder and online teacher path | Store packed 3Di IDs. Generate contextual 128-dimensional targets online through the frozen teacher unless an explicitly compatible precomputed latent exists. |
| Training lifecycle | 5M trainer and launcher | Keep separate raw/weighted metrics, rolling `latest.pt`, atomic final checkpoint, resume counters, source revision, seed, throughput, elapsed time, and peak memory. Require train/validation/test before training. |
| Mutation validation | ProteinGym preparation and mutation benchmark scripts | Keep masked-marginal single-substitution scoring and per-assay/aggregate Spearman outputs. |
| Structure validation | coordinate preparation, `src/minifold.py`, MiniFold trainer | Keep packed experimental CA coordinates, frozen PLM embeddings, a small reproducible probe, and a separate CASP15 replication manifest. |

## Historical material excluded

- Engram, pair-bias, residual top-k, and obsolete sparse-target experiments.
- Old 15k/30k/200k launch queues, systemd units, machine-specific GPU queues,
  live plotting loops, archived reports, and historical outputs.
- Training data, existing targets, checkpoints, database files, logs, figures,
  and caches.
- Absolute paths and mutable paths embedded in manifests.

Dormant parameter tensors that are required to load the established 5M
checkpoints are retained in the model state schema, but they are not exposed as
pipeline objectives.

## Defects corrected during extraction

1. Replace the 5M all-train split with deterministic `train`, `valid`, and
   `test` assignments. A stable family/cluster key is preferred; accession is
   the fallback. Preflight rejects any empty required split.
2. Replace absolute manifest paths with POSIX paths relative to `data_root`.
3. Replace whole-CSV loading and global Python sorts with an indexed SQLite
   manifest plus independently consumable JSONL partitions.
4. Make every external source resolve through one YAML registry plus documented
   environment overrides.
5. Make batch stages atomic, resumable, idempotent, and auditable through
   resolved configuration, counts, failures, timings, inventories, and a final
   completion marker.

## Implemented package plan

1. Built a typed configuration/source registry, bootstrap helpers, and a
   preflight command that reports all missing contracts in one pass.
2. Vendored and tested the target codecs, packed-shard reader/writer, relative-path
   manifest, stable splitting, Foldseek readers, and streaming FASTA/A3M
   reduction.
3. Exposed `pipeline data build` with `smoke` and `production` backends. The
   production orchestrator emits bounded batch work plans; local workers and
   Slurm arrays execute the same idempotent batch command.
4. Added explicit `100m` and `300m` model presets, configurable objective weights,
   mixed-supervision masks, training/resume/checkpoint logic, and held-out
   metrics.
5. Added one `pipeline validate` orchestration command for held-out targets,
   ProteinGym, MiniFold, CASP15, and optional pair diagnostics.
6. Added local ROCm and Slurm launch examples, data movement verification, source
   contracts, scaling/storage guidance, and migration notes.
7. Verified in a fresh uv environment with unit tests and an end-to-end 5-20
   protein smoke build, two-step model exercises, held-out evaluation, one
   ProteinGym assay, a tiny MiniFold probe, scratch cleanup, and an absolute-path
   scan. Record exact commands and results in the README.

## Scale and population contract

The `--count` value is a desired unique sequence population, not permission to
duplicate AFDB records. Structure-supervised examples are capped by the chosen
AFDB/Foldseek release. Larger populations are mixed: all valid sequences receive
MLM, MSA-backed sequences may receive PSSM/MI, and only structure-backed
sequences receive distance/contact/3Di/latent losses. Objective availability is
stored per example and masked independently during training.
