# Data format

## Indexed manifest

`manifest.sqlite` is authoritative. `manifest.jsonl` is a streaming interchange
view, not a training dependency. Each `examples` row contains:

- `example_id`, stable `group_id`, and deterministic `split`;
- `split_rank`, an indexed integer used for bounded random access;
- relative `shard_path`, `byte_offset`, `byte_length`, and record SHA-256;
- sequence length and `has_pssm`, `has_mi`, `has_structure`, `has_3di` flags;
- compact JSON provenance.

Paths are POSIX-style and relative to `DATA_ROOT`. The writer rejects absolute
paths and `..`. Moving all of `DATA_ROOT` therefore requires no manifest edit.
SQLite indexes support split and supervision queries at 500M scale. Builders
stream partitions/manifests and do no global Python sort.

## Packed shards

A shard is a binary concatenation of independently compressed NPZ records. Its
JSONL sidecar maps each record to a byte range and checksum; `inventory.json`
lists every shard and checksum. This preserves random access while avoiding
millions of filesystem entries. A worker writes under `.part`, validates it,
then atomically renames the directory.

Each record may contain:

| array | encoding | availability |
|---|---|---|
| `input_ids` | int16, padded to `max_len` | always |
| `attention_mask` | bool | always |
| `pssm_log_probs` | float16, `[max_len,20]` | MSA-backed |
| `gap_fraction` | float16 | MSA-backed |
| MI components | compressed codec arrays | MSA-backed |
| `ca_coords_fp16` | fp16 `[L,3]` | structure-backed |
| `three_di_ids` | uint8, padded with 20 | structure-backed |
| `valid_length`, `msa_depth` | scalar | always/MSA-backed |

## Dense MI codec

The target is 21-state mutual information computed from an aligned MSA and
corrected by average product correction (APC). It is symmetric with an exact
zero diagonal. Compression stores a low-frequency real FFT block, quantized
32×32 residual blocks, diagonal-line residuals, and three eigen components.
Loading reconstructs a dense float matrix, symmetrizes it, and resets the
diagonal. Unit tests enforce shape, finite values, symmetry, diagonal, and
bounded reconstruction error.

## Structure targets

CA coordinates are the portable primitive. The loader computes
`log1p(clamp(cdist(CA), 64 Å))`, symmetrizes, and resets its diagonal. Contacts
are derived online from unlogged distance `<8 Å` (configurable) and are not
stored separately. Missing structure is represented by availability flags—not
zero-valued supervision.

3Di is stored as token IDs. The frozen StructEncoder consumes those IDs during
training to create a 128-dimensional target in blocks; the shards do not store
the dense latent target.

## Experimental coordinate benchmark

MiniFold does not train from the AFDB training-target manifest. Its separate
coordinate root has `manifest.sqlite`, packed NPZ byte records, relative shard
paths and checksums. Each record holds padded sequence IDs/mask and observed CA
coordinates plus PDB method/resolution provenance. Ordinary experimental rows
use train/valid/test; a separate root contains only `casp15`. The smoke builder
creates the same schema with deterministic fixture coordinates.

## Splits

SHA-256 of `seed + group_id` maps a group to `[0,1)`. The bottom test fraction
is `test`, the next validation fraction is `valid`, and the remainder is
`train`. When no group is known the accession is the group. Smoke construction
uses paired synthetic families; production uses the UniClust membership inverse
index. All three splits must be non-empty.

The historical 5M all-train manifest is intentionally not copied. Migration
must assign a real stable split before training.
