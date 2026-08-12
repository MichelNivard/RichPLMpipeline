# ProteinLoss

ProteinLoss is a pipeline for training better **single-sequence protein language
models** by using evolutionary and structural information as supervision.

A protein is written as one amino-acid sequence, but much of what makes that
sequence meaningful is easier to see in other data:

- homologous sequences reveal which amino acids evolution tolerates;
- correlated changes across homologs reveal coupled residue positions;
- protein structures reveal long-range geometry and local structural states;
- mutation experiments reveal whether a model's scores track biological
  function.

The usual choices are to ignore those resources and train only on raw sequence,
or to give an MSA/structure to the model every time it is used. ProteinLoss
tests a third route:

> Use MSAs and structures as training targets, not as inference inputs.

During training, a shared Transformer is asked to recover masked amino acids,
evolutionary profiles, coevolution maps, distances, contacts, and structural
representations. After training, the teachers disappear. Embedding a protein or
scoring a mutation still requires only an amino-acid sequence and a checkpoint.

This repository contains the complete portable implementation: public-source
contracts, target builders, packed data formats, 100M/300M model presets,
training and resume logic, validation, tests, local ROCm launchers, and Slurm
examples. It deliberately contains no source databases, training targets,
checkpoints, logs, or historical run outputs.

## The scientific idea

Masked-language modelling (MLM) asks a useful but narrow question: given one
sequence with a residue hidden, which amino acid belongs there? A sufficiently
large sequence model can learn a great deal this way, but it has to rediscover
signals that an MSA or structure makes explicit.

ProteinLoss asks several biological questions of the same hidden states:

| objective | question posed to the model | teacher information |
|---|---|---|
| MLM | Which residue fits this sequence context? | the original sequence |
| PSSM | Which amino acids are evolutionarily tolerated here? | an MSA column |
| dense MI | Which residue positions co-vary beyond background conservation? | pairs of MSA columns |
| distance | How far apart are two residues in the fold? | C-alpha coordinates |
| contact | Are two nonlocal residues spatially close? | distance threshold |
| StructEncoder latent | What contextual local structural environment does this residue occupy? | frozen transformer over 3Di |
| 3Di | Which local structural state does this residue occupy? | Foldseek's structural alphabet |

A **position-specific scoring matrix (PSSM)** is used here as a 20-amino-acid
probability profile for every aligned position. **Mutual information (MI)**
measures statistical dependence between two alignment columns. Average product
correction (APC) subtracts broad conservation/phylogeny background so the
remaining value is a more useful coupling target. MI is not simply a contact
map: it can reflect physical packing, functional coupling, compensation,
phylogeny, or a shallow/noisy alignment.

The objectives are not separate post-hoc probes. Their gradients train the same
sequence trunk from the beginning. The hypothesis is that a representation
which must explain these complementary views will be more biologically useful
than one trained on masked residues alone.

For example, a wild-type leucine at one position gives MLM one correct label.
An MSA may additionally show that leucine, isoleucine, and valine are tolerated
there but charged residues are not. MI may show that this site changes only
when a distant site compensates. A structure may show that the two sites pack
together in the protein core. ProteinLoss turns each of those observations into
a training signal while keeping the deployed input unchanged.

### Why this particular target mix

The design grew out of a controlled ladder of MLM-only, evolutionary, sparse
pair, structure, and finally dense-pair models with the same 110M-class trunk.
The later dense route was retained because it delivered much more supervision
per protein without changing inference. In the historical experiments that
motivated this extraction, the shared trunk could optimize all of the tasks at
once, and dense MI plus distance helped not only their own heads but also MLM,
PSSM, structure-latent, and 3Di readouts.

The strongest completed historical 5M configuration—MLM, PSSM, dense MI,
dense distance, StructEncoder latent, and 3Di—was evaluated on 97 compatible
ProteinGym assays containing 316,736 valid substitutions. Its reported mean
Spearman was 0.1392, compared with 0.0305 for the same-scale MLM-only model and
0.0541 for MLM + PSSM + dense MI. These are motivating results from the
pre-portability run, not outputs reproduced by the tiny smoke test in this
repository. That old 5M manifest also had no held-out rows, which is why this
implementation now requires deterministic non-empty validation and test
splits.

An explicit contact head did not improve the earlier dense-distance model under
simple equal loss weighting. It remains independently configurable because it
is a useful diagnostic and may behave differently with other weights or data,
but distance is the richer geometric teacher and contacts can always be derived
from it.

## What goes in, and why

There are three classes of data: training population, training teachers, and
independent validation. They should not be conflated.

### 1. UniProtKB: the sequence population

[UniProtKB](https://www.uniprot.org/help/downloads) supplies the amino-acid
sequences. The same pinned release is used in two forms:

- a streaming FASTA from which unique proteins of length at most 512 are
  selected; and
- an MMseqs database from which homologous family members are retrieved.

Every selected sequence supports online MLM. If its homology search produces a
usable alignment, it also supports PSSM and MI. The release must be pinned:
`current_release` is convenient for downloading but is not a reproducible
version identifier.

### 2. UniClust30: a fast route to protein families

[UniClust30](https://uniclust.mmseqs.com/) groups related UniProt sequences and
provides representative sequences plus membership mappings. Searching all of
UniProt independently for every query is prohibitively slow. The production
route instead:

1. searches the smaller UniClust30 representative database;
2. expands matching representatives back to their member accessions;
3. retrieves those members from the pinned UniProtKB database;
4. realigns them to the query; and
5. creates a temporary A3M alignment.

The representative search is an accelerator, not the final MSA. Expansion and
realignment preserve within-family variation, which is the information needed
for PSSM and MI. A3Ms and batch MMseqs databases are scratch: after they are
reduced into compact targets, they are deleted.

The mature workflow uses UniClust30 2018_08 for experimental comparability. It
is old and may require a mirror. Replacing it is valid, but changes the teacher
distribution and must be recorded as a new data version.

### 3. AlphaFold DB through Foldseek: structure teachers at scale

The [AlphaFold Protein Structure Database](https://alphafold.ebi.ac.uk/download)
provides broad predicted-structure coverage. [Foldseek](https://github.com/steineggerlab/foldseek)
stores those structures in databases that support indexed access to:

- the corresponding amino-acid sequence;
- C-alpha coordinates; and
- one 3Di structural token per residue.

The C-alpha atom provides one coordinate per residue, giving a compact trace of
the protein backbone. It is enough to construct residue-residue distances
without storing a full atomic model in every training record.

C-alpha coordinates create the dense log-distance target. Contacts are derived
from the same distances, so a second contact dataset is unnecessary. The 3Di
tokens provide direct local-structure labels and input to the frozen
StructEncoder teacher.

These are **training teachers**, not experimental ground truth for the
structure benchmark. The benchmark uses RCSB structures separately so that it
does not merely test recovery of AlphaFold-derived labels.

### 4. StructEncoder: a contextual structure teacher

Foldseek 3Di is a small categorical alphabet. The project StructEncoder is a
frozen Transformer trained over 3Di sequences. It turns each token into a
128-dimensional contextual representation, allowing the amino-acid model to
learn more than a local class label.

The teacher checkpoint is a project artifact and is not redistributed here.
The mature checkpoint contract and SHA-256 are documented in
[data sources](docs/DATA_SOURCES.md). During training the teacher encodes 3Di in
bounded blocks. Its dense residue latents are never stored in the dataset.

### 5. ProteinGym: mutation-effect validation

[ProteinGym](https://github.com/OATML-Markslab/ProteinGym) contains experimental
deep-mutational-scanning assays. It is not a training source. For every
compatible single substitution in a protein no longer than 512 residues, the
pipeline scores:

```text
log p(mutant amino acid | masked sequence)
- log p(wild-type amino acid | masked sequence)
```

Assay-level Spearman correlation tests whether a sequence-only checkpoint ranks
mutations in the same direction as experimental functional effects. No MSA,
structure, or assay-specific fitting is supplied at evaluation time.

### 6. RCSB PDB and CASP15: representation-level structure validation

The MiniFold benchmark uses observed C-alpha coordinates from experimental
[RCSB PDB](https://www.rcsb.org/) mmCIF structures. The PLM is frozen and the
same small coordinate probe is trained on top of its embeddings. This asks a
different question from ProteinGym: does the representation already contain
geometry that a fixed-capacity structure head can use?

CASP15 targets are excluded from the ordinary PDB train/validation/test set and
prepared as a separate replication split. ProteinGym, PDB, and CASP15 therefore
measure downstream behaviour independently of the AFDB target construction.

The full source registry, upstream URLs, versions, checksum policy, conversion
commands, and local file contracts are in
[docs/DATA_SOURCES.md](docs/DATA_SOURCES.md).

## From public data to a checkpoint

```text
UniProtKB FASTA
  -> stream valid unique sequences (length <= 512)
  -> look up UniClust family and AFDB availability
  -> stable family-aware train / valid / test split
  -> bounded source partitions

each worker partition
  -> MMseqs search against UniClust30 representatives
  -> expand representatives to UniProtKB members
  -> realign members and create transient A3Ms
  -> reduce A3Ms to PSSM + dense APC-corrected MI
  -> read matching Foldseek CA + 3Di records when available
  -> write compressed records into packed binary shards
  -> delete A3M and MMseqs scratch

completed workers
  -> indexed SQLite manifest with relative shard paths
  -> validated train / valid / test coverage
  -> 100M or 300M multi-objective training
  -> rolling resumable and final checkpoints
  -> held-out + ProteinGym + MiniFold/PDB + separate CASP15 reports
```

The planner streams the source FASTA and performs indexed SQLite lookups. It
does not create a 500M-row in-memory table or globally sort the population.
Workers are bounded and independent, so the same work plan can run in a local
loop or a Slurm array. Finalization streams worker manifests into indexed
SQLite and refuses missing workers, empty splits, or an unexplained shortfall.

## Why the target formats look unusual

At this scale, a mathematically simple representation can be an operationally
impossible one.

For a maximum-length protein, one dense `512 x 512` fp16 matrix is 0.5 MiB. At
five million proteins, storing just one such matrix would require about 2.6 TB;
at 500 million it would require about 262 TB. MI and distance would each incur
that cost, before sequences, profiles, indexes, replication, or checkpoints.
Millions of individual files would also overwhelm filesystem metadata and make
distributed training I/O erratic.

ProteinLoss preserves dense supervision while avoiding dense persistent maps:

| target | naive persistent form | stored form | reconstruction |
|---|---|---|---|
| PSSM | `L x 20` profile | fp16 profile | directly loaded |
| dense APC-MI | `L x L` float matrix | FFT low frequencies + 32×32 block residuals + diagonal-line residuals + three eigen components | decoded per batch, symmetrized, diagonal reset |
| distance | `L x L` matrix | fp16 C-alpha coordinates, `L x 3` | `log1p(cdist)` computed per batch |
| contact | another `L x L` map | nothing extra | threshold reconstructed distance, normally at 8 Å |
| 3Di | token per residue | packed uint8 IDs | directly loaded |
| StructEncoder latent | `L x 128` float vectors | nothing extra | frozen teacher encodes stored 3Di online |

Distances are capped and transformed with `log1p` before regression. This
reduces the leverage of very large separations while retaining resolution among
nearby residues, where fold and contact geometry are most informative.

The savings are large. At length 512, fp16 C-alpha coordinates occupy about
3 KiB rather than 0.5 MiB for a dense fp16 distance map. A stored fp16
StructEncoder latent would be about 128 KiB per protein—roughly 655 GB for five
million examples—so online teacher inference trades bounded compute for a major
storage and I/O reduction.

Compressed MI is intentionally a storage codec, not a sparse training target.
The loader reconstructs a full symmetric field and the model learns against
all eligible nonlocal pairs. The codec retains broad low-frequency structure,
local block residuals, near-diagonal patterns, and dominant low-rank residuals.
Tests enforce finite values, symmetry, a zero diagonal, and bounded
reconstruction error.

Records are concatenated into binary shards. A small index stores each record's
relative shard path, byte offset, byte length, and checksum. This gives random
access without millions of loose files and allows the whole dataset directory
to move to another machine unchanged.

See [docs/DATA_FORMAT.md](docs/DATA_FORMAT.md) for the exact schema and codec
contract.

## Mixed supervision and the real population ceiling

Not every valid UniProt sequence has every teacher:

1. every selected sequence can train MLM;
2. a usable MSA is required for PSSM and MI;
3. a matching Foldseek CA/3Di record is required for distance, contact, 3Di,
   and StructEncoder latent targets.

The third population is bounded by the chosen AFDB/Foldseek release and is much
smaller than an aspirational 500M unique-sequence corpus. The pipeline does not
silently duplicate structure-backed records to reach a requested count.
Instead, the manifest records target availability per example and losses are
masked independently. MSA-backed, structure-missing examples still contribute
MLM/PSSM/MI; structure-backed examples contribute the additional geometry and
structure objectives.

If selection cannot reach the requested number of unique sequences, planning
reports the population ceiling. If worker failures create a target shortfall,
finalization withholds `COMPLETED` unless the operator explicitly accepts it.

## Quickstart: exercise the complete design without downloading databases

The smoke backend creates 12 deterministic proteins and synthetic teachers. It
uses the same manifest, codecs, packed shards, mixed-supervision masks,
training, and validation interfaces as production. It is a software acceptance
test, not a miniature claim about biological performance or remote source
availability.

Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/) are required.

For conventional Linux CPU/NVIDIA:

```bash
git clone https://github.com/MichelNivard/RichPLMpipeline.git
cd RichPLMpipeline
uv sync --extra test
uv run pytest
uv run pipeline data build --config configs/data/smoke.yaml
uv run pipeline train \
  --config configs/data/smoke.yaml \
  --model 100m --objectives mlm \
  --run-name smoke-100m --max-steps 2 --device auto
uv run pipeline validate \
  --config configs/data/smoke.yaml \
  --model 100m --objectives mlm \
  --checkpoint smoke_artifacts/runs/smoke-100m/final.pt \
  --output smoke_artifacts/validation/heldout
```

For AMD ROCm, install the accelerator-specific Torch wheel through the provided
script. PyTorch exposes an AMD device through its `cuda` API. Invoke the created
environment directly so a later default-index sync does not replace ROCm Torch:

```bash
ROCM_WHEEL_TAG=rocm6.4 TORCH_VERSION=2.9.1 scripts/setup_rocm_env.sh
.venv/bin/pytest
.venv/bin/pipeline data build --config configs/data/smoke.yaml
.venv/bin/pipeline train \
  --config configs/data/smoke.yaml \
  --config configs/cluster/local_rocm.yaml \
  --model 100m --objectives mlm \
  --run-name smoke-100m --max-steps 2 --device auto
.venv/bin/pipeline validate \
  --config configs/data/smoke.yaml \
  --model 100m --objectives mlm \
  --checkpoint smoke_artifacts/runs/smoke-100m/final.pt \
  --output smoke_artifacts/validation/heldout
```

The build must report non-empty train/valid/test splits, mixed structure
coverage, and `scratch_remaining: []`. Delete `smoke_artifacts/` afterward.

## Building a dataset from public sources

### 1. Create one source registry

All external paths and versions live in one YAML file. Copy the example outside
version control, fill its environment variables, and pin mutable releases:

```bash
cp configs/sources.example.yaml sources.local.yaml
export PROTEINLOSS_SOURCE_ROOT=/path/to/external/sources
scripts/bootstrap_public_sources.sh          # dry-run plan
scripts/bootstrap_public_sources.sh --execute
```

The bootstrap script may download redistributable UniProtKB and ProteinGym
inputs. It deliberately does not initiate the large AFDB/Foldseek download or
invent a StructEncoder checkpoint. Those require an explicit storage/version
decision. For sources that cannot be bootstrapped, the detailed source guide
defines the exact required local artifact.

Preflight reports every missing source/tool in one response:

```bash
uv run pipeline preflight --sources sources.local.yaml --all
```

The source registry includes UniProtKB FASTA and MMseqs DB, UniClust30 seed and
membership index, AFDB/Foldseek sequence/lookup/CA/3Di stores, StructEncoder,
ProteinGym, RCSB/CASP15 inputs, MMseqs, and Foldseek.

### 2. Build portable indexes

Large membership and lookup tables are converted once into streaming-friendly
SQLite indexes:

```bash
uv run pipeline bootstrap index-membership \
  --mapping /staged/uniclust_members.tsv.gz \
  --output /staged/uniclust_members.sqlite
uv run pipeline bootstrap index-lookups --sources sources.local.yaml
```

### 3. Plan, execute, and finalize

```bash
export PROTEINLOSS_SOURCES_CONFIG=$PWD/sources.local.yaml
export PROTEINLOSS_DATA_ROOT=/data/proteinloss/5m
export PROTEINLOSS_SCRATCH_ROOT=/scratch/proteinloss/5m

uv run pipeline data build \
  --config configs/data/5m.yaml \
  --sources "$PROTEINLOSS_SOURCES_CONFIG"
```

`data build` is the streaming planner. It writes bounded source partitions and
`work_plan.jsonl`; it does not run thousands of expensive searches in the
planning process. Run one worker per plan row locally:

```bash
workers=$(wc -l < "$PROTEINLOSS_DATA_ROOT/work_plan.jsonl")
for worker in $(seq 0 $((workers - 1))); do
  uv run pipeline data worker \
    --config configs/data/5m.yaml \
    --sources "$PROTEINLOSS_SOURCES_CONFIG" \
    --worker-id "$worker"
done

uv run pipeline data finalize \
  --config configs/data/5m.yaml \
  --sources "$PROTEINLOSS_SOURCES_CONFIG"
```

For a cluster, submit `examples/slurm/data_array.sbatch` over the worker range,
then finalize only after the array is complete. The same worker code is used in
both cases; see [docs/CLUSTER.md](docs/CLUSTER.md).

### 4. Choose the scale

```bash
uv run pipeline data build --config configs/data/5m.yaml \
  --sources "$PROTEINLOSS_SOURCES_CONFIG"
uv run pipeline data build --config configs/data/50m.yaml \
  --sources "$PROTEINLOSS_SOURCES_CONFIG"
uv run pipeline data build --config configs/data/500m.yaml \
  --sources "$PROTEINLOSS_SOURCES_CONFIG"

# Exact one-off override:
uv run pipeline data build --config configs/data/5m.yaml \
  --count 5000000 --sources "$PROTEINLOSS_SOURCES_CONFIG"
```

The same commands and formats apply, but the hardware plan does not. A 500M
configuration is a distributed streaming contract—not a claim that one node
can finish it or that 500M structure-backed proteins exist.

## Where builds spend time and disk

The dominant cost is constructing MSA-derived targets, not extracting C-alpha
coordinates. Historical dense-route measurements provide useful orders of
magnitude, although different databases, MMseqs builds, storage, and worker
sizes will change them:

| measured stage | historical rate | why it is slow |
|---|---:|---|
| representative search | 24.8 h for 1.3M queries (~19.1 h/M) | MMseqs prefilter against seed representatives |
| family expansion + MSA + dense targets | 23.9 h for 1M queries | member DB slicing, `result2msa`, MI and packed writing |
| older structure-side targets | ~4.7 h for 1M | indexed CA/3Di lookup and teacher-related reduction |

Inside the measured dense MSA stage, `result2msa` and local member-database
creation each consumed about 35% of wall time; PSSM/MI reduction and encoding
consumed about 19%; membership/accession lookup consumed about 7%. In the
separate representative search, MMseqs prefilter was about 22.2 of 24.8 hours.

The practical consequences are:

- Put MMseqs source databases and scratch on fast local NVMe when possible.
- Distribute representative searches and independent worker partitions.
- Keep membership/accession SQLite indexes warm and reuse compatible search
  results rather than paying the first-stage search repeatedly.
- Treat `result2msa` and per-worker database construction as first-class
  bottlenecks; increasing GPUs alone does not eliminate them.
- Batch MI work and measure CPU versus accelerator reduction on the actual
  cluster. Exact 21-state MI scales quadratically with sequence length.
- Do not retain A3Ms or worker MMseqs databases. They are reproducible scratch
  and can dwarf the final targets.
- Measure the first completed packed shards before reserving full capacity.
  Target size depends strongly on length and MI compressibility.

The mature dense experiment wrote about 8.8 GB of compressed pair records per
one million queries (about 11 GB as a complete target directory). A broader
planning range for all packed targets is roughly 10–40 KB per accepted example:
about 0.05–0.2 TB for 5M, 0.5–2 TB for 50M, and 5–20 TB for 500M, before
replication. These are capacity estimates, not quotas or guarantees.

At the historical single-machine rate, fresh representative search plus dense
target construction was roughly 43 hours per million proteins. A naive linear
5M run was therefore about nine days; 50M and 500M require sharding and
parallelism rather than patience on one workstation. See
[docs/SCALING.md](docs/SCALING.md) for the distributed contract and remaining
bottlenecks.

## Training models

The model always receives amino-acid tokens and an attention mask. All teacher
data are targets, never encoder inputs.

| preset | maximum length | width | layers | heads | feed-forward | pair rank | exact all-head parameters |
|---|---:|---:|---:|---:|---:|---:|---:|
| `100m` | 512 | 768 | 16 | 12 | 3072 | 128 | 115,669,163 |
| `300m` | 512 | 1024 | 24 | 16 | 4096 | 128 | 305,588,395 |

The names describe their approximate model class. Exact counts include every
current head and are enforced by tests.

```bash
uv run pipeline model info all

export PROTEINLOSS_DATA_ROOT=/data/proteinloss/5m
export PROTEINLOSS_RUN_ROOT=/runs/proteinloss/5m

# MLM-only baseline
uv run pipeline train --config configs/data/5m.yaml \
  --config configs/objectives/mlm.yaml \
  --config configs/cluster/slurm.yaml \
  --model 100m --run-name 100m-mlm

# Evolutionary sequence + pair supervision
uv run pipeline train --config configs/data/5m.yaml \
  --config configs/objectives/msa.yaml \
  --config configs/cluster/slurm.yaml \
  --model 100m --run-name 100m-msa

# Every implemented objective on the 300M preset
uv run pipeline train --config configs/data/50m.yaml \
  --config configs/objectives/all.yaml \
  --config configs/cluster/slurm.yaml \
  --model 300m --run-name 300m-all
```

Every objective weight is independently configurable in YAML. A CLI list such
as `--objectives mlm,pssm,dense_mi,distance,3di` is shorthand for weight 1.0
on each named objective. Availability masks ensure a structure loss is not
applied to a sequence without structure targets.

Training writes separate raw losses and weighted contributions, target-specific
metrics, optimizer steps and cumulative examples, throughput, elapsed time,
peak GPU allocation/reservation, seed and source revision metadata, a rolling
optimizer-resumable checkpoint, and a final checkpoint. See
[docs/TRAINING.md](docs/TRAINING.md).

### Resume a stopped run

```bash
uv run pipeline train --config configs/data/5m.yaml \
  --model 100m --objectives mlm,pssm,dense_mi \
  --run-name 100m-msa \
  --resume "$PROTEINLOSS_RUN_ROOT/100m-msa/latest.pt"
```

`latest.pt` includes model, optimizer, AMP scaler, RNG state, epoch, step, and
example counters. Checkpoints are written atomically. `COMPLETED` is created
only after the run's held-out evaluation succeeds.

## Validating what the representations learned

One command can produce held-out, mutation-effect, and structure-probe reports:

```bash
uv run pipeline proteingym prepare \
  --sources sources.local.yaml \
  --output validation_inputs/proteingym

uv run pipeline structure prepare --mode pdb \
  --sources sources.local.yaml \
  --output validation_inputs/pdb_coordinates \
  --download-missing

uv run pipeline structure prepare --mode casp15 \
  --sources sources.local.yaml \
  --output validation_inputs/casp15_coordinates \
  --download-missing

uv run pipeline validate \
  --config configs/data/5m.yaml \
  --config configs/objectives/all.yaml \
  --model 100m \
  --checkpoint "$PROTEINLOSS_RUN_ROOT/100m-all/final.pt" \
  --output validation/100m-all \
  --proteingym-metadata validation_inputs/proteingym/selected_assays.csv \
  --proteingym-assay-dir validation_inputs/proteingym/assays \
  --minifold \
  --minifold-data validation_inputs/pdb_coordinates \
  --casp15-data validation_inputs/casp15_coordinates \
  --pair-diagnostics
```

The components answer different questions:

| component | what it tests | important separation |
|---|---|---|
| held-out targets | generalization of MLM/PSSM/MI/distance/contact/latent/3Di heads | deterministic non-empty family-aware test split |
| ProteinGym | sequence-only mutation-effect ranking against experiments | no MSA/structure or assay fitting at inference |
| MiniFold PDB | whether frozen PLM embeddings support a coordinate probe | experimental structures separate from AFDB teachers |
| CASP15 replication | structure-probe behaviour on a named external target set | excluded from ordinary probe training/splits |
| pair diagnostics | direct dense MI/distance/contact head behaviour | optional; not a replacement for downstream tests |

Outputs are machine-readable CSV/JSON plus a concise Markdown report. The
portable MiniFold is a reproducible low-rank frozen-embedding diagnostic, not a
general-purpose structure predictor. Detailed metrics and input preparation are
in [docs/VALIDATION.md](docs/VALIDATION.md).

## Resumability, audit trail, and data movement

Every major stage writes resolved configuration, source/runtime versions,
attempted/written/skipped/failed counts, timings and throughput, failure rows,
checksums or shard inventories, and a completion marker only after validation.
Work is first written to `.part` paths and atomically promoted.

```text
DATA_ROOT/
  metadata/                 resolved config and source/runtime versions
  source_partitions/        bounded planner output
  workers/worker-N/         packed target shards, indexes, worker reports
  manifest.sqlite           authoritative indexed manifest
  manifest.jsonl            streaming interchange view
  failure_rows.jsonl
  build_summary.json
  COMPLETED

SCRATCH_ROOT/                transient MMseqs DBs and A3Ms; empty after success
RUN_ROOT/run-name/           metrics, rolling/final checkpoints, metadata
```

To move a dataset, copy the complete `DATA_ROOT`, verify the shard inventory,
and point `PROTEINLOSS_DATA_ROOT` at the copy. Manifest paths are relative to
the data root, so they require no rewrite.

Safe to delete after successful reduction/verification:

- transient A3Ms and MMseqs worker databases;
- `.part` directories from failed workers after their failure rows are saved;
- downloaded archives after extraction, version, and checksum are recorded;
- old rolling checkpoints and regenerable per-variant validation rows.

Keep:

- the authoritative manifest, packed shards, indexes, and shard inventories;
- resolved configs, source/version metadata, failure reports, and split policy;
- final checkpoints and benchmark split definitions;
- the StructEncoder teacher if latent training must be resumed.

## What has actually been verified

On 2026-08-12 the portable implementation was tested with deterministic local
fixtures and an existing ProteinGym input bundle. It intentionally did not
download or rebuild the very large UniProt/UniClust/AFDB sources.

| gate | recorded result |
|---|---|
| fresh UV/ROCm environment | Python 3.12.3, Torch 2.9.1+rocm6.4, AMD device visible |
| tests | 13 passed |
| smoke target build | 12/12 written; train 6, valid 2, test 4; 11 structure-backed + 1 structure-missing; scratch empty |
| 100M MLM | 2 optimizer steps; exact 115,669,163 parameters; peak allocated 2,327.6 MiB |
| 100M seven-loss | 2 steps; every objective/metric emitted; peak allocated 2,344.0 MiB |
| 300M MLM | 2 steps; exact 305,588,395 parameters; peak allocated 5,927.6 MiB; final checkpoint reloaded on CPU |
| rolling resume | 100M resumed from step 2 and completed step 3 |
| held-out | 4 test examples and all active head metrics |
| ProteinGym smoke | 1 real assay, 922 substitutions, mean Spearman 0.0714 |
| MiniFold smoke | frozen PLM; 4 PDB-schema test examples + 2 separately loaded CASP15 examples |
| cleanup and portability | no retained A3M/MMseqs scratch, generated outputs, checkpoints, or machine-specific paths in this repository |

Exact acceptance commands and fuller outputs are recorded in
[docs/VALIDATION.md](docs/VALIDATION.md). The production
Internet/MMseqs/Foldseek path is implemented and documented but was not run
cradle-to-grave as part of the portable smoke test; doing so would be a
multi-day, hundreds-of-gigabytes experiment rather than a useful software test.

## Important limitations and gotchas

- **The source snapshot defines the experiment.** UniProt, UniClust, AFDB,
  Foldseek formats, and ProteinGym change. Pin versions and checksums; never
  treat a mutable download URL as provenance.
- **AFDB is not experimental ground truth.** It is broad training supervision.
  Use the separate experimental PDB/CASP15 path for structure evaluation.
- **Homology leakage is a real risk.** Stable family/cluster hashing is preferred
  over accession hashing. The fallback is deterministic but does not guarantee
  remote-homolog separation.
- **A requested count is not guaranteed coverage.** MSA failures and AFDB
  coverage reduce eligible objectives. Inspect coverage before training.
- **The 512-residue cap is part of the model contract.** Longer proteins need a
  documented crop/chunk policy or a different preset; silent truncation changes
  targets and benchmark comparability.
- **MI is quadratic.** Compression solves persistent storage, not the cost of
  computing or decoding an `L x L` target.
- **A3M cleanup is intentional.** Keeping every alignment makes storage and
  metadata handling prohibitive. Preserve only small debug samples if needed.
- **The 300M preset is construction-tested and two-step ROCm-tested, not
  capacity-free.** Full-length production training needs substantially more
  memory than the tiny smoke and may require a higher-memory GPU and gradient
  accumulation.
- **Single-GPU training is the directly implemented path.** Slurm arrays
  distribute target workers. Multi-node DDP is an integration point, not a
  feature silently claimed by this release.
- **The MiniFold probe is diagnostic.** Its compact low-rank portable form is
  designed for controlled embedding comparisons, not full-atom prediction.

## Documentation map

- [Data sources](docs/DATA_SOURCES.md): upstreams, licenses/download contracts,
  versions, checksums, conversion, and storage estimates.
- [Data format](docs/DATA_FORMAT.md): manifest schema, packed shards, MI codec,
  structure targets, and stable splitting.
- [Model](docs/MODEL.md): exact presets, forward path, pair heads, checkpoint
  compatibility, and inference.
- [Training](docs/TRAINING.md): losses, mixed-supervision masks, logging,
  checkpoints, and resume semantics.
- [Validation](docs/VALIDATION.md): held-out targets, ProteinGym, experimental
  MiniFold/PDB, separate CASP15, and verified commands/results.
- [Cluster operation](docs/CLUSTER.md): path overrides, AMD ROCm, local workers,
  and Slurm arrays.
- [Scaling](docs/SCALING.md): 5M–500M architecture, population ceilings,
  bottlenecks, and capacity planning.
- [Migration](docs/MIGRATION.md): what was retained or replaced from historical
  ProteinLoss experiments and how to treat old manifests/checkpoints.
- [Inventory](docs/INVENTORY.md): extraction evidence and scope decisions.
