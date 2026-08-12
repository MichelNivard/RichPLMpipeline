# Data sources and local contracts

`configs/sources.example.yaml` is the only registry for external files and
tools. Copy it to an untracked local file. `${VARIABLE}` values are expanded at
runtime; relative paths are interpreted relative to the registry file. Run
`pipeline preflight --sources sources.local.yaml --all` to list every missing
item in one pass.

The sources have distinct roles:

| source | role in the experiment | artifact made from it |
|---|---|---|
| UniProtKB | unique sequence population and homolog member store | input IDs, online MLM population |
| UniClust30 | representative search accelerator and family membership | transient A3Ms, stable family grouping |
| MSA derived from UniClust + UniProt | evolutionary teacher | PSSM and compressed dense APC-MI |
| AFDB/Foldseek | broad predicted-structure teacher | fp16 CA coordinates and packed 3Di |
| StructEncoder | contextual 3Di teacher | online 128-dimensional latent target |
| ProteinGym | independent functional benchmark | mutation-effect CSV/JSON reports |
| RCSB PDB / CASP15 | independent experimental structure benchmark | packed coordinate-probe datasets |

AFDB is deliberately a training teacher; RCSB/CASP15 is deliberately an
evaluation source. Mixing those roles would make the structure benchmark much
less informative.

Never label a mutable `current_release` URL as an immutable version. Store its
release identifier, retrieval date, upstream checksum where offered, and a
local SHA-256. `metadata/resolved_config.*` and preflight output preserve this
information with a build.

## UniProtKB

- Upstream: `https://ftp.uniprot.org/pub/databases/uniprot/`
- Files: the reviewed Swiss-Prot and unreviewed TrEMBL FASTA files from one
  pinned release, plus that directory's `RELEASE.metalink`/checksum metadata.
- Registry contract `uniprot_fasta`: one FASTA or gzip-compressed FASTA with
  headers whose accession is either the first token or the middle field of
  `sp|ACCESSION|...` / `tr|ACCESSION|...`.
- Registry contract `uniprot_mmseqs_db`: MMseqs database prefix built from the
  exact same FASTA; `<prefix>.dbtype` and `<prefix>.lookup` must exist.

Concatenate gzip members without recompression, then index:

```bash
cat uniprot_sprot.fasta.gz uniprot_trembl.fasta.gz > uniprotkb.fasta.gz
mmseqs createdb uniprotkb.fasta.gz uniprotkb
```

The portable accession index is built from the MMseqs lookup by
`pipeline bootstrap index-lookups --sources sources.local.yaml`. This is a
streaming SQLite conversion.

## UniClust30

- Upstream: `https://uniclust.mmseqs.com/`
- Historical comparison release: UniClust30 2018_08, mirrored under
  `https://wwwuser.gwdg.de/~compbiol/uniclust/2018_08/` when available.
- Registry contract `uniclust30_seed_db`: MMseqs representative database prefix.
- Registry contract `uniclust30_membership`: portable SQLite containing
  representative-to-members and accession-to-representative indexes.

The raw mapping is two tab-separated columns, representative and member, grouped
by representative. Convert it without holding the mapping in memory:

```bash
uv run pipeline bootstrap index-membership \
  --mapping uniclust30_2018_08_members.tsv.gz \
  --output uniclust30_2018_08_members.sqlite
```

UniClust30 is not redistributed here. If the 2018 mirror is unavailable, record
the replacement snapshot and expect different MSA targets; do not claim an
exact historical replication.

Known-good local artifacts used by the mature dense route had these SHA-256
values. They are local identity checks, not a claim that the upstream server
publishes the same digest:

| file | SHA-256 |
|---|---|
| `uniclust30_2018_08.tar.gz` | `c8923e6f1bf86b8f57516197f402697d3fc0882994ca053144a8284beaaa1c03` |
| `uniclust_uniprot_mapping.tsv.gz` | `729edf3e48efd62bb4822da0a413d5f191e1c52169dc2bca00f5bd1a0b13fb7a` |

## AlphaFold DB / Foldseek

- Foldseek source/releases: `https://github.com/steineggerlab/foldseek`
- AlphaFold DB downloads: `https://alphafold.ebi.ac.uk/download`
- Supported Foldseek bootstrap form:
  `foldseek databases Alphafold/UniProt50 <PREFIX> <TMP>`
- Registry contract `afdb_foldseek`: one prefix with `.lookup`, `.index`, `_ca`,
  `_ca.index`, `_ss`, and `_ss.index`. The CA store is required; a sequence/3Di
  database without CA cannot produce distance targets.

Foldseek documents about 151 GB RAM for the AFDB/UniProt50 database with CA and
about 35 GB without CA. Pin a Foldseek release: database format compatibility
has changed between major releases. Then create the portable accession lookup:

```bash
uv run pipeline bootstrap index-lookups --sources sources.local.yaml
```

The worker reads byte offsets directly from Foldseek indexes, writes CA as
fp16, and writes packed 3Di token IDs. Coordinates are decoded into symmetric
`log1p(angstrom)` matrices only in the loader.

## StructEncoder teacher

The mature project teacher is `checkpoint_step69438.pt`, with 20 3Di tokens,
pad ID 20, width 768, 12 layers, 8 heads, 128 latent dimensions, and block size
256. It is a project artifact, not redistributed in this directory. Put it at
`structencoder_teacher.path`, record its SHA-256, and keep it frozen. Targets
are computed online in bounded blocks, so no latent array is stored in shards.

Known-good checkpoint identity:

```text
size:    344,380,845 bytes
SHA-256: 342fefdee451122a3daffa8c8b101872375968da547df7f029e4330d6c6ee0e0
```

## ProteinGym

- Code/metadata: `https://github.com/OATML-Markslab/ProteinGym`
- Metadata is pinned in the bootstrap script to repository commit
  `144fe22b07dfaeec2b366f2346203a9838a55b4c`.
- Pinned substitutions bundle:
  `https://marks.hms.harvard.edu/proteingym/ProteinGym_v1.3/DMS_ProteinGym_substitutions.zip`
- Metadata contract: `DMS_substitutions.csv` from the matching release.
- Registry contract `proteingym_bundle`: ZIP containing the `DMS_filename`
  CSVs. Preparation selects
  single-substitution assays, canonical sequences, and length at most 512.

The full substitutions bundle is an input archive, never copied into this
package. The bootstrap script can download it where redistribution permits.
Pass the registry directly with
`pipeline proteingym prepare --sources sources.local.yaml --output <DIR>`.

Known-good v1.3 files used for the portability acceptance run:

| file | SHA-256 |
|---|---|
| `DMS_substitutions.csv` | `a8f498011532a74aa9fe556a50555a75e928c5837d19c06a87592ae04049b308` |
| `DMS_ProteinGym_substitutions.zip` | `3a83766254ac9ac9984ec25cb73c6e010ea4418f5e35f143933e6b6e6473b921` |

## RCSB PDB and CASP15

- PDB sequence snapshot:
  `https://files.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt.gz`
- Individual mmCIF:
  `https://files.rcsb.org/download/<PDB_ID>.cif.gz`
- CASP15 target list: `https://predictioncenter.org/casp15/targetlist.cgi`
- Registry contracts: `rcsb_seqres`, `rcsb_mmcif_cache`,
  `casp15_target_table`, and `casp15_mmcif_cache`.

Record the mutable PDB snapshot date and SHA-256. Prepare a chain-level dataset
of canonical sequences (length ≤512) with experimental CA coordinates, stable
family/cluster splits, and the packed NPZ contract described in
`DATA_FORMAT.md`. CASP15 target identifiers must be removed from the ordinary
PDB train/valid/test population and written to a separate packed CASP15
coordinate dataset. This
prevents the replication split from becoming probe-training data.

The self-contained conversion/downloader is:

```bash
uv run pipeline structure prepare --mode pdb \
  --output /data/validation/pdb --cache /data/rcsb-cache \
  --seqres /data/rcsb-cache/pdb_seqres.txt.gz \
  --casp15-table /data/rcsb-cache/casp15_targetlist.csv \
  --download-missing --max-chains 250000
uv run pipeline structure prepare --mode casp15 \
  --output /data/validation/casp15 --cache /data/rcsb-cache \
  --casp15-table /data/rcsb-cache/casp15_targetlist.csv --download-missing
```

Omit `--download-missing` on an offline cluster after staging the exact files.
The parser accepts observed CA atoms from mmCIF, filters the ordinary PDB set to
X-ray structures at configurable resolution, splits coordinate gaps into
segments, packs records with relative offsets/checksums, and requires
train/valid/test. CASP15 mmCIF records are written only to split `casp15`.

## Storage planning

Approximate, deliberately conservative ranges depend on release and target
compressibility:

| input/output | typical order of magnitude |
|---|---:|
| UniProtKB FASTA + MMseqs database | hundreds of GB to low TB |
| UniClust30 representative/membership data | tens to hundreds of GB |
| AFDB/UniProt50 Foldseek with CA | hundreds of GB; ~151 GB RAM to load |
| transient MMseqs worker scratch | 1–10 GB per 2,000-query worker |
| packed ProteinLoss targets | roughly 10–40 KB/example, content dependent |
| 5M / 50M / 500M targets | roughly 0.05–0.2 / 0.5–2 / 5–20 TB |
| ProteinGym v1.3 substitutions, extracted | about 1 GB |

Measure the first completed shards before capacity planning a full run. Scratch
and final targets should live on different quotas when possible.
