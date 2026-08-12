# Scaling contract

## Bounded planning and workers

Planning streams UniProtKB one record at a time, performs indexed SQLite family
and AFDB lookups, and writes source partitions of `worker_batch_size`. It keeps
only the current partition handle and counters. There is no global DataFrame,
CSV load, or Python sort.

Each worker handles one bounded partition. MMseqs searches UniClust30
representatives, expands matching families through the indexed membership map,
subsets the UniProt database, aligns members, writes transient A3Ms, and reduces
each A3M immediately to PSSM and dense APC-MI. Foldseek CA/3Di byte records are
read by indexed offset. Final targets are packed; A3Ms and MMseqs scratch are
removed in `finally`.

Finalization streams completed worker JSONL into indexed SQLite and a streaming
JSONL view. It requires every planned worker and, by default, the requested
written count.

## Scale presets

| plan | workers at 2,000 rows | validation/test | MSA depth | operational intent |
|---|---:|---:|---:|---|
| 5M | 2,500 | 2% / 2% | 2,048 | established dense-pair scale |
| 50M | 25,000 | 1% / 1% | 2,048 | cluster production |
| 500M | 250,000 | 0.5% / 0.5% | 1,024 | streaming/index stress scale |

These are planning defaults, not promises of available unique proteins or a
fixed disk footprint. Measure early shards and tune partition size, concurrency,
depth, and local scratch.

## Measured dense-route reference

The portable architecture is based on the mature dense route, whose timings
show which optimizations matter. These measurements are historical evidence,
not performance promises for a different database, filesystem, MMseqs build,
or cluster:

| stage | measured scale | wall time | approximate time per million |
|---|---:|---:|---:|
| representative search against UniClust30 seeds | 1.3M queries | 24.8 h | 19.1 h/M |
| family-batched MSA + PSSM + dense MI packing | 1.0M queries | 23.9 h | 23.9 h/M |
| older CA/contact/latent structure reduction | 1.0M proteins | 4.7 h | 4.7 h/M |

Within the 23.9-hour MSA/target stage, `result2msa` and repeated local member
database creation each accounted for about 35%, PSSM/MI computation and packed
encoding about 19%, membership/accession mapping about 7%, and everything else
less than 4%. In the separate 24.8-hour representative search, prefiltering
accounted for about 22.2 hours.

Consequently, a naive fresh single-machine run was roughly 43 hours per million
before the smaller structure-side cost: about nine days for 5M and already
impractical at 50M. The 50M/500M presets exist to exercise the same streaming
contract with distributed workers, not to suggest waiting for one process.

The historical dense writer produced about 8.8 GB of compressed pair records
per million queries and about 11 GB for the complete pair-target directory.
All-target planning should still use the wider 10–40 KB/example range because
length distribution, shard overhead, profiles, and compression vary.

## Population ceiling and mixed supervision

Three populations differ:

1. valid unique UniProtKB sequences no longer than 512;
2. rows for which the UniClust expansion produces a usable MSA;
3. rows that additionally map to AFDB/Foldseek CA and 3Di records.

Full structure supervision is limited by population 3. The pipeline never
duplicates structure rows to make a requested count. Rows in population 2 but
not 3 train MLM/PSSM/MI; population 3 also trains distance/contact/3Di/latent.
Availability flags mask losses at batch time. If production selection cannot
reach the requested number of unique sequences, it reports
`population_ceiling_reached` and stops. If target failures produce a shortfall,
finalization withholds `COMPLETED`; setting `allow_shortfall` is an explicit,
recorded policy choice.

At extreme scale, a database-native sharding strategy may be preferable to a
single UniProt FASTA scan and one SQLite manifest file. The current architecture
keeps memory bounded and supports distributed workers, but metadata IOPS and a
single final SQLite insertion stream remain expected 500M bottlenecks.

## Expected bottlenecks

- MMseqs search/align and remote source staging dominate wall time and scratch.
- Exact 21-state MI is quadratic in length; CPU reduction may bottleneck GPUs.
- AFDB CA storage and random reads require high-throughput shared/local disks.
- Packed target compression ratio varies strongly with length and MSA content.
- Single-GPU Transformer memory is quadratic in sequence length; 300M generally
  requires a high-memory accelerator and accumulation.

## Useful scaling levers

- Keep MMseqs databases and worker scratch on fast local NVMe; copy compact
  finished shards to shared storage after validation.
- Reuse compatible representative-hit stores. Repeating a fresh prefilter can
  cost roughly 19 hours per million queries at the historical CPU rate.
- Distribute bounded source partitions. Cap array concurrency by scratch IOPS
  and member-database memory, not only by available CPU cores.
- Amortize `createsubdb` and `result2msa` with measured worker sizes; more
  workers can become slower when they contend for the same database.
- Batch or accelerate exact MI reduction, but remember that compression reduces
  disk size rather than the quadratic computation needed to create the map.
- Delete A3Ms and MMseqs scratch only after the packed record, index, checksum,
  counts, and failure row have been committed.
