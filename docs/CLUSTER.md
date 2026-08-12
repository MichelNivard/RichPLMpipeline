# Local Linux, ROCm, and Slurm

## Path configuration

The only runtime path knobs are YAML plus these environment overrides:

| variable | config key |
|---|---|
| `PROTEINLOSS_DATA_ROOT` | `paths.data_root` |
| `PROTEINLOSS_RUN_ROOT` | `paths.run_root` |
| `PROTEINLOSS_SCRATCH_ROOT` | `paths.scratch_root` |
| `PROTEINLOSS_SOURCES_CONFIG` | `sources_config` |
| `PROTEINLOSS_DEVICE` | `training.device` |
| `PROTEINLOSS_NUM_WORKERS` | `training.num_workers` |
| `PROTEINLOSS_SEED` | `seed` |

No username, mount point, GPU number, scheduler account, or systemd service is
embedded. `scripts/run_pipeline.sh` locates the project relative to itself and
uses uv when available.

## AMD ROCm workstation

Install the ROCm PyTorch wheel compatible with the host through the included
setup script. The defaults match the verified host and are explicit overrides:

```bash
ROCM_WHEEL_TAG=rocm6.4 TORCH_VERSION=2.9.1 scripts/setup_rocm_env.sh
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))'
examples/local_rocm/smoke.sh
```

PyTorch's device spelling remains `cuda`. Do not hard-code `HIP_VISIBLE_DEVICES`
inside the pipeline; select a device in the invoking shell when needed.

## Slurm data arrays

First run planning on a login/service node. Read `worker_partitions` from its
JSON or count `work_plan.jsonl`, then submit the exact array range:

```bash
export PIPELINE_PROJECT=/shared/code/portable_pipeline
export PROTEINLOSS_SOURCES_CONFIG=/shared/config/sources.local.yaml
export PROTEINLOSS_DATA_ROOT=/shared/data/proteinloss/5m
export PROTEINLOSS_SCRATCH_ROOT=/local-scratch/$USER/proteinloss-5m
cd "$PIPELINE_PROJECT"
uv run pipeline data build --config configs/data/5m.yaml --sources "$PROTEINLOSS_SOURCES_CONFIG"
workers=$(wc -l < "$PROTEINLOSS_DATA_ROOT/work_plan.jsonl")
sbatch --array="0-$((workers - 1))%32" examples/slurm/data_array.sbatch
```

After the array succeeds, run `pipeline data finalize`. It lists all incomplete
worker IDs rather than accepting a partial manifest. Local scratch is deleted
inside each successful/failed worker `finally` block when `cleanup_scratch` is
true; packed shards remain under the shared data root.

Adjust Slurm partitions, accounts, GPU resource syntax, memory, wall time, and
concurrency to the site. `configs/cluster/slurm.yaml` contains application
settings only. The example batch files contain no site policy assumptions.

## Training jobs

Submit `examples/slurm/train.sbatch` after finalization. Use scheduler signal
handling/wall-time controls to leave enough time for the regular atomic rolling
checkpoint; restart with `--resume`. Systemd may supervise a local invocation,
but it is neither generated nor required.
