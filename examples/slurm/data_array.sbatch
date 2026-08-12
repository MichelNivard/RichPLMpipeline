#!/usr/bin/env bash
#SBATCH --job-name=proteinloss-data
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
# Set --array=0-(N-1) after `pipeline data build` reports worker_partitions.

set -euo pipefail
: "${PIPELINE_PROJECT:?Set PIPELINE_PROJECT to the portable_pipeline directory}"
: "${PROTEINLOSS_SOURCES_CONFIG:?Set PROTEINLOSS_SOURCES_CONFIG}"
: "${PROTEINLOSS_DATA_ROOT:?Set PROTEINLOSS_DATA_ROOT}"
: "${PROTEINLOSS_SCRATCH_ROOT:?Set PROTEINLOSS_SCRATCH_ROOT}"
: "${SLURM_ARRAY_TASK_ID:?Submit this file as an array}"

cd "${PIPELINE_PROJECT}"
uv run pipeline data worker --config configs/data/5m.yaml --config configs/cluster/slurm.yaml --sources "${PROTEINLOSS_SOURCES_CONFIG}" --worker-id "${SLURM_ARRAY_TASK_ID}"
