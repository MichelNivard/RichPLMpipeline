#!/usr/bin/env bash
#SBATCH --job-name=proteinloss-train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=7-00:00:00

set -euo pipefail
: "${PIPELINE_PROJECT:?Set PIPELINE_PROJECT to the portable_pipeline directory}"
: "${PROTEINLOSS_DATA_ROOT:?Set PROTEINLOSS_DATA_ROOT}"
: "${PROTEINLOSS_RUN_ROOT:?Set PROTEINLOSS_RUN_ROOT}"

cd "${PIPELINE_PROJECT}"
uv run pipeline train --config configs/data/5m.yaml --config configs/objectives/msa.yaml --config configs/cluster/slurm.yaml --model 100m --run-name proteinloss-100m
