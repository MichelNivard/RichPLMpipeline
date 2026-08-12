#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

: "${PROTEINLOSS_DATA_ROOT:?Set PROTEINLOSS_DATA_ROOT}"
: "${PROTEINLOSS_RUN_ROOT:?Set PROTEINLOSS_RUN_ROOT}"
.venv/bin/pipeline train --config configs/data/5m.yaml --config configs/objectives/msa.yaml --config configs/cluster/local_rocm.yaml --model 100m --run-name proteinloss-100m
