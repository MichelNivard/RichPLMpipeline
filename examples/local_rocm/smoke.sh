#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"

scripts/setup_rocm_env.sh
.venv/bin/pytest
.venv/bin/pipeline data build --config configs/data/smoke.yaml
.venv/bin/pipeline train --config configs/data/smoke.yaml --config configs/cluster/local_rocm.yaml --model 100m --objectives mlm --run-name smoke-100m --max-steps 2 --device auto
.venv/bin/pipeline validate --config configs/data/smoke.yaml --config configs/cluster/local_rocm.yaml --model 100m --objectives mlm --checkpoint smoke_artifacts/runs/smoke-100m/final.pt --output smoke_artifacts/validation/heldout
