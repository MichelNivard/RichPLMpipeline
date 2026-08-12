#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
ROCM_WHEEL_TAG="${ROCM_WHEEL_TAG:-rocm6.4}"
TORCH_VERSION="${TORCH_VERSION:-2.9.1}"

cd "${PROJECT_DIR}"
uv venv --clear .venv
uv pip install --python .venv/bin/python \
  'numpy>=2,<3' 'PyYAML>=6,<7' 'pytest>=8.3,<9' 'hatchling>=1.26'
uv pip install --python .venv/bin/python \
  --index-url "https://download.pytorch.org/whl/${ROCM_WHEEL_TAG}" \
  "torch==${TORCH_VERSION}"
uv pip install --python .venv/bin/python --no-deps --editable .
.venv/bin/python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")'
