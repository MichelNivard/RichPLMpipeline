#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

if [[ -x "${PROJECT_DIR}/.venv/bin/pipeline" ]]; then
  exec "${PROJECT_DIR}/.venv/bin/pipeline" "$@"
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run --project "${PROJECT_DIR}" pipeline "$@"
fi

if command -v pipeline >/dev/null 2>&1; then
  exec pipeline "$@"
fi

echo "Neither uv nor an installed 'pipeline' command is available." >&2
exit 127
