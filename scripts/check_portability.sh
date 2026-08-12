#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
historical_user='mic''helnivard'
historical_ssd='SSD''1'
historical_hdd='HDD''1'
pattern="/home/${historical_user}|/media/${historical_user}|${historical_ssd}|${historical_hdd}"

if rg -n "${pattern}" "${PROJECT_DIR}" --glob '!uv.lock'; then
  echo "Machine-specific path found." >&2
  exit 1
fi

for forbidden in '*.pt' '*.ckpt' '*.bin' '*.a3m' '*.log'; do
  if find "${PROJECT_DIR}" -type f -name "${forbidden}" -not -path '*/.venv/*' -print -quit | grep -q .; then
    echo "Generated/training artifact found matching ${forbidden}." >&2
    exit 1
  fi
done

echo "Portability scan passed."
