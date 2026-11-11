#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:-main_dota}
NPROC=${NPROC:-4}

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --standalone --nproc_per_node="${NPROC}" \
    -m georl.cli.train experiment="${EXPERIMENT}"
else
  georl-train experiment="${EXPERIMENT}"
fi
