#!/usr/bin/env bash
set -euo pipefail

EXPERIMENT=${1:-main_dota}
CKPT=${2:-outputs/${EXPERIMENT}/last.pt}

georl-eval experiment="${EXPERIMENT}" "++ckpt=${CKPT}"
