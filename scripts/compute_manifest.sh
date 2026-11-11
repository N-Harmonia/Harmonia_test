#!/usr/bin/env bash
set -euo pipefail

ROOT=${1:?usage: compute_manifest.sh <data_root>}
python - <<PY
from pathlib import Path
from georl.data.manifests import write_manifest
out = write_manifest(Path("${ROOT}"))
print(out)
PY
