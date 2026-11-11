#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT=${DATA_ROOT:-data}
mkdir -p "${DATA_ROOT}"

cat <<'NOTE'
Datasets used by GeoRL:

  - DOTA v1.0    : https://captain-whu.github.io/DOTA/
  - iSAID v1.0   : https://captain-whu.github.io/iSAID/
  - LoveDA       : https://github.com/Junjue-Wang/LoveDA
  - xView v1     : https://xviewdataset.org/

Download each archive, extract under ${DATA_ROOT}, then run:
  python -m georl.data.manifests <dataset_root>

That will produce a manifest.json with SHA-256s for verification.
NOTE
