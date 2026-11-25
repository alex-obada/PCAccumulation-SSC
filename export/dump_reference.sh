#!/usr/bin/env bash
# Convert the selected frames to .bin inputs (+ reference outputs) for pcacc_cpp/data/.
# Extra arguments are passed to dump_reference.py, e.g.:  bash export/dump_reference.sh --ref 0 50 --only-ref
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
source venv_pcaccumulation/bin/activate
python -W ignore export/dump_reference.py "$@"
