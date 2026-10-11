#!/usr/bin/env bash
set -euo pipefail
VALIDATION_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$VALIDATION_DIR/../activate.sh"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python "$VALIDATION_DIR/run-validation.py"
