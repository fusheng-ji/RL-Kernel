#!/usr/bin/env bash
# gzip *.log files over 90MB in a tree; leaves original removed.
set -euo pipefail
ROOT=${1:?tree}
THRESH=$((90*1024*1024))
find "$ROOT" -type f -name '*.log' -size +90M -print0 2>/dev/null | while IFS= read -r -d '' f; do
  gzip -9 -f "$f"
  echo "gzipped: $f"
done