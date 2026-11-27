#!/usr/bin/env bash
# Post-redaction checks before push. Sources stage_push.local.sh for LEAK_GREP_I.
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
ROOT=${1:?tree root}
: "${LEAK_GREP_I:?}"
PCI_RE='[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]'
LEAK_RE="$LEAK_GREP_I"

grep -raE "$LEAK_RE" "$ROOT" >/tmp/validate_leak.txt 2>/dev/null || true
if [ -s /tmp/validate_leak.txt ]; then
  echo "== LEAK contents =="; head -30 /tmp/validate_leak.txt; exit 1
fi
grep -raE "$PCI_RE" "$ROOT" >/tmp/validate_pci.txt 2>/dev/null || true
if [ -s /tmp/validate_pci.txt ]; then
  echo "== PCI leak =="; head -20 /tmp/validate_pci.txt; exit 1
fi

while IFS= read -r -d '' f; do
  python3 -m json.tool "$f" >/dev/null || { echo "BAD JSON: $f"; exit 1; }
done < <(find "$ROOT" -name '*.json' ! -name '*.digests.json' -type f -print0 2>/dev/null)

python3 - "$ROOT" <<'PY'
import json, re, sys, pathlib
root = pathlib.Path(sys.argv[1])
hex64 = re.compile(r"^[0-9a-fA-F]{64}$")
bad = []
for f in root.rglob("*.digests.json"):
    try:
        d = json.load(open(f))
    except json.JSONDecodeError as e:
        bad.append(f"{f}: {e}"); continue
    def walk(obj, p=""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{p}.{k}" if p else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{p}[{i}]")
        elif isinstance(obj, str) and len(obj) >= 32:
            if len(obj) == 64 and not hex64.match(obj):
                bad.append(f"{f} {p}: not 64-hex digest {obj[:20]}...")
            elif "<" in obj and "JOB" in obj:
                bad.append(f"{f} {p}: placeholder in digest {obj}")
    walk(d)
if bad:
    print("\n".join(bad[:30])); sys.exit(1)
PY

echo "validate OK: $ROOT"