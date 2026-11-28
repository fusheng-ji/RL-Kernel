#!/usr/bin/env bash
# Post-redaction checks before push. Sources stage_push.local.sh for LEAK_GREP_I.
# usage: stage_push_validate.sh DIR                      check a stage tree on disk
#        stage_push_validate.sh --git WORKTREE [PATH...]  check what would be committed: the index version of every
#          staged change (added, copied, modified, renamed; previously committed files included), plus every
#          index file under PATH... when given. *.gz files are decompressed and scanned as well.
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
: "${LEAK_GREP_I:?}"
PCI_RE='[0-9A-Fa-f]{8}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-7]'
LEAK_RE="$LEAK_GREP_I"
TMPD=$(mktemp -d); trap 'rm -rf "$TMPD"' EXIT

if [ "${1:-}" = --git ]; then
  WT=${2:?worktree}; shift 2
  { git -C "$WT" diff --cached --name-only -z --diff-filter=ACMR
    [ $# -gt 0 ] && git -C "$WT" ls-files -z -- "$@"
  } | sort -zu > "$TMPD/paths"
  n=$(tr -cd '\0' < "$TMPD/paths" | wc -c)
  ROOT=$TMPD/index; mkdir -p "$ROOT"
  # write the index (staged) version of each path, not the working-tree file
  [ "$n" -gt 0 ] && (cd "$WT" && git checkout-index -z --stdin --prefix="$ROOT/" < "$TMPD/paths")
  LABEL="index of $WT ($n files)"
else
  ROOT=${1:?tree root}; LABEL=$ROOT
fi

# decompressed copies of *.gz so their contents are scanned too
GZ=$TMPD/gz; mkdir -p "$GZ"
while IFS= read -r -d '' f; do
  rel=${f#$ROOT/}; mkdir -p "$GZ/$(dirname "$rel")"
  gzip -dc "$f" > "$GZ/${rel%.gz}" 2>/dev/null || { echo "BAD GZ: $rel"; exit 1; }
done < <(find "$ROOT" -name '*.gz' -type f -print0 2>/dev/null)

grep -raE "$LEAK_RE" "$ROOT" "$GZ" >"$TMPD/leak.txt" 2>/dev/null || true
if [ -s "$TMPD/leak.txt" ]; then
  echo "== LEAK contents ($LABEL) =="; sed "s#$TMPD/##" "$TMPD/leak.txt" | head -30 | cut -c1-300; exit 1
fi
grep -raE "$PCI_RE" "$ROOT" "$GZ" >"$TMPD/pci.txt" 2>/dev/null || true
if [ -s "$TMPD/pci.txt" ]; then
  echo "== PCI leak ($LABEL) =="; sed "s#$TMPD/##" "$TMPD/pci.txt" | head -20 | cut -c1-300; exit 1
fi

while IFS= read -r -d '' f; do
  python3 -m json.tool "$f" >/dev/null || { echo "BAD JSON: ${f#$TMPD/}"; exit 1; }
done < <(find "$ROOT" "$GZ" -name '*.json' ! -name '*.digests.json' -type f -print0 2>/dev/null)

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

echo "validate OK: $LABEL"
