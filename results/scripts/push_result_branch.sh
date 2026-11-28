#!/usr/bin/env bash
# Push one handoff results branch: fetch tip, copy redacted tree, leak-check, commit -s, push.
# usage: push_result_branch.sh <remote-branch> <worktree-dir> <dest-path-in-repo> <src-dir> "<msg>"
# env:   REFRESH="glob ..."  committed files (paths relative to <dest>) that may be overwritten from <src>,
#                            e.g. REFRESH="checks/steps.txt checks/gates-snapshot.txt". Every other file already
#                            committed under <dest> is kept as committed (hand fixes survive); only new files are added.
#        ALLOW_DELETE=1      also delete committed files that are missing from <src> (off by default).
# <src> must be a redacted stage tree (stage_push.sh / redact_tree.py); never point it at raw $OUT.
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
BR=$1; WT=$2; DEST=$3; SRC=$4; MSG=$5
MAIN=${GIT_MAIN:-<WORK>/wt/push}
: "${LEAK_GREP_I:?}"
git -C "$MAIN" fetch origin "$BR"
if [ -d "$WT/.git" ] || [ -f "$WT/.git" ]; then
  git -C "$WT" fetch origin "$BR" && git -C "$WT" reset --hard "origin/$BR"
else
  git -C "$MAIN" worktree add -f "$WT" "origin/$BR"
fi
mkdir -p "$WT/$DEST"

# committed files under DEST that are not in REFRESH are excluded from the copy
EXC=$(mktemp); trap 'rm -f "$EXC"' EXIT
git -C "$WT" ls-files -- "$DEST" | sed "s#^$DEST/##" | while IFS= read -r rel; do
  keep=1
  for g in ${REFRESH:-}; do
    # shellcheck disable=SC2053
    [[ "$rel" == $g ]] && keep=0 && break
  done
  [ $keep = 1 ] && echo "/$rel"
done > "$EXC"
nkept=$(wc -l < "$EXC")
# report committed files whose stage copy differs (kept as committed unless listed in REFRESH)
while IFS= read -r rel; do
  rel=${rel#/}
  [ -f "$SRC/$rel" ] && ! cmp -s "$SRC/$rel" "$WT/$DEST/$rel" && echo "kept committed (stage differs): $DEST/$rel"
done < "$EXC"
del=""; [ "${ALLOW_DELETE:-0}" = 1 ] && del=--delete
rsync -a $del --exclude-from="$EXC" "$SRC/" "$WT/$DEST/"
echo "copied from $SRC; kept $nkept committed files as is; refresh: ${REFRESH:-none}"

cd "$WT"
git add -f "$DEST"
if git diff --cached --quiet; then echo "nothing to commit in $BR"; exit 0; fi
# scan the index version of every staged change, plus all of DEST, before committing
bash "$D/stage_push_validate.sh" --git "$WT" "$DEST"
git commit -s -m "$MSG"
git push origin "HEAD:$BR"
