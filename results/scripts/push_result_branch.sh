#!/usr/bin/env bash
# Push one handoff results branch: fetch tip, copy redacted tree, leak-check, commit -s, push.
# usage: push_result_branch.sh <remote-branch> <worktree-dir> <dest-path-in-repo> <src-dir> "<msg>"
set -euo pipefail
D=$(dirname "$0")
# shellcheck source=/dev/null
[ -f "$D/stage_push.local.sh" ] && source "$D/stage_push.local.sh"
BR=$1; WT=$2; DEST=$3; SRC=$4; MSG=$5
MAIN=${SCHED_ROOT:-<WORK>/wt/push}
: "${LEAK_GREP_I:?}"
git -C "$MAIN" fetch origin "$BR"
if [ -d "$WT/.git" ] || [ -f "$WT/.git" ]; then
  git -C "$WT" fetch origin "$BR" && git -C "$WT" reset --hard "origin/$BR"
else
  git -C "$MAIN" worktree add -f "$WT" "origin/$BR"
fi
mkdir -p "$WT/$DEST"
rsync -a --delete "$SRC/" "$WT/$DEST/"
cd "$WT"
git add -f "$DEST"
if git diff --cached --quiet; then echo "nothing to commit in $BR"; exit 0; fi
git commit -s -m "$MSG"
if git grep -nIiE "$LEAK_GREP_I" HEAD -- "$DEST" 2>/dev/null | head -5 | grep -q .; then
  echo "LEAK in commit:"; git grep -nIiE "$LEAK_GREP_I" HEAD -- "$DEST" | head -20
  exit 1
fi
git push origin "HEAD:$BR"