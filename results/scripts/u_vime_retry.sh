#!/usr/bin/env bash
# retry vime V0 then V2 with the updated handoff (cd4fda86: pip pinned to B200 freeze)
D=$(dirname "$0"); source $D/common.sh
log "vime retry V0 start (handoff cd4fda86)"
bash $D/u_vime.sh V0 && bash $D/u_vime.sh V2
log "vime retry done: V0=$(cat $GATES/vime_V0 2>/dev/null) V2=$(cat $GATES/vime_V2 2>/dev/null)"
