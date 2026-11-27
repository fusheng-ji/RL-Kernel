#!/usr/bin/env bash
# build several checkouts one after another on the one card of this step: u_build_seq.sh WQ WB
D=$(dirname "$0"); for k in "$@"; do bash $D/u_build.sh $k; done
