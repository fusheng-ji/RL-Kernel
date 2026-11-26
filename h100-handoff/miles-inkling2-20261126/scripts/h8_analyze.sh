#!/bin/bash
# H8 (one node, CPU work inside the container): every summary file for the report. args: none; reads the stage dirs by TAG names below.
# The stage directories come from $WORK/h8.env (H1 H2_DIRS H3 H4T H4S H5A H5B H6 H7F H7R), e.g. H3=$WORK/runs/h3_smoke-r0
set -x
source $WORK/h8.env
O=$WORK/runs/report; mkdir -p $O; cd /root
python3 $THIS/scripts/probe_summary.py $H2_DIRS > $O/h2_probe_summary.json
python3 $THIS/scripts/analyze_dump.py $H3/dump --out $O/h3_consistency.json > /dev/null
cat $H4T/replay_bitwise.txt > $O/h4_replay_bitwise.txt
for p in probe_r3 probe_top; do grep -nE "Error|Traceback|assert|refus|not supported" $H4T/$p/train.log | head -20 > $O/h4_$p.txt; done
cp $H4S/sglang_default.json $H4S/sglang_deterministic.json $O/ 2>/dev/null
python3 $THIS/scripts/collect.py $H5A --stage $H5A --name run10_a --out $O --h1 $H1
python3 $THIS/scripts/collect.py $H5B --stage $H5B --name run10_b --out $O --h1 $H1
python3 $THIS/scripts/collect.py $H6/run --stage $H6 --name run100 --out $O --h1 $H1 --dump $H6/run/dump
python3 $THIS/scripts/summ.py $O run100 > $O/run100_summary.txt
python3 $THIS/scripts/drift.py $H6/run/dump $O/run100_drift_rows.pt > $O/run100_drift.txt
python3 $THIS/scripts/sync_compare.py $H6 $H7F 0:w32_r33 1:w65_r66 2:w98_r99 > $O/h7_sync_compare.json
D=$H6/run/dump
python3 $THIS/scripts/cmp_points.py late_w32_r33 $D 33 $H7F/w32_r33.pt > $O/h7_w32_r33.json
python3 $THIS/scripts/cmp_points.py late_w65_r66 $D 66 $H7F/w65_r66.pt > $O/h7_w65_r66.json
python3 $THIS/scripts/cmp_points.py late_w98_r99 $D 99 $H7F/w98_r99.pt > $O/h7_w98_r99.json
python3 $THIS/scripts/cmp_points.py orig_r0 $D 0 $H7F/orig_r0.pt > $O/h7_orig_r0.json
python3 $THIS/scripts/cmp_points.py orig_r99 $H7R/r99/dump 0 $H7F/orig_r99.pt > $O/h7_orig_r99.json
ls -la $O
