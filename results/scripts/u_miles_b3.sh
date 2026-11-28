#!/usr/bin/env bash
# Round 3: H2 probe b3, then H3 if pass (stop after H3).
D=$(dirname "$0"); source "$D/common.sh"
J=${1:-$JB}
export WORK=$SCHED/miles THIS=$WT/h_mi/h100-handoff/miles-inkling2-20261126
export TAG=${TAG:-b3} SLURM_OVERLAP=1
RS=$THIS/scripts/run_stage.sbatch
gate_ok miles_H1 || exit 1
set +e
timeout -k 60 5400 bash "$RS" "$THIS/scripts/h2_memprobe.sh" "$TAG" 1 0.80 >"$OUT/miles/h2_memprobe-${TAG}.log" 2>&1
stagerc=$?
# stay in set +e: the verdict heredoc below exits 1 on FAIL and b3rc must still reach the gate
echo "stage_rc=$stagerc" >>"$OUT/miles/h2_memprobe-${TAG}.log"
if [ "$stagerc" = 96 ]; then
  gate_set miles_H2_busy "BUSY rc=96 TAG=$TAG"
  exit 96
fi
python3 - "$WORK" "$OUT/miles" "$TAG" "$stagerc" <<'PY'
import glob, re, sys
work, out, tag, stagerc = sys.argv[1:5]
st = f"{work}/runs/h2_memprobe-{tag}"
logs = sorted(glob.glob(st + "/*/train.log"))
open(f"{out}/h2_{tag}_verdict.txt", "w").write(
    f"stage_rc={stagerc} train_log={'yes' if logs else 'no'}\n"
)
if not logs:
    sys.exit(1)
txt = open(logs[-1], errors="replace").read()
rc = re.findall(r"^train rc=(\d+)", txt, re.M)
trainer_step = bool(re.search(r" - step 0: \{", txt))
oom = bool(re.search(r"OutOfMemory|out of memory", txt, re.I))
trc = int(rc[-1]) if rc else 127
fail_stage = int(stagerc) not in (0, 124) and int(stagerc) != 96
verdict = "PASS" if not fail_stage and trc == 0 and trainer_step and not oom else "FAIL"
open(f"{out}/h2_{tag}_verdict.txt", "a").write(f"{verdict} train_rc={trc}\n")
sys.exit(0 if verdict == "PASS" else 1)
PY
b3rc=$?
for gc in "$WORK"/runs/h2_memprobe-"$TAG"/gpu_check_node*.txt; do
  [ -f "$gc" ] || continue
  # gpu_check writes "nproc 64" (space); accept "nproc=64" too
  grep -A5 '== cpus' "$gc" | grep -qE '^nproc[ =]64$' || { log "STOP: b3 gpu_check cpus not 64 in $gc"; exit 1; }
done
if [ $b3rc != 0 ]; then
  gate_set miles_H2 "FAIL $TAG"
  exit 1
fi
echo 'export SG_TP=1 SG_MEM=0.80' >> "$WORK/h100.env"
gate_set miles_H2 "OK setting=$TAG"
for n in $(nodes_of "$J"); do
  other=$(squeue -h -w "$n" -o '%i' 2>/dev/null | grep -v "^$J" | grep -c . || true)
  [ "$other" = 0 ] || { log "STOP: not exclusive before miles H3"; exit 1; }
done
set +e
timeout -k 60 5400 bash "$RS" "$THIS/scripts/h3_smoke.sh" >"$OUT/miles/h3_smoke.log" 2>&1
h3rc=$?
set -e
gate_set miles_H3 "done rc=$h3rc"
log "miles H3 finished rc=$h3rc (round3 stop)"
[ "$h3rc" = 0 ] || exit 1