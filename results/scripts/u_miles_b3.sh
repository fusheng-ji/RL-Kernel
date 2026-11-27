#!/usr/bin/env bash
# Round 3: H2 probe b3 only, then H3 if pass (stop after H3 per review).
D=$(dirname "$0"); source "$D/common.sh"
J=$1
export WORK=$SCHED/miles THIS=$WT/h_mi/h100-handoff/miles-inkling2-20261126
export TAG=${TAG:-b3} SLURM_OVERLAP=1
RS=$THIS/scripts/run_stage.sbatch
gate_ok miles_H1 || exit 1
timeout -k 60 5400 bash "$RS" "$THIS/scripts/h2_memprobe.sh" b3 1 0.80 >"$OUT/miles/h2_memprobe-b3.log" 2>&1 || true
python3 - "$WORK" "$OUT/miles" <<'PY'
import glob, re, sys
work, out = sys.argv[1:3]
st = f"{work}/runs/h2_memprobe-b3"
logs = glob.glob(st + "/*/train.log")
if not logs:
    open(f"{out}/h2_b3_verdict.txt", "w").write("FAIL no train.log\n"); sys.exit(1)
txt = open(logs[-1], errors="replace").read()
rc = re.findall(r"^train rc=(\d+)", txt, re.M)
trainer_step = bool(re.search(r" - step 0: \{", txt))
oom = bool(re.search(r"OutOfMemory|out of memory", txt, re.I))
verdict = "PASS" if rc and int(rc[-1]) == 0 and trainer_step and not oom else "FAIL"
open(f"{out}/h2_b3_verdict.txt", "w").write(f"{verdict} rc={rc[-1] if rc else '?'}\n")
sys.exit(0 if verdict == "PASS" else 1)
PY
b3rc=$?
if [ $b3rc = 0 ]; then
  echo 'export SG_TP=1 SG_MEM=0.80' >> "$WORK/h100.env"
  gate_set miles_H2 "OK setting=b3"
  timeout -k 60 5400 bash "$RS" "$THIS/scripts/h3_smoke.sh" >"$OUT/miles/h3_smoke.log" 2>&1
  gate_set miles_H3 done
  log "miles round3: H3 done — stopping for review"
else
  gate_set miles_H2 "FAIL b3"
  log "STOP: miles H2 b3 failed — see train.log stacks fr ray_logs"
  exit 1
fi