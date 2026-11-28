#!/usr/bin/env bash
# miles H1..H7 by running the handoff stage scripts inside the allocation. usage: u_miles.sh H1|H2|H3|H4|H5|H6|H7
# gates: miles_H0 (digest ok, set in PLAN prep) -> H1 -> H2(b|c|d) -> H3.. ; H5 sets miles_H5_ok only if H6 est <= 16 h.
source "$(dirname "$0")/common.sh"
export WORK=$SCHED/miles THIS=$WT/h_mi/h100-handoff/miles-inkling2-20261126 SLURM_OVERLAP=1
SC=$THIS/scripts; RS=$SC/run_stage.sbatch; R1=$SC/run_stage_1node.sbatch; mkdir -p $OUT/miles
gate_ok miles_H0 || { echo "miles H0 digest gate not OK"; exit 1; }
st() { local t=$1 tag=$2 stage=$3; shift 3; TAG=$tag timeout -k 60 $t bash $RS $SC/$stage "$@" > $OUT/miles/$stage-$tag.log 2>&1; echo "$stage-$tag exit=$?" >> $OUT/steps.txt; }
s1() { local t=$1 tag=$2 stage=$3; shift 3; TAG=$tag timeout -k 60 $t bash $R1 $SC/$stage "$@" > $OUT/miles/$stage-$tag.log 2>&1; echo "$stage-$tag exit=$?" >> $OUT/steps.txt; }
case $1 in
 H1) s1 1h r0 h1_env.sh; h1rc=$(grep -o 'h1_env.sh-r0 exit=[0-9]*' $OUT/steps.txt | tail -1 | cut -d= -f2)
     # README expects a clean tree in the image; a non-empty git status is a fact to report, not a failure
     dirty=$(grep -cE '^ ?[MADRCU?]{1,2} ' $WORK/runs/h1_env-r0/env.txt 2>/dev/null)
     if [ "$h1rc" = 0 ]; then gate_set miles_H1 "done exit=0 git_status_lines=${dirty:-?}"; else gate_set miles_H1 "FAIL exit=${h1rc:-none}"; exit 1; fi;;
 H2) gate_ok miles_H1 || exit 1
     st 90m a h2_memprobe.sh a 1 0.55; st 90m b h2_memprobe.sh b 1 0.80; st 90m c h2_memprobe.sh c 2 0.55; st 90m d h2_memprobe.sh d 2 0.70
     python3 - "$WORK" "$OUT/miles" <<'PY'
import glob, os, re, sys
work, out = sys.argv[1:3]
ERR = re.compile(r"OutOfMemory|out of memory|CUDA error|Traceback|RuntimeError", re.I)
def probe_pass(letter):
    st = f"{work}/runs/h2_memprobe-{letter}"
    logs = sorted(glob.glob(st + "/*/train.log"))
    if not logs: return False, "no train.log"
    txt = open(logs[-1], errors="replace").read()
    rc = re.findall(r"^train rc=(\d+)", txt, re.M)
    if not rc or int(rc[-1]) != 0: return False, "train rc"
    # README round 3 item 5: count the trainer's " - step 0: {" line, not "perf 0:" (the rollout logs one too)
    if not re.search(r" - step 0: \{", txt): return False, "no trainer step 0"
    if ERR.search(txt): return False, "error in log"
    return True, logs[-1]
chosen = None
for letter in "bcd":
    ok, why = probe_pass(letter)
    if ok:
        chosen = letter
        break
path = f"{out}/h2_choice.txt"
if chosen:
    open(path, "w").write(f"setting={chosen}\n")
    print(f"OK setting={chosen}")
    sys.exit(0)
open(path, "w").write("FAIL b,c,d\n")
print("FAIL b,c,d")
sys.exit(1)
PY
     h2rc=$?
     if [ $h2rc = 0 ]; then gate_set miles_H2 "OK $(tail -1 $OUT/miles/h2_choice.txt | tr -d '\n')"; else gate_set miles_H2 "FAIL b,c,d"; exit 1; fi;;
 H3) gate_ok miles_H2 || exit 1; st 90m r0 h3_smoke.sh; gate_set miles_H3 done;;
 H4) gate_ok miles_H3 || exit 1; st 3h r0 h4_trainer.sh $WORK/runs/h3_smoke-r0/dump; s1 90m r0 h4_sglang.sh $WORK/runs/h3_smoke-r0/dump; gate_set miles_H4 done;;  # self-check on one node, after the trainer
 H5) gate_ok miles_H3 || exit 1; st 3h a h5_run.sh 10; st 3h b h5_run.sh 10
     h5verdict=$(python3 - "$WORK" <<'PY'
import ast, glob, re, sys
work = sys.argv[1]
pat = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+) [^\]]*\][^\n]*?(step|rollout|perf) (\d+): (\{[^\n]*\})"
)
def mean_step(run_dir):
    logs = glob.glob(run_dir + "/*/train.log") or glob.glob(run_dir + "/train.log")
    if not logs:
        return None
    txt = open(logs[0], errors="replace").read()
    rows = {}
    for m in pat.finditer(txt):
        n, body = int(m.group(3)), m.group(4)
        body = re.sub(r"\b(nan|inf)\b", "None", body.rstrip())
        try:
            rows.setdefault(n, {}).update(ast.literal_eval(body))
        except (ValueError, SyntaxError):
            continue
    times = [rows[s].get("perf/step_time") for s in range(1, 10) if rows.get(s, {}).get("perf/step_time") is not None]
    if not times:
        return None
    return sum(times) / len(times)
means = [mean_step(f"{work}/runs/h5_run-{tag}") for tag in ("a", "b")]
means = [m for m in means if m is not None]
if not means:
    print("GATE:FAIL unparsed"); sys.exit(2)
mean = sum(means) / len(means)
est_h = (100 * mean + 4 * 120) / 3600
if est_h <= 16:
    print(f"GATE:OK est={est_h:.2f}h mean_step_s={mean:.1f}")
    sys.exit(0)
print(f"GATE:FAIL H6>{est_h:.2f}h mean_step_s={mean:.1f}")
sys.exit(1)
PY
)
     h5rc=$?
     echo "$h5verdict" >> $OUT/miles/h5_estimate.txt
     case "$h5verdict" in
       GATE:OK*) gate_set miles_H5 "${h5verdict#GATE:}";;
       GATE:FAIL*) gate_set miles_H5 "${h5verdict#GATE:}"; exit 1;;
       *) gate_set miles_H5 "FAIL unparsed"; exit 1;;
     esac
     [ $h5rc = 0 ] || exit $h5rc;;
 H6) gate_ok miles_H5 || exit 1; st 18h ${TAG6:-r0} h6_run100.sh;;     # requeue: move the partial run to not-green-attempts and TAG6=r1
 H7) D=$WORK/runs/h6_run100-${TAG6:-r0}/run
     s1 3h r0 h7_fresh.sh $D/hf/32:$D/dump/rollout_data/33.pt:w32_r33 $D/hf/65:$D/dump/rollout_data/66.pt:w65_r66 $D/hf/98:$D/dump/rollout_data/99.pt:w98_r99 \
        $WORK/models/Inkling-2-Mini-bf16:$D/dump/rollout_data/0.pt:orig_r0 $WORK/models/Inkling-2-Mini-bf16:$D/dump/rollout_data/99.pt:orig_r99
     st 90m r0 h7_replay.sh $D/dump 99;;
esac
