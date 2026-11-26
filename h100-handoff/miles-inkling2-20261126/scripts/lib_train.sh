# sourced by the stage scripts: run_miles <num_rollout> "<extra args>" [subdir]  -> logs to $RUN[/subdir]/train.log
# Two nodes x 4 GPUs on an external Ray cluster (node_entry.sh). The recipe sets --expert-model-parallel-size from
# num_gpus_per_node (4), so EP8 is appended; the SGLang engine size and memory fraction come from $WORK/h100.env (H2).
[ -f "$WORK/h100.env" ] && source "$WORK/h100.env"
SG_TP=${SG_TP:-1}; SG_MEM=${SG_MEM:-0.55}
h100_args() { echo "--expert-model-parallel-size 8 --rollout-num-gpus-per-engine $SG_TP --sglang-mem-fraction-static $SG_MEM"; }
# Between runs: stop any Ray job still running (a timed-out launcher leaves its job behind), then kill leftover
# SGLang servers on both nodes (they are subprocesses of Ray actors; the launcher only cleans node 0).
kill_engines() {
  python3 -c "
from ray.job_submission import JobSubmissionClient
c = JobSubmissionClient('http://127.0.0.1:8265')
for j in c.list_jobs():
    if str(j.status).split('.')[-1] in ('PENDING', 'RUNNING'): print('stopping', j.submission_id, c.stop_job(j.submission_id))
" >> "$RUN/kill_engines.log" 2>&1 || true
  python3 -c "import miles.utils.external_utils.command_utils.legacy as U; U.exec_command_multi_node('pkill -9 -f \"[s]glang.launch_server\"; sleep 3; true')" > /dev/null 2>&1 || true
}
run_miles() {
  local O=$RUN; [ -n "${3:-}" ] && O=$RUN/$3; mkdir -p "$O"
  cd /root/miles
  kill_engines
  echo "SG_TP=$SG_TP SG_MEM=$SG_MEM" > "$O/h100_config.txt"
  # RUN_TIMEOUT (seconds, 0 = none): probes set it, since a refused option can retry forever (R3 did on B200)
  timeout --kill-after=60 ${RUN_TIMEOUT:-0} python3 -c "
import sys
from scripts.run_inkling2_mini import ScriptArgs, execute
execute(ScriptArgs(num_nodes=2, num_gpus_per_node=4, num_rollout=int(sys.argv[1]), model_dir=sys.argv[2], data_dir=sys.argv[3], output_dir=sys.argv[4], extra_args=sys.argv[5]))
" "$1" "$WORK/models" "$WORK/datasets" "$O/out" "$2 $(h100_args)" > "$O/train.log" 2>&1
  local rc=$?
  echo "train rc=$rc" | tee -a "$O/train.log"
  kill_engines
  return $rc
}
