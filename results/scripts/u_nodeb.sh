#!/usr/bin/env bash
# node-b: one process, four fixed GPU lanes (CUDA_VISIBLE_DEVICES + taskset). No overlapping srun steps.
source "$(dirname "$0")/common.sh"
D=$(dirname "$0")
vime_skip=0; gate_ok vime_local || vime_skip=1
lane() {  # lane GPU CPUSPEC script...
  local gpu=$1 spec=$2 ncpus=$3; shift 3
  taskset -c "$spec" env CUDA_VISIBLE_DEVICES=$gpu MAX_JOBS=$ncpus "$@"
}
mkdir -p "$OUT/logs"
mapfile -t SPECS < <(cpu_lane_specs 16 8 8 32) || { gate_set nodeb_done "FAIL cpuspec"; exit 1; }
NCOUNT=(); for sp in "${SPECS[@]}"; do NCOUNT+=( $(echo "$sp" | tr ',' '\n' | wc -l) ); done
ALLOWED=$(read_step_cpus)
{
  echo "Cpus_allowed_list=$ALLOWED"
  echo "lane0=${SPECS[0]} MAX_JOBS=${NCOUNT[0]}"
  echo "lane1=${SPECS[1]} MAX_JOBS=${NCOUNT[1]}"
  echo "lane2=${SPECS[2]} MAX_JOBS=${NCOUNT[2]}"
  echo "lane3=${SPECS[3]} MAX_JOBS=${NCOUNT[3]}"
} > "$OUT/logs/nodeb-lanes.env.txt"
echo "[$(date -u +%FT%TZ)] node-b lanes start vime_skip=$vime_skip" >> $OUT/steps.txt
pids=()
lane 0 "${SPECS[0]}" "${NCOUNT[0]}" bash -c "bash $D/u_build.sh W10 && bash $D/u_t10a.sh" > "$OUT/logs/lane0.log" 2>&1 &
pids+=($!)
lane 1 "${SPECS[1]}" "${NCOUNT[1]}" bash -c '
  source '"$D"'/common.sh
  gate_wait build_W10 28800 || exit 1
  while [ $((SECONDS)) -lt 28800 ]; do
    gate_ok build_W10 && [ -f $OUT/build/W10/ext.txt ] && head -c4 $OUT/build/W10/ext.txt | grep -q True && break
    sleep 60
  done
  bash '"$D"'/u_t10edit.sh
' > "$OUT/logs/lane1.log" 2>&1 &
pids+=($!)
lane 2 "${SPECS[2]}" "${NCOUNT[2]}" bash -c "bash $D/u_build.sh WQ && bash $D/u_build.sh WB && bash $D/u_qk2.sh" > "$OUT/logs/lane2.log" 2>&1 &
pids+=($!)
if [ $vime_skip = 0 ]; then
  lane 3 "${SPECS[3]}" "${NCOUNT[3]}" bash -c "bash $D/u_vime.sh V0 && bash $D/u_vime.sh V2" > "$OUT/logs/lane3.log" 2>&1 &
  pids+=($!)
else
  gate_set vime_V0 "SKIP vime_local"; log "vime lanes skipped (vime_local gate)"
fi
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
gate_set nodeb_done "$(date -u +%FT%TZ) rc=$rc"
exit $rc