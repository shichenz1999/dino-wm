#!/bin/bash
# Autonomous run of all 22 (ckpt, setting) jobs: plan (trace on) + standard vis,
# distributed across a GPU list, with per-job verification and retry. Safe to
# leave unattended. Usage:  bash run_all_auto.sh "1 6 7"
set -u
cd /local_data/sz4968/world-model/experiments/dino-wm/dino-wm
export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=(${1:-"1 6 7"})
CKPT_BASE=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints
RESULTS=/local_data/sz4968/world-model/experiments/dino-wm/results
LOGDIR=$RESULTS/logs
EVAL=$RESULTS/point_maze/eval
mkdir -p "$LOGDIR"
MAX_RETRY=5
MIN_FREE_MIB=19000   # only start a job on a GPU with at least this much free VRAM

# Block until the given physical GPU has >= MIN_FREE_MIB free (avoids the
# run-19min-then-OOM waste under heavy contention from other users).
wait_for_gpu_mem() {
  local gpu=$1
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ] && return 0
    sleep 30
  done
}

# "ckpt_id:model_epoch:setting:variant_type:variant"  (dispatch order A->D)
JOBS=(
  "straight_aggcos1e-1:20:u_maze:baseline:baseline"      "dinowm_released:latest:u_maze:baseline:baseline"
  "straight_aggcos1e-1:20:u_maze_eval:baseline:baseline" "dinowm_released:latest:u_maze_eval:baseline:baseline"
  "straight_aggcos1e-1:20:medium:baseline:baseline"      "dinowm_released:latest:medium:baseline:baseline"
  "straight_aggcos1e-1:20:large:baseline:baseline"       "dinowm_released:latest:large:baseline:baseline"
  "straight_aggcos1e-1:20:u_maze:task:avoid_goal"        "dinowm_released:latest:u_maze:task:avoid_goal"
  "straight_aggcos1e-1:20:small:baseline:baseline"       "dinowm_released:latest:small:baseline:baseline"
  "straight_aggcos1e-1:20:open:baseline:baseline"        "dinowm_released:latest:open:baseline:baseline"
  "straight_aggcos1e-1:20:u_maze:color:purpleball"       "dinowm_released:latest:u_maze:color:purpleball"
  "straight_aggcos1e-1:20:u_maze:color:graywall"         "dinowm_released:latest:u_maze:color:graywall"
  "straight_aggcos1e-1:20:u_maze:color:whitefloor"       "dinowm_released:latest:u_maze:color:whitefloor"
  "straight_aggcos1e-1:20:u_maze:color:all"              "dinowm_released:latest:u_maze:color:all"
)

# resolve the result dir for a job spec (latest timestamp under its setting/variant)
result_dir() {
  IFS=':' read -r ckpt epoch setting vtype variant <<< "$1"
  local base="$EVAL/$ckpt/$setting"
  if [ "$vtype" = "baseline" ]; then base="$base/baseline"; else base="$base/$vtype/$variant"; fi
  find "$base" -maxdepth 2 -name evals.json -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2
}

run_job() {
  local gpu=$1 spec=$2
  IFS=':' read -r ckpt epoch setting vtype variant <<< "$spec"
  local extra=""
  [ "$vtype" != "baseline" ] && extra="variant_type=$vtype variant=$variant"
  [ "$vtype" = "task" ] && [ "$variant" = "avoid_goal" ] && extra="$extra task=avoid_goal"
  local tag="${ckpt}_${setting}_${vtype}_${variant}"
  # skip if already fully done (resumable / don't redo the validation run)
  local done_rd; done_rd=$(result_dir "$spec")
  if [ -n "$done_rd" ] && [ -f "$done_rd/evals.json" ] && [ -f "$done_rd/summary.png" ]; then
    echo "$(date +%H:%M:%S) [GPU $gpu] SKIP  $tag (already complete)"; return 0
  fi
  echo "$(date +%H:%M:%S) [GPU $gpu] WAIT-MEM $tag (need ${MIN_FREE_MIB}MiB free)"
  wait_for_gpu_mem "$gpu"
  echo "$(date +%H:%M:%S) [GPU $gpu] PLAN  $tag"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n dino_wm python -u plan.py --config-name plan_point_maze.yaml \
    env_name=point_maze ckpt_id=$ckpt model_epoch=$epoch ckpt_base_path=$CKPT_BASE \
    setting=$setting $extra n_evals=50 planner.max_iter=5 trace.enabled=true hydra/launcher=basic \
    > "$LOGDIR/plan_${tag}.log" 2>&1
  local rd; rd=$(result_dir "$spec")
  if [ -z "$rd" ] || [ ! -f "$rd/evals.json" ]; then
    echo "$(date +%H:%M:%S) [GPU $gpu] PLAN-FAIL $tag"; return 1
  fi
  echo "$(date +%H:%M:%S) [GPU $gpu] VIS   $tag -> $(basename "$rd")"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n dino_wm python scripts/visualize_plan.py "$rd" \
    > "$LOGDIR/vis_${tag}.log" 2>&1
  if [ ! -f "$rd/summary.png" ]; then
    echo "$(date +%H:%M:%S) [GPU $gpu] VIS-FAIL $tag (plan ok, vis failed)"; return 2
  fi
  echo "$(date +%H:%M:%S) [GPU $gpu] OK    $tag"; return 0
}

# job-queue with retry: each slot runs one job, refills from the queue
declare -A PID2GPU PID2SPEC
declare -A RETRY
queue=("${JOBS[@]}")
qi=0
launch() {  # $1=gpu
  local g=$1 spec="${queue[$qi]}"; qi=$((qi+1))
  run_job "$g" "$spec" & PID2GPU[$!]=$g; PID2SPEC[$!]=$spec
}
for g in "${GPUS[@]}"; do [ $qi -lt ${#queue[@]} ] && launch "$g"; done
while [ ${#PID2GPU[@]} -gt 0 ]; do
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?
      g=${PID2GPU[$pid]}; spec=${PID2SPEC[$pid]}
      unset PID2GPU[$pid] PID2SPEC[$pid]
      if [ $rc -ne 0 ]; then
        r=${RETRY[$spec]:-0}
        if [ "$r" -lt "$MAX_RETRY" ]; then
          RETRY[$spec]=$((r+1)); queue+=("$spec")
          echo "$(date +%H:%M:%S) requeue ($((r+1))/$MAX_RETRY): $spec"
        else
          echo "$(date +%H:%M:%S) GIVE-UP after $MAX_RETRY: $spec"
        fi
      fi
      [ $qi -lt ${#queue[@]} ] && launch "$g"
    fi
  done
  sleep 20
done

echo "=================== FINAL VERIFICATION ==================="
ok=0; bad=0
for spec in "${JOBS[@]}"; do
  rd=$(result_dir "$spec")
  if [ -n "$rd" ] && [ -f "$rd/evals.json" ] && [ -f "$rd/summary.png" ]; then
    sr=$(conda run -n dino_wm python3 -c "import json;print(json.load(open('$rd/evals.json'))['summary']['final']['success_rate'])" 2>/dev/null)
    echo "OK   $spec  SR=$sr"; ok=$((ok+1))
  else
    echo "BAD  $spec  (evals=$([ -f "$rd/evals.json" ]&&echo y||echo n) summary=$([ -f "$rd/summary.png" ]&&echo y||echo n))"; bad=$((bad+1))
  fi
done
echo "DONE: $ok ok, $bad bad, of ${#JOBS[@]}"
