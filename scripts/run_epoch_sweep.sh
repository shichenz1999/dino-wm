#!/bin/bash
# Epoch-sweep eval: reproduce_eN checkpoints across several mazes, plan (trace ON)
# + vis, distributed over a GPU list with VRAM gate + retry + skip-if-done.
# Usage:  bash run_epoch_sweep.sh "0 1"
set -u
cd /local_data/sz4968/world-model/experiments/dino-wm/dino-wm
export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPUS=(${1:-"0 1"})
CKPT_BASE=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints
RESULTS=/local_data/sz4968/world-model/experiments/dino-wm/results
LOGDIR=$RESULTS/logs
EVAL=$RESULTS/point_maze/eval
mkdir -p "$LOGDIR"
MAX_RETRY=10
MIN_FREE_MIB=19000

EPOCHS=(5 10 20 50 100)
SETTINGS=(u_maze u_maze_eval small open)

# build job list: "epoch:setting"
JOBS=()
for e in "${EPOCHS[@]}"; do for s in "${SETTINGS[@]}"; do JOBS+=("$e:$s"); done; done

wait_for_gpu_mem() {
  local gpu=$1
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu" 2>/dev/null | tr -d ' ')
    [ -n "$free" ] && [ "$free" -ge "$MIN_FREE_MIB" ] && return 0
    sleep 30
  done
}

result_dir() {  # epoch:setting -> latest run dir with evals.json
  IFS=':' read -r e s <<< "$1"
  find "$EVAL/reproduce_e$e/$s/baseline" -maxdepth 2 -name evals.json -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2
}

run_job() {
  local gpu=$1 spec=$2
  IFS=':' read -r e s <<< "$spec"
  local tag="reproduce_e${e}_${s}"
  local rd; rd=$(result_dir "$spec")
  if [ -n "$rd" ] && [ -f "$rd/evals.json" ] && [ -f "$rd/summary.png" ]; then
    echo "$(date +%H:%M:%S) [GPU $gpu] SKIP $tag (done)"; return 0
  fi
  echo "$(date +%H:%M:%S) [GPU $gpu] WAIT-MEM $tag"; wait_for_gpu_mem "$gpu"
  echo "$(date +%H:%M:%S) [GPU $gpu] PLAN $tag (epoch $e, $s)"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n dino_wm python -u plan.py --config-name plan_point_maze.yaml \
    env_name=point_maze ckpt_id=reproduce_e$e model_epoch=$e ckpt_base_path=$CKPT_BASE \
    setting=$s n_evals=50 planner.max_iter=5 trace.enabled=true hydra/launcher=basic \
    > "$LOGDIR/sweep_${tag}.log" 2>&1
  rd=$(result_dir "$spec")
  if [ -z "$rd" ] || [ ! -f "$rd/evals.json" ]; then echo "$(date +%H:%M:%S) [GPU $gpu] PLAN-FAIL $tag"; return 1; fi
  echo "$(date +%H:%M:%S) [GPU $gpu] VIS $tag"
  CUDA_VISIBLE_DEVICES=$gpu conda run -n dino_wm python scripts/visualize_plan.py "$rd" >> "$LOGDIR/sweep_${tag}.log" 2>&1
  [ ! -f "$rd/summary.png" ] && { echo "$(date +%H:%M:%S) [GPU $gpu] VIS-FAIL $tag"; return 2; }
  local sr; sr=$(conda run -n dino_wm python3 -c "import json;print(json.load(open('$rd/evals.json'))['summary']['final']['success_rate'])" 2>/dev/null)
  echo "$(date +%H:%M:%S) [GPU $gpu] OK $tag SR=$sr"; return 0
}

declare -A PID2GPU PID2SPEC RETRY
queue=("${JOBS[@]}"); qi=0
launch() { local g=$1 spec="${queue[$qi]}"; qi=$((qi+1)); run_job "$g" "$spec" & PID2GPU[$!]=$g; PID2SPEC[$!]=$spec; }
for g in "${GPUS[@]}"; do [ $qi -lt ${#queue[@]} ] && launch "$g"; done
while [ ${#PID2GPU[@]} -gt 0 ]; do
  for pid in "${!PID2GPU[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid"; rc=$?; g=${PID2GPU[$pid]}; spec=${PID2SPEC[$pid]}; unset PID2GPU[$pid] PID2SPEC[$pid]
      if [ $rc -ne 0 ]; then r=${RETRY[$spec]:-0}; if [ "$r" -lt "$MAX_RETRY" ]; then RETRY[$spec]=$((r+1)); queue+=("$spec"); echo "$(date +%H:%M:%S) requeue ($((r+1))): $spec"; else echo "GIVE-UP $spec"; fi; fi
      [ $qi -lt ${#queue[@]} ] && launch "$g"
    fi
  done
  sleep 20
done
echo "=== EPOCH SWEEP DONE ==="
for e in "${EPOCHS[@]}"; do for s in "${SETTINGS[@]}"; do
  rd=$(result_dir "$e:$s"); sr=$([ -n "$rd" ]&&conda run -n dino_wm python3 -c "import json;print(json.load(open('$rd/evals.json'))['summary']['final']['success_rate'])" 2>/dev/null||echo "-")
  echo "  reproduce_e$e / $s: SR=$sr"
done; done
