#!/usr/bin/env bash
# Re-run baseline plans for every maze setting (except u_maze, which is
# already covered by run_color_gen.sh). Pinned to GPUs 5/6/7, each window
# runs its share of settings sequentially with chained visualize.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="baselines"
N_EVALS=50
SEED=99

# Group settings across 3 GPUs to balance workload (small/open are fastest).
BATCHES=(
  "5  u_maze_eval small open"
  "6  medium medium_eval"
  "7  large large_eval"
)

COMMON="--config-name plan_point_maze.yaml \
  model_name=point_maze \
  ckpt_base_path=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints/pretrained \
  n_evals=${N_EVALS} seed=${SEED} hydra/launcher=basic"

ENV_SETUP="source /local_data/sz4968/miniforge3/etc/profile.d/conda.sh && \
  conda activate dino_wm && \
  export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data && \
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia && \
  export WANDB_MODE=disabled"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n control

for entry in "${BATCHES[@]}"; do
  read -ra parts <<< "$entry"
  gpu="${parts[0]}"
  settings=("${parts[@]:1}")
  win_name="gpu${gpu}"
  tmux new-window -t "$SESSION" -n "$win_name"
  # Build a chained command: for each setting, plan → grep saved dir → visualize.
  cmd="$ENV_SETUP && export CUDA_VISIBLE_DEVICES=$gpu"
  for setting in "${settings[@]}"; do
    log="/tmp/plan_${setting}.log"
    viz="/tmp/viz_${setting}.log"
    # Wipe stale outputs for that setting first (old format).
    cmd="$cmd && rm -rf plan_outputs/${setting}/baseline/2025* plan_outputs/${setting}/baseline/2026*"
    cmd="$cmd && echo '== plan: $setting (GPU $gpu)' && \
      python plan.py $COMMON setting=$setting 2>&1 | tee $log && \
      RUN_DIR=\$(grep -oP 'Planning result saved dir: \\K.*' $log | head -1) && \
      echo '== viz: '\$RUN_DIR && \
      python scripts/visualize_plan.py \$RUN_DIR --per-iter 2>&1 | tee $viz"
  done
  tmux send-keys -t "$SESSION:$win_name" "$cmd" C-m
  echo "[GPU $gpu] launched: ${settings[*]}"
done

echo ""
echo "Monitor with:  tmux attach -t $SESSION"
echo "Logs at /tmp/plan_<setting>.log and /tmp/viz_<setting>.log"
