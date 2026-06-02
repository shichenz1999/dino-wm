#!/usr/bin/env bash
# Run task-variant ablations in parallel, one GPU each.
# Same WM checkpoint, different planner objective + success criterion per task.
set -euo pipefail
cd "$(dirname "$0")/.."

SESSION="task_gen"
N_EVALS=50
SEED=99

# (window_name, gpu, plan-override-args...) for each task variant.
# Each entry bundles the variant tag (for dashboard schema) AND the planner
# overrides needed to flip the task (objective.invert, success.mode, etc.).
RUNS=(
  "avoid_goal  3  setting=u_maze variant_type=task variant=avoid_goal objective.invert=true"
  # Add more task variants here, e.g.:
  # "multi_goal  1  setting=u_maze variant_type=task variant=multi_goal ..."
)

COMMON="--config-name plan_point_maze.yaml \
  env_name=point_maze \
  ckpt_id=dinowm_released \
  ckpt_base_path=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints \
  n_evals=${N_EVALS} seed=${SEED} hydra/launcher=basic"

ENV_SETUP="source /local_data/sz4968/miniforge3/etc/profile.d/conda.sh && \
  conda activate dino_wm && \
  export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data && \
  export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia && \
  export WANDB_MODE=disabled"

tmux kill-session -t "$SESSION" 2>/dev/null || true
tmux new-session -d -s "$SESSION" -n "control"
tmux send-keys -t "$SESSION:control" "echo 'task_gen launcher — waiting for runs to finish.'" C-m

for entry in "${RUNS[@]}"; do
  read -ra parts <<< "$entry"
  name="${parts[0]}"
  gpu="${parts[1]}"
  overrides="${parts[*]:2}"

  tmux new-window -t "$SESSION" -n "$name"
  tmux send-keys -t "$SESSION:$name" "$ENV_SETUP && \
    export CUDA_VISIBLE_DEVICES=$gpu && \
    LOG=/tmp/plan_${name}.log && \
    python plan.py $COMMON $overrides 2>&1 | tee \$LOG && \
    RUN_DIR=\$(grep -oP 'Planning result saved dir: \\K.*' \$LOG | head -1) && \
    echo '== plan done, visualizing' \$RUN_DIR && \
    python scripts/visualize_plan.py \$RUN_DIR --per-iter 2>&1 | tee /tmp/viz_${name}.log" C-m
  echo "[$name] launched on GPU $gpu (tmux window: $SESSION:$name)"
done

echo ""
echo "Monitor with:  tmux attach -t $SESSION"
echo "Logs at /tmp/plan_<name>.log and /tmp/viz_<name>.log"
