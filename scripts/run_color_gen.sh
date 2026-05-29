#!/usr/bin/env bash
# Run baseline + 4 color variants in parallel, one GPU each.
# Each invocation launches in its own tmux window for easy monitoring.
set -euo pipefail

cd "$(dirname "$0")/.."

SESSION="color_gen"
N_EVALS=50
SEED=99

# (window_name, gpu, plan-override-args...) for each variant
RUNS=(
  "baseline    0  setting=u_maze"
  "purpleball  1  setting=u_maze variant_type=color variant=purpleball"
  "graywall    2  setting=u_maze variant_type=color variant=graywall"
  "whitefloor  3  setting=u_maze variant_type=color variant=whitefloor"
  "all         4  setting=u_maze variant_type=color variant=all"
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
tmux new-session -d -s "$SESSION" -n "control"
tmux send-keys -t "$SESSION:control" "echo 'color_gen launcher — waiting for runs to finish.'" C-m

for entry in "${RUNS[@]}"; do
  read -ra parts <<< "$entry"
  name="${parts[0]}"
  gpu="${parts[1]}"
  overrides="${parts[*]:2}"

  tmux new-window -t "$SESSION" -n "$name"
  # plan.py → grep its saved dir from the log → visualize_plan.py on it.
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
echo "All 5 runs started. Monitor with:"
echo "  tmux attach -t $SESSION"
echo "Logs at /tmp/plan_<name>.log"
