#!/bin/bash
set -e
cd /local_data/sz4968/world-model/experiments/dino-wm/dino-wm

export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia

CKPT_ID=straight_aggcos1e-1
CKPT_BASE=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints
N_EVALS=50
EPOCH=20

for SETTING in u_maze u_maze_eval medium large; do
    echo "========================================"
    echo "Running: setting=${SETTING}, ckpt_id=${CKPT_ID}, n_evals=${N_EVALS}"
    echo "========================================"
    python plan.py --config-name plan_point_maze.yaml \
        env_name=point_maze \
        ckpt_id=${CKPT_ID} \
        ckpt_base_path=${CKPT_BASE} \
        model_epoch=${EPOCH} \
        setting=${SETTING} \
        n_evals=${N_EVALS} \
        hydra/launcher=basic
    echo "Done: ${SETTING}"
    echo ""
done

echo "All settings complete."
