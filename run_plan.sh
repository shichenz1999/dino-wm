#!/bin/bash
export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia

python plan.py --config-name plan_point_maze.yaml \
  model_name=point_maze \
  ckpt_base_path=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints/pretrained \
  n_evals=5 \
  hydra/launcher=basic
