#!/bin/bash
set -e
cd /local_data/sz4968/world-model/experiments/dino-wm/dino-wm

export DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/sz4968/.mujoco/mujoco210/bin:/usr/lib/nvidia

GPU=$1
shift

CKPT_ID=straight_aggcos1e-1
CKPT_BASE=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints
N_EVALS=50
EPOCH=20

for SPEC in "$@"; do
    SETTING=$(echo "$SPEC" | cut -d: -f1)
    VARIANT_TYPE=$(echo "$SPEC" | cut -d: -f2)
    VARIANT=$(echo "$SPEC" | cut -d: -f3)

    EXTRA_ARGS=""
    if [ "$VARIANT_TYPE" != "baseline" ]; then
        EXTRA_ARGS="variant_type=${VARIANT_TYPE} variant=${VARIANT}"
    fi
    if [ "$VARIANT_TYPE" = "task" ] && [ "$VARIANT" = "avoid_goal" ]; then
        EXTRA_ARGS="$EXTRA_ARGS task=avoid_goal"
    fi

    echo "========================================"
    echo "GPU=${GPU} | setting=${SETTING} | variant_type=${VARIANT_TYPE} | variant=${VARIANT}"
    echo "========================================"

    CUDA_VISIBLE_DEVICES=${GPU} python plan.py --config-name plan_point_maze.yaml \
        env_name=point_maze \
        ckpt_id=${CKPT_ID} \
        ckpt_base_path=${CKPT_BASE} \
        model_epoch=${EPOCH} \
        setting=${SETTING} \
        ${EXTRA_ARGS} \
        n_evals=${N_EVALS} \
        hydra/launcher=basic

    # Find the latest run dir and visualize
    RUN_DIR=$(find ../results/point_maze/eval/${CKPT_ID}/${SETTING}/ -maxdepth 3 -name "evals.json" -printf '%T@ %h\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2)
    if [ -n "$RUN_DIR" ]; then
        echo "Visualizing: ${RUN_DIR}"
        CUDA_VISIBLE_DEVICES=${GPU} python scripts/visualize_plan.py "$RUN_DIR"
    fi

    echo "Done: ${SETTING}/${VARIANT_TYPE}/${VARIANT}"
    echo ""
done

echo "GPU ${GPU} all tasks complete."
