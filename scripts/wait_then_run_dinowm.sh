#!/bin/bash
echo "Waiting for all straightening eval sessions (se_g*) to finish..."

while true; do
    RUNNING=$(tmux list-sessions 2>/dev/null | grep -c "^se_g" || true)
    if [ "$RUNNING" -eq 0 ]; then
        break
    fi
    echo "$(date '+%H:%M:%S') - $RUNNING straightening sessions still running, waiting..."
    sleep 120
done

echo "$(date '+%H:%M:%S') - All straightening sessions done. Starting dinowm_released re-eval..."

# Delete old dinowm_released results
rm -rf /local_data/sz4968/world-model/experiments/dino-wm/results/point_maze/eval/dinowm_released/
echo "Cleaned old dinowm_released results."

LOG=/local_data/sz4968/world-model/experiments/dino-wm/results
SCR=/local_data/sz4968/world-model/experiments/dino-wm/dino-wm/scripts/run_dinowm_all.sh

# Same distribution as straightening: 7 GPUs, 13 runs
tmux new-session -d -s dw_g0 "conda run -n dino_wm bash $SCR 0 u_maze:baseline:baseline u_maze:color:purpleball 2>&1 | tee $LOG/dw_g0.log"
tmux new-session -d -s dw_g1 "conda run -n dino_wm bash $SCR 1 u_maze_eval:baseline:baseline u_maze:color:graywall 2>&1 | tee $LOG/dw_g1.log"
tmux new-session -d -s dw_g2 "conda run -n dino_wm bash $SCR 2 medium:baseline:baseline u_maze:color:whitefloor 2>&1 | tee $LOG/dw_g2.log"
tmux new-session -d -s dw_g3 "conda run -n dino_wm bash $SCR 3 medium_eval:baseline:baseline u_maze:color:all 2>&1 | tee $LOG/dw_g3.log"
tmux new-session -d -s dw_g4 "conda run -n dino_wm bash $SCR 4 large:baseline:baseline small:baseline:baseline 2>&1 | tee $LOG/dw_g4.log"
tmux new-session -d -s dw_g5 "conda run -n dino_wm bash $SCR 5 large_eval:baseline:baseline open:baseline:baseline 2>&1 | tee $LOG/dw_g5.log"
tmux new-session -d -s dw_g6 "conda run -n dino_wm bash $SCR 6 u_maze:task:avoid_goal 2>&1 | tee $LOG/dw_g6.log"

echo "All 7 dinowm_released sessions launched."
