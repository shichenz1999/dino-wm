"""Register the 5x5 maze pool used by the diversity / quantity / grok studies.

23 mazes registered as `maze2d-5x5-<name>-v0`: the 18 distinct training mazes
(the 20-maze pool minus `umaze` and `nmaze`, which are byte-identical to d4rl
U_MAZE / U_MAZE_EVAL and stay reachable via maze2d-umaze-v1 / maze2d-eval-umaze-v1)
plus the 5 held-out unseen test mazes.

Layouts mirror data/scripts/gen_maze_data.MAZES + gen_maze_evals.VAL_MAZES
(# = wall, O = open, G = default goal). Kept here so the env package is
self-contained (no dependency on data/scripts).
"""
from gym.envs.registration import register

# 18 training mazes (pool minus umaze, nmaze) + 5 unseen held-out test mazes.
LAYOUTS = {
    # --- training pool (18; umaze & nmaze omitted = d4rl U_MAZE / U_MAZE_EVAL) ---
    "mcmaze":  ["#####", "#O#O#", "#O#O#", "#OOG#", "#####"],
    "smaze":   ["#####", "##OG#", "#OOO#", "#OO##", "#####"],
    "msmaze":  ["#####", "#OO##", "#OOO#", "##OG#", "#####"],
    "mlmaze":  ["#####", "#OOO#", "#O###", "#G###", "#####"],
    "imaze":   ["#####", "#O#O#", "#OOO#", "#O#G#", "#####"],
    "plus":    ["#####", "##O##", "#OOO#", "##G##", "#####"],
    "open":    ["#####", "#OOO#", "#OOO#", "#OOG#", "#####"],
    "pillar":  ["#####", "#OOO#", "#O#O#", "#OOG#", "#####"],
    "cmaze":   ["#####", "#OOO#", "#O#O#", "#O#G#", "#####"],
    "zmaze":   ["#####", "#O###", "#OOO#", "###G#", "#####"],
    "mzmaze":  ["#####", "###O#", "#OOG#", "#O###", "#####"],
    "barmaze": ["#####", "#####", "#OOG#", "#####", "#####"],
    "lmaze":   ["#####", "#OOO#", "###O#", "###G#", "#####"],
    "tmaze":   ["#####", "#O###", "#OOG#", "#O###", "#####"],
    "mtmaze":  ["#####", "###O#", "#OOO#", "###G#", "#####"],
    "hmaze":   ["#####", "#OOO#", "##O##", "#OOG#", "#####"],
    "emaze":   ["#####", "#OOO#", "#OOO#", "#O#G#", "#####"],
    "fmaze":   ["#####", "#OOG#", "#OO##", "#O###", "#####"],
    # --- held-out unseen test mazes (5) ---
    "rzmaze":     ["#####", "#OO##", "##O##", "##OG#", "#####"],
    "cornermaze": ["#####", "#O###", "#O#O#", "#OOG#", "#####"],
    "jmaze":      ["#####", "#O#O#", "#OOG#", "#O###", "#####"],
    "vmaze":      ["#####", "#OO##", "##OO#", "#OG##", "#####"],
    "pmaze":      ["#####", "#OOO#", "#OO##", "#OG##", "#####"],
}

for _name, _rows in LAYOUTS.items():
    register(
        id=f"maze2d-5x5-{_name}-v0",
        entry_point="env.pointmaze:PointMazeWrapper",
        max_episode_steps=150,
        kwargs={
            "maze_spec": "\\".join(_rows),   # d4rl spec: rows joined by '\'
            "reward_type": "sparse",
            "reset_target": False,
            "ref_min_score": 0.0,
            "ref_max_score": 1.0,
        },
    )
