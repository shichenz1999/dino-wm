"""Task variants for objective-flip experiments.

Each entry registers a gym id that maps to a STANDARD MazeEnv (same env as
baseline) — task semantics differ only in the planner's objective + success
criterion, which the plan config handles separately.
"""
from gym.envs.registration import register
from .maze_model import U_MAZE

TASK_VARIANTS = {
    'maze2d-umaze-avoidgoal-v1': {},   # same env as baseline; differs in objective
}

for env_id, kwargs in TASK_VARIANTS.items():
    register(
        id=env_id,
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={'maze_spec': U_MAZE, **kwargs},
    )
