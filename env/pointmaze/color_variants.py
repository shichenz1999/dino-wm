"""Color variants for generalization experiments.

Each entry in COLOR_VARIANTS registers a new gym ID with overridden color
kwargs forwarded to MazeEnv. Defaults stay as in maze_model.py — only
listed args differ.
"""
from gym.envs.registration import register
from .maze_model import U_MAZE

COLOR_VARIANTS = {
    'maze2d-umaze-purpleball-v1': {'ball_rgba': '0.8 0.2 0.8 1'},
    'maze2d-umaze-graywall-v1':   {'wall_rgba': '0.5 0.5 0.5 1'},
    'maze2d-umaze-whitefloor-v1': {'floor_rgb1': '0.95 0.95 0.95',
                                   'floor_rgb2': '0.85 0.85 0.85'},
    'maze2d-umaze-all-v1':        {'ball_rgba': '0.8 0.2 0.8 1',
                                   'wall_rgba': '0.5 0.5 0.5 1',
                                   'floor_rgb1': '0.95 0.95 0.95',
                                   'floor_rgb2': '0.85 0.85 0.85'},
}

for env_id, color_kwargs in COLOR_VARIANTS.items():
    register(
        id=env_id,
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={'maze_spec': U_MAZE, **color_kwargs},
    )
