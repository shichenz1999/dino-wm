import os
import numpy as np
import gym
from env.pointmaze.maze_model import MazeEnv
from utils import aggregate_dct

STATE_RANGES = np.array([
    [0.39318362, 3.2198412],  # Range for first dimension
    [0.62660956, 3.2187355],  # Range for second dimension
    [-5.2262554, 5.2262554],  # Range for third dimension
    [-5.2262554, 5.2262554],  # Range for fourth dimension
    # [0.90001136, 3.0999563],  # Range for first dimension of target
    # [0.9000267, 3.0999668]    # Range for second dimension of target
])

class PointMazeWrapper(MazeEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """Sample init + goal states. Maze-agnostic: picks from valid cells in
        self.reset_locations (auto-computed from maze_arr by MazeEnv).
        Jitter matches MazeEnv's built-in reset (±0.1) to stay clear of walls.
        """
        rs = np.random.RandomState(seed)
        cells = self.reset_locations + self.goal_locations  # all walkable cells

        def generate_state():
            w, h = cells[rs.randint(len(cells))]
            x = w + rs.uniform(-0.1, 0.1)
            y = h + rs.uniform(-0.1, 0.1)
            return np.array([
                x, y,
                rs.uniform(STATE_RANGES[2][0], STATE_RANGES[2][1]),
                rs.uniform(STATE_RANGES[3][0], STATE_RANGES[3][1]),
            ])

        return generate_state(), generate_state()
    
    def update_env(self, env_info):
        pass 
    
    def eval_state(self, goal_state, cur_state):
        pos_dist = np.linalg.norm(goal_state[..., :2] - cur_state[..., :2], axis=-1)
        vel_dist = np.linalg.norm(goal_state[..., 2:] - cur_state[..., 2:], axis=-1)
        state_dist = np.linalg.norm(goal_state - cur_state, axis=-1)
        success = pos_dist < 0.5
        return {
            'success': success,
            'state_dist': state_dist,
            'pos_dist': pos_dist,
            'vel_dist': vel_dist,
        }

    def prepare(self, seed, init_state):
        """
        Reset with controlled init_state
        obs: (H W C)
        state: (state_dim)
        """
        self.prepare_for_render()
        self.seed(seed)
        self.set_init_state(init_state)
        obs, state = self.reset()
        return obs, state

    def step_multiple(self, actions):
        """
        infos: dict, each key has shape (T, ...)
        """
        obses = []
        rewards = []
        dones = []
        infos = []
        for action in actions:
            o, r, d, info = self.step(action)
            obses.append(o)
            rewards.append(r)
            dones.append(d)
            infos.append(info)
        obses = aggregate_dct(obses)
        rewards = np.stack(rewards)
        dones = np.stack(dones)
        infos = aggregate_dct(infos)
        return obses, rewards, dones, infos

    def rollout(self, seed, init_state, actions):
        """
        only returns np arrays of observations and states
        seed: int
        init_state: (state_dim, )
        actions: (T, action_dim)
        obses: dict (T, H, W, C)
        states: (T, D)
        """
        obs, state = self.prepare(seed, init_state)
        obses, rewards, dones, infos = self.step_multiple(actions)
        for k in obses.keys():
            obses[k] = np.vstack([np.expand_dims(obs[k], 0), obses[k]])
        states = np.vstack([np.expand_dims(state, 0), infos["state"]])
        states = np.stack(states)
        return obses, states
