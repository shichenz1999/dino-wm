import os
import numpy as np
import gym
from env.pointmaze.maze_model import (
    MazeEnv, WALL, COLLISION_RADIUS, wall_boxes, clearance_to_walls,
)
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
    REACH_THRESH = 0.5

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.action_dim = self.action_space.shape[0]
    
    def sample_random_init_goal_states(self, seed):
        """Sample init + goal states by CONTINUOUS rejection sampling over the
        walkable area: draw a random (x, y) in the maze's walkable bounding box
        and keep it only if the ball (radius COLLISION_RADIUS) sits clear of
        every WALL cell. Maze-agnostic (reads maze_arr), so init/goal can land
        anywhere in the corridor instead of snapping to cell centres.

        Cell index vs qpos: walls sit at world (w+1, h+1) and the ball body base
        is at (1.2, 1.2), so a cell (w, h) centre is at qpos = (w, h) - OFF where
        OFF = ball_base - 1.0 = 0.2. We must convert in BOTH directions, else the
        sampled positions are offset ~0.2 from the real corridor (and leak into
        walls). OFF is read from the model so it stays correct if the maze
        construction changes.

        Reproducible: a seeded RandomState drives every draw, so the same seed
        yields the same accepted points (rejected draws advance the stream
        deterministically too).
        """
        rs = np.random.RandomState(seed)
        cells = np.array(self.reset_locations + self.goal_locations)  # (K, 2) row,col
        (rmin, cmin), (rmax, cmax) = cells.min(0), cells.max(0)
        m = COLLISION_RADIUS                 # keep the ball's footprint clear of walls
        arr = self.maze_arr
        bid = self.model.body_name2id('particle')
        off = self.model.body_pos[bid][:2] - 1.0     # qpos = cell - off  (≈0.2)

        # Exact disk-vs-wall clearance in the qpos frame, shared with the
        # geodesic erosion (clearance_to_walls). off is symmetric (0.2, 0.2).
        boxes = wall_boxes(arr, offset=off[0])

        def free(x, y):
            return clearance_to_walls((x, y), boxes) >= m - 1e-9  # eps: boundary at ==m

        def generate_state():
            while True:
                # walkable cell range -> qpos range (cell - off), inset by m.
                x = rs.uniform(rmin - 0.5 - off[0] + m, rmax + 0.5 - off[0] - m)
                y = rs.uniform(cmin - 0.5 - off[1] + m, cmax + 0.5 - off[1] - m)
                if free(x, y):
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
        success = pos_dist < self.REACH_THRESH
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
