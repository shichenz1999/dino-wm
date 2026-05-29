"""Continuous geodesic distance-from-goal field over the ball-center space
(free space eroded by the ball radius), solved with the eikonal method (skfmm)."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from typing import Tuple
import numpy as np

from .maze_model import parse_maze, WALL, COLLISION_RADIUS

Cell = Tuple[int, int]

# Geodesic-field grid spacing (qpos units); smaller = more accurate but slower.
RESOLUTION = 0.02


@dataclass
class GeoField:
    phi: np.ndarray   # geodesic distance from goal; np.inf where unreachable
    max_d: float      # max finite distance = farthest reachable point from goal
    x0: float         # world coord of pixel (0, .)
    y0: float         # world coord of pixel (., 0)
    res: float        # grid spacing (qpos units)
    nearest: tuple    # (ii, jj) index maps -> nearest finite pixel (robust query)

    def distance(self, qpos) -> float:
        """Geodesic distance from goal to a continuous ball position."""
        i = int(round((float(qpos[0]) - self.x0) / self.res))
        j = int(round((float(qpos[1]) - self.y0) / self.res))
        i = min(max(i, 0), self.phi.shape[0] - 1)
        j = min(max(j, 0), self.phi.shape[1] - 1)
        d = self.phi[i, j]
        if not np.isfinite(d):  # ball landed in/just past a blocked pixel
            d = self.phi[self.nearest[0][i, j], self.nearest[1][i, j]]
        return float(d)


@lru_cache(maxsize=256)
def geo_field(maze_spec: str, goal_pix: Cell,
              ball_radius: float = COLLISION_RADIUS,
              resolution: float = RESOLUTION) -> GeoField:
    """Geodesic field from goal_pix (goal's grid index); cached per pixel so
    one solve is reused across MPC iterations."""
    import skfmm
    from scipy.ndimage import distance_transform_edt

    maze_arr = parse_maze(maze_spec)
    width, height = maze_arr.shape
    res = float(resolution)
    x0 = y0 = -0.5  # outer edge of the border cells
    nx = int(round(width / res)) + 1
    ny = int(round(height / res)) + 1

    xs = x0 + res * np.arange(nx)
    ys = y0 + res * np.arange(ny)
    cw = np.clip(np.floor(xs + 0.5).astype(int), 0, width - 1)
    ch = np.clip(np.floor(ys + 0.5).astype(int), 0, height - 1)
    open_mask = (maze_arr != WALL)[cw[:, None], ch[None, :]]

    # Erode free space by the ball radius -> where the ball CENTER may sit.
    edt = distance_transform_edt(open_mask) * res  # dist to nearest wall pixel
    center_free = edt >= ball_radius

    # Goal must sit in the ball-center free space (a goal too close to a wall
    # is a bug -> fail loudly, don't silently fix).
    gi, gj = goal_pix
    assert 0 <= gi < nx and 0 <= gj < ny, \
        f"goal pixel {goal_pix} out of grid bounds ({nx}x{ny})"
    assert center_free[gi, gj], \
        f"goal pixel {goal_pix} not in ball-center free space (too close to a wall)"

    # Eikonal solve from the goal over the center-free region.
    phi0 = np.ones((nx, ny))
    phi0[gi, gj] = -1.0
    dist = skfmm.distance(np.ma.MaskedArray(phi0, mask=~center_free), dx=res)
    phi = np.ma.filled(np.abs(dist), np.inf)

    finite = np.isfinite(phi)
    max_d = float(phi[finite].max())
    ii, jj = distance_transform_edt(~finite, return_distances=False,
                                    return_indices=True)
    return GeoField(phi, max_d, x0, y0, res, (ii, jj))


def field_for(maze_spec: str, goal_qpos: np.ndarray,
              ball_radius: float = COLLISION_RADIUS,
              resolution: float = RESOLUTION) -> GeoField:
    """Goal qpos -> cached geodesic field, seeded at the goal's true grid pixel
    (no goal-noise assumption)."""
    res = float(resolution)
    gi = int(round((float(goal_qpos[0]) + 0.5) / res))
    gj = int(round((float(goal_qpos[1]) + 0.5) / res))
    return geo_field(maze_spec, (gi, gj), ball_radius, resolution)


def avoid_eval(maze_spec, goal_qpos, ball_qpos, margin=0.5, frac=None):
    """Avoid-goal criterion. Returns (dist_to_far, thr, success):
    dist_to_far = max_d - geo(ball, goal); success = dist_to_far < thr."""
    field = field_for(maze_spec, goal_qpos)
    far = field.max_d - field.distance(ball_qpos)
    thr = (field.max_d * (1.0 - frac)) if frac is not None else float(margin)
    return far, thr, bool(far < thr)
