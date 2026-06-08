#!/usr/bin/env python
"""Visualize how the CEM search converges, from a plan_trace.pkl.

For a chosen (eval, MPC round), rolls every candidate action sequence of each
optimization step out in the real maze and plots the resulting ball paths,
coloured by cost — so you watch the 300-candidate cloud narrow toward the goal
over the opt steps, with the finally-chosen sequence highlighted. Produces, per
(eval, round):
    cem_eval{E}_round{R}.mp4   — one frame per opt step (the narrowing cloud)
    cem_eval{E}_round{R}.png   — small-multiples grid of the opt steps
and a per-eval cem_eval{E}_rounds.png summarizing the chosen path each round.

Usage:
    python scripts/visualize_cem_search.py <run_dir> [--evals 0 1] [--max-cands 300]
"""
import os
import sys
import pickle
import argparse
from pathlib import Path
import numpy as np
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.animation as animation
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import env.pointmaze  # noqa: F401  registers gym envs
import gym
from env.pointmaze.maze_model import WALL, wall_boxes

# setting -> registered baseline env id (mirror of plan_point_maze env_id_map)
ENV_ID = {
    "u_maze": "maze2d-umaze-v1", "u_maze_eval": "maze2d-eval-umaze-v1",
    "medium": "maze2d-medium-v1", "medium_eval": "maze2d-eval-medium-v1",
    "large": "maze2d-large-v1", "large_eval": "maze2d-eval-large-v1",
    "small": "maze2d-small-v0", "open": "maze2d-open-v0",
}


def load_action_stats(env_name, data_root):
    """Action mean/std used to denormalize planner actions (per raw 2-dim)."""
    import torch
    from datasets.point_maze_dset import PointMazeDataset
    dpath = Path(data_root) / env_name
    dset = PointMazeDataset(data_path=str(dpath), normalize_action=True)
    return dset.action_mean.numpy(), dset.action_std.numpy()


def draw_maze(ax, maze_arr, off):
    """Draw wall cells as rectangles in the qpos frame (matches candidate coords)."""
    boxes = wall_boxes(maze_arr, offset=off)  # (M,4) x0,x1,y0,y1
    for x0, x1, y0, y1 in boxes:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0,
                               facecolor="0.6", edgecolor="0.45", linewidth=0.3))
    free = np.argwhere(maze_arr != WALL).astype(float)
    fb = wall_boxes(maze_arr, offset=off)  # reuse extents for limits via all cells
    allc = np.argwhere(np.ones_like(maze_arr)).astype(float)
    xs = allc[:, 0] - off - 0.5
    ys = allc[:, 1] - off - 0.5
    ax.set_xlim(xs.min(), xs.max() + 1)
    ax.set_ylim(ys.min(), ys.max() + 1)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([]); ax.set_yticks([])


def rollout_candidates(env, start_state, pop_actions, action_mean, action_std, frameskip):
    """Roll each candidate (S,H,A) out from start_state -> (S, H*frameskip+1, 2) qpos xy."""
    S, H, A = pop_actions.shape
    d = A // frameskip
    # (S, H, f*d) -> (S, H*f, d) env actions, then denormalize
    acts = rearrange(pop_actions.astype(np.float32), "s h (f d) -> s (h f) d", f=frameskip, d=d)
    acts = acts * action_std[None, None, :] + action_mean[None, None, :]
    trajs = np.zeros((S, acts.shape[1] + 1, 2), dtype=np.float32)
    raw = env.unwrapped
    for s in range(S):
        raw.set_state(np.asarray(start_state[:2], float), np.asarray(start_state[2:4], float))
        trajs[s, 0] = raw.sim.data.qpos[:2]
        for t in range(acts.shape[1]):
            raw.step(acts[s, t])
            trajs[s, t + 1] = raw.sim.data.qpos[:2]
    return trajs


def cost_colors(costs, cmap="viridis_r"):
    """Map costs -> colors (low cost = bright). Returns (N,4) RGBA and norm."""
    c = np.asarray(costs, float)
    lo, hi = np.percentile(c, 2), np.percentile(c, 98)
    norm = plt.Normalize(lo, max(hi, lo + 1e-6))
    return plt.get_cmap(cmap)(norm(c)), norm


def render_round_video(trace, meta, env, maze_arr, off, action_mean, action_std,
                       eval_idx, round_idx, out_path, max_cands=300):
    cfg = trace["config"]
    fs = cfg["frameskip"]; n_taken = cfg["n_taken_actions"]
    rnd = trace["rounds"][round_idx]
    opt_steps = rnd["opt_steps"]
    e_states = np.asarray(meta["e_states"])           # (N, T, 4)
    goal = np.asarray(meta["state_g"])[eval_idx]
    # start state of this round = state after executing the previous rounds
    start_env_step = round_idx * n_taken * fs
    start_state = e_states[eval_idx, min(start_env_step, e_states.shape[1] - 1)]

    # pre-roll every opt step's candidates (cache for animation)
    per_step = []
    for o in opt_steps:
        pop = o["population"][eval_idx][:max_cands]      # (S,H,A)
        costs = o["costs"][eval_idx][:max_cands]
        trajs = rollout_candidates(env, start_state, pop, action_mean, action_std, fs)
        chosen = None
        per_step.append((trajs, costs))
    chosen_mu = rnd["chosen_mu"][eval_idx][None]        # (1,H,A)
    chosen_traj = rollout_candidates(env, start_state, chosen_mu, action_mean, action_std, fs)[0]

    fig, ax = plt.subplots(figsize=(6, 6))

    def draw_frame(k):
        ax.clear()
        draw_maze(ax, maze_arr, off)
        trajs, costs = per_step[k]
        colors, _ = cost_colors(costs)
        # order so low-cost (bright) drawn on top
        order = np.argsort(-costs)
        segs = [np.column_stack([trajs[s, :, 0], trajs[s, :, 1]]) for s in order]
        lc = LineCollection(segs, colors=colors[order], linewidths=0.6, alpha=0.5)
        ax.add_collection(lc)
        ax.plot(chosen_traj[:, 0], chosen_traj[:, 1], "-", color="red", lw=2.2, alpha=0.95, zorder=5)
        ax.plot(start_state[0], start_state[1], "o", color="lime", ms=9, zorder=6, mec="k")
        ax.plot(goal[0], goal[1], "*", color="red", ms=18, zorder=6, mec="k")
        ax.set_title(f"eval {eval_idx}  round {round_idx}  opt step {k+1}/{len(per_step)}  "
                     f"(best cost {per_step[k][1].min():.3f})", fontsize=10)

    anim = animation.FuncAnimation(fig, draw_frame, frames=len(per_step), interval=600)
    anim.save(out_path, writer="ffmpeg", fps=2, dpi=120)
    plt.close(fig)

    # small-multiples grid
    n = len(per_step)
    cols = min(n, 5); rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.6, rows * 2.6))
    axes = np.atleast_1d(axes).ravel()
    for k in range(len(axes)):
        ax = axes[k]
        if k >= n:
            ax.axis("off"); continue
        draw_maze(ax, maze_arr, off)
        trajs, costs = per_step[k]
        colors, _ = cost_colors(costs)
        order = np.argsort(-costs)
        segs = [np.column_stack([trajs[s, :, 0], trajs[s, :, 1]]) for s in order]
        ax.add_collection(LineCollection(segs, colors=colors[order], linewidths=0.4, alpha=0.5))
        ax.plot(chosen_traj[:, 0], chosen_traj[:, 1], "-", color="red", lw=1.5, zorder=5)
        ax.plot(goal[0], goal[1], "*", color="red", ms=10, zorder=6, mec="k")
        ax.set_title(f"step {k+1}", fontsize=8)
    fig.suptitle(f"CEM search — eval {eval_idx}, round {round_idx}", fontsize=11)
    fig.tight_layout()
    fig.savefig(str(out_path).replace(".mp4", ".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--evals", type=int, nargs="*", default=[0])
    ap.add_argument("--rounds", type=int, nargs="*", default=None, help="default: all")
    ap.add_argument("--max-cands", type=int, default=300)
    ap.add_argument("--data-root", default=os.environ.get("DATASET_DIR",
                    str(PROJECT_ROOT.parent / "data")))
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    trace = pickle.load(open(run_dir / "plan_trace.pkl", "rb"))
    meta = trace["meta"]
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    setting = trace["meta"].get("setting") or cfg.get("setting")
    env_name = trace["meta"].get("env_name") or cfg.get("env_name")

    env = gym.make(ENV_ID[setting])
    env.unwrapped.prepare_for_render()
    maze_arr = env.unwrapped.maze_arr
    off = float((env.unwrapped.model.body_pos[env.unwrapped.model.body_name2id("particle")][:2] - 1.0)[0])
    action_mean, action_std = load_action_stats(env_name, args.data_root)

    out_dir = run_dir / "cem_search"
    out_dir.mkdir(exist_ok=True)
    n_rounds = len(trace["rounds"])
    print(f"trace: {n_rounds} rounds, planner={trace['planner']}, setting={setting}")
    for e in args.evals:
        rounds = args.rounds if args.rounds is not None else range(n_rounds)
        for r in rounds:
            out = out_dir / f"cem_eval{e}_round{r}.mp4"
            print(f"  rendering eval {e} round {r} -> {out.name}")
            render_round_video(trace, meta, env, maze_arr, off, action_mean, action_std,
                               e, r, str(out), max_cands=args.max_cands)
    print("Done.")


if __name__ == "__main__":
    main()
