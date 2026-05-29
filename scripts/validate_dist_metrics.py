#!/usr/bin/env python
"""Validate that append_dist_metrics reproduces planning's observations
bit-exact, for a plan_outputs/<run>/ dir (point_maze, random_state goal).

Asserts three invariants and writes a PASS/FAIL report to the run dir:
  1. replay states  == saved e_states            (env.rollout is deterministic)
  2. goal set_state+marker == env.prepare render  (random_state goal origin)
  3. pixel_dist_steps[-1] == original visual_dist  (end-to-end, every iter)

Exit code 0 = all pass, 1 = any fail. Report saved to <run>/dist_validation.txt.

Usage:
    python scripts/validate_dist_metrics.py plan_outputs/<run>/ [--max-evals N]
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
from einops import rearrange
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import env.pointmaze  # noqa: F401
from visualize_plan import create_render_env
from datasets.point_maze_dset import PointMazeDataset
import append_dist_metrics as adm

# tolerances: states/goal are exact; pixel cutoff carries evals.json's 3-dp round
STATE_TOL = 1e-9
GOAL_TOL = 0.0
PIXEL_TOL = 0.01


def _load(run_dir):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    train_cfg = adm.load_train_cfg(cfg)
    meta = pickle.load(open(run_dir / "plan_meta.pkl", "rb"))
    return cfg, train_cfg, meta


def check_replay_states(cfg, train_cfg, meta, n_proc):
    """Invariant 1: replayed states == saved e_states."""
    fs = int(train_cfg.frameskip)
    base_seed = int(cfg.get("seed", 0))
    amean, astd = adm.load_action_stats(train_cfg)
    raw = create_render_env(cfg, train_cfg).unwrapped
    worst = 0.0
    for i in range(n_proc):
        al = int(meta["action_len"][i])
        exec_a = (rearrange(meta["actions"][i], "t (f d) -> (t f) d", f=fs) * astd + amean)
        exec_a = exec_a[: al * fs].astype(np.float32)
        _, states = raw.rollout(base_seed * i + 1, meta["state_0"][i], exec_a)
        ref = meta["e_states"][i][: states.shape[0]]
        worst = max(worst, float(np.abs(states - ref).max()))
    return worst <= STATE_TOL, worst


def check_goal_render(cfg, train_cfg, meta, n_proc):
    """Invariant 2: set_state+marker goal == env.prepare goal."""
    base_seed = int(cfg.get("seed", 0))
    raw = create_render_env(cfg, train_cfg).unwrapped
    worst = 0.0
    for i in range(n_proc):
        sg = meta["state_g"][i]
        obs_orig, _ = raw.prepare(base_seed * i + 1, sg)
        g_orig = obs_orig["visual"].astype(np.int64)
        g_mine = adm.render_goal(raw, sg).astype(np.int64)
        worst = max(worst, float(np.linalg.norm((g_orig - g_mine).ravel())))
    return worst <= GOAL_TOL, worst


def check_pixel_end_to_end(run_dir, n_proc):
    """Invariant 3: pixel_dist_steps[-1] == original visual_dist, every iter."""
    adm.main(run_dir, in_place=False, sanity=False, max_evals=n_proc)
    orig = json.load(open(run_dir / "evals.json"))
    new = json.load(open(run_dir / "evals_dist.json"))
    worst, n = 0.0, 0
    for i in range(n_proc):
        for o_it, n_it in zip(orig["evals"][str(i)]["iters"],
                              new["evals"][str(i)]["iters"]):
            d = abs(n_it["pixel_dist_steps"][-1] - o_it["visual_dist"])
            worst = max(worst, d)
            n += 1
    return worst <= PIXEL_TOL, worst, n


def main(run_dir, max_evals=5):
    run_dir = Path(run_dir).resolve()
    cfg, train_cfg, meta = _load(run_dir)
    n_proc = min(max_evals, meta["state_0"].shape[0])

    results = []
    ok1, w1 = check_replay_states(cfg, train_cfg, meta, n_proc)
    results.append(("replay_states == e_states", ok1, f"max|diff|={w1:.2e}  (tol {STATE_TOL:.0e})"))
    ok2, w2 = check_goal_render(cfg, train_cfg, meta, n_proc)
    results.append(("goal set_state+marker == env.prepare", ok2, f"max L2={w2:.3f}  (tol {GOAL_TOL})"))
    ok3, w3, ncmp = check_pixel_end_to_end(run_dir, n_proc)
    results.append((f"pixel_steps[-1] == visual_dist ({ncmp} iters)", ok3, f"max|diff|={w3:.4f}  (tol {PIXEL_TOL})"))

    all_ok = all(ok for _, ok, _ in results)
    lines = [f"validation: {run_dir}", f"evals checked: {n_proc}", ""]
    for name, ok, detail in results:
        lines.append(f"[{'PASS' if ok else 'FAIL'}] {name}\n        {detail}")
    lines += ["", f"RESULT: {'ALL PASS' if all_ok else 'FAIL'}"]
    report = "\n".join(lines)
    print(report)
    (run_dir / "dist_validation.txt").write_text(report + "\n")
    print(f"\nsaved {run_dir / 'dist_validation.txt'}")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--max-evals", type=int, default=5)
    a = p.parse_args()
    main(a.run_dir, max_evals=a.max_evals)
