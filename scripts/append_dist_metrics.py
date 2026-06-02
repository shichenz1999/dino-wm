#!/usr/bin/env python
"""Post-process a plan_outputs/<run>/ dir: compute per-iter, per-step
distance-to-goal curves in four spaces and append them to evals.json.

Per eval i, per iter k (matching evals[i]["iters"]):
  Distances are per-frame — sampled at every env step in the iter
  (n_taken*frameskip points):
    pixel_dist_steps          : ||obs_step - obs_goal||_2  (int32)
    geo_dist_steps            : geodesic dist (eikonal) ball->goal  (point_maze)
    latent_dist_steps         : ||encoder(obs_step) - z_obs_g||_2  (visual, to goal)
  wm_pred_mse stays per-action (n_taken points; WM predicts one latent/action):
    wm_pred_mse_steps         : mean((imagined_visual - actual_visual)**2)

Bit-exact reproduction of planning's observations (no planner re-run):
  * trajectory obs come from REPLAYING env.rollout with the saved actions
    (same do_simulation+render path planning used). set_state-based re-rendering
    is NOT bit-exact (MuJoCo mj_forward vs mj_step give a ~1px ball shift), so
    we replay instead.
  * the goal obs (random_state goal_source) was produced by env.prepare/reset
    -> set_state render, so it IS reproduced bit-exact by set_state + set_marker.
  * the goal latent z_obs_g is read straight from plan_latents.pkl.
The actual visual latent is encoded once per step and shared by latent_dist +
wm_pred_mse.

Usage:
    python scripts/append_dist_metrics.py plan_outputs/<run>/ [--in-place] [--sanity] [--max-evals N]
"""
import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import env.pointmaze  # noqa: F401  (registers gym envs)
from plan import load_model
from utils import move_to_device
from datasets.img_transforms import default_transform
from datasets.point_maze_dset import PointMazeDataset
from visualize_plan import create_render_env
from env.pointmaze.maze_model import (
    U_MAZE, U_MAZE_EVAL, MEDIUM_MAZE, MEDIUM_MAZE_EVAL,
    LARGE_MAZE, LARGE_MAZE_EVAL, SMALL_MAZE, OPEN,
)

SPEC_MAP = {
    "u_maze": U_MAZE, "u_maze_eval": U_MAZE_EVAL,
    "medium": MEDIUM_MAZE, "medium_eval": MEDIUM_MAZE_EVAL,
    "large": LARGE_MAZE, "large_eval": LARGE_MAZE_EVAL,
    "small": SMALL_MAZE, "open": OPEN,
}


def load_train_cfg(cfg):
    model_path = Path(str(cfg.ckpt_base_path)) / str(cfg.env_name) / str(cfg.ckpt_id)
    return OmegaConf.load(model_path / "hydra.yaml")


def load_wm(cfg, train_cfg, device):
    model_ckpt = (Path(str(cfg.ckpt_base_path)) / str(cfg.env_name) / str(cfg.ckpt_id)
                  / "checkpoints" / f"model_{cfg.model_epoch}.pth")
    model = load_model(model_ckpt, train_cfg, train_cfg.num_action_repeat, device)
    model.eval()
    return model


def load_action_stats(train_cfg):
    """action_mean/std used to denormalize the saved (normalized) actions."""
    data_path = OmegaConf.to_container(train_cfg.env.dataset, resolve=True)["data_path"]
    ds = PointMazeDataset(data_path=data_path, normalize_action=True)
    return ds.action_mean.numpy(), ds.action_std.numpy()


def encode_visual(wm, transform, frames, device):
    """frames: (B, H, W, C) uint8 -> visual latent (B, P, D) on cpu.

    Reproduces the planning encode path: preprocess + dataset transform, then
    wm.encode_obs (its own encoder_transform + encoder). proprio is irrelevant
    to the visual latent, so we pass zeros.
    """
    B = frames.shape[0]
    vis = torch.tensor(np.asarray(frames))[:, None]            # (B, 1, H, W, C)
    vis = rearrange(vis, "b t h w c -> b t c h w") / 255.0
    vis = transform(vis)
    obs = {"visual": vis.to(device),
           "proprio": torch.zeros(B, 1, 4, device=device)}
    with torch.no_grad():
        z = wm.encode_obs(obs)["visual"]                       # (B, 1, P, D)
    return z[:, 0].cpu()


def render_goal(raw, state_g):
    """Bit-exact goal frame: set_state + set_marker (matches env.prepare/reset).
    Renders twice and keeps the second to dodge mujoco_py's one-call-stale buffer.
    """
    raw.set_state(state_g[:2], state_g[2:4])
    raw.set_marker()
    raw.sim.render(224, 224)
    return raw.sim.render(224, 224)


def main(run_dir, in_place=False, sanity=False, max_evals=None):
    run_dir = Path(run_dir).resolve()
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    train_cfg = load_train_cfg(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    frameskip = int(train_cfg.frameskip)
    n_taken = int((cfg.get("planner") or {}).get("n_taken_actions") or cfg.get("goal_H"))
    base_seed = int(cfg.get("seed", 0))
    setting = str(cfg.get("setting", "u_maze"))
    goal_source = str(cfg.get("goal_source", ""))
    is_point_maze = (train_cfg.env.name == "point_maze")
    if not is_point_maze:
        raise SystemExit(f"only point_maze supported; got {train_cfg.env.name}")
    if goal_source != "random_state":
        # other sources build the goal via a do_simulation rollout; the goal
        # frame would need that rollout replayed (gt_actions). Not handled here.
        raise SystemExit(f"goal_source={goal_source} not supported (only random_state)")
    maze_spec = SPEC_MAP.get(setting, U_MAZE)

    meta = pickle.load(open(run_dir / "plan_meta.pkl", "rb"))
    latents = pickle.load(open(run_dir / "plan_latents.pkl", "rb"))
    evals = json.load(open(run_dir / "evals.json"))

    state_0 = meta["state_0"]
    state_g = meta["state_g"]
    actions = meta["actions"]                   # (n, T, f*d) normalized WM actions
    action_len = meta["action_len"].astype(int)
    n_evals = state_0.shape[0]
    rollout = latents["wm_rollout_latents"]     # list[iter] {"visual": (n, n_taken+1, P, D)}
    zg_visual = latents["z_obs_g"]["visual"]    # (n, 1, P, D)

    print(f"run={run_dir.name}  n_evals={n_evals}  frameskip={frameskip}  "
          f"n_taken={n_taken}  goal_source={goal_source}")

    print("  loading model + action stats + env ...")
    wm = load_wm(cfg, train_cfg, device)
    transform = default_transform(img_size=train_cfg.img_size)
    amean, astd = load_action_stats(train_cfg)
    the_env = create_render_env(cfg, train_cfg)
    raw = the_env.unwrapped
    from env.pointmaze.geo_dist import field_for

    n_proc = n_evals if max_evals is None else min(max_evals, n_evals)
    sanity_blob = None

    for i in range(n_proc):
        al = action_len[i]
        # replay the executed actions to get the real per-step obs (bit-exact)
        exec_a = (rearrange(actions[i], "t (f d) -> (t f) d", f=frameskip) * astd + amean)
        exec_a = exec_a[: al * frameskip].astype(np.float32)
        obses, states = raw.rollout(base_seed * i + 1, state_0[i], exec_a)
        vis = obses["visual"]                                  # (al*fs+1, H, W, C) uint8
        T_env = vis.shape[0]

        goal_frame = render_goal(raw, state_g[i])
        vg = goal_frame.reshape(-1).astype(np.int32)
        zg_i = zg_visual[i, 0]                                 # (P, D)
        field = field_for(maze_spec, state_g[i, :2])

        iters = evals["evals"][str(i)]["iters"]
        n_fine = n_taken * frameskip
        for k, iter_entry in enumerate(iters):
            # distances: every env step (frame) in the iter
            est = [min(k * n_fine + s, T_env - 1) for s in range(1, n_fine + 1)]
            frames = vis[est]                                  # (n_fine, H, W, C)

            pix = np.linalg.norm(
                frames.reshape(len(est), -1).astype(np.int32) - vg, axis=-1)
            geo = np.array([field.distance(states[s, :2]) for s in est])

            z_act = encode_visual(wm, transform, frames, device)   # (n_fine, P, D)
            lat = (z_act - zg_i).flatten(1).norm(dim=1).numpy()
            # wm_mse: per-action — subsample action-boundary frames from z_act
            act_cols = [frameskip * j - 1 for j in range(1, n_taken + 1)]
            z_act_a = z_act[act_cols]                              # (n_taken, P, D)
            z_img = rollout[k]["visual"][i, 1:n_taken + 1]         # (n_taken, P, D)
            wm_mse = ((z_img - z_act_a) ** 2).flatten(1).mean(dim=1).numpy()

            iter_entry["pixel_dist_steps"] = [round(float(x), 3) for x in pix]
            iter_entry["geo_dist_steps"] = [round(float(x), 3) for x in geo]
            iter_entry["latent_dist_steps"] = [round(float(x), 3) for x in lat]
            iter_entry["wm_pred_mse_steps"] = [round(float(x), 4) for x in wm_mse]

            if sanity and i == 0 and k == 0:
                sanity_blob = {"goal_frame": goal_frame, "frames": frames,
                               "est": est, "pix": pix, "geo": geo,
                               "lat": lat, "wm_mse": wm_mse}

    out_path = run_dir / ("evals.json" if in_place else "evals_dist.json")
    json.dump(evals, open(out_path, "w"), indent=2)
    print(f"  wrote {out_path}")

    if sanity and sanity_blob:
        _save_sanity(run_dir, sanity_blob)
    return out_path


def _save_sanity(run_dir, s):
    """Montage: goal frame + eval0/iter0 replay frames, annotated with metrics."""
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()

    goal, frames = s["goal_frame"], s["frames"]
    H, W = goal.shape[:2]
    pad, label_h = 6, 56
    cells = [("GOAL", goal, None)]
    for idx, fr in enumerate(frames):
        cells.append((
            f"step {s['est'][idx]}", fr,
            f"geo={s['geo'][idx]:.2f}\npix={s['pix'][idx]:.0f}\n"
            f"lat={s['lat'][idx]:.1f}\nmse={s['wm_mse'][idx]:.3f}",
        ))

    n = len(cells)
    cw, ch = W + 2 * pad, H + label_h + 2 * pad
    canvas = Image.new("RGB", (n * cw, ch), (30, 30, 30))
    draw = ImageDraw.Draw(canvas)
    for ci, (title, img, info) in enumerate(cells):
        x0 = ci * cw + pad
        canvas.paste(Image.fromarray(np.asarray(img)), (x0, pad))
        draw.text((x0, pad + H + 2), title, font=font, fill=(255, 255, 255))
        if info:
            draw.text((x0, pad + H + 16), info, font=font, fill=(180, 220, 180))
    out = run_dir / "dist_sanity.png"
    canvas.save(out)
    print(f"  wrote {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--in-place", action="store_true",
                   help="overwrite evals.json (default: write evals_dist.json)")
    p.add_argument("--sanity", action="store_true",
                   help="save dist_sanity.png montage for eval 0 / iter 0")
    p.add_argument("--max-evals", type=int, default=None,
                   help="only process the first N evals (smoke test)")
    a = p.parse_args()
    main(a.run_dir, in_place=a.in_place, sanity=a.sanity, max_evals=a.max_evals)
