#!/usr/bin/env python
"""
Annotated visualizations for a plan_outputs/<run>/ directory.

Reads:
    .hydra/config.yaml
    plan_meta.pkl      (state_0/g, gt_actions, actions, action_len, successes, e_states)
    plan_latents.pkl   (wm_rollout_latents, z_obs_g)
    evals.json

Writes:
    summary.png
    evals/eval_N/trajectory.png
    evals/eval_N/frames.png
    evals/eval_N/video.mp4

Usage:
    python scripts/visualize_plan.py plan_outputs/<run>/
"""

import sys
import os
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import imageio
import torch
from einops import rearrange

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import env.pointmaze  # noqa: F401  (registers gym envs)
import gym


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

U_MAZE_OPEN_REGIONS = [
    (0.5, 1.1, 0.5, 3.1),
    (2.5, 3.1, 0.5, 3.1),
    (1.1, 2.5, 2.5, 3.1),
]
MAZE_X_RANGE = (0.0, 3.6)
MAZE_Y_RANGE = (0.0, 3.6)

COLOR_SUCCESS = (0, 160, 0)
COLOR_FAILURE = (200, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (30, 30, 30)
COLOR_PANEL_BG = (50, 50, 50)

N_TIME_COLUMNS = 6

MARKER_RADIUS = 6


def make_crit_label_fn(cfg, reach_thresh=0.5):
    """Return fn(state, goal) -> criterion text for the banner.

    reach: "pos_dist=X < thr"; avoid: "dist_to_far=X < margin". Single source for
    the criterion text, used by every renderer so they can't drift. reach_thresh
    comes from the env's REACH_THRESH (the same value its eval_state uses).
    """
    s = dict(cfg.get("success") or {})
    inv = bool((cfg.get("objective") or {}).get("invert", False))
    mode = s.get("mode", "default")
    if mode == "default" and inv:
        mode = "avoid"        # mirror plan.py auto-pair

    if mode != "avoid":       # default reach: Euclidean distance to goal
        thr = reach_thresh
        def fn(state, goal):
            d = float(np.linalg.norm(np.asarray(state)[:2] - np.asarray(goal)[:2]))
            cmp = "<" if d < thr else "≥"
            return f"pos_dist={d:.2f} {cmp} {thr:.2f}"
        return fn

    from env.pointmaze.geo_dist import avoid_eval
    from env.pointmaze.maze_model import (
        U_MAZE, U_MAZE_EVAL, MEDIUM_MAZE, MEDIUM_MAZE_EVAL,
        LARGE_MAZE, LARGE_MAZE_EVAL, SMALL_MAZE, OPEN,
    )
    spec_map = {
        "u_maze": U_MAZE, "u_maze_eval": U_MAZE_EVAL,
        "medium": MEDIUM_MAZE, "medium_eval": MEDIUM_MAZE_EVAL,
        "large":  LARGE_MAZE,  "large_eval":  LARGE_MAZE_EVAL,
        "small":  SMALL_MAZE,  "open":        OPEN,
    }
    maze_spec = spec_map.get(str(cfg.get("setting", "")), U_MAZE)
    margin = float(s.get("margin", 0.5))
    frac = s.get("frac", None)

    def fn(state, goal):
        far, thr, ok = avoid_eval(maze_spec, np.asarray(goal)[:2],
                                  np.asarray(state)[:2], margin, frac)
        cmp = "<" if ok else "≥"
        return f"dist_to_far={far:.2f} {cmp} {thr:.2f}"

    return fn



# ---------------------------------------------------------------------------
# Font loading
# ---------------------------------------------------------------------------

def get_font(size=14, bold=False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Data loading & model/env setup
# ---------------------------------------------------------------------------

def load_run(run_dir: Path):
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")
    with open(run_dir / "plan_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(run_dir / "plan_latents.pkl", "rb") as f:
        latents = pickle.load(f)
    with open(run_dir / "evals.json") as f:
        evals = json.load(f)
    return cfg, meta, latents, evals


def load_train_cfg(cfg):
    model_path = Path(str(cfg.ckpt_base_path)) / str(cfg.env_name) / str(cfg.ckpt_id)
    return OmegaConf.load(model_path / "hydra.yaml")


def load_decoder(cfg, device="cpu"):
    model_path = Path(str(cfg.ckpt_base_path)) / str(cfg.env_name) / str(cfg.ckpt_id)
    train_cfg = OmegaConf.load(model_path / "hydra.yaml")
    torch.hub.load("facebookresearch/dinov2", train_cfg.encoder.name)
    ckpt_path = model_path / "checkpoints" / f"model_{cfg.model_epoch}.pth"
    ckpt = torch.load(str(ckpt_path), map_location=device)
    decoder = ckpt["decoder"].to(device)
    decoder.eval()
    return decoder


def render_clean_bg(the_env):
    """Render the maze with the ball site temporarily hidden (alpha=0).

    The result is a TRULY ball-free background, used as the reference for
    diff-based ball markers. Median-over-time picks up ghosts wherever the
    ball lingered, which then bleed into start/final markers as artifacts.
    """
    raw = the_env.unwrapped
    site_id = raw.model.site_name2id("particle_site")
    saved_alpha = float(raw.model.site_rgba[site_id, 3])
    raw.model.site_rgba[site_id, 3] = 0.0
    img = raw.sim.render(224, 224).copy()
    raw.model.site_rgba[site_id, 3] = saved_alpha
    return img


def create_render_env(cfg, train_cfg):
    setting = cfg.get("setting")
    variant_type = cfg.get("variant_type", "baseline")
    variant = cfg.get("variant", "baseline")
    env_id_map = cfg.get("env_id_map") or {}
    if setting and setting in env_id_map:
        if variant_type == "baseline":
            env_id = env_id_map[setting].get("baseline", train_cfg.env.name)
        else:
            env_id = env_id_map[setting].get(variant_type, {}).get(variant, train_cfg.env.name)
    else:
        env_id = train_cfg.env.name
    env_kwargs = dict(train_cfg.env.get("kwargs", {}))
    the_env = gym.make(env_id, **env_kwargs)
    the_env.unwrapped.prepare_for_render()
    return the_env


# ---------------------------------------------------------------------------
# Coordinate mapping (state → pixel)
# ---------------------------------------------------------------------------

def calibrate_state_to_pixel(the_env):
    """Render ball at two known walkable qpos values to find the linear
    state→pixel mapping.

    Calibration points are auto-picked as the two diagonal-extreme WALKABLE
    cells (qpos == maze cell index), so the ball is never occluded by a wall.
    Hardcoded (1,1)/(3,3) broke on mazes where (3,3) is a wall (e.g.
    large_eval): the occluded ball produced 0 diff pixels → NaN calibration →
    invisible trajectory line.
    """
    from env.pointmaze.maze_model import WALL
    raw = the_env.unwrapped
    zero_vel = np.zeros(2)

    raw.set_state(np.array([50.0, 50.0]), zero_vel)
    bg = raw.sim.render(224, 224)

    def _find_center(pos):
        raw.set_state(np.asarray(pos, dtype=float), zero_vel)
        frame = raw.sim.render(224, 224)
        diff = np.abs(frame.astype(float) - bg.astype(float)).sum(axis=-1)
        ys, xs = np.where(diff > 30)
        if len(xs) == 0:
            return None
        return np.array([xs.mean(), ys.mean()])

    # Walkable cells, then pick the two diagonal extremes (min/max of w+h)
    # so they differ in both axes → stable per-axis linear fit.
    walkable = np.argwhere(raw.maze_arr != WALL).astype(float)
    pos_a = walkable[np.argmin(walkable.sum(axis=1))]
    pos_b = walkable[np.argmax(walkable.sum(axis=1))]
    px_a, px_b = _find_center(pos_a), _find_center(pos_b)

    # Fallback to legacy points if something went wrong (e.g. degenerate maze).
    if (px_a is None or px_b is None
            or np.any(pos_b == pos_a)):
        pos_a, pos_b = np.array([1.0, 1.0]), np.array([3.0, 3.0])
        px_a, px_b = _find_center(pos_a), _find_center(pos_b)

    scale = (px_b - px_a) / (pos_b - pos_a)
    offset = px_a - scale * pos_a
    return scale, offset


def state_to_pixel(xy, scale, offset):
    """Map state (x, y) coordinates to pixel (px, py). xy shape: (..., 2+)."""
    return xy[..., :2] * scale + offset


# ---------------------------------------------------------------------------
# Frame generation
# ---------------------------------------------------------------------------

def render_env_frames(the_env, e_states):
    """Render env frames by setting state at each timestep."""
    raw = the_env.unwrapped
    n_evals, n_frames = e_states.shape[:2]
    frames = np.zeros((n_evals, n_frames, 224, 224, 3), dtype=np.uint8)
    for i in range(n_evals):
        for t in range(n_frames):
            raw.set_state(e_states[i, t, :2], e_states[i, t, 2:4])
            frames[i, t] = raw.sim.render(224, 224)
    return frames


def render_single_frame(the_env, state):
    raw = the_env.unwrapped
    raw.set_state(state[:2], np.zeros(2))
    return raw.sim.render(224, 224)


def decode_wm_latents(decoder, z_visual, device="cpu"):
    """Decode visual latents → (b, t, H, W, 3) uint8."""
    z = z_visual.to(device)
    b, t = z.shape[:2]
    with torch.no_grad():
        decoded, _ = decoder(z)
    decoded = rearrange(decoded, "(b t) c h w -> b t c h w", b=b, t=t)
    uint8 = (((decoded.clamp(-1, 1) + 1) / 2) * 255).cpu().numpy().astype(np.uint8)
    return np.transpose(uint8, (0, 1, 3, 4, 2))


def generate_wm_frames(decoder, latents, device="cpu"):
    """Decode all WM latents. Returns (wm_frames_all, goal_recon_all) in HWC uint8."""
    rollout_latents = latents["wm_rollout_latents"]
    per_iter = [decode_wm_latents(decoder, z["visual"], device) for z in rollout_latents]

    wm_obs_0_recon = per_iter[0][:, 0:1]
    wm_imagined = np.concatenate([d[:, 1:] for d in per_iter], axis=1)
    wm_frames_all = np.concatenate([wm_obs_0_recon, wm_imagined], axis=1)

    goal_recon_all = decode_wm_latents(decoder, latents["z_obs_g"]["visual"], device)
    return wm_frames_all, goal_recon_all


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def chw_to_hwc(img):
    if img.ndim == 3 and img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    return img


def blend_ball_marker(canvas, source_frame, clean_bg, color, alpha=1.0):
    """Use (source - clean_bg) magnitude as a soft mask to paint `color` onto canvas.

    Preserves the ball's anti-aliased edges and is color-agnostic — works
    regardless of the actual ball color in source_frame.
    """
    diff = source_frame.astype(np.float32) - clean_bg.astype(np.float32)
    signal = np.linalg.norm(diff, axis=-1, keepdims=True) / 255.0
    signal = np.clip(signal * 2.0, 0, 1)
    color_arr = np.array(color, dtype=np.float32)
    result = canvas.astype(np.float32) * (1 - signal * alpha) + color_arr * (signal * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Config formatting
# ---------------------------------------------------------------------------

def cfg_to_strings(cfg):
    if cfg is None:
        return ("", "", "")
    planner_cfg = cfg.get("planner", {}) or {}
    is_mpc = "MPC" in str(planner_cfg.get("_target_", ""))
    sub_cfg = planner_cfg.get("sub_planner", {}) if is_mpc else planner_cfg

    line1 = (
        f"{cfg.get('env_name', '?')}/{cfg.get('ckpt_id', '?')} | n_evals={cfg.get('n_evals')} | "
        f"goal_source={cfg.get('goal_source')} | seed={cfg.get('seed')}"
    )
    if is_mpc:
        line2 = (
            f"MPC (max_iter={planner_cfg.get('max_iter')}, "
            f"n_taken_actions={planner_cfg.get('n_taken_actions')})"
        )
        sub_name = str(sub_cfg.get("target", "")).split(".")[-1] or "sub"
    else:
        line2 = f"{str(planner_cfg.get('_target_', '')).split('.')[-1] or 'planner'} (open-loop)"
        sub_name = ""
    line3 = (
        f"  └→ {sub_name or 'planner'} (horizon={sub_cfg.get('horizon', cfg.get('goal_H'))}, "
        f"num_samples={sub_cfg.get('num_samples')}, topk={sub_cfg.get('topk')}, "
        f"opt_steps={sub_cfg.get('opt_steps')})"
    )
    return line1, line2, line3


def estimate_mpc_iters(action_len, cfg=None):
    planner_cfg = (cfg or {}).get("planner", {}) or {}
    n_taken_cfg = planner_cfg.get("n_taken_actions")
    if n_taken_cfg is None:
        return 1
    return max(1, int(np.ceil(int(action_len) / max(1, n_taken_cfg))))


# ---------------------------------------------------------------------------
# Trajectory plot
# ---------------------------------------------------------------------------

def render_trajectory_plot(e_states, state_g, success, bg_img,
                           start_frame, final_frame, goal_frame,
                           px_scale, px_offset, scale=2,
                           eval_idx=None, action_len=None,
                           cfg=None,
                           start_color=(40, 200, 60),
                           crit_label=None,
                           show_title=True, show_legend=True):
    final_color = (50, 130, 240)
    goal_color = (220, 50, 50)

    # Blend goal / start / final balls at native resolution to keep smooth edges.
    canvas = bg_img.copy()
    canvas = blend_ball_marker(canvas, goal_frame, bg_img, color=goal_color)
    canvas = blend_ball_marker(canvas, start_frame, bg_img, color=start_color)
    canvas = blend_ball_marker(canvas, final_frame, bg_img, color=final_color)

    H, W = canvas.shape[:2]
    pil = Image.fromarray(canvas).resize(
        (W * scale, H * scale), Image.NEAREST
    ).convert("RGBA")
    draw = ImageDraw.Draw(pil)

    traj_px = state_to_pixel(e_states[:, :2], px_scale, px_offset) * scale
    line_w = max(2, int(MARKER_RADIUS * scale * 0.3))
    s_arr = np.array(start_color, dtype=np.float32)
    f_arr = np.array(final_color, dtype=np.float32)
    n = len(traj_px)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        c = tuple(int(v) for v in (1 - t) * s_arr + t * f_arr)
        draw.line([tuple(traj_px[i]), tuple(traj_px[i + 1])],
                  fill=c + (255,), width=line_w)

    pil = pil.convert("RGB")

    if show_legend:
        legend_font = get_font(13, bold=True)
        pad = 8
        line_h = 18
        legend_items = [
            (start_color, "Start"),
            (final_color, "Final"),
            (goal_color, "Goal"),
        ]
        max_label_w = max(legend_font.getbbox(label)[2] for _, label in legend_items)
        box_w = 24 + max_label_w + 2 * pad
        box_h = line_h * len(legend_items) + 2 * pad
        box_x = pil.width - box_w - pad
        box_y = pil.height - box_h - pad
        draw_rgb = ImageDraw.Draw(pil)
        draw_rgb.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                           fill=(0, 0, 0), outline=(200, 200, 200))
        r = 6
        for i, (color, label) in enumerate(legend_items):
            cy = box_y + pad + i * line_h + line_h // 2
            cx = box_x + pad + 6
            draw_rgb.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            draw_rgb.text((box_x + pad + 18, cy - 7), label,
                          font=legend_font, fill=(255, 255, 255))

    if not show_title:
        return np.array(pil)

    # title strip
    title_h = 90
    title_color = (0, 160, 0) if success else (200, 0, 0)
    eval_str = f"Eval {eval_idx} — " if eval_idx is not None else ""
    status_text = f"{eval_str}{'SUCCESS' if success else 'FAILURE'}"
    info_parts = []
    if action_len is not None:
        action_len = int(action_len)
        info_parts.append(f"mpc_iters={estimate_mpc_iters(action_len, cfg)}")
        info_parts.append(f"action_len={action_len}")
    if crit_label is not None:
        info_parts.append(crit_label)
    info_text = " | ".join(info_parts)
    _, sub_mpc, sub_cem = cfg_to_strings(cfg) if cfg is not None else ("", "", "")

    final = Image.new("RGB", (pil.width, pil.height + title_h), color=(40, 40, 40))
    final.paste(pil, (0, title_h))
    title_draw = ImageDraw.Draw(final)
    font_status = get_font(16, bold=True)
    font_info = get_font(11)
    title_draw.text((10, 4), status_text, font=font_status, fill=title_color)
    if info_text:
        title_draw.text((10, 28), info_text, font=font_info, fill=(200, 200, 200))
    if sub_mpc:
        title_draw.text((10, 48), sub_mpc, font=font_info, fill=(180, 180, 180))
    if sub_cem:
        title_draw.text((10, 66), sub_cem, font=font_info, fill=(180, 180, 180))

    return np.array(final)


# ---------------------------------------------------------------------------
# Frames grid (PNG)
# ---------------------------------------------------------------------------

def render_frames_grid(env_frames, wm_frames, goal_real, goal_recon,
                       eval_idx, success, action_len, frameskip,
                       cfg=None, crit_label=None):
    H, W = env_frames.shape[1:3]
    n_env_frames = env_frames.shape[0]
    n_wm_frames = wm_frames.shape[0]

    env_indices = np.linspace(0, n_env_frames - 1, N_TIME_COLUMNS).astype(int)
    wm_indices = (env_indices / frameskip).round().clip(0, n_wm_frames - 1).astype(int)

    env_row = [env_frames[i] for i in env_indices] + [goal_real]
    wm_row = [wm_frames[i] for i in wm_indices] + [goal_recon]

    n_cols = len(env_row)

    label_w = 80
    cell_pad = 4
    header_h = 30
    title_h = 100
    legend_h = 30

    cell_w = W + 2 * cell_pad
    cell_h = H + 2 * cell_pad

    total_w = label_w + n_cols * cell_w
    total_h = title_h + header_h + 2 * cell_h + legend_h

    canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

    status_color = COLOR_SUCCESS if success else COLOR_FAILURE
    status_text = f"Eval {eval_idx} — {'SUCCESS' if success else 'FAILURE'}"
    action_len = int(action_len)
    step_parts = [
        f"mpc_iters={estimate_mpc_iters(action_len, cfg)}",
        f"action_len={action_len}",
    ]
    if crit_label is not None:
        step_parts.append(crit_label)
    step_info = "  |  ".join(step_parts)

    _, sub_mpc, sub_cem = cfg_to_strings(cfg) if cfg is not None else ("", "", "")

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font_title = get_font(18, bold=True)
    font_sub = get_font(12)
    font_label = get_font(14)
    font_header = get_font(13)
    font_small = get_font(11)

    draw.rectangle([0, 0, total_w, title_h], fill=COLOR_PANEL_BG)
    draw.text((16, 6), status_text, font=font_title, fill=status_color)
    draw.text((16, 30), step_info, font=font_sub, fill=(200, 200, 200))
    if sub_mpc:
        draw.text((16, 50), sub_mpc, font=font_sub, fill=(180, 180, 180))
    if sub_cem:
        draw.text((16, 68), sub_cem, font=font_sub, fill=(180, 180, 180))

    header_y = title_h
    draw.rectangle([0, header_y, total_w, header_y + header_h], fill=COLOR_PANEL_BG)
    for ci, env_idx in enumerate(env_indices):
        x = label_w + ci * cell_w + cell_pad
        text = f"t={env_idx}"
        bbox = draw.textbbox((0, 0), text, font=font_header)
        tw = bbox[2] - bbox[0]
        draw.text((x + (W - tw) // 2, header_y + 8), text, font=font_header, fill=COLOR_TEXT)
    x = label_w + N_TIME_COLUMNS * cell_w + cell_pad
    bbox = draw.textbbox((0, 0), "Goal", font=font_header)
    tw = bbox[2] - bbox[0]
    draw.text((x + (W - tw) // 2, header_y + 8), "Goal", font=font_header, fill=COLOR_TEXT)

    for ri, label in enumerate(["Env (MuJoCo)", "Imagined (WM)"]):
        y = title_h + header_h + ri * cell_h + cell_h // 2
        for li, ln in enumerate(label.split()):
            draw.text((8, y - 10 + li * 14), ln, font=font_label, fill=COLOR_TEXT)

    canvas = np.array(pil)

    for ri, row in enumerate([env_row, wm_row]):
        for ci, img in enumerate(row):
            y0 = title_h + header_h + ri * cell_h + cell_pad
            x0 = label_w + ci * cell_w + cell_pad
            canvas[y0:y0 + H, x0:x0 + W] = img

    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    legend_y = total_h - legend_h
    draw.rectangle([0, legend_y, total_w, total_h], fill=COLOR_PANEL_BG)
    legend = (
        "Env = MuJoCo real render  |  "
        f"Imagined = WM latent → decoder (predicts every {frameskip} env frames)  |  "
        "Goal: real (top row), WM-reconstructed (bottom row)"
    )
    draw.text((20, legend_y + 8), legend, font=font_small, fill=(200, 200, 200))

    return np.array(pil)


# ---------------------------------------------------------------------------
# Annotated video (MP4)
# ---------------------------------------------------------------------------

def render_video(env_frames, wm_frames, goal_real, env_bg,
                 eval_idx, success, action_len, frameskip, out_path, fps=12,
                 cfg=None, crit_label=None):
    n_env_frames = env_frames.shape[0]
    H, W = env_frames.shape[1:3]
    # Goal mask comes from the real env render — same pixel position works
    # for both panels since the camera/layout is identical.

    status_text = f"Eval {eval_idx} — {'SUCCESS' if success else 'FAILURE'}"
    status_color = COLOR_SUCCESS if success else COLOR_FAILURE
    action_len = int(action_len)
    stats_line1 = f"mpc_iters={estimate_mpc_iters(action_len, cfg)} | action_len={action_len}"
    dist_line = crit_label if crit_label is not None else ""

    status_h = 22
    frame_h = 16
    stats_h = 16
    dist_h = 16
    top_banner_h = status_h + frame_h + stats_h + dist_h
    mid_banner_h = 20

    total_w = W
    total_h = top_banner_h + H + mid_banner_h + H

    _probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    font_status = get_font(15, bold=True)
    for _sz in (15, 13, 12, 11, 10):
        font_status = get_font(_sz, bold=True)
        bbox = _probe.textbbox((0, 0), status_text, font=font_status)
        if bbox[2] - bbox[0] <= total_w - 8:
            break
    font_top = get_font(10)
    font_mid = get_font(9)

    n_preds = wm_frames.shape[0] - 1

    writer = imageio.get_writer(out_path, fps=fps)

    for t in range(n_env_frames):
        i_idx = min(t // frameskip, wm_frames.shape[0] - 1)
        pred_segment = min(t // frameskip, n_preds)

        env_panel = blend_ball_marker(env_frames[t], goal_real, env_bg,
                                      color=(255, 50, 50), alpha=0.7)
        wm_panel = blend_ball_marker(wm_frames[i_idx], goal_real, env_bg,
                                     color=(255, 50, 50), alpha=0.7)

        canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

        canvas[0:top_banner_h] = COLOR_PANEL_BG
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)

        bbox = draw.textbbox((0, 0), status_text, font=font_status)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, 3), status_text,
                  font=font_status, fill=status_color)

        frame_text = f"Frame {t}/{n_env_frames - 1} | pred {pred_segment}/{n_preds}"
        bbox = draw.textbbox((0, 0), frame_text, font=font_top)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, status_h + 1), frame_text,
                  font=font_top, fill=COLOR_TEXT)

        bbox = draw.textbbox((0, 0), stats_line1, font=font_top)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, status_h + frame_h + 1), stats_line1,
                  font=font_top, fill=(200, 200, 200))

        if dist_line:
            bbox = draw.textbbox((0, 0), dist_line, font=font_top)
            tw = bbox[2] - bbox[0]
            draw.text(((total_w - tw) // 2, status_h + frame_h + stats_h + 1),
                      dist_line, font=font_top, fill=(200, 200, 200))

        canvas = np.array(pil)
        canvas[top_banner_h:top_banner_h + H, 0:W] = env_panel

        y_mid = top_banner_h + H
        canvas[y_mid:y_mid + mid_banner_h] = COLOR_PANEL_BG
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        mid_text = "↑ Env  |  ↓ Imagined (WM)  |  red = goal"
        bbox = draw.textbbox((0, 0), mid_text, font=font_mid)
        mw = bbox[2] - bbox[0]
        draw.text(((total_w - mw) // 2, y_mid + 4), mid_text, font=font_mid, fill=COLOR_TEXT)

        canvas = np.array(pil)
        y_imag = y_mid + mid_banner_h
        canvas[y_imag:y_imag + H, 0:W] = wm_panel

        writer.append_data(canvas)

    writer.close()


# ---------------------------------------------------------------------------
# Summary PNG
# ---------------------------------------------------------------------------

def render_summary(run_dir, cfg, meta, out_path,
                   env_bgs, env_frames_all, goal_real_all,
                   px_scale, px_offset,
                   crit_fn=None,
                   start_color=(40, 200, 60)):
    e_states = meta["e_states"]
    successes = meta["successes"]
    action_len = meta["action_len"]
    state_g = meta["state_g"]
    n_evals = e_states.shape[0]
    frameskip = cfg.get("frameskip", 5)

    def _final_idx(i):
        return min(int(action_len[i]) * frameskip + 1, e_states.shape[1]) - 1

    crit_labels = [
        crit_fn(e_states[i, _final_idx(i)], state_g[i]) if crit_fn else None
        for i in range(n_evals)
    ]

    traj_imgs = []
    for i in range(n_evals):
        n_frames = min(int(action_len[i]) * frameskip + 1, e_states.shape[1])
        traj_img = render_trajectory_plot(
            e_states=e_states[i, :n_frames],
            state_g=state_g[i],
            success=successes[i],
            bg_img=env_bgs[i],
            start_frame=env_frames_all[i, 0],
            final_frame=env_frames_all[i, n_frames - 1],
            goal_frame=goal_real_all[i],
            px_scale=px_scale, px_offset=px_offset, scale=1,
            eval_idx=i, action_len=action_len[i],
            cfg=cfg, start_color=start_color, crit_label=crit_labels[i],
            show_title=False, show_legend=False,
        )
        traj_imgs.append(traj_img)

    traj_H, traj_W = traj_imgs[0].shape[:2]

    title_h = 80
    legend_h = 24
    info_w = 190
    row_h = traj_H
    min_title_w = 460
    total_w = max(traj_W + info_w, min_title_w)
    total_h = title_h + legend_h + n_evals * row_h

    canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

    n_success = int(successes.sum())
    success_rate = n_success / n_evals
    title = f"Planning Summary — {n_success}/{n_evals} success ({success_rate * 100:.1f}%)"
    sub1, sub2, sub3 = cfg_to_strings(cfg)

    canvas[0:title_h] = COLOR_PANEL_BG
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    font_title = get_font(14, bold=True)
    font_sub = get_font(10)
    font_info = get_font(12)
    font_info_bold = get_font(12, bold=True)

    draw.text((12, 6), title, font=font_title, fill=COLOR_TEXT)
    draw.text((12, 28), sub1, font=font_sub, fill=(180, 180, 180))
    draw.text((12, 44), sub2, font=font_sub, fill=(180, 180, 180))
    draw.text((12, 60), sub3, font=font_sub, fill=(180, 180, 180))

    legend_items = [
        (start_color, "Start"),
        ((50, 130, 240), "Final"),
        ((220, 50, 50), "Goal"),
    ]
    legend_font = get_font(12, bold=True)
    draw.rectangle([0, title_h, total_w, title_h + legend_h], fill=(50, 50, 50))
    lx = 20
    ly = title_h + legend_h // 2
    r = 5
    for color, label in legend_items:
        draw.ellipse([lx - r, ly - r, lx + r, ly + r], fill=color)
        text_x = lx + 10
        draw.text((text_x, ly - 8), label, font=legend_font, fill=COLOR_TEXT)
        lx = text_x + legend_font.getbbox(label)[2] + 24

    canvas = np.array(pil)

    for i in range(n_evals):
        y0 = title_h + legend_h + i * row_h
        canvas[y0:y0 + traj_H, 0:traj_W] = traj_imgs[i]

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        status_text = "✓ SUCCESS" if successes[i] else "✗ FAILURE"
        status_color = COLOR_SUCCESS if successes[i] else COLOR_FAILURE
        info_x = traj_W + 20
        info_y = y0 + 30

        action_len_i = int(action_len[i])
        lines = [
            (f"Eval {i}", font_info_bold, COLOR_TEXT),
            (status_text, font_info_bold, status_color),
            (f"mpc_iters = {estimate_mpc_iters(action_len_i, cfg)}", font_info, COLOR_TEXT),
            (f"action_len = {action_len_i}", font_info, COLOR_TEXT),
        ]
        if crit_labels[i] is not None:
            lines.append((crit_labels[i], font_info, COLOR_TEXT))
        for li, (txt, font, color) in enumerate(lines):
            draw.text((info_x, info_y + li * 22), txt, font=font, fill=color)

        canvas = np.array(pil)

    Image.fromarray(canvas).save(out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(run_dir: str, per_iter: bool = True):
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    print(f"Processing: {run_dir}")
    cfg, meta, latents, evals_data = load_run(run_dir)

    n_evals = meta["e_states"].shape[0]
    frameskip = cfg.get("frameskip", 5)
    print(f"  n_evals = {n_evals}, frameskip = {frameskip}")

    # --- Setup: load decoder, env, calibrate coordinates ---
    print("  Loading decoder and env ...")
    train_cfg = load_train_cfg(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    decoder = load_decoder(cfg, device)
    the_env = create_render_env(cfg, train_cfg)
    px_scale, px_offset = calibrate_state_to_pixel(the_env)

    # --- Generate pixel data from latents and states ---
    print("  Rendering env frames ...")
    env_frames_all = render_env_frames(the_env, meta["e_states"])
    goal_real_all = np.stack([
        render_single_frame(the_env, meta["state_g"][i]) for i in range(n_evals)
    ])
    # Single clean bg (rendered with ball hidden) shared across all evals —
    # the maze itself doesn't vary per eval, only the ball trajectory does.
    env_bg = render_clean_bg(the_env)
    env_bgs = np.tile(env_bg[None], (n_evals, 1, 1, 1))

    # Start marker color = actual ball color (so the variant is visible at a
    # glance in trajectory.png). Read directly from the env's particle_site rgba.
    raw = the_env.unwrapped
    site_id = raw.model.site_name2id("particle_site")
    start_color = tuple(int(c * 255) for c in raw.model.site_rgba[site_id, :3])

    # Per-run preview.png — env + ball at eval 0's starting position. No
    # trajectory/markers; lets the dashboard show variant appearance (ball
    # color + wall/floor colors) at a glance without any trajectory clutter.
    preview = render_single_frame(the_env, meta["e_states"][0, 0])
    Image.fromarray(preview).save(run_dir / "preview.png")
    print(f"  Saved preview.png")

    print("  Decoding WM latents ...")
    wm_frames_all, goal_recon_all = generate_wm_frames(decoder, latents, device)

    successes = meta["successes"]
    action_len = meta["action_len"]
    state_g = meta["state_g"]

    # reach threshold comes from the env itself (REACH_THRESH on its wrapper),
    # so the label can't drift from the env's actual eval_state.
    reach_thresh = getattr(the_env.unwrapped, "REACH_THRESH", 0.5)
    crit_fn = make_crit_label_fn(cfg, reach_thresh)  # one source for the label

    # --- Summary ---
    print("  Rendering summary.png ...")
    summary_path = run_dir / "summary.png"
    render_summary(run_dir, cfg, meta, summary_path,
                   env_bgs=env_bgs, env_frames_all=env_frames_all,
                   goal_real_all=goal_real_all,
                   px_scale=px_scale, px_offset=px_offset,
                   crit_fn=crit_fn,
                   start_color=start_color)
    print(f"    saved {summary_path}")

    # --- Per-eval outputs ---
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(exist_ok=True)

    for i in range(n_evals):
        eval_dir = evals_dir / f"eval_{i}"
        eval_dir.mkdir(exist_ok=True)
        print(f"  Eval {i} ({'success' if successes[i] else 'failure'}) ...")

        n_env = int(action_len[i]) * frameskip + 1
        n_env = min(n_env, env_frames_all.shape[1])
        n_wm = min(int(action_len[i]) + 1, wm_frames_all.shape[1])
        env_frames = env_frames_all[i, :n_env]
        wm_frames = wm_frames_all[i, :n_wm]
        goal_real = goal_real_all[i]
        goal_recon = goal_recon_all[i, 0]

        e_states_i = meta["e_states"][i, :n_env]
        crit_i = crit_fn(e_states_i[-1], state_g[i])

        # --- trajectory.png ---
        traj_img = render_trajectory_plot(
            e_states=e_states_i, state_g=state_g[i],
            success=successes[i], bg_img=env_bgs[i],
            start_frame=env_frames[0], final_frame=env_frames[-1],
            goal_frame=goal_real,
            px_scale=px_scale, px_offset=px_offset, scale=2,
            eval_idx=i, action_len=action_len[i],
            cfg=cfg, start_color=start_color, crit_label=crit_i,
        )
        Image.fromarray(traj_img).save(eval_dir / "trajectory.png")

        # --- frames.png ---
        frames_img = render_frames_grid(
            env_frames, wm_frames, goal_real, goal_recon,
            eval_idx=i, success=successes[i],
            action_len=action_len[i], frameskip=frameskip,
            cfg=cfg, crit_label=crit_i,
        )
        Image.fromarray(frames_img).save(eval_dir / "frames.png")

        # --- video.mp4 ---
        render_video(
            env_frames, wm_frames, goal_real, env_bgs[i],
            eval_idx=i, success=successes[i],
            action_len=action_len[i], frameskip=frameskip,
            out_path=str(eval_dir / "video.mp4"),
            cfg=cfg, crit_label=crit_i,
        )

        # --- per-iter snapshots (optional) ---
        if per_iter:
            n_taken_cfg = (cfg.get("planner", {}) or {}).get("n_taken_actions")
            if n_taken_cfg is None:
                continue
            n_taken_int = int(n_taken_cfg)
            env_full = env_frames_all[i]
            wm_full = wm_frames_all[i]
            actual_iters = max(1, int(np.ceil(int(action_len[i]) / n_taken_int)))

            iters_data = evals_data["evals"][str(i)]["iters"]
            for k in range(actual_iters):
                env_end = min((k + 1) * n_taken_int * frameskip + 1, env_full.shape[0])
                wm_end = min((k + 1) * n_taken_int + 1, wm_full.shape[0])
                env_k = env_full[:env_end]
                wm_k = wm_full[:wm_end]
                e_states_k = meta["e_states"][i, :env_end]

                iter_metrics = iters_data[k]
                action_len_k = min(int(action_len[i]), (k + 1) * n_taken_int)
                success_k = iter_metrics["success"]
                iter_label = f"{i} [{k + 1}/{actual_iters}]"
                crit_k = crit_fn(e_states_k[-1], state_g[i])

                iter_dir = eval_dir / f"iter_{k + 1}"
                iter_dir.mkdir(exist_ok=True)

                traj_img = render_trajectory_plot(
                    e_states=e_states_k, state_g=state_g[i],
                    success=success_k, bg_img=env_bgs[i],
                    start_frame=env_k[0], final_frame=env_k[-1],
                    goal_frame=goal_real,
                    px_scale=px_scale, px_offset=px_offset, scale=2,
                    eval_idx=iter_label, action_len=action_len_k,
                    cfg=cfg, start_color=start_color, crit_label=crit_k,
                )
                Image.fromarray(traj_img).save(iter_dir / "trajectory.png")

                frames_img = render_frames_grid(
                    env_k, wm_k, goal_real, goal_recon,
                    eval_idx=iter_label, success=success_k,
                    action_len=action_len_k, frameskip=frameskip,
                    cfg=cfg, crit_label=crit_k,
                )
                Image.fromarray(frames_img).save(iter_dir / "frames.png")

                render_video(
                    env_k, wm_k, goal_real, env_bgs[i],
                    eval_idx=iter_label, success=success_k,
                    action_len=action_len_k, frameskip=frameskip,
                    out_path=str(iter_dir / "video.mp4"),
                    cfg=cfg, crit_label=crit_k,
                )

    print("Done.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Render visualizations for a plan_outputs/<run>/ directory."
    )
    parser.add_argument("run_dir", help="Path to plan_outputs/<run>/")
    parser.add_argument(
        "--per-iter",
        action="store_true",
        default=True,
        help="Generate cumulative per-iter snapshots (on by default).",
    )
    args = parser.parse_args()
    main(args.run_dir, per_iter=args.per_iter)
