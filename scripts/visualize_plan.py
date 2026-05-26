#!/usr/bin/env python
"""
Annotated visualizations for a plan_outputs/<run>/ directory.

Reads:
    .hydra/config.yaml (single source of truth for config)
    plan_meta.pkl      (state_0/g, gt_actions, actions, action_len, successes, e_states)
    plan_visuals.pkl   (obs_g, env_frames, wm_obs_0_recon, wm_obs_g_recon, wm_imagined)

Writes:
    summary.png                              global overview
    evals/eval_N/trajectory.png              one eval's trajectory plot
    evals/eval_N/frames.png                  7-column grid (env vs imagined)
    evals/eval_N/video.mp4                   annotated video with goal overlay

Usage:
    python scripts/visualize_plan.py plan_outputs/<run>/
"""

import sys
import json
import pickle
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import cm
import imageio


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

# PointMaze U-maze open regions (in MuJoCo (x, y) coordinates).
# Derived from env/pointmaze/point_maze_wrapper.py:
#   left corridor:  x ∈ [0.5, 1.1], y ∈ [0.5, 3.1]
#   right corridor: x ∈ [2.5, 3.1], y ∈ [0.5, 3.1]
#   top connector:  x ∈ [1.1, 2.5], y ∈ [2.5, 3.1]
U_MAZE_OPEN_REGIONS = [
    (0.5, 1.1, 0.5, 3.1),   # left corridor (x_min, x_max, y_min, y_max)
    (2.5, 3.1, 0.5, 3.1),   # right corridor
    (1.1, 2.5, 2.5, 3.1),   # top connector
]
# Plot extent (slightly larger than the open region)
MAZE_X_RANGE = (0.0, 3.6)
MAZE_Y_RANGE = (0.0, 3.6)

# Color constants
COLOR_SUCCESS = (0, 160, 0)
COLOR_FAILURE = (200, 0, 0)
COLOR_TEXT = (255, 255, 255)
COLOR_BG = (30, 30, 30)
COLOR_PANEL_BG = (50, 50, 50)

N_TIME_COLUMNS = 6   # detail PNG samples 6 time points + 1 goal column

# Success threshold per env (for the "pos_dist < thresh" comparator on viz titles).
# evals.json already has the bool success (computed by env.eval_state), but the
# numeric threshold is not in evals.json — kept here so titles can show "X < 0.5".
SUCCESS_THRESH = {
    "point_maze": 0.5,
    "wall":       4.5,
}


# -----------------------------------------------------------------------------
# Font loading
# -----------------------------------------------------------------------------

def get_font(size=14, bold=False):
    """Try common font paths."""
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


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_run(run_dir: Path):
    """Load all data needed for visualization from a plan_outputs/<run>/ directory.

    Format:
      .hydra/config.yaml - full Hydra config (single source of truth)
      plan_meta.pkl      - per-traj data + algorithm results
      plan_visuals.pkl   - per-traj pixel data
      evals.json         - per-eval per-iter metrics (success, pos_dist, etc.)
    """
    cfg = OmegaConf.load(run_dir / ".hydra" / "config.yaml")

    with open(run_dir / "plan_meta.pkl", "rb") as f:
        meta = pickle.load(f)
    with open(run_dir / "plan_visuals.pkl", "rb") as f:
        visuals = pickle.load(f)
    with open(run_dir / "evals.json") as f:
        evals = json.load(f)

    return cfg, meta, visuals, evals


def cfg_to_strings(cfg):
    """Format hydra cfg as 3 short header lines for viz labels:
    1) task / eval settings
    2) MPC (outer) with full config field names
    3) sub-planner (inner, indented under MPC) — typically CEM
    """
    if cfg is None:
        return ("", "", "")
    planner_cfg = cfg.get("planner", {}) or {}
    is_mpc = "MPC" in str(planner_cfg.get("_target_", ""))
    sub_cfg = planner_cfg.get("sub_planner", {}) if is_mpc else planner_cfg

    line1 = (
        f"{cfg.get('model_name', '?')} | n_evals={cfg.get('n_evals')} | "
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
    """How many MPC outer iterations to slice action_len into.

    For MPC: ceil(action_len / n_taken_actions). For non-MPC planners
    (single open-loop plan), always 1.
    """
    planner_cfg = (cfg or {}).get("planner", {}) or {}
    n_taken_cfg = planner_cfg.get("n_taken_actions")
    if n_taken_cfg is None:
        return 1  # non-MPC: single open-loop plan, no iters
    return max(1, int(np.ceil(int(action_len) / max(1, n_taken_cfg))))


# -----------------------------------------------------------------------------
# Image utilities
# -----------------------------------------------------------------------------

def chw_to_hwc(img):
    """Convert (C, H, W) uint8 → (H, W, C) uint8."""
    if img.ndim == 3 and img.shape[0] == 3:
        return np.transpose(img, (1, 2, 0))
    return img


def detect_ball_center(img):
    """Return (px, py) centroid of the green ball, or None if not found."""
    img_n = img.astype(np.float32) / 255.0
    r, g, b = img_n[..., 0], img_n[..., 1], img_n[..., 2]
    mask = (g > 0.4) & (r < 0.5) & (b < 0.5) & (g > r) & (g > b)
    if mask.sum() < 10:
        return None
    ys, xs = np.where(mask)
    return float(xs.mean()), float(ys.mean())


def detect_trajectory_pixels(env_frames):
    """Detect ball center in each env_frame; interpolate over missing detections.

    env_frames: (T, H, W, 3) uint8
    Returns: (T, 2) array of (px, py).
    """
    raw = [detect_ball_center(f) for f in env_frames]
    arr = np.full((len(raw), 2), np.nan, dtype=np.float32)
    for i, p in enumerate(raw):
        if p is not None:
            arr[i] = p
    # Linear interpolation over NaNs
    for col in range(2):
        nans = np.isnan(arr[:, col])
        if nans.any() and (~nans).any():
            arr[nans, col] = np.interp(
                np.where(nans)[0], np.where(~nans)[0], arr[~nans, col]
            )
    return arr


def overlay_goal_ghost(frame, goal_frame, clean_bg,
                       color=(255, 50, 50), alpha=1.0):
    """Diff-based colored overlay: place a colored goal ghost on the frame.

    Steps:
      1. diff = goal_frame - clean_bg  (non-zero only where the goal ball is)
      2. ball_signal = diff's green channel (the ball is green in MuJoCo,
         so its green channel is the cleanest "ball-ness" indicator)
      3. paint with the given color * signal strength

    Result: current frame stays untouched, a colored ghost appears at the
    goal ball location. No mask extraction — works on any environment as
    long as a clean background is available.
    """
    diff = goal_frame.astype(np.float32) - clean_bg.astype(np.float32)
    ball_signal = np.clip(diff[..., 1:2], 0, 255)  # (H, W, 1) using green channel
    color_arr = np.array(color, dtype=np.float32) / 255.0  # normalized RGB weights
    colored = ball_signal * color_arr  # (H, W, 3) colored ghost
    result = frame.astype(np.float32) + colored * alpha
    return np.clip(result, 0, 255).astype(np.uint8)


def compute_clean_bg(frames):
    """Median over the time dimension → ball-free background.
    frames: (T, H, W, 3) uint8
    """
    return np.median(frames, axis=0).astype(np.uint8)


# -----------------------------------------------------------------------------
# Trajectory plot (matplotlib)
# -----------------------------------------------------------------------------

def estimate_ball_radius_px(env_frames):
    """Estimate ball radius (in pixels) by averaging detection across frames.

    Accepts shape (..., H, W, 3): flattens leading dims to per-frame.
    """
    # Flatten leading dims so each "frame" is (H, W, 3)
    frames = env_frames.reshape(-1, *env_frames.shape[-3:])
    img_n = frames.astype(np.float32) / 255.0
    r, g, b = img_n[..., 0], img_n[..., 1], img_n[..., 2]
    mask = (g > 0.4) & (r < 0.5) & (b < 0.5) & (g > r) & (g > b)
    # Pixel count per frame; convert to radius via area = π r²
    counts = mask.sum(axis=(-2, -1))  # (N,) per-frame counts
    counts = counts[counts >= 10]
    if len(counts) == 0:
        return 5.0  # fallback default
    avg_area = counts.mean()
    return float(np.sqrt(avg_area / np.pi))


def _color_blend_ball(canvas_np, source_frame, clean_bg, color, alpha=1.0):
    """Blend the ball from `source_frame` into `canvas_np`, retinted with `color`.

    Uses the magnitude of the per-pixel difference (across all 3 RGB channels)
    as the ball "signal" — this is more robust than using one channel because
    the orange maze floor has non-zero green. The signal magnitude is then
    used as a mask weight to REPLACE (not add) the background pixel with
    the marker color. This preserves the ball's smooth anti-aliased edges
    while making the marker visually solid.
    """
    diff = source_frame.astype(np.float32) - clean_bg.astype(np.float32)
    # Magnitude of color change across all 3 channels, normalized to [0, 1]
    signal = np.linalg.norm(diff, axis=-1, keepdims=True) / 255.0   # (H, W, 1)
    signal = np.clip(signal * 2.0, 0, 1)                            # boost contrast
    color_arr = np.array(color, dtype=np.float32)                   # (3,)
    # Replace pixels with marker color, weighted by signal strength
    result = canvas_np.astype(np.float32) * (1 - signal * alpha) + color_arr * (signal * alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def render_trajectory_plot(env_frames, goal_real, success, bg_img,
                           ball_radius_px, scale=2,
                           eval_idx=None, action_len=None,
                           pos_dist=None, success_thresh=0.5,
                           frameskip=5, cfg=None,
                           show_title=True, show_legend=True):
    """Draw a trajectory on top of a clean MuJoCo render.

    Trajectory points come directly from color-detecting the ball in each
    env_frame — no MuJoCo→pixel fitting; the trajectory passes through the
    exact pixel the ball occupied in each frame.

    Start/Final/Goal markers are placed by **blending** the actual ball
    appearance from env_frames[0], env_frames[-1], goal_real onto the
    background — this preserves the smooth anti-aliased ball edges and
    just re-tints them with marker colors (blue / red / green).

    env_frames:  (T, H, W, 3) uint8 — actual env render frames
    goal_real:   (H, W, 3) uint8    — goal-state env render
    success:     bool               — title color
    bg_img:      (H, W, 3) uint8    — ball-free maze background
    Returns:     (H', W', 3) uint8
    """
    # Compose the marker overlays BEFORE upscaling.
    # Trajectory colors: green (Start) → blue (Final). Goal red (video convention).
    start_color = (40, 200, 60)    # green (matches the ball's natural color)
    final_color = (50, 130, 240)   # blue
    goal_color = (220, 50, 50)     # red
    canvas = bg_img.copy()
    canvas = _color_blend_ball(canvas, goal_real, bg_img, color=goal_color)
    canvas = _color_blend_ball(canvas, env_frames[0], bg_img, color=start_color)
    canvas = _color_blend_ball(canvas, env_frames[-1], bg_img, color=final_color)

    # Upscale for sharper line drawing
    pil = Image.fromarray(canvas).resize(
        (canvas.shape[1] * scale, canvas.shape[0] * scale), Image.NEAREST,
    ).convert("RGB")
    draw = ImageDraw.Draw(pil, "RGBA")

    # Detect ball pixel position directly in each frame (for trajectory line)
    traj_px = detect_trajectory_pixels(env_frames) * scale  # (T, 2)
    goal_pix = detect_ball_center(goal_real)
    if goal_pix is None:
        goal_pix = (bg_img.shape[1] / 2, bg_img.shape[0] / 2)
    goal_px = np.array(goal_pix) * scale

    # Marker size matches actual ball radius in the upscaled image
    R = ball_radius_px * scale

    # Trajectory line: linear interpolation between start_color and final_color
    n = len(traj_px)
    line_w = max(2, int(R * 0.3))
    start_arr = np.array(start_color, dtype=np.float32)
    final_arr = np.array(final_color, dtype=np.float32)
    for i in range(n - 1):
        t = i / max(n - 1, 1)
        c = (1 - t) * start_arr + t * final_arr
        rgb = tuple(int(v) for v in c)
        draw.line(
            [tuple(traj_px[i]), tuple(traj_px[i + 1])],
            fill=rgb + (255,), width=line_w,
        )

    # Start/Final/Goal markers already blended onto bg before upscaling
    # (see _color_blend_ball above), preserving smooth ball edges.

    # Legend box (bottom-right corner) — explains markers
    if show_legend:
        legend_font = get_font(13, bold=True)
        pad = 8
        line_h = 18
        legend_items = [
            ((40, 200, 60), "Start"),
            ((50, 130, 240), "Final"),
            ((220, 50, 50), "Goal"),
        ]
        max_label_w = max(
            legend_font.getbbox(label)[2] for _, label in legend_items
        )
        box_w = 24 + max_label_w + 2 * pad
        box_h = line_h * len(legend_items) + 2 * pad
        box_x = pil.width - box_w - pad
        box_y = pil.height - box_h - pad
        draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                       fill=(0, 0, 0, 180), outline=(255, 255, 255, 200))
        r = 6
        for i, (color, label) in enumerate(legend_items):
            cy = box_y + pad + i * line_h + line_h // 2
            cx = box_x + pad + 6
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color)
            draw.text((box_x + pad + 18, cy - 7), label,
                      font=legend_font, fill=(255, 255, 255, 255))

    # If title disabled (e.g. for summary thumbnails), return early
    if not show_title:
        return np.array(pil)

    # Title strip — 4 lines (status / step+pos_dist / MPC / CEM), matching frames.png
    title_h = 90
    title_color = (0, 160, 0) if success else (200, 0, 0)
    eval_str = f"Eval {eval_idx} — " if eval_idx is not None else ""
    status_text = f"{eval_str}{'SUCCESS' if success else 'FAILURE'}"
    info_parts = []
    if action_len is not None:
        action_len = int(action_len)
        info_parts.append(f"mpc_iters={estimate_mpc_iters(action_len, cfg)}")
        info_parts.append(f"action_len={action_len}")
    if pos_dist is not None:
        cmp = "<" if pos_dist < success_thresh else "≥"
        info_parts.append(f"pos_dist={pos_dist:.2f} {cmp} {success_thresh}")
    info_text = " | ".join(info_parts)
    _, sub_mpc, sub_cem = (cfg_to_strings(cfg)
                            if cfg is not None else ("", "", ""))

    final = Image.new("RGB", (pil.width, pil.height + title_h), color=(40, 40, 40))
    final.paste(pil, (0, title_h))
    title_draw = ImageDraw.Draw(final)
    font_status = get_font(16, bold=True)
    font_info = get_font(11)
    # All 4 lines left-aligned (consistent with MPC/CEM tree structure)
    title_draw.text((10, 4), status_text, font=font_status, fill=title_color)
    if info_text:
        title_draw.text((10, 28), info_text, font=font_info, fill=(200, 200, 200))
    if sub_mpc:
        title_draw.text((10, 48), sub_mpc, font=font_info, fill=(180, 180, 180))
    if sub_cem:
        title_draw.text((10, 66), sub_cem, font=font_info, fill=(180, 180, 180))

    return np.array(final)


# -----------------------------------------------------------------------------
# Frames grid (PNG)
# -----------------------------------------------------------------------------

def render_frames_grid(env_frames, wm_frames, goal_real, goal_recon,
                       eval_idx, success, action_len, frameskip,
                       pos_dist=None, success_thresh=0.5, cfg=None):
    """
    Build the 7-column detail PNG: 6 time samples + 1 goal column, 2 rows (env / imagined).

    env_frames: (T*frameskip+1, H, W, 3) uint8  e.g. (26, 224, 224, 3)
    wm_frames:  (T+1, H, W, 3) uint8            e.g. (6, 224, 224, 3)
    goal_real:  (H, W, 3) uint8
    goal_recon: (H, W, 3) uint8
    """
    H, W = env_frames.shape[1:3]
    n_env_frames = env_frames.shape[0]
    n_wm_frames = wm_frames.shape[0]

    # Sample N_TIME_COLUMNS evenly across env time
    env_indices = np.linspace(0, n_env_frames - 1, N_TIME_COLUMNS).astype(int)
    # Map each env index to nearest wm index
    wm_indices = (env_indices / frameskip).round().clip(0, n_wm_frames - 1).astype(int)

    # Build rows
    env_row = [env_frames[i] for i in env_indices] + [goal_real]
    wm_row = [wm_frames[i] for i in wm_indices] + [goal_recon]

    n_cols = len(env_row)  # 7

    # Layout constants
    label_w = 80        # left-side label column
    cell_pad = 4        # padding around each cell
    header_h = 30       # column header
    title_h = 100       # top title (4 lines: status / step+dist / MPC / CEM)
    legend_h = 30       # bottom legend

    cell_w = W + 2 * cell_pad
    cell_h = H + 2 * cell_pad

    total_w = label_w + n_cols * cell_w
    total_h = title_h + header_h + 2 * cell_h + legend_h

    canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

    # Draw title (4 lines: status / step+pos_dist / MPC / CEM)
    status_color = COLOR_SUCCESS if success else COLOR_FAILURE
    status_text = f"Eval {eval_idx} — {'SUCCESS' if success else 'FAILURE'}"
    action_len = int(action_len)
    step_parts = [
        f"mpc_iters={estimate_mpc_iters(action_len, cfg)}",
        f"action_len={action_len}",
    ]
    if pos_dist is not None:
        cmp = "<" if pos_dist < success_thresh else "≥"
        step_parts.append(f"pos_dist = {pos_dist:.2f} {cmp} {success_thresh}")
    step_info = "  |  ".join(step_parts)

    _, sub_mpc, sub_cem = (cfg_to_strings(cfg)
                            if cfg is not None else ("", "", ""))

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

    # Column headers
    header_y = title_h
    draw.rectangle([0, header_y, total_w, header_y + header_h], fill=COLOR_PANEL_BG)
    for ci, env_idx in enumerate(env_indices):
        x = label_w + ci * cell_w + cell_pad
        text = f"t={env_idx}"
        bbox = draw.textbbox((0, 0), text, font=font_header)
        tw = bbox[2] - bbox[0]
        draw.text((x + (W - tw) // 2, header_y + 8), text, font=font_header, fill=COLOR_TEXT)
    # Goal column header
    x = label_w + N_TIME_COLUMNS * cell_w + cell_pad
    bbox = draw.textbbox((0, 0), "Goal", font=font_header)
    tw = bbox[2] - bbox[0]
    draw.text((x + (W - tw) // 2, header_y + 8), "Goal", font=font_header, fill=COLOR_TEXT)

    # Row labels
    for ri, label in enumerate(["Env (MuJoCo)", "Imagined (WM)"]):
        y = title_h + header_h + ri * cell_h + cell_h // 2
        # multi-line
        for li, ln in enumerate(label.split()):
            draw.text((8, y - 10 + li * 14), ln, font=font_label, fill=COLOR_TEXT)

    canvas = np.array(pil)

    # Place images
    for ri, row in enumerate([env_row, wm_row]):
        for ci, img in enumerate(row):
            y0 = title_h + header_h + ri * cell_h + cell_pad
            x0 = label_w + ci * cell_w + cell_pad
            canvas[y0:y0 + H, x0:x0 + W] = img

    # Bottom legend
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


# -----------------------------------------------------------------------------
# Annotated video (MP4)
# -----------------------------------------------------------------------------

def render_video(env_frames, wm_frames, goal_real, goal_recon,
                 eval_idx, success, action_len, frameskip, out_path, fps=12,
                 pos_dist=None, success_thresh=0.5, cfg=None):
    """
    Build annotated mp4 video.

    env_frames: (T*frameskip+1, H, W, 3) uint8
    wm_frames:  (T+1, H, W, 3) uint8
    """
    n_env_frames = env_frames.shape[0]
    H, W = env_frames.shape[1:3]

    status_text = f"Eval {eval_idx} — {'SUCCESS' if success else 'FAILURE'}"
    status_color = COLOR_SUCCESS if success else COLOR_FAILURE
    # Static info split into 2 lines for readability
    action_len = int(action_len)
    stats_line1 = f"mpc_iters={estimate_mpc_iters(action_len, cfg)} | action_len={action_len}"
    if pos_dist is not None:
        cmp = "<" if pos_dist < success_thresh else "≥"
        dist_line = f"pos_dist = {pos_dist:.2f} {cmp} {success_thresh}"
    else:
        dist_line = ""

    # Top banner: 4 lines — status (large) / frame counter (dynamic) / stats / pos_dist
    status_h = 22  # large status line
    frame_h = 16   # dynamic frame counter
    stats_h = 16   # static stats (mpc_iters + action_len)
    dist_h = 16    # static pos_dist
    top_banner_h = status_h + frame_h + stats_h + dist_h
    mid_banner_h = 20

    panel_w = W
    panel_h = H

    # No side padding — canvas width = panel width
    total_w = W
    panel_x_offset = 0
    total_h = top_banner_h + panel_h + mid_banner_h + panel_h

    # Auto-fit status font: shrink size until status_text fits in video width.
    # status_text can grow when --per-iter adds "[K/N]" to eval_idx.
    from PIL import ImageDraw as _ImageDraw, Image as _Image
    _probe = _ImageDraw.Draw(_Image.new("RGB", (1, 1)))
    font_status = get_font(15, bold=True)
    for _sz in (15, 13, 12, 11, 10):
        font_status = get_font(_sz, bold=True)
        bbox = _probe.textbbox((0, 0), status_text, font=font_status)
        if bbox[2] - bbox[0] <= total_w - 8:  # 8px margin
            break
    font_top = get_font(10)
    font_mid = get_font(9)

    # Precompute clean backgrounds (median over time → ball-free maze)
    env_clean = compute_clean_bg(env_frames)
    wm_clean = compute_clean_bg(wm_frames)

    writer = imageio.get_writer(out_path, fps=fps)

    # Total number of WM predictions = wm_frames - 1 (frame 0 is recon, not pred)
    n_preds = wm_frames.shape[0] - 1

    for t in range(n_env_frames):
        # Imagined frame index (every frameskip env frames, hold in between)
        i_idx = min(t // frameskip, wm_frames.shape[0] - 1)
        # pred 0 = reconstruction; pred 1..n_preds = forward predictions
        pred_segment = min(t // frameskip, n_preds)

        # Diff-based overlay: keeps current frame intact, adds goal-ball signal as ghost.
        env_panel = overlay_goal_ghost(env_frames[t], goal_real, env_clean)
        wm_panel = overlay_goal_ghost(wm_frames[i_idx], goal_recon, wm_clean)

        # Compose canvas
        canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

        # Top banner — 4 lines: status (large, prominent) / frame (dynamic) / stats / dist
        canvas[0:top_banner_h] = COLOR_PANEL_BG
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)

        # Line 1: status (large, colored)
        bbox = draw.textbbox((0, 0), status_text, font=font_status)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, 3), status_text,
                  font=font_status, fill=status_color)

        # Line 2: dynamic frame counter
        frame_text = f"Frame {t}/{n_env_frames - 1} | pred {pred_segment}/{n_preds}"
        bbox = draw.textbbox((0, 0), frame_text, font=font_top)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, status_h + 1), frame_text,
                  font=font_top, fill=COLOR_TEXT)

        # Line 3: static stats (mpc_iters + action_len)
        bbox = draw.textbbox((0, 0), stats_line1, font=font_top)
        tw = bbox[2] - bbox[0]
        draw.text(((total_w - tw) // 2, status_h + frame_h + 1), stats_line1,
                  font=font_top, fill=(200, 200, 200))

        # Line 4: pos_dist (static)
        if dist_line:
            bbox = draw.textbbox((0, 0), dist_line, font=font_top)
            tw = bbox[2] - bbox[0]
            draw.text(((total_w - tw) // 2, status_h + frame_h + stats_h + 1),
                      dist_line, font=font_top, fill=(200, 200, 200))

        # Env panel (centered)
        canvas = np.array(pil)
        canvas[top_banner_h:top_banner_h + panel_h,
               panel_x_offset:panel_x_offset + panel_w] = env_panel

        # Mid banner (label between env and imag)
        y_mid = top_banner_h + panel_h
        canvas[y_mid:y_mid + mid_banner_h] = COLOR_PANEL_BG
        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        mid_text = "↑ Env  |  ↓ Imagined (WM)  |  red = goal"
        bbox = draw.textbbox((0, 0), mid_text, font=font_mid)
        mw = bbox[2] - bbox[0]
        draw.text(((total_w - mw) // 2, y_mid + 4), mid_text, font=font_mid, fill=COLOR_TEXT)

        # Imag panel (centered)
        canvas = np.array(pil)
        y_imag = y_mid + mid_banner_h
        canvas[y_imag:y_imag + panel_h,
               panel_x_offset:panel_x_offset + panel_w] = wm_panel

        writer.append_data(canvas)

    writer.close()


# -----------------------------------------------------------------------------
# Summary PNG
# -----------------------------------------------------------------------------

def render_summary(run_dir, cfg, meta, visuals, out_path,
                   env_bgs, env_frames_all, goal_real_all, ball_radius_px,
                   pos_dists=None, success_thresh=0.5):
    """Render summary.png: trajectory plots for all evals + status info."""
    e_states = meta["e_states"]
    successes = meta["successes"]
    action_len = meta["action_len"]
    n_evals = e_states.shape[0]

    # Generate trajectory plot per eval (returns numpy image)
    traj_imgs = []
    for i in range(n_evals):
        n_frames = int(action_len[i]) * cfg.get("frameskip", 5) + 1
        n_frames = min(n_frames, env_frames_all.shape[1])
        pd = pos_dists[i] if pos_dists is not None else None
        traj_img = render_trajectory_plot(
            env_frames=env_frames_all[i, :n_frames],
            goal_real=goal_real_all[i, 0],
            success=successes[i],
            bg_img=env_bgs[i],
            ball_radius_px=ball_radius_px, scale=1,
            eval_idx=i, action_len=action_len[i],
            pos_dist=pd, success_thresh=success_thresh,
            frameskip=cfg.get("frameskip", 5),
            show_title=False, show_legend=False,
        )
        traj_imgs.append(traj_img)

    traj_H, traj_W = traj_imgs[0].shape[:2]

    # Layout — title now has 4 lines (result, task, MPC, CEM-indented)
    title_h = 80
    legend_h = 24
    info_w = 190
    row_h = traj_H
    min_title_w = 460  # enough for CEM line at font 10 (full param names)
    total_w = max(traj_W + info_w, min_title_w)
    total_h = title_h + legend_h + n_evals * row_h

    canvas = np.full((total_h, total_w, 3), COLOR_BG, dtype=np.uint8)

    # 4-line title: result | task | MPC | CEM (indented under MPC)
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

    # Global legend strip (horizontal, below title) — applies to all eval thumbnails
    legend_items = [
        ((40, 200, 60), "Start"),
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

    # Per-eval rows
    for i in range(n_evals):
        y0 = title_h + legend_h + i * row_h
        # Trajectory image
        canvas[y0:y0 + traj_H, 0:traj_W] = traj_imgs[i]

        # Info panel
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
        if pos_dists is not None:
            cmp = "<" if pos_dists[i] < success_thresh else "≥"
            lines.append((
                f"pos_dist = {pos_dists[i]:.2f} {cmp} {success_thresh}",
                font_info, COLOR_TEXT,
            ))
        for li, (txt, font, color) in enumerate(lines):
            draw.text((info_x, info_y + li * 22), txt, font=font, fill=color)

        canvas = np.array(pil)

    Image.fromarray(canvas).save(out_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main(run_dir: str, per_iter: bool = False):
    run_dir = Path(run_dir).resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run dir not found: {run_dir}")

    print(f"Processing: {run_dir}")
    cfg, meta, visuals, evals_data = load_run(run_dir)

    n_evals = meta["e_states"].shape[0]
    frameskip = cfg.get("frameskip", 5)
    print(f"  n_evals = {n_evals}, frameskip = {frameskip}")

    # Real goal frames (env-rendered): obs_g['visual'] is now flat in visuals
    goal_real_all = visuals["obs_g"]                              # (n, 1, H, W, 3) uint8

    env_frames_all = visuals["env_frames"]                        # (n, T*f+1, H, W, 3) uint8
    # Assemble the WM row: obs_0 reconstruction + forward imaginations → (n, T+1, 3, H, W)
    wm_frames_all = np.concatenate(
        [visuals["wm_obs_0_recon"], visuals["wm_imagined"]], axis=1
    )                                                              # (n, T+1, 3, H, W) uint8
    goal_recon_all = visuals["wm_obs_g_recon"]                    # (n, 1, 3, H, W) uint8

    successes = meta["successes"]
    action_len = meta["action_len"]

    # Pre-compute per-eval clean MuJoCo backgrounds (median over frames → ball-free)
    env_bgs_all = np.stack([
        compute_clean_bg(env_frames_all[i]) for i in range(n_evals)
    ])  # (n_evals, H, W, 3)

    # Estimate the actual ball radius (in pixels) for sizing markers
    ball_radius_px = estimate_ball_radius_px(env_frames_all)
    print(f"  Estimated ball radius: {ball_radius_px:.1f} px")

    # Pull per-eval pos_dist directly from evals.json (last iter the eval went through
    # = the eval's effective stopping point). No on-the-fly state-slicing needed.
    pos_dists = np.array([
        evals_data["evals"][str(i)]["iters"][-1]["pos_dist"]
        for i in range(n_evals)
    ])
    # Success threshold is env-specific and not in evals.json (which only has the
    # bool success). Looked up here so titles can show "pos_dist X < thresh".
    success_thresh = SUCCESS_THRESH.get(cfg.get("model_name"), 0.5)

    # --- Summary ---
    print("  Rendering summary.png ...")
    summary_path = run_dir / "summary.png"
    render_summary(run_dir, cfg, meta, visuals, summary_path,
                   env_bgs=env_bgs_all, env_frames_all=env_frames_all,
                   goal_real_all=goal_real_all,
                   ball_radius_px=ball_radius_px,
                   pos_dists=pos_dists, success_thresh=success_thresh)
    print(f"    saved {summary_path}")

    # --- Per-eval outputs ---
    evals_dir = run_dir / "evals"
    evals_dir.mkdir(exist_ok=True)

    for i in range(n_evals):
        eval_dir = evals_dir / f"eval_{i}"
        eval_dir.mkdir(exist_ok=True)
        print(f"  Eval {i} ({'success' if successes[i] else 'failure'}) ...")

        # Prepare per-eval data
        env_frames = env_frames_all[i]                            # (T*f+1, H, W, 3)
        wm_frames = np.stack([chw_to_hwc(f) for f in wm_frames_all[i]])  # (T+1, H, W, 3)
        goal_real = goal_real_all[i, 0]                           # (H, W, 3)
        goal_recon = chw_to_hwc(goal_recon_all[i, 0])             # (H, W, 3)

        # Trim frames to the per-eval action length.
        n_env = int(action_len[i]) * frameskip + 1
        n_env = min(n_env, env_frames.shape[0])
        n_wm = min(int(action_len[i]) + 1, wm_frames.shape[0])
        env_frames = env_frames[:n_env]
        wm_frames = wm_frames[:n_wm]

        # --- trajectory.png ---
        n_traj = env_frames.shape[0]
        traj_img = render_trajectory_plot(
            env_frames=env_frames, goal_real=goal_real,
            success=successes[i],
            bg_img=env_bgs_all[i],
            ball_radius_px=ball_radius_px, scale=2,
            eval_idx=i, action_len=action_len[i],
            pos_dist=pos_dists[i], success_thresh=success_thresh,
            frameskip=frameskip, cfg=cfg,
        )
        Image.fromarray(traj_img).save(eval_dir / "trajectory.png")

        # --- frames.png ---
        frames_img = render_frames_grid(
            env_frames, wm_frames, goal_real, goal_recon,
            eval_idx=i, success=successes[i],
            action_len=action_len[i], frameskip=frameskip,
            pos_dist=pos_dists[i], success_thresh=success_thresh,
            cfg=cfg,
        )
        Image.fromarray(frames_img).save(eval_dir / "frames.png")

        # --- video.mp4 ---
        render_video(
            env_frames, wm_frames, goal_real, goal_recon,
            eval_idx=i, success=successes[i],
            action_len=action_len[i], frameskip=frameskip,
            out_path=str(eval_dir / "video.mp4"),
            pos_dist=pos_dists[i], success_thresh=success_thresh,
            cfg=cfg,
        )

        # --- per-iter cumulative snapshots (optional) ---
        if per_iter:
            n_taken_cfg = (cfg.get("planner", {}) or {}).get("n_taken_actions")
            if n_taken_cfg is None:
                # Non-MPC: only one "iter" exists; skip (full viz already covers it)
                continue
            n_taken_int = int(n_taken_cfg)
            # Use the full (un-trimmed) per-eval arrays for slicing
            env_full = env_frames_all[i]
            wm_full = np.stack([chw_to_hwc(f) for f in wm_frames_all[i]])
            actual_iters = max(1, int(np.ceil(int(action_len[i]) / n_taken_int)))

            iters_data = evals_data["evals"][str(i)]["iters"]
            for k in range(actual_iters):
                env_end = (k + 1) * n_taken_int * frameskip + 1
                env_end = min(env_end, env_full.shape[0])
                wm_end = (k + 1) * n_taken_int + 1
                wm_end = min(wm_end, wm_full.shape[0])
                env_k = env_full[:env_end]
                wm_k = wm_full[:wm_end]

                # Read iter-k metrics from evals.json (already cap-adjusted in plan.py)
                iter_metrics = iters_data[k]
                action_len_k = min(int(action_len[i]), (k + 1) * n_taken_int)
                success_k = iter_metrics["success"]
                pos_dist_k = iter_metrics["pos_dist"]
                iter_label = f"{i} [{k + 1}/{actual_iters}]"

                iter_dir = eval_dir / f"iter_{k + 1}"
                iter_dir.mkdir(exist_ok=True)

                traj_img = render_trajectory_plot(
                    env_frames=env_k, goal_real=goal_real,
                    success=success_k,
                    bg_img=env_bgs_all[i],
                    ball_radius_px=ball_radius_px, scale=2,
                    eval_idx=iter_label, action_len=action_len_k,
                    pos_dist=pos_dist_k, success_thresh=success_thresh,
                    frameskip=frameskip, cfg=cfg,
                )
                Image.fromarray(traj_img).save(iter_dir / "trajectory.png")

                frames_img = render_frames_grid(
                    env_k, wm_k, goal_real, goal_recon,
                    eval_idx=iter_label, success=success_k,
                    action_len=action_len_k, frameskip=frameskip,
                    pos_dist=pos_dist_k, success_thresh=success_thresh,
                    cfg=cfg,
                )
                Image.fromarray(frames_img).save(iter_dir / "frames.png")

                render_video(
                    env_k, wm_k, goal_real, goal_recon,
                    eval_idx=iter_label, success=success_k,
                    action_len=action_len_k, frameskip=frameskip,
                    out_path=str(iter_dir / "video.mp4"),
                    pos_dist=pos_dist_k, success_thresh=success_thresh,
                    cfg=cfg,
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
        help="Also generate cumulative per-iter snapshots "
             "(eval_N/iter_K/{trajectory,frames,video}). Each iter K shows "
             "the trajectory from t=0 up to the end of MPC iter K.",
    )
    args = parser.parse_args()
    main(args.run_dir, per_iter=args.per_iter)
