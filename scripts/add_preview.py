#!/usr/bin/env python
"""Add preview.png to existing run dirs without re-running visualize_plan.

For each run dir under plan_outputs/, if preview.png is missing:
  - read .hydra/config.yaml → resolve gym env_id
  - read plan_meta.pkl OR plan_visuals.pkl → pick a valid qpos
  - render the env at that qpos with no markers → save preview.png

Usage:
    python scripts/add_preview.py [--plan-outputs PATH] [--force]
"""
import argparse
import pickle
import sys
from pathlib import Path
import numpy as np
import gym
from omegaconf import OmegaConf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import env.pointmaze  # noqa: F401  (registers gym ids)


def resolve_env_id(cfg) -> str:
    """Handle both old flat env_id_map (str leaf) and new nested format."""
    setting = cfg.get("setting")
    vtype = cfg.get("variant_type", "baseline")
    vname = cfg.get("variant", "baseline")
    eim = cfg.get("env_id_map") or {}
    sub = eim.get(setting)
    if isinstance(sub, str):
        return sub  # old flat format
    if not isinstance(sub, dict):
        return None
    if vtype == "baseline":
        return sub.get("baseline")
    return (sub.get(vtype) or {}).get(vname)


def pick_qpos(run_dir: Path) -> np.ndarray | None:
    """Use eval 0's initial qpos — guaranteed valid for the maze."""
    for meta_name in ("plan_meta.pkl", "plan_visuals.pkl"):
        p = run_dir / meta_name
        if not p.exists():
            continue
        with open(p, "rb") as f:
            data = pickle.load(f)
        if "e_states" in data:
            return np.asarray(data["e_states"][0, 0, :2], dtype=np.float32)
        if "state_0" in data:
            return np.asarray(data["state_0"][0, :2], dtype=np.float32)
    return None


def render_preview(env_id: str, qpos: np.ndarray) -> np.ndarray:
    env = gym.make(env_id)
    env.unwrapped.prepare_for_render()
    raw = env.unwrapped
    raw.set_state(qpos, np.zeros(2, dtype=np.float32))
    return raw.sim.render(224, 224)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan-outputs",
                    default=str(REPO_ROOT / "plan_outputs"))
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing preview.png.")
    args = ap.parse_args()

    plan_outputs = Path(args.plan_outputs).resolve()
    cfg_paths = sorted(plan_outputs.rglob(".hydra/config.yaml"))
    print(f"Scanning {plan_outputs} — {len(cfg_paths)} runs.")

    for cfg_path in cfg_paths:
        run_dir = cfg_path.parent.parent
        preview_path = run_dir / "preview.png"
        if preview_path.exists() and not args.force:
            print(f"  skip (exists): {run_dir.relative_to(plan_outputs)}")
            continue

        cfg = OmegaConf.load(cfg_path)
        env_id = resolve_env_id(cfg)
        if not env_id:
            print(f"  SKIP no env_id: {run_dir.relative_to(plan_outputs)}")
            continue

        qpos = pick_qpos(run_dir)
        if qpos is None:
            print(f"  SKIP no qpos: {run_dir.relative_to(plan_outputs)}")
            continue

        try:
            img = render_preview(env_id, qpos)
        except Exception as e:
            print(f"  ERR  {run_dir.relative_to(plan_outputs)}: {e}")
            continue
        Image.fromarray(img).save(preview_path)
        print(f"  wrote {run_dir.relative_to(plan_outputs)} (env={env_id}, qpos={qpos.tolist()})")


if __name__ == "__main__":
    main()
