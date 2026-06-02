"""Relocate already-migrated plan runs into the task-first output layout.

A migrated run dir (has .hydra/config.yaml with env_name + ckpt_id) currently
lives at e.g.:
    plan_outputs/u_maze/baseline/<ts>_gH5
This moves it under its env_name/ckpt_id prefix to match new runs:
    plan_outputs/point_maze/dinowm_released/u_maze/baseline/<ts>_gH5

os.walk runs WITHOUT followlinks, so the results/ symlink (ablation runs) is
never entered -> ablation is left untouched.

Dry-run by default. Pass --apply to move (and prune emptied old dirs).
"""
import argparse
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

PLAN = Path(
    "/local_data/sz4968/world-model/experiments/dino-wm/dino-wm/plan_outputs"
)


def collect() -> list[tuple[Path, Path]]:
    moves = []
    for dirpath, _dirs, files in os.walk(PLAN):  # no followlinks -> skips results/
        if os.path.basename(dirpath) != ".hydra" or "config.yaml" not in files:
            continue
        run_dir = Path(dirpath).parent
        cfg = OmegaConf.load(Path(dirpath) / "config.yaml")
        env_name = cfg.get("env_name")
        ckpt_id = cfg.get("ckpt_id")
        if env_name is None or ckpt_id is None:
            continue  # not migrated
        rel = run_dir.relative_to(PLAN)
        if rel.parts[:2] == (str(env_name), str(ckpt_id)):
            continue  # already in task-first place
        new_dir = PLAN / str(env_name) / str(ckpt_id) / rel
        moves.append((run_dir, new_dir))
    return sorted(moves)


def prune_empty():
    # Remove now-empty dirs, deepest first; never touch the plan_outputs root.
    for dirpath, _dirs, _files in sorted(os.walk(PLAN), reverse=True):
        p = Path(dirpath)
        if p == PLAN:
            continue
        try:
            if not any(p.iterdir()):
                p.rmdir()
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry-run)")
    args = ap.parse_args()

    moves = collect()
    print(f"{len(moves)} run(s) to relocate\n")
    for src, dst in moves:
        print(f"[move] {src.relative_to(PLAN)}")
        print(f"    -> {dst.relative_to(PLAN)}")
        if args.apply:
            if dst.exists():
                print(f"    !! target exists, skipping")
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))

    if args.apply:
        prune_empty()
        print(f"\nAPPLIED: moved {len(moves)} run(s), pruned empty dirs.")
    else:
        print(f"\nDRY-RUN (nothing moved).")


if __name__ == "__main__":
    main()
