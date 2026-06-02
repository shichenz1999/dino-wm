"""One-time migration: rewrite old plan-run .hydra/config.yaml to task-first schema.

Old schema (DINO-WM original, model_name doubles as task):
    ckpt_base_path: <root>/checkpoints/pretrained
    model_name:     point_maze
    model_epoch:    latest

New schema (env_name / ckpt_id separated):
    ckpt_base_path: <root>/checkpoints
    env_name:       point_maze         # = old model_name
    ckpt_id:        dinowm_released     # the released-checkpoint label
    model_epoch:    latest             # unchanged

Only these ckpt fields are touched; every other key is preserved verbatim.
Dry-run by default. Pass --apply to write (backs up each file to .orig first).
"""
import argparse
import os
import shutil
from pathlib import Path

from omegaconf import OmegaConf

PLAN_OUTPUTS = Path(
    "/local_data/sz4968/world-model/experiments/dino-wm/dino-wm/plan_outputs"
)
CKPT_ID = "dinowm_released"


def find_configs(root: Path) -> list[Path]:
    out = []
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        if os.path.basename(dirpath) == ".hydra" and "config.yaml" in files:
            out.append(Path(dirpath) / "config.yaml")
    return sorted(out)


def plan_change(cfg) -> dict | None:
    """Return {field: (old, new)} for a config that needs migrating, else None."""
    if "model_name" not in cfg or "ckpt_id" in cfg:
        return None  # already migrated or not an old-schema plan config
    old_base = str(cfg.get("ckpt_base_path", ""))
    old_name = str(cfg.get("model_name"))
    new_base = str(Path(old_base).parent)  # strip trailing "/pretrained"
    return {
        "ckpt_base_path": (old_base, new_base),
        "model_name -> env_name": (old_name, old_name),
        "ckpt_id (new)": (None, CKPT_ID),
        "model_epoch": (str(cfg.get("model_epoch")), str(cfg.get("model_epoch"))),
    }


def apply_change(cfg):
    """Rebuild the mapping so env_name/ckpt_id sit where model_name was
    (right after ckpt_base_path), instead of being appended at the end."""
    old = OmegaConf.to_container(cfg, resolve=False)  # preserves key order
    new: dict = {}
    for k, v in old.items():
        if k == "ckpt_base_path":
            new[k] = str(Path(str(v)).parent)  # strip trailing "/pretrained"
        elif k == "model_name":
            new["env_name"] = v                # take model_name's slot
            new["ckpt_id"] = CKPT_ID
        else:
            new[k] = v
    return OmegaConf.create(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry-run)")
    args = ap.parse_args()

    configs = find_configs(PLAN_OUTPUTS)
    print(f"Found {len(configs)} .hydra/config.yaml under {PLAN_OUTPUTS}\n")

    n_migrate = 0
    for path in configs:
        rel = path.relative_to(PLAN_OUTPUTS)
        # Ablation runs use a separate (curated-card) schema; leave them alone.
        if "ablation" in rel.parts:
            print(f"[skip]    {rel}  (ablation run)")
            continue
        cfg = OmegaConf.load(path)
        change = plan_change(cfg)
        if change is None:
            print(f"[skip]    {rel}  (no model_name / already migrated)")
            continue
        n_migrate += 1
        print(f"[migrate] {rel}")
        for field, (old, new) in change.items():
            arrow = "(unchanged)" if old == new else f"->  {new}"
            print(f"            {field:22s}: {old}  {arrow}")
        if args.apply:
            backup = path.with_suffix(".yaml.orig")
            if not backup.exists():
                shutil.copy2(path, backup)
            cfg = apply_change(cfg)
            OmegaConf.save(cfg, path)
            print(f"            written (backup: {backup.name})")

    mode = "APPLIED" if args.apply else "DRY-RUN (no files written)"
    print(f"\n{mode}: {n_migrate} config(s) to migrate, "
          f"{len(configs) - n_migrate} skipped.")


if __name__ == "__main__":
    main()
