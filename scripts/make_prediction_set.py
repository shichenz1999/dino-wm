"""Build a fixed prediction-eval set (held-out windows) for k-step latent rollout MSE.

Each window = (context frames, action sequence, future target frames). The same set
is reused across all WMs so the only variable is the predictor.

IMPORTANT — held-out correctness:
  dino-wm's TrajSlicerDataset loads trajectories by *local* index and ignores the
  random-split mapping (see memory: project-dinowm-split-leak). So a model trained
  with split_ratio=r actually SAW episodes 0..int(r*N)-1 and NEVER saw the LAST
  N-int(r*N) episodes. We therefore build windows ONLY from those last (unseen)
  trajectories, not from the buggy `val_slices`.

Window geometry (model-steps at frameskip):
  num_frames = num_hist + horizon           frames per window
  context  = first num_hist frames          (fed to wm.rollout as obs_0)
  target h = frame at step (num_hist-1+h)    for h = 1..horizon
  raw frame index of step k = start + frameskip*k

Output: a pickle {meta, windows}; windows store dataset indices only (frames/actions
are pulled from the frozen dataset at eval time, kept encoder-agnostic).

Run (from the dino-wm working copy root):
  DATASET_DIR=/local_data/sz4968/world-model/experiments/dino-wm/data \
  python scripts/make_prediction_set.py
"""
import os
import sys
import argparse
import pickle
from pathlib import Path

import numpy as np

# allow running as `python scripts/make_prediction_set.py` from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets.point_maze_dset import PointMazeDataset


def pick_starts(rng, max_start, k):
    """Exactly k DISTINCT, evenly-spread start indices in [0, max_start] (one per bin).

    Guarantees k distinct values whenever max_start+1 >= k (otherwise returns all
    available). Bins give spread; a used-set + fallback pool guarantees no dedup loss.
    """
    if max_start <= 0:
        return [0]
    k = min(k, max_start + 1)
    edges = np.linspace(0, max_start, k + 1)
    used = set()
    for j in range(k):
        lo, hi = int(np.ceil(edges[j])), int(np.floor(edges[j + 1]))
        pool = [c for c in range(lo, hi + 1) if c not in used]
        if not pool:  # bin exhausted -> draw any remaining distinct index
            pool = [c for c in range(0, max_start + 1) if c not in used]
        used.add(int(rng.choice(pool)))
    return sorted(used)


def main():
    ap = argparse.ArgumentParser()
    # input: dinowm official dataset (via DATASET_DIR); output: my own eval data,
    # kept separate under world-model/data/ (NOT under the dinowm dataset dir).
    default_data = os.path.join(os.environ.get("DATASET_DIR", "data"), "point_maze")
    default_out = "/local_data/sz4968/world-model/data/eval_sets/point_maze/prediction_set.pkl"
    ap.add_argument("--data_path", default=default_data)
    ap.add_argument("--out", default=default_out)
    ap.add_argument("--split_ratio", type=float, default=0.9,
                    help="must match the trained WM's split_ratio (point_maze=0.9)")
    ap.add_argument("--num_hist", type=int, default=3)
    ap.add_argument("--frameskip", type=int, default=5)
    ap.add_argument("--horizon", type=int, default=5)
    ap.add_argument("--per_traj", type=int, default=5)
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    dset = PointMazeDataset(data_path=args.data_path, transform=None, normalize_action=False)
    n_total = len(dset)
    seq_lens = [int(x) for x in dset.seq_lengths.tolist()]

    # Episodes 0..train_len-1 were SEEN by training (slicer local-index quirk);
    # the LAST block is genuinely unseen.
    train_len = int(args.split_ratio * n_total)
    heldout_ids = list(range(train_len, n_total))

    num_frames = args.num_hist + args.horizon
    span = num_frames * args.frameskip  # raw frames spanned by a window [start, start+span)

    rng = np.random.default_rng(args.seed)
    windows = []
    for tid in heldout_ids:
        T = seq_lens[tid]
        max_start = T - span  # last frame used = start + frameskip*(num_frames-1) = start+span-frameskip
        if max_start < 0:
            continue
        for s in pick_starts(rng, max_start, args.per_traj):
            ctx_idx = [s + args.frameskip * k for k in range(args.num_hist)]
            tgt_idx = [s + args.frameskip * (args.num_hist - 1 + h)
                       for h in range(1, args.horizon + 1)]
            # actions driving the window: raw rows [s, s+span) (folded to (num_frames, f*adim) at eval)
            act_slice = [s, s + span]
            windows.append(dict(
                traj_id=tid, start=int(s),
                ctx_idx=ctx_idx, tgt_idx=tgt_idx, act_slice=act_slice,
            ))

    meta = dict(
        env="point_maze",
        dataset="point_maze",
        split_ratio=args.split_ratio,
        train_len=train_len,
        heldout_id_range=[heldout_ids[0], heldout_ids[-1]],
        n_heldout_traj=len(heldout_ids),
        num_hist=args.num_hist,
        frameskip=args.frameskip,
        horizon=args.horizon,
        num_frames=num_frames,
        span=span,
        per_traj=args.per_traj,
        seed=args.seed,
        n_windows=len(windows),
        target_steps={f"h={h}": args.frameskip * (args.num_hist - 1 + h)
                      for h in range(1, args.horizon + 1)},
        note=("Windows built ONLY from trajectories the WM never trained on "
              "(episodes >= train_len), because TrajSlicerDataset loads by local "
              "index and ignores the random split. Indices reference the frozen "
              "point_maze dataset; frames/actions loaded at eval."),
    )

    out = dict(meta=meta, windows=windows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print(f"[make_prediction_set] wrote {len(windows)} windows -> {args.out}")
    print(f"  held-out trajectories: {heldout_ids[0]}..{heldout_ids[-1]} "
          f"({len(heldout_ids)} traj), per_traj={args.per_traj}")
    print(f"  num_hist={args.num_hist} frameskip={args.frameskip} horizon={args.horizon} "
          f"num_frames={num_frames} span={span} (valid start range [0,{seq_lens[heldout_ids[0]]-span}])")
    print(f"  target raw-step offsets per h: {meta['target_steps']}")


if __name__ == "__main__":
    main()
