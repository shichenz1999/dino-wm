"""Records the planner's inner search so it can be replayed/visualized later.

Captures, for every MPC round and every optimization step, the full candidate
population and its costs (CEM) or the action iterate and gradient (GD), plus the
chosen action of each round. Off by default (no tracer attached) so the planners
run byte-identically; when attached, the planner emits one cheap `add_*` call per
step. Dumped to `plan_trace.pkl` as a nested structure of fp16 arrays.

Schema (plan_trace.pkl):
    {
      'planner': 'cem' | 'gd',
      'config':  {num_samples, topk, opt_steps, horizon, action_dim,
                  max_iter, n_taken_actions, frameskip},
      'rounds': [                                  # one per MPC round
        {
          'mpc_iter': int,
          'opt_steps': [                           # variable length (early break)
            # CEM:
            {'population': (N,S,H,A) fp16, 'costs': (N,S) fp16,
             'topk_idx': (N,topk) int16, 'mu': (N,H,A) fp16, 'sigma': (N,H,A) fp16}
            # GD:
            {'actions': (N,H,A) fp16, 'loss': (N,) fp16, 'grad': (N,H,A) fp16}
          ],
          'chosen_mu':     (N,H,A) fp16,           # action sequence returned this round
          'taken_actions': (N,n_taken,A) fp16,     # the prefix actually executed
        }, ...
      ],
      'meta': {...}                                # filled by plan.py (states, seeds)
    }
"""
import pickle
import numpy as np


def _np(x, dtype):
    """Detach a torch tensor (or pass an array) to a cpu numpy array of dtype."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x).astype(dtype)


class PlanTracer:
    def __init__(self, planner="cem", dtype="float16"):
        self.planner = planner
        self.dtype = np.float16 if dtype == "float16" else np.float32
        self.config = {}
        self.rounds = []
        self.meta = {}
        self._cur = None                 # current round buffer
        self._pending = {}               # opt_step -> {traj -> record}  (CEM per-traj buffer)

    # --- round boundaries (driven by MPCPlanner) ---
    def start_round(self, mpc_iter):
        self._flush_pending()
        self._cur = {"mpc_iter": int(mpc_iter), "opt_steps": [],
                     "chosen_mu": None, "taken_actions": None}
        self.rounds.append(self._cur)
        self._pending = {}

    def _ensure_round(self):
        if self._cur is None:            # pure CEM/GD without an MPC wrapper
            self.start_round(0)

    # --- CEM: one call per (opt_step, traj) inside the per-eval loop ---
    def add_cem(self, opt_step, traj, population, costs, topk_idx, mu, sigma):
        self._ensure_round()
        rec = {
            "population": _np(population, self.dtype),     # (S, H, A)
            "costs": _np(costs, self.dtype),               # (S,)
            "topk_idx": _np(topk_idx, np.int16),           # (topk,)
            "mu": _np(mu, self.dtype),                     # (H, A)
            "sigma": _np(sigma, self.dtype),               # (H, A)
        }
        self._pending.setdefault(int(opt_step), {})[int(traj)] = rec

    # --- GD: one call per opt_step (already vectorized over evals) ---
    def add_gd(self, opt_step, actions, loss, grad):
        self._ensure_round()
        self._cur["opt_steps"].append({
            "actions": _np(actions, self.dtype),           # (N, H, A)
            "loss": _np(loss, self.dtype),                 # (N,)
            "grad": _np(grad, self.dtype),                 # (N, H, A)
        })

    def _flush_pending(self):
        """Stack the CEM per-traj records of each opt_step into (N, ...) arrays."""
        if not self._pending or self._cur is None:
            return
        for opt_step in sorted(self._pending):
            per_traj = self._pending[opt_step]
            trajs = sorted(per_traj)
            stack = lambda k: np.stack([per_traj[t][k] for t in trajs], axis=0)
            self._cur["opt_steps"].append({
                "population": stack("population"),         # (N, S, H, A)
                "costs": stack("costs"),                   # (N, S)
                "topk_idx": stack("topk_idx"),             # (N, topk)
                "mu": stack("mu"),                         # (N, H, A)
                "sigma": stack("sigma"),                   # (N, H, A)
            })
        self._pending = {}

    # --- chosen action of the round (driven by MPCPlanner) ---
    def record_chosen(self, chosen_mu, taken_actions):
        self._ensure_round()
        self._flush_pending()
        self._cur["chosen_mu"] = _np(chosen_mu, self.dtype)
        self._cur["taken_actions"] = _np(taken_actions, self.dtype)

    # --- finalize ---
    def dump(self, path, config=None, meta=None):
        self._flush_pending()
        if config:
            self.config.update(config)
        if meta:
            self.meta.update(meta)
        payload = {
            "planner": self.planner,
            "config": self.config,
            "rounds": self.rounds,
            "meta": self.meta,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f)
        return path
