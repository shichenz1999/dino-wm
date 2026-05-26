import os
import time
import gym
import json
from datetime import datetime
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed, move_to_device

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
]

def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device
        # Wall-clock timing of the full planning run
        self._t0 = time.time()
        self._start_iso = datetime.now().isoformat(timespec="seconds")

        # have different seeds for each planning instances
        self.eval_seed = [cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])]
        print("eval_seed: ", self.eval_seed)
        self.n_evals = cfg_dict["n_evals"]
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=self.frameskip * self.goal_H + 1)
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def sample_traj_segment_from_dset(self, traj_len):
        states = []
        actions = []
        observations = []
        env_info = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        # sample init_states from dset
        for i in range(self.n_evals):
            max_offset = -1
            while max_offset < 0:  # filter out traj that are not long enough
                traj_id = random.randint(0, len(self.dset) - 1)
                obs, act, state, e_info = self.dset[traj_id]
                max_offset = obs["visual"].shape[0] - traj_len
            state = state.numpy()
            offset = random.randint(0, max_offset)
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def _wm_decode_uint8(self, z):
        """WM decode → uint8 frames. Decoder outputs in [-1, 1], we map to [0, 255]."""
        visuals = self.wm.decode_obs(z)[0]["visual"]
        return (((visuals.clamp(-1, 1) + 1) / 2) * 255).cpu().numpy().astype(np.uint8)

    def _generate_wm_visuals(self, actions, e_obses):
        """
        Generate WM visualization frames for plan_visuals.pkl.

        For MPC: rollout WM separately per iter, using the real env obs at iter
        start (extracted from e_obses). Mirrors MPC's true predictions — iter k
        starts from the env state that resulted from executing actions through
        iter k-1.
        For non-MPC (open-loop CEM/GD): single rollout from initial state.

        Returns dict with keys: wm_obs_0_recon, wm_obs_g_recon, wm_imagined.
        """
        n_taken = getattr(self.planner, "n_taken_actions", None)
        # Side effect: save iter-end latents (last frame of each iter's rollout)
        # so the evals.json metric loop can compute env-vs-imagined divergence
        # without re-running WM.
        self._iter_end_latents = []

        with torch.no_grad():
            if n_taken:
                # MPC path: per-iter rollout. Each iter starts from the real env
                # state at its starting frame (index k * n_taken * frameskip in e_obses).
                n_iters = actions.shape[1] // n_taken
                per_iter_uint8 = []
                for k in range(n_iters):
                    start_idx = k * n_taken * self.frameskip
                    iter_obs = {
                        key: arr[:, start_idx : start_idx + 1]
                        for key, arr in e_obses.items()
                    }
                    iter_acts = actions[:, k * n_taken : (k + 1) * n_taken].detach()
                    trans_iter_obs = move_to_device(
                        self.data_preprocessor.transform_obs(iter_obs),
                        self.evaluator.device,
                    )
                    iter_z, _ = self.wm.rollout(obs_0=trans_iter_obs, act=iter_acts)
                    per_iter_uint8.append(self._wm_decode_uint8(iter_z))
                    # Save the iter-end imagined latent (last timestep), keeping
                    # the dict-of-tensors structure that wm.encode_obs also returns.
                    self._iter_end_latents.append({
                        key: arr[:, -1:].detach() for key, arr in iter_z.items()
                    })
                # iter 0 frame 0 = obs_0 recon; concat all iters' frames 1+ as imaginations
                wm_obs_0_recon = per_iter_uint8[0][:, 0:1]
                wm_imagined = np.concatenate(
                    [v[:, 1:] for v in per_iter_uint8], axis=1
                )
            else:
                # Open-loop fallback: single rollout from initial state
                trans_obs_0 = move_to_device(
                    self.data_preprocessor.transform_obs(self.obs_0),
                    self.evaluator.device,
                )
                full_z, _ = self.wm.rollout(obs_0=trans_obs_0, act=actions.detach())
                full_uint8 = self._wm_decode_uint8(full_z)
                wm_obs_0_recon = full_uint8[:, 0:1]
                wm_imagined = full_uint8[:, 1:]
                # Single "iter" — save the final latent so the metric loop works
                self._iter_end_latents.append({
                    key: arr[:, -1:].detach() for key, arr in full_z.items()
                })

            # Goal reconstruction (same for both paths)
            trans_obs_g = move_to_device(
                self.data_preprocessor.transform_obs(self.obs_g),
                self.evaluator.device,
            )
            z_obs_g = self.wm.encode_obs(trans_obs_g)
            wm_obs_g_recon = self._wm_decode_uint8(z_obs_g)

        return {
            "wm_obs_0_recon": wm_obs_0_recon,
            "wm_obs_g_recon": wm_obs_g_recon,
            "wm_imagined":    wm_imagined,
        }

    def _per_iter_metrics(self, k, n_taken, action_len_int, e_obses, e_states):
        """Compute the 8 per-eval metrics at iter k (1-indexed at k+1).

        Per-eval cutoff = min(action_len[i], (k+1)*n_taken) — already-succeeded
        trajs are evaluated at their success moment, matching logs.json semantics.
        """
        n = self.n_evals
        cutoff_step = np.minimum(action_len_int, (k + 1) * n_taken)
        end_step = np.minimum(cutoff_step * self.frameskip, e_states.shape[1] - 1)
        iter_idx = np.minimum(
            np.maximum(np.ceil(cutoff_step / n_taken).astype(int) - 1, 0), k
        )
        ii = np.arange(n)

        metrics = dict(self.env.eval_state(self.state_g, e_states[ii, end_step]))

        # Observation-level (int32 cast avoids uint8 wrap-around).
        v_now = e_obses["visual"][ii, end_step].reshape(n, -1).astype(np.int32)
        v_goal = self.obs_g["visual"][:, 0].reshape(n, -1).astype(np.int32)
        metrics["visual_dist"] = np.linalg.norm(v_now - v_goal, axis=-1)
        metrics["proprio_dist"] = np.linalg.norm(
            e_obses["proprio"][ii, end_step] - self.obs_g["proprio"][:, 0], axis=-1
        )

        # WM latent divergence per-eval.
        cur_obs = {key: arr[ii, end_step][:, None] for key, arr in e_obses.items()}
        trans_cur = move_to_device(
            self.data_preprocessor.transform_obs(cur_obs), self.evaluator.device
        )
        with torch.no_grad():
            env_z = self.wm.encode_obs(trans_cur)
        imagined_z = {
            key: torch.stack(
                [self._iter_end_latents[int(iter_idx[i])][key][i, 0] for i in range(n)]
            ).unsqueeze(1)
            for key in self._iter_end_latents[0].keys()
        }
        for key in ("visual", "proprio"):
            diff = env_z[key] - imagined_z[key]
            metrics[f"div_{key}_emb"] = diff.flatten(1).norm(dim=1).cpu().numpy()
        return metrics

    def _compute_evals_json(self, actions, e_obses, e_states, action_len, successes):
        """Build the {evals, per_iter, final} payload for evals.json."""
        n_taken = getattr(self.planner, "n_taken_actions", None) or int(actions.shape[1])
        n_taken = int(n_taken)
        action_len_int = action_len.astype(int)
        max_iters = max(1, int(np.ceil(action_len.max() / n_taken)))

        per_iter_raw = [
            self._per_iter_metrics(k, n_taken, action_len_int, e_obses, e_states)
            for k in range(max_iters)
        ]

        def _format_val(key, v):
            return bool(v) if key == "success" else round(float(v), 3)

        evals = {}
        for i in range(self.n_evals):
            n_iters_i = max(1, int(np.ceil(action_len_int[i] / n_taken)))
            iters = [
                {"iter": k + 1, **{key: _format_val(key, m[key][i]) for key in m}}
                for k, m in enumerate(per_iter_raw[:n_iters_i])
            ]
            evals[str(i)] = {
                "success":    bool(successes[i]),
                "action_len": int(action_len[i]),
                "iters":      iters,
            }

        def _mean_key(key):
            return "success_rate" if key == "success" else f"mean_{key}"

        per_iter = [
            {
                "iter": k + 1,
                **{_mean_key(key): round(float(np.asarray(m[key]).astype(float).mean()), 3)
                   for key in m},
            }
            for k, m in enumerate(per_iter_raw)
        ]
        final = {k: v for k, v in per_iter[-1].items() if k != "iter"}
        return {"evals": evals, "per_iter": per_iter, "final": final}

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        action_len = np.where(np.isfinite(action_len), action_len, actions.shape[1])
        logs, successes, e_obses, e_states = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )

        # === plan_meta.pkl: per-trajectory data + algorithm results (~KB) ===
        with open("plan_meta.pkl", "wb") as f:
            pickle.dump(
                {
                    # task (per-traj, state-level)
                    "state_0":    self.state_0,                               # (n, state_dim)
                    "state_g":    self.state_g,                               # (n, state_dim)
                    "gt_actions": self.gt_actions,                            # (n, T, ...) | None
                    # algorithm output (per-traj)
                    "actions":    actions.detach().cpu().numpy(),             # (n, T, action_dim)
                    "action_len": action_len,                                 # (n,) int, steps executed
                    "successes":  np.asarray(successes),                      # (n,) bool, reached goal
                    "e_states":   e_states,                                   # (n, T*frameskip+1, state_dim)
                },
                f,
            )
        print(f"Dumped plan meta to {os.path.abspath('plan_meta.pkl')}")

        # === plan_visuals.pkl: per-trajectory pixel data (~MB) ===
        viz = self._generate_wm_visuals(actions, e_obses)
        with open("plan_visuals.pkl", "wb") as f:
            pickle.dump(
                {
                    # real env (MuJoCo)
                    "obs_g":      self.obs_g["visual"],                       # (n, 1, H, W, 3) uint8
                    "env_frames": e_obses["visual"],                          # (n, T*frameskip+1, H, W, 3) uint8
                    # WM reconstructions + imaginations (from _generate_wm_visuals)
                    **viz,
                },
                f,
            )
        print(f"Dumped plan visuals to {os.path.abspath('plan_visuals.pkl')}")

        # === evals.json: per-eval per-iter metrics ===
        eval_data = self._compute_evals_json(actions, e_obses, e_states, action_len, successes)
        result = {
            "summary": {
                "n_evals":      self.n_evals,
                "start_time":   self._start_iso,
                "duration_sec": round(time.time() - self._t0, 1),
                "final":        eval_data["final"],
                "per_iter":     eval_data["per_iter"],
            },
            "evals": eval_data["evals"],
        }
        with open("evals.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Dumped evals to {os.path.abspath('evals.json')}")

        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = {
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        }
        with open(self.log_filename, "a") as file:
            file.write(json.dumps(logs_entry) + "\n")
        return logs


def load_ckpt(snapshot_path, device):
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=device)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device):
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not train_cfg.has_decoder:
        result["decoder"] = None

    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
    )
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model = load_model(model_ckpt, model_cfg, num_action_repeat, device=device)

    # Resolve env id: cfg.setting → cfg.env_id_map[setting] overrides model's env.
    # Lets one model checkpoint be evaluated on multiple maze variants.
    setting = cfg_dict.get("setting")
    env_id_map = cfg_dict.get("env_id_map") or {}
    env_id = env_id_map.get(setting, model_cfg.env.name) if setting else model_cfg.env.name

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(env_id, *model_cfg.env.args, **model_cfg.env.kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(env_id, *model_cfg.env.args, **model_cfg.env.kwargs)
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict["wandb_logging"] = True
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
