# Phase 1 TODO: 测试 U_MAZE_EVAL

每一步都列出**具体改哪个文件、哪一行、改什么**。

---

## TODO 1：让 `point_maze` 这个 env 接受可变 `maze_spec`

### 1.1 改 `env/__init__.py`（第 9-21 行）

**当前**（写死 U_MAZE）：
```python
register(
    id='point_maze',
    entry_point='env.pointmaze:PointMazeWrapper',
    max_episode_steps=300,
    kwargs={
        'maze_spec':U_MAZE,           # ← 写死
        ...
    }
)
```

**改成**：注册多个版本，每个对应一个 maze：

```python
from .pointmaze import (U_MAZE, U_MAZE_EVAL, SMALL_MAZE,
                        MEDIUM_MAZE, MEDIUM_MAZE_EVAL,
                        LARGE_MAZE, LARGE_MAZE_EVAL)

POINT_MAZE_VARIANTS = {
    "point_maze":              U_MAZE,        # 兼容旧名（默认）
    "point_maze_umaze":        U_MAZE,
    "point_maze_umaze_eval":   U_MAZE_EVAL,
    "point_maze_small":        SMALL_MAZE,
    "point_maze_medium":       MEDIUM_MAZE,
    "point_maze_medium_eval":  MEDIUM_MAZE_EVAL,
    "point_maze_large":        LARGE_MAZE,
    "point_maze_large_eval":   LARGE_MAZE_EVAL,
}

for env_id, maze_spec in POINT_MAZE_VARIANTS.items():
    register(
        id=env_id,
        entry_point='env.pointmaze:PointMazeWrapper',
        max_episode_steps=300,
        kwargs={
            'maze_spec': maze_spec,
            'reward_type':'sparse',
            'reset_target': False,
            'ref_min_score': 23.85,
            'ref_max_score': 161.86,
            'dataset_url':'',
        }
    )
```

**好处**：之前的 `gym.make("point_maze")` 不变（保持兼容）。新 maze 用 `gym.make("point_maze_umaze_eval")` 等。

---

## TODO 2：让 plan.py 能选择 maze

### 2.1 改 `plan.py` 第 548 和 557 行

**当前**：env 名字写在训练时的 `model_cfg.env.name`，是固定的 `point_maze`。

**改成**：允许命令行覆盖。修改 `planning_main` 里的 env 实例化逻辑：

```python
# 大约第 540 行附近
# Allow overriding the env name from the planning config
env_name_override = cfg_dict.get("env_name", None)
env_name = env_name_override if env_name_override else model_cfg.env.name

# 然后下面的 gym.make 用 env_name 而不是 model_cfg.env.name
if env_name == "wall" or env_name == "deformable_env":
    ...
else:
    env = SubprocVectorEnv(
        [lambda: gym.make(env_name, *model_cfg.env.args, **model_cfg.env.kwargs)
         for _ in range(cfg_dict["n_evals"])]
    )
```

### 2.2 改 `conf/plan_point_maze.yaml`

加一行配置（顶部加）：

```yaml
# 在最上面 (defaults: 之后) 加：
env_name: null   # null = use the same env as training (point_maze = U_MAZE)
```

这样用命令行覆盖：
```bash
python plan.py --config-name plan_point_maze.yaml \
  model_name=point_maze \
  ckpt_base_path=... \
  n_evals=10 \
  env_name=point_maze_umaze_eval   # ← 这里指定
```

---

## TODO 3：让 `sample_random_init_goal_states` 通用化

### 3.1 改 `env/pointmaze/point_maze_wrapper.py` 第 21-44 行

**当前**：硬编码了 U_MAZE 的合法区域：
```python
valid = ((0.5 <= x <= 1.1 or 2.5 <= x <= 3.1) and (0.5 <= y <= 3.1))\
        or ((1.1 < x < 2.5) and (2.5 <= y <= 3.1))
```

**改成**：基于 `self.maze_arr` 找 EMPTY/GOAL 格子，对所有 maze 通用：

```python
from env.pointmaze.maze_model import EMPTY, GOAL

def sample_random_init_goal_states(self, seed):
    rs = np.random.RandomState(seed)
    
    # 找所有"可走"的格子（EMPTY 或 GOAL 都算可走）
    walkable = np.argwhere(
        (self.maze_arr == EMPTY) | (self.maze_arr == GOAL)
    )  # shape (N, 2), 每行是 (w, h) maze_arr 索引

    def generate_state():
        # 1. 随机选一个 cell
        idx = rs.choice(len(walkable))
        cell_w, cell_h = walkable[idx]
        # 2. mujoco coord: maze_arr (w, h) → pos (w+1, h+1) 但 ball pos 是 (x=col?, y=row?)
        #    根据 maze_model.py 的 wall placement: pos=[w+1, h+1] 但 ball 也是 (x, y)
        #    经验：state[0] 是 x，对应 maze_arr 的列；state[1] 是 y，对应 maze_arr 的行
        #    为安全起见，先用 cell 中心 + 小偏移
        x = (cell_h + 1) + rs.uniform(-0.3, 0.3)   # 沿 cell 内随机
        y = (cell_w + 1) + rs.uniform(-0.3, 0.3)
        state = np.array([
            x, y,
            rs.uniform(low=STATE_RANGES[2][0], high=STATE_RANGES[2][1]),
            rs.uniform(low=STATE_RANGES[3][0], high=STATE_RANGES[3][1]),
        ])
        return state

    init_state = generate_state()
    goal_state = generate_state()
    return init_state, goal_state
```

**注意 x/y 对应 maze_arr 哪个轴需要测试一下**：写完先 print 几个采样位置，然后 render 一下看球是否落在通道里（不在墙里）。

---

## TODO 4：单元测试 sampler

### 4.1 新建 `scripts/test_maze_sampler.py`（小脚本）

```python
"""验证 sampler 在不同 maze 上都能采到合法点（不在墙里）。"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ['MUJOCO_PY_MUJOCO_PATH'] = '/local_data/sz4968/.mujoco/mujoco210'

import gym
import env  # registers
from env.pointmaze.maze_model import WALL

for env_id in ["point_maze_umaze", "point_maze_umaze_eval", "point_maze_small",
               "point_maze_medium", "point_maze_large"]:
    env_obj = gym.make(env_id)
    for seed in range(10):
        init_state, goal_state = env_obj.sample_random_init_goal_states(seed)
        # 验证不在墙里
        for name, s in [("init", init_state), ("goal", goal_state)]:
            x, y = s[0], s[1]
            # 反推 maze_arr 索引
            cell_h = int(x - 1)
            cell_w = int(y - 1)
            in_bounds = 0 <= cell_w < env_obj.maze_arr.shape[0] and 0 <= cell_h < env_obj.maze_arr.shape[1]
            if in_bounds:
                in_wall = env_obj.maze_arr[cell_w, cell_h] == WALL
                status = "WALL!" if in_wall else "ok"
            else:
                status = "OUT OF BOUNDS"
            print(f"{env_id:30} seed={seed} {name}=({x:.2f},{y:.2f}) → {status}")
    print()
```

跑：
```bash
python scripts/test_maze_sampler.py
```

如果都是 `ok` 就放心，有 `WALL!` 或 `OUT OF BOUNDS` 就要修 sampler 逻辑。

---

## TODO 5：快速跑 n_evals=2 验证

```bash
WANDB_MODE=disabled python plan.py --config-name plan_point_maze.yaml \
  model_name=point_maze \
  ckpt_base_path=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints/pretrained \
  n_evals=2 \
  env_name=point_maze_umaze_eval
```

看：
1. ✅ 没有 MuJoCo 报错（球没生成在墙里）
2. ✅ `plan_results.pkl`、`plan_visuals.pkl` 正常生成
3. ✅ 跑可视化 `python scripts/visualize_plan.py plan_outputs/<run>/`
4. ✅ trajectory.png 里看到球在 U_MAZE_EVAL 的镜像走廊里走

---

## TODO 6：正式跑 n_evals=50

```bash
tmux new-session -d -s plan_umaze_eval \
  'WANDB_MODE=disabled python plan.py --config-name plan_point_maze.yaml \
   model_name=point_maze \
   ckpt_base_path=/local_data/sz4968/world-model/experiments/dino-wm/checkpoints/pretrained \
   n_evals=50 \
   env_name=point_maze_umaze_eval ; touch /tmp/plan_done'
```

跑完后：
```bash
python scripts/visualize_plan.py plan_outputs/<run>/
```

记录结果到 `docs/results/01_umaze_eval.md`（success rate, 失败 case 的特征等）。

---

## 改动清单总览

| # | 文件 | 改动 |
|---|---|---|
| 1 | `env/__init__.py` | 新增 7 个 maze 变体的 gym register |
| 2 | `plan.py` 第 540-560 行 | 加 `env_name` 覆盖逻辑 |
| 3 | `conf/plan_point_maze.yaml` | 加 `env_name: null` 配置项 |
| 4 | `env/pointmaze/point_maze_wrapper.py` 第 21-44 行 | sampler 通用化 |
| 5 | `scripts/test_maze_sampler.py` | **新文件**，验证 sampler |
| 6 | `docs/results/01_umaze_eval.md` | **新文件**，记录结果 |

**总代码改动**：约 50 行（大部分是 env/__init__ 的注册扩展）+ 2 个新文件。
