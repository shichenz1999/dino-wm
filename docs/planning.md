# DINO-WM Planning 规划流程

> 以 PointMaze + 当前配置为例：`n_evals=2, horizon=5, n_taken_actions=5, num_samples=100, topk=10, opt_steps=10, max_iter=10, frameskip=5`

---

## 1. 整体结构

```
plan.py
  └── MPCPlanner (外层：滚动重规划)
        └── CEMPlanner (内层：每轮优化一段动作序列)
              └── World Model (评分：想象动作后果)
```

- **CEM**：在世界模型里"想象"很多动作序列，挑出最优的一组
- **MPC**：拿 CEM 的结果在真实环境执行几步，再用新状态作为起点重新规划

### MPC 和 CEM 的嵌套关系（一次 plan.py 跑下来的最坏情况）

```
plan.py 跑 1 次
│
├─ MPC 第 1 轮:
│   ├─ CEM 想动作:
│   │   ├─ opt_step 1:  采 100 个候选 × 5 步 → 选 top 10 → 更新分布
│   │   ├─ opt_step 2:  采 100 个 × 5 步 → 选 top 10 → 更新分布
│   │   ├─ opt_step 3:  ...
│   │   ├─ ...
│   │   └─ opt_step 10: 输出最优 5 步动作
│   │   (内部完成: 10 轮 × 100 候选 = 1000 次"想象 5 步")
│   └─ 真实环境执行 5 个高层动作 (= 25 env 帧)
│   ↓
│   是否所有 eval 都成功?
│     是 → 结束
│     否 → 进入下一轮
│
├─ MPC 第 2 轮: (还有 eval 没成功)
│   ├─ CEM 重新想 → 又是 10 轮 × 100 候选
│   └─ 再执行 5 个高层动作
│
├─ ...
│
└─ MPC 第 10 轮: (max_iter 上限)
    └─ 还没成功 → 这次 eval 失败 (action_len = ∞)
```

### 类比

- **CEM = 头脑风暴**：在 WM 里想象 100 种走法、挑最好的 10 个、来回精炼 10 次
- **MPC = 走一步看一步**：拿 CEM 的最优 5 步去真实环境执行，到了再决定下一段怎么走

### 步数对照

| 层级 | 数量 | 配置参数 |
|---|---|---|
| MPC 最大轮数 | 10 | `max_iter` |
| MPC 每轮执行步数 | 5（高层动作）| `n_taken_actions` |
| 每个高层动作 = env 帧 | 5 | `frameskip` |
| MPC 每轮 env 帧数 | 25 = 5 × 5 | - |
| MPC 10 轮上限 env 帧 | 250 | - |
| CEM 每次调用的优化步数 | 10 | `opt_steps` |
| CEM 每步采样数 | 100 | `num_samples` |
| CEM 每步保留 top | 10 | `topk` |
| CEM 每个候选长度 | 5 步 | `horizon` |
| CEM 单次调用"想象"次数 | 1000 = 10 × 100 | - |

跑满 50 evals 最坏情况：**50 × 10 × 1000 = 500,000 次世界模型想象**（但每次很便宜，在 latent 空间）。

---

## 2. 动作的两种表示

PointMaze 原始动作维度 = 2（fx, fy），`frameskip=5`。

| | planner / 世界模型 | 真实环境（MuJoCo） |
|---|---|---|
| 1 个动作 | 10 维向量 | 5 个连续的 [fx, fy] |
| `horizon=5` 个动作 | 5 × 10 维 | 25 帧物理步 |

```
高层动作 a_t (10维) = [fx₁ fy₁ | fx₂ fy₂ | fx₃ fy₃ | fx₄ fy₄ | fx₅ fy₅]
                       帧1     帧2     帧3     帧4     帧5
```

世界模型直接预测 5 帧后的状态（跨越式），真实环境逐帧执行。

---

## 3. 算法细节

### CEM 单次 opt_step 的数学流程

```
输入：mu (5, 10), sigma (5, 10)    ← 当前动作分布的均值和标准差

1. 采样:      action = randn(100, 5, 10) * sigma + mu       # 100 个候选 × 5 步动作
2. WM Rollout: z_traj = wm.rollout(obs_0, action)            # 100 条想象轨迹
3. 评分:      loss = ||z_traj[-1] - z_goal||²                # 最终 latent 与 goal 距离
4. 选 top-K:  topk_action = action[argsort(loss)[:10]]       # 保留 10 个最好的
5. 更新分布:  mu    = topk_action.mean(0)
              sigma = topk_action.std(0)

输出：更优的 mu, sigma（采样分布向更好的动作收紧）
```

每一轮的 mu/sigma 都比上一轮"窄"，最多 `opt_steps=10` 轮，全 eval 提前 success 就 break。

### MPC 外层循环代码

```python
init_obs  ← 环境初始观察
while not all_success and iter < max_iter:           # max_iter=10
    actions = CEM.plan(cur_obs, goal)                # 优化出 5 步动作
    taken   = actions[:n_taken_actions]              # 取前 5 步
    env.execute(taken)                                # 真实环境执行
    cur_obs ← 环境新状态
    iter += 1
```

`n_taken_actions = horizon = 5`，所以一轮 MPC = 一段完整的 CEM 规划。
`max_iter=10` 表示最多重规划 10 次后强制结束（之前是 `null` 表示无限）。

---

## 5. 输出文件全景

工作流分两步：先跑 `plan.py` 得到数据，再跑 `scripts/visualize_plan.py` 得到可视化。

### Step 1: `plan.py` 跑完后的文件

```
plan_outputs/{时间戳}_{model}_gH{n}/
├── .hydra/                       完整 hydra 配置（参考用）
├── plan.log                      文本日志
├── logs.json                     每轮的 success_rate、距离指标
├── planning_settings.json        ← 算法超参数（n_evals, MPC, CEM 参数）
├── plan_targets.pkl              输入：起点和目标 state/obs
├── plan_results.pkl              输出：actions, e_states, successes, action_len
└── plan_visuals.pkl              像素数据：env_frames, wm_imagined_frames, goal_wm_reconstructed
```

| 文件 | 含义 |
|---|---|
| `plan_targets.pkl` | 起始/目标状态（输入条件）|
| `plan_results.pkl` | 规划核心结果（动作 + 轨迹 + 成功标记，几 KB）|
| `plan_visuals.pkl` | 可视化用的图像数据（~10 MB / n_evals=2）|
| `planning_settings.json` | 干净的算法超参数 JSON（方便对比/查询）|
| `logs.json` | metrics（每帧 success_rate、距离指标）|

**注意**：原代码 (`_plot_rollout_compare` in evaluator.py) 会生成 `plan_0_output_*.png`、`plan{iter}.png`、`output_final*.mp4` 等 legacy 可视化文件，目前已**注释禁用**，统一改用下面的 visualize_plan.py。

### Step 2: `scripts/visualize_plan.py` 跑完后追加的文件

```
plan_outputs/{时间戳}_{model}_gH{n}/
├── summary.png                  ← 全局概览（所有 eval 的迷宫缩略图 + 状态）
└── evals/
    └── eval_{N}/
        ├── trajectory.png       ← 单 eval 的轨迹图（在 MuJoCo 渲染上画线）
        ├── frames.png           ← 7 列时间快照（env / imagined 各一行 + goal）
        └── video.mp4            ← 标注视频（带 goal overlay）
```

| 文件 | 含义 |
|---|---|
| `summary.png` | 1 张图看完所有 eval（标题含算法配置 + Start/Final/Goal legend）|
| `eval_N/trajectory.png` | 在干净 MuJoCo 背景上画的轨迹（Start 绿/Final 蓝/Goal 红） |
| `eval_N/frames.png` | 上行 env 真实 / 下行 WM 想象，对比 6 个时间点 + goal 列 |
| `eval_N/video.mp4` | 动态视频，env 和 imagined 双 panel，叠加红色 goal 标记 |

完整命令：

```bash
# 1. 跑 plan.py
python plan.py --config-name plan_point_maze.yaml \
  model_name=point_maze \
  ckpt_base_path=... \
  n_evals=50

# 2. 跑可视化
python scripts/visualize_plan.py plan_outputs/<时间戳>/
```

---

## 6. 关键参数一览

| 参数 | 当前值 | 含义 |
|---|---|---|
| `n_evals` | 2 (测试时) | 同时规划的独立任务数（不同起点+目标） |
| `goal_H` | 5 | 目标距离起点几步可达（在 random_state 模式下也用作 horizon） |
| `frameskip` | 5 | 一个高层动作 = 几帧 env 物理步 |
| `horizon` | 5 | CEM 每次优化几步高层动作（= goal_H） |
| `num_samples` | 100 | CEM 每个 iteration 采样几组动作 |
| `topk` | 10 | 选 loss 最小的多少组来更新分布 |
| `opt_steps` | 10 | CEM 最大 iteration 数 |
| `var_scale` | 1 | 初始 sigma |
| `n_taken_actions` | 5 | MPC 每轮在 CEM 5 步里实际执行几步 |
| `max_iter` | 10 | MPC 最大轮数（之前是 null = 无限）|
