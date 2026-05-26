# Plan: WM Transfer to Different Mazes

## 目标

测试在 U_MAZE 上训练的 WM 能否泛化到其他 maze 布局。**Planning 参数保持一致**，只换 env，观察 success rate 的变化。

## 可用的 maze（在 env/pointmaze/maze_model.py）

| Maze | Size | 备注 |
|---|---|---|
| U_MAZE | 5×5 | 训练用，baseline |
| U_MAZE_EVAL | 5×5 | 镜像翻转 |
| SMALL_MAZE | 6×6 | 正方形回字 |
| MEDIUM_MAZE | 8×8 | 完全不同布局 |
| MEDIUM_MAZE_EVAL | 8×8 | 不同布局 |
| LARGE_MAZE | 9×12 | 大迷宫 |
| LARGE_MAZE_EVAL | 9×12 | 大迷宫 |

## 实验设定

- WM：**用现成的 pretrained**（model_latest.pth），不重训
- Planning 参数：和 baseline 一致（`max_iter=10, horizon=5, num_samples=100, opt_steps=10`）
- 每个 maze 跑 `n_evals=50`，对比 success rate

## Phase 1：U_MAZE_EVAL（先做这个）

**为什么先做这个**：
- 改动最小（同样 5×5，只是镜像）
- 几何对称，理论上 WM 应该能处理
- 快速验证整套实验 pipeline 可行

### 代码改动清单

1. **添加 yaml 选项让 maze 可配置**
   - `conf/env/point_maze.yaml` 加 `maze_spec: U_MAZE`
   - 默认 U_MAZE，可命令行覆盖

2. **环境初始化时使用配置的 maze**
   - `MazeEnv.__init__` 已经接受 `maze_spec` 参数
   - 主要是把 config 传进去

3. **改 `sample_random_init_goal_states`**
   - 当前硬编码了 U_MAZE 的合法区域
   - 改成：从 `self.maze_arr` 里找 EMPTY 格子随机采样
   - 通用做法（对所有 maze 适用）：
     ```python
     empty_cells = np.argwhere(self.maze_arr == EMPTY)
     idx = rs.choice(len(empty_cells))
     cell_w, cell_h = empty_cells[idx]
     # cell to mujoco coord: (cell_w + 1 - 0.5, cell_h + 1 - 0.5) ± 偏移
     x = (cell_h + 1) - 0.5 + rs.uniform(-0.3, 0.3)
     y = (cell_w + 1) - 0.5 + rs.uniform(-0.3, 0.3)
     ```

4. **运行命令**
   ```bash
   python plan.py --config-name plan_point_maze.yaml \
     model_name=point_maze \
     env.kwargs.maze_spec=U_MAZE_EVAL \  # 或者类似的覆盖方式
     n_evals=50
   ```

### 预期与判断标准

| Success rate | 含义 |
|---|---|
| ≥ 0.9 | WM 完全泛化到镜像（DINOv2 几何鲁棒） |
| 0.6 - 0.9 | 部分泛化（有些朝向 WM 处理不好）|
| < 0.6 | WM 强依赖训练时的具体方向 |

记录到 `docs/results/01_umaze_eval.md`。

## Phase 2：SMALL_MAZE（看 Phase 1 结果再定）

如果 Phase 1 顺利，跑 SMALL_MAZE：
- 布局完全不同（正方形 + 中心障碍）
- 测试 WM 能否处理"未见过的布局"

## Phase 3：MEDIUM / LARGE_MAZE（最后）

更大迷宫，可能需要：
- 加大 `max_iter`（更多重规划机会）
- 加大 `horizon`（看得更远）

预期 success rate 会显著下降。

