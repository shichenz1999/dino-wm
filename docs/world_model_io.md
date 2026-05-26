# 世界模型 I/O（以 PointMaze 为例）

## 一句话总结

> **输入 3 帧历史 latent + 3 个动作（每个动作 = 5 帧力的 concat），输出 1 个未来 latent（env 5 帧后）。要预测更远就滑窗 + 再 predict。**

---

## 最清晰版本

假设当前 env 帧是 0，要预测 env 5 帧后：

```
输入：
  ▸ 3 帧 latent：z(env=-10),  z(env=-5),  z(env=0)
  ▸ 3 个动作：a₋₂, a₋₁, a₀
    每个动作 = 5 帧 [fx, fy] 力的 concat（10 维）:
      a₋₂ = [f₋₁₀, f₋₉, f₋₈, f₋₇, f₋₆]   ← env -10 到 -5 期间的 5 个力
      a₋₁ = [f₋₅,  f₋₄, f₋₃, f₋₂, f₋₁]   ← env -5 到 0 期间
      a₀  = [f₀,   f₁,  f₂,  f₃,  f₄]    ← env 0 到 5 期间（驱动预测）

输出：
  z(env=+5)
```

预测更远（如 env +25 帧）= 自回归 5 次：
- 把 z(+5) 加到 history（带上对应动作 a₁）
- 滑窗：[z(-5), z(0), z(+5)] + [a₋₁, a₀, a₁]
- 再 predict → z(+10)
- 重复 5 次得到 z(+5), z(+10), z(+15), z(+20), z(+25)

---

## 关键数字对照

| | Planner / WM 视角 | 真实环境视角 |
|---|---|---|
| 1 action | 10 维 | 5 个 [fx, fy] 力 |
| 1 forward | 1 次 predict | 跨 5 env 帧 |
| 5 步 rollout | 5 次串行 forward | 25 env 帧 |
| history 长度 | 3 latent | 跨 15 env 帧 |

---

## 张量形状

| 项 | 形状 | 含义 |
|---|---|---|
| **history latent** | `(B, 3, 196, 384)` | 3 帧 × 14×14=196 个 patch × DINOv2 384 维 |
| **action** | `(B, 3, 10)` | 3 个动作，每个 10 维（= 5 个 env 力打包） |
| **predicted latent** | `(B, 1, 196, 384)` | 下一帧 latent |

---

## 训练

- **数据**：把轨迹切成长度 `H+1=4` 的片段
- **目标**：给前 3 帧 + 它们的动作，预测第 4 帧的 DINOv2 latent
- **损失**：`||predict(z₀, z₁, z₂, a₀, a₁, a₂) - z₃||²`（latent 空间 MSE，不重建像素）
- **encoder**：DINOv2 ViT-S/14（19M 参数，冻结）
- **decoder**：独立训练的转置卷积，仅用于可视化，不参与 planning
