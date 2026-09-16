# Forward-Inverse Consistency Diagnostic Protocol

日期：2026-09-16

状态：设计冻结，未实现代码，未有实验数字。

## 目的

验证一个明确的问题：对于 MaMi 的 hand-object candidate，预测出的 future interaction 是否不仅能达到目标，还能反向解释产生它的 hand action。

该诊断不是“时间环形 WM”的名称创新。通用 forward-inverse consistency 已由 ACID、WAV 等机器人 world-model 工作覆盖；整轨迹共同优化也不是双向时间因果。这里检查的是该机制是否能给 hand-object contact refinement 提供新的候选排序信息。

## 当前数据约束

现有 34D state layout：

```text
0:3    object translation relative to anchor
3:9    object rotation 6D
9:15   left/right palm position in object frame
15:21  left/right palm velocity
21:24  object translation velocity
24:26  left/right palm signed clearance
26:32  left/right palm surface normal
32:34  left/right contact probability
```

现有 15D action layout：

```text
0:6    left/right palm delta
6:9    object translation delta
9:15   object rotation delta 6D
```

直接对当前完整 15D action 做 inverse consistency 是同义重复，因为 object future 已经写在 action 中。

因此本协议强制定义：

```text
x_t = state history
h_t = action[..., 0:6]                 # hand action, model input
y_t = action[..., 6:15] + future contact state
                                        # hidden consequence, never input
```

`y_t` 只能作为训练标签、离线 evaluated target 或计划中的 object response；不能进入 forward model 的输入，也不能用于在线 observation。

## 模型

### Forward model

```text
F(x_t, h_t) -> y_hat_t
```

输出：

- object translation delta；
- object rotation delta；
- future contact probability 或 contact transition；
- 可选的预测不确定性。

第一版使用小型 MLP/GRU，不使用大 Transformer。

### Inverse model

```text
G(x_t, y_t) -> h_hat_t
```

输出 6D palm action。必须同时报告 dense inverse 和 sparse inverse 消融，以检查 action-relevant feature 是否低维可识别。

### Candidate score

```text
score = geometry_contact_score
        - lambda * action_reachability_residual
```

`lambda` 只能在 train-only 数据上确定。validation 只能评价，不能选择最有利的 lambda。

## 实验 arms

| Arm | 内容 |
|---|---|
| A0 | constant-velocity / zero-object response baseline |
| A1 | geometry-only SDF scorer |
| A2 | forward-only object/contact prediction |
| A3 | inverse-only action reachability |
| A4 | forward + inverse consistency |
| A5 | A4 + geometry contact scorer |
| A6 | oracle over fixed candidates |

所有 arm 使用同一 state history、candidate set、object split、seed 和 contact threshold。

## 数据与切分

主数据：

- 现有 v8 `train.npz` 和 `val.npz`；
- object-held-out split 保持不变；
- sequence index 作为 bootstrap group；
- H=1、2、4、8，H=8 为 planning-relevant primary horizon。

第二阶段，在服务器恢复后复用冻结 L1 candidate：

- candidate types：hold、inward、outward、zero、random smooth；
- primary endpoint：candidate regret、top-1 success、dense contact F1、penetration；
- 所有 candidate identity、action 和 score 必须落盘。

## 控制与压力测试

- action shuffle：交换 candidate action，但保持 state 和 planned consequence。
- history shuffle：保持 action，交换 state history。
- future permutation：打乱 object/contact consequence 与 state 的对应关系。
- static control：低动态、背景占优、无物体响应候选不得获得高 consistency。
- collapse test：交换 latent/action features 后，如果 future prediction 不变，说明模型绕过 action。
- many-to-one test：检查多个不同 hand action 是否映射到近似相同 consequence。

## 主指标

- inverse action reconstruction error；
- forward object translation/rotation error；
- future contact F1、precision、recall 和 calibration；
- candidate top-1 accuracy；
- candidate regret；
- dense contact F1；
- mean penetration；
- stalled/background false-confidence rate。

所有差值使用 sequence-clustered paired bootstrap。

## 晋级门控

### Gate 1: identifiability

- inverse action error 明显优于 constant action 和 future shuffle；
- action excitation 不能接近零；
- 正确配对必须明显优于 action/history shuffle。

### Gate 2: decision utility

- 相对 geometry-only，top-1 candidate accuracy 至少提高 3 个百分点，或 candidate regret 至少降低 10%；
- 95% sequence-clustered CI 下界大于零；
- contact F1 不下降，penetration 不恶化。

### Gate 3: false confidence

- static、stalled、无物体响应候选不能获得高于真实交互的 consistency；
- inverse consistency 不能只奖励低动态轨迹。

任一 gate 失败即停止该线，不把它接入 finger-aware refinement。

## 实现边界

- 不修改现有 `ContactActionTransition` 和冻结 checkpoint。
- 新增独立 consistency experiment module 和脚本。
- 保留完整 input hash、split、seed、precision、checkpoint 和 evaluation artifact。
- 不声称真实物理、力、摩擦、反事实因果或双向时间。

## 条件后续

只有 Gate 1-3 通过后，才允许在第二阶段使用 BEHAVE/GRAB 和 HandX 构造 finger-level forward-inverse consistency。第二阶段必须使用独立 action schema，不能复用当前 15D object action。

