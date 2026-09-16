# MaMi-HOI 下一阶段实验路线

日期：2026-09-16

本文是仓库内当前阶段的计划入口。详细证据仍来自对应的实验报告和审计目录，本文只记录已经冻结的状态、实验依赖、冲突和晋级条件。

## 当前结论

### 已有正结果

- L1 bounded palm action-chunk selection 在 43 条 validation 序列、78 个手部事件上把 dense hand-vertex SDF contact F1 从 `0.4131` 提升到 `0.4909`，平均 SDF penetration 从 `9.787 mm` 降到 `7.363 mm`。该结果仍待 canonical analytic rerun 复核。
- Stage 2 local geometry B 在 object-held-out split 上通过 matched controls：三 seed mean onset AUC `0.8650 -> 0.9125`，release AUC `0.7933 -> 0.8431`。contact F1 基本不变，B 仍未进入最终动作选择。

### 已停止的版本

- 当前 34D/15D learned residual 为 NO-GO。H1 palm error 已经恶化，不能只归因于长 rollout。
- 当前 phase contrastive 和 predictive contrastive C2 均为 NO-GO，没有超过 B。
- K=4 整段 seed selection 最终为 NO-GO。

### 当前保留配置

```text
34D engineered state
+ shared left/right 5x5x5 local object-SDF PointNet
+ residual_scale = 0
+ contrastive_weight = 0
```

B 当前是事件表征分支，不是端到端 MaMi correction，也没有 articulated hand。

## 三个计划层

### A. 冻结现有主基线

目标：让后续所有实验有同一比较基准，避免继续在未复核的 L1 结果上叠加模型。

必须完成：

1. 运行 canonical analytic contact baseline，确认 selected 高于 base/random，oracle 高于 selected。
2. 把 contact 主指标从 frame-wise F1 扩展为 stable contact episode：onset、stable hold、release、late release、false contact 和 dropout。
3. 将独立左右手排序升级为 `(H_L,H_R,O)` joint bimanual state。独立 top-K 只保留为 baseline。
4. 让 B 的 learned event head 真正进入候选评分，并增加 geometry-only、state-extrapolation、learned-event 三臂隔离。

晋级条件：

- canonical rerun 与现有 L1 方向一致；
- joint state 相对独立左右手排序改善 episode 指标或候选 regret；
- learned component 在正确 action/state pairing 下优于固定几何和 shuffle controls。

### B. Forward-inverse consistency 诊断

目标：验证“预测的 future 是否能反向解释原 hand action”是否提供超过已有 forward/geometry 分数的额外候选排序信息。

约束：

- 当前 15D action 直接包含 palm delta 和 object translation/rotation delta，直接做 cycle consistency 基本是同义重复。
- 第一轮必须把 object channels `6:15` 从 action 输入中移除，作为 hidden consequence。
- 该实验是前置诊断，不是直接发明环形时间 WM。ACID/WAV 已覆盖通用机制。

详细协议见 [forward-inverse-consistency-protocol-2026-09-16.md](forward-inverse-consistency-protocol-2026-09-16.md)。

### C. Finger-aware hand-object refinement

目标：把 palm proxy 升级为 articulated hand，并用 finger contact 候选改善 MaMi 生成结果的局部接触。

当前阻塞：

- 当前 220D 表示没有 `pose_hand`，渲染手为 neutral pose。
- 原始 BEHAVE 是否仍保留 `smpl_fit_all.npz` 的 90D hand pose 尚未确认。
- GRAB/ARCTIC contact ontology、MANO/SMPL-X topology 和许可证尚未完成实际数据门。

推荐顺序：

```text
恢复或获取 finger supervision
-> HandX / ContactOpt zero-shot teacher gate
-> 冻结 MaMi 的独立 hand refiner
-> bounded contact correction
-> 只有稳定后才扩展 WM 的 finger state/action
```

第一版保持 MaMi、wrist 和 object trajectory 冻结。成熟手部模型只允许修改 finger residual 和 interaction-local body residual，不重训 220D 主干。

## 新论文带来的设计约束

- Bench2Dex 提示 contact representation 应关注局部接触几何和持续交互，不应继续只优化全局 SDF 距离。
- Single-Query Bimanual HOI 支持把 `(H_L,H_R,O)` 作为基本 interaction state。
- ReCHOIR 支持把 contact 用作 local residual control，而不是覆盖整个 motion prior。
- EPIC-Contact 提供 dense hand-object correspondence，适合作为 correspondence head 的外部监督、teacher 或 evaluator；它不是 MaMi 的主训练数据。
- WLA3 提示可以从 state transition 学习 latent interaction action，但必须验证 action identifiability，不能把 latent compression 当作物理因果。

## 执行顺序与依赖

| 阶段 | 内容 | 依赖 | 可并行 |
|---|---|---|---|
| E0 | canonical analytic rerun | GPU 服务器 | 数据许可证审计 |
| E1 | contact episode + joint bimanual state | E0 | E2 协议实现 |
| E2 | forward-inverse consistency diagnostic | frozen 34D/15D split | E0/E3 数据检查 |
| E3 | BEHAVE/GRAB/EPIC-Contact data gate | 数据访问与许可证 | E0/E1/E2 |
| E4 | HandX/ContactOpt teacher gate | E3 | 无 |
| E5 | frozen MaMi finger-aware local refinement | E1/E2 至少保留一个正机制，且 E4 通过 | 无 |
| E6 | 论文主线合并 | E5 | 无 |

## 冲突规则

- 当前 palm 6D action 与 finger action 不能直接拼成一个未定义 action vector。E2 使用 palm action，finger branch 使用独立 MANO/SMPL-X action schema。
- 独立左右手排序与 joint bimanual state 冲突。新主实验使用 joint state，独立排序作为 baseline。
- frame-wise F1 与 stable episode metric 冲突。episode metric 作为主指标，frame F1 只用于历史比较。
- SDF refinement 与 correspondence learning 冲突。SDF 保留为几何引擎和解析 baseline，correspondence 用于局部状态和评价。
- 全局动作修正与 local residual control 冲突。MaMi 主干继续冻结，contact module 只修改 interaction-relevant residual。
- kinematic HOI 与真实物理操作冲突。没有 simulator、force、friction、mass 或闭环反馈时，不能声称真实搬动或物理因果。
- 整轨迹 energy planning 与实时交互 WM 冲突。它只能作为候选生成器或慢速 planner，不能宣传为双向时间因果。

## 当前遗漏项

- canonical analytic rerun 尚未执行。
- contact episode 协议尚未建立。
- joint bimanual state 尚未正式实验。
- B 的 learned event head 尚未进入最终 scorer。
- 原始 BEHAVE `pose_hand` 尚未确认。
- GRAB/ARCTIC/EPIC-Contact 的字段和许可证尚未数据门。
- object response 与 action leakage 的隔离协议尚未实现。
- receding-horizon 或真实执行反馈尚未开始。
- 独立外部 evaluator 尚未接入。

## 停止条件

- E2 若不能证明 inverse consistency 超过 forward-only/geometry-only，停止该线。
- E3 若数据 topology、坐标、fps、hand identity 或许可证不可复现，停止 finger 模型训练。
- E4 若 HandX/ContactOpt 不能超过 neutral 和简单 retargeting，论文主线留在 structured contact refinement。
- E5 若 contact episode 改善伴随 wrist drift、object response 或 motion quality 退化，退回 E1 主线。

## 证据入口

- [本周研究成果评价](weekly-research-assessment-2026-09-16.md)
- [Analytic contact baseline](../experiments/analytic-contact-baseline.md)
- [WM Stage 1 learned residual](../experiments/wm-stage1-learned-residual.md)
- [WM Stage 1 failure mechanisms](../experiments/wm-stage1-failure-mechanisms.md)
- [表示能力边界](representation-scope.md)
