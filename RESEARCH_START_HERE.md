# HOI 研究当前入口

更新：2026-09-22；证据快照基于整理前提交 `c06dd0a`。这是当前研究入口，不是新实验结果。用户已要求按本轮讨论整理下一阶段文件；执行范围为下述有边界的试验路线，尚不代表任何新机制有效。

## 主问题与本轮选择

导师关切：生成动作的手仍浮空、穿模、接触不自然。当前要检验：**在完整接触建立—保持—释放过程中，短时未来信息能否改善可执行的手—臂修正，超过相同预算的几何前瞻优化？**

保留几何查询、E5、现有人体/手模型和评价资产。WM 是候选机制，不是本轮必须产出的标签。下一步只推进 **CTP-01 接触转换试验的 T0：资产与任务契约**，之后按协议检查 T1/T2 入口。不要启动新 WM 训练、P1 主动探测、全身物理系统或新的 VLM 支线。

为什么这不是重跑旧排序器：必须新增完整事件、实际运动学执行和公平的时域比较。若只能使用原 43 序列/78 事件的平移候选或两候选交接集合，不得另换名字重复训练。

## 新会话阅读顺序

先读本页，再读 [CTP-01 协议](docs/experiments/contact-transition-pilot-v1.md) 与 [交接记录](docs/research/implementation-handoff.md)。需要操作边界时看 [实施工作约定](docs/research/research-workflow.md)。可直接复制 [DeepSeek 任务提示词](docs/research/deepseek-prompts.md)。只按需要回查下面的证据。

## 当前证据账本

| 分支 | 已有证据与边界 | 当前动作 |
|---|---|---|
| E5 | 343/376 联合通过；五点平滑 312/376；train/dev 重叠短窗口 | 保留为基线，不能称完整手—臂已解决；[报告](docs/experiments/contactopt-e5-sequence-solver-result-2026-09-19.md) |
| A 方向约束投影 | 351/376，但未与普通平滑/强接触对照显著分离，时序有代价 | 当前机制 NO-GO，保留工程实现；[结果](docs/experiments/contactopt-e5t-direction-projection-result-2026-09-22.md) |
| B EPIC 接触记忆 | 代码、测试、smoke 已有；本地日志最后记录全量 manifest 启动 | 独立进行中，服务器实时状态未核验；不重启、不作为 CTP 必经门槛；[已提交实现](scripts/evaluate_epic_contact_memory_correction.py) |
| C0-R2 交接排序 | 82 targets、41 groups、每组两个候选；主条件没有改变排序 | 当前版本 NO-GO，不启动原完整优化器；[结果](docs/experiments/oakink-c0r2-decoupled-reranking-result-2026-09-21.md) |
| 旧 residual / E2 | 残差退化；E2 真实/打乱动作 H8 F1 0.7099/0.7102 | 原版本停止；[残差](docs/experiments/wm-stage1-learned-residual.md)、[E2](docs/experiments/forward-inverse-consistency-2026-09-18.md) |
| D0 / DWM P0 | 已有 MuJoCo 动作分支数据；P0 漏目标输入已修，修后 top-1 0.1028 vs geometry 0.2306 | 不再写“尚无动作分支数据”；P0 仍 NO-GO；[诊断](docs/experiments/dwm-p0-failure-analysis-2026-09-22.md) |
| P0-R2 | top-1 0.2769 vs 0.2306，有正向点估计，冻结晋级门槛与区间未通过 | 不抹掉局部正信号，也不翻转 NO-GO；[结果](docs/experiments/dwm-p0r2-result-2026-09-22.md) |
| MaMi residual reranker | 报告均值 F1 0.4830 vs geometry 0.4909，未建立正收益 | 当前版本停止；报告文字百分点差与显示均值有出入，精确效应需查原 JSON；[结果](docs/experiments/dwm-mami-reranker-result-2026-09-22.md) |
| CTP-01 | 只有本轮协议和实施提示词 | T0 待执行；没有新实验结果 |

B 的本地未提交协议与计划仍由原任务持有：`docs/experiments/epic-contact-memory-protocol-2026-09-23.md`、`.planning/2026-09-22-epic-contact-memory/`。整理时未纳入本次提交；文件日期不能证明未来实验已完成。

## 已知资产限制

现有 E5 固定手腕和物体，只优化手指系数；不能直接充当手—臂执行器。MaMi 处理缓存缺少 `pose_hand`；HandX 投影不等于真实手指监督。EPIC 有手物几何，不应假定它提供同一人的完整上肢。见 [E5 协议](docs/experiments/contactopt-e5-sequence-solver-protocol-2026-09-19.md)、[E3 数据检查](docs/experiments/e3-data-gate-2026-09-20.md)、[HandX 投影](docs/experiments/e3d-handx-mano-projection-2026-09-19.md)。T0 必须落实数据与人体参数映射，不能跨数据集拼成伪真值。

旧 28 个 E5 test 窗口在方向实验报告中仍未读；本轮未检查其实时使用情况。OakInk/EPIC 已查看的测试结果不重新标为盲测。新试验先用开发资料，最终独立验证另立协议。

## 文档优先级与历史入口

事实以可核对的代码、配置和运行产物为依据；研究范围以用户最新指示及本页为入口。协议控制对应实验，结果报告记录对应版本，旧建议不覆盖新完成结果。发现冲突先记录，不能仅按修改时间判真伪。

[主线失败分析](docs/research/wm-mainline-failure-analysis-2026-09-22.md)保留概念讨论，但其本轮前未纳入的 DWM/A/C 后续状态由上表补充。[9 月 21 日审计](docs/research/astra-920-progress-audit-2026-09-21.md)保留当时证据范围；[9 月 20 日候选方向](docs/research/idea-revisit-and-research-options-2026-09-20.md)是历史选项，不是并行执行清单。

## 更新规则

每个实验里程碑结束，实施者只更新对应行与交接记录，并附结果路径/commit。改变主问题、主要评价或启动新分支，应明确写出方案与原因，回到研究讨论决定。不要为每个新会话创建另一个“总计划”。
