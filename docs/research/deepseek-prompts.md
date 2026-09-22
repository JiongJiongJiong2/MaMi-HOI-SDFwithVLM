# DeepSeek 实施提示词

使用顺序：1 → 2 → 3 → 4。每次只复制一个代码框到一个新会话；2/3 必须满足上一步记录的前置条件。5 用于中断恢复。不是要求同时开五个任务，也不要求某一个模型完成所有阶段。当前所有 CTP 阶段尚未执行。

每段均指向同一个 [研究入口](../../RESEARCH_START_HERE.md)、[协议](../experiments/contact-transition-pilot-v1.md) 和 [交接](implementation-handoff.md)。可在现有主项目中串行执行；若使用 worktree，把工具返回的真实路径作为执行路径，并保留相同的仓库内文档结构。

## 1. T0：资产核对与任务契约

```text
你负责 HOI 项目的 CTP-01/T0：确认完整接触转换试验的最小资产，产出可实施契约。这是本会话唯一任务，不训练模型、不开展正式对照。

主仓库 E:/HOI/MaMi-HOI-SDFwithVLM。先检查 git status、branch、HEAD 和未提交文件，再读取 RESEARCH_START_HERE.md、docs/experiments/contact-transition-pilot-v1.md、docs/research/implementation-handoff.md、docs/research/research-workflow.md。不要把旧报告的“下一步”当成本次任务。

简短复述 T0 问题和终点后直接工作。检查现有 MaMi/E5/HandX 与可用手物数据，核对完整事件、时间单位、手—臂参数映射、FK/网格执行路径和标签来源。最多整理 12 个开发片段；选择只依赖数据完整性和预定规则，不能按方法收益筛选。若需现有服务器，只在已有访问授权下只读核验环境和任务，勿重复启动 EPIC B 的 manifest，勿开新付费资源或删除数据。

输出 docs/experiments/ctp01-t0-asset-contract.md 及紧凑清单，按协议冻结时间范围、修正空间、预算和评价定义。区分已核验与未知。缺上肢或完整事件就交付可复现缺口，不能拼接跨数据集真值、不能把手部诊断写成全链通过。

更新研究入口中 CTP 行及交接，提交本次聚焦成果。报告 T1 是否具备条件和下一条具体操作。本会话结束于 T0，不启动 T1/T2，不重跑旧 DWM/C0 排序器。
```

## 2. T1：执行接口及两个片段 smoke

```text
你负责 CTP-01/T1：落实一个可重建的手—臂修正接口，并在固定两个片段上做 smoke，不训练 WM、不做全量实验。

主仓库 E:/HOI/MaMi-HOI-SDFwithVLM。先检查 git 状态，读取 RESEARCH_START_HERE.md、docs/experiments/contact-transition-pilot-v1.md、docs/research/implementation-handoff.md 及 docs/experiments/ctp01-t0-asset-contract.md。若 T0 文件不存在或其完整事件、手—臂映射、评价/预算未就绪，只记录具体缺口，不擅自扩成数据工程。

按 T0 契约复用现有实现，完成有界姿态修正到 FK/蒙皮输出的最小路径。验证零修正重建、已知方向动作生效、坐标与单位、腕连接、骨长/关节限制、物体固定和非目标部位偏移。不要只平移导出网格。使用 E5 工具时注明它原本只动手指，不把新自由度收益归给预测。

正常实施 bug 自主修复；涉及实验定义变化，记录版本变化和理由，不静默换目标。只执行冻结的两个片段和必要定向测试，保留修正前后完整视图、参数与小型数值汇总。

输出 docs/experiments/ctp01-t1-executor-smoke.md，更新入口和交接，聚焦提交代码、测试与报告，保留其他任务文件。给出 T2 是否就绪。本会话到 smoke 结论结束，不自动训练或扩大样本。
```

## 3. T2：完整过程的无学习对照

```text
你负责 CTP-01/T2：检验完整接触过程里，前瞻几何与接触阶段信息是否改善实际修正。这次是无学习对照，正负结果都算任务完成。

主仓库 E:/HOI/MaMi-HOI-SDFwithVLM。先检查 git 状态，再读取 RESEARCH_START_HERE.md、docs/experiments/contact-transition-pilot-v1.md、docs/research/implementation-handoff.md、T0 契约与 T1 smoke 报告。前置未通过则定位缺口，不越过门槛。

只使用已冻结的开发片段、预算、主要端点与守护指标。实现协议的 U 不修正、R 反应几何、G 解析前瞻、O 阶段 oracle 四组，全部共享执行器。G 也能读取任务允许的未来参考，不能削弱它来制造优势。明确 O 额外使用不可部署的核验标签，只作收益上限。按需要报告共同候选池的乐观最优上限，不重用旧 78 事件池包装新实验。

核对完整动作和独立表面指标，逐序列配对报告接触建立、保持、释放与身体质量。样本不足如实写证据不足；不得改阈值、删难例、把窗口当独立样本，或查看保留 test。不要新增残差网络、GRU、VLM 或主动探测。

输出 docs/experiments/ctp01-t2-outcome.md、紧凑可复算结果与真实命令，更新入口和交接并聚焦提交。解释这次支持或不支持哪个具体假设，停止于 T2，不自动开始学习阶段。
```

## 4. T3：独立复核与下一步决策

```text
请独立复核 CTP-01 是否值得进入学习阶段，不直接训练。主仓库 E:/HOI/MaMi-HOI-SDFwithVLM。

先检查 git 状态，读取 RESEARCH_START_HERE.md、docs/experiments/contact-transition-pilot-v1.md、docs/research/implementation-handoff.md 及 CTP T0/T1/T2 产物。不要只复述上一会话最终回答。核对实际代码 diff、配置、原始紧凑指标、样本选择、预算和可视化；仅做必要的小规模重算，不重新跑全套或修改冻结数据。

回答：候选动作是否真正由运动学链执行？G 是否得到公平的未来信息与预算？O 是否存在实质且一致的额外收益？改善是否来自更多自由度、更多计算或标签泄漏？小样本能支持多强结论？是否只是复现旧 MaMi residual/C0-R2 的负结果？

输出 docs/research/ctp01-t3-decision.md，结论为收束几何方案、补充独立证据、修复无效实验，或提出新的学习协议之一。若建议学习，写清未知预测目标、合法输入、监督来源、强基线和最终动作终点，不能仅写“加 WM”。保留既有 NO-GO，不修改旧门槛。不运行新训练。

更新入口和交接，按项目规则提交聚焦复核成果；将需要研究决策的取舍带回讨论会话。
```

## 5. 中断/上下文压缩/修 bug 后恢复

```text
恢复当前已领取的 CTP-01 阶段，主仓库 E:/HOI/MaMi-HOI-SDFwithVLM。不要根据上一段聊天猜进度，也不要重启已经运行的任务。

先检查 git 状态，读取 RESEARCH_START_HERE.md、docs/research/implementation-handoff.md、docs/experiments/contact-transition-pilot-v1.md 及交接指向的当前阶段产物。核对本地/远端版本、实际输出和进程；远端不可访问就记录未知。区分实现完成、smoke、正式运行和已复核结果。

用一小段说明原研究问题、当前阶段、中断点和最小恢复动作，然后继续已授权范围。必要 bug 自主修复；修好后回到原实验。反复失败先总结已排除原因，不无限重构。更改标签/划分/指标/基线/核心机制需记录为协议变化，不用改门槛制造 PASS。

阶段结束或再次受阻前更新交接中的真实命令、版本、运行状态、产物和下一条操作。保留其他任务变更，提交本次已验证成果，不自动启动下一个阶段。
```
