# 本周研究成果评价与下一步取舍

日期：2026-09-16。审计基点：`f8241ee`。范围：9 月 9–16 日，必要时回溯此前的问题定义。本文评价研究成果、证据和方向；没有连接服务器、运行新训练或修改实验实现。

## 判断摘要

**这一周形成了真实的研究积累，尚未形成“学习型 WM 已改善 MaMi-HOI 全身生成”的结果。值得继续，但下一阶段应围绕一个可检验的协调机制收敛。**

最有价值的进展有三项：第一，逐步建立了区分几何查询、接触判定、事件预测、动作选择与最终动作实现的实验体系；第二，解析几何选择短动作块在限定窗口内获得可复核的改善；第三，局部物体几何对接触事件排序有独立于模型容量的贡献。两项正结果目前属于不同链路，不能相加成 learned WM 的端到端成功。

我赞同“逐步实验、放弃无效实现、慢慢打磨”的工作方式。应保持稳定的是问题——让生成动作在接触切换时保持贴合与身体协调；允许更换的是机制。若每次只更换模块名称而没有更新可证伪的解释，实验会越来越多，论文主线却不会更清楚。

## 1. 本次如何作出判断

本次冻结三个问题：RQ1，这一周留下了什么可靠成果？RQ2，现有改善应归因于几何、候选搜索、学习表征还是预测动力学？RQ3，怎样从这些成果走向一个改善全身协调的明确贡献？

材料包括主仓库提交与核心执行路径、多个 HOI/WM 会话、9 月 11–16 日实验报告、原始 JSON/JSONL/CSV，以及公开论文正文。三个并行调查分别覆盖几何证据、WM 证据、研究史与全身 HOI 文献；综合时再次核查关键代码和指标。既有报告是线索，原始结果和实际计算定义优先。同一报告内按后续实验的最终结论判断，不沿用页首旧状态。

主仓库从 `b176f73` 到 `f8241ee` 涉及 78 个文件，约 1.73 万行新增。它已经从单项 SDF/几何实验，发展为包含 contact event schema、手表面评估、HOI-Dyn adapter、action-conditioned predictor、局部几何编码、候选选择和消融脚本的平台。这说明研究工具发生了实质变化，但文件量、测试通过或代码接入不能作为方法有效性的证据。提交日期也不能替代实验执行日期。

本次为离线证据复核：未独立重训历史模型；对关键统计进行了重算；不存在新的 test 集结果。详细证据、会话来源及复算文件保存在 [本次审计目录](C:/Users/何炯乐/Documents/HOI项目/audit-week-20260916)。

## 2. 这一周的演进：问题在变清楚

最初的关切来自导师看到的浮空、穿模和动作不自然。随后工作从“多加几何约束”推进到“当前指标是否真的测到可见手部接触”。代理关节、手表面和真实抓握被区分开；冻结 U0、同预算 U0-FT 和 U1U 的比较，也开始区分继续训练本身与新增损失的贡献。这是后续所有研究判断的基础。

截面方向经历了重要的任务澄清。用户想要的是给双手共享的一份局部结构提示，不能用单手面积覆盖直接替代这个问题。后续同候选池消费者、拓扑、内在描述子与时序比较，比最初的覆盖探针更接近检验原始想法。因此，应同时承认早期结论的适用范围有限，以及后来的受控负结果确实减少了继续投入同一实现的理由。

WM 方向也发生了实质变化：从只识别当前接触状态的 CEWM 分类器，到具有显式 action、future target 和 rollout 的模型，再从挑选 MaMi 随机 seed 转向控制局部动作块。用户在历史会话中坚持区分 classifier 与 WM，是有价值的问题把关。最近又把解析分支与 learned residual 分开比较，并保留 B、停止当前 C/C2。这些取舍让研究开始能回答“究竟哪个部分有用”。

对这一周工作最公允的评价是：**实验执行能力和机制辨别能力明显进步；方法贡献的闭环仍待完成。** 不宜将连续 NO-GO 解释为没有进展，也不宜把若干局部 GO 拼接成已经成功的方法。

## 3. 实验成果账本

下表中的“停止”仅针对已测试版本与预算；“保留”表示有依据进入下一次检验，不代表论文主张已经成立。pp 表示百分点。各行终点、数据和平均方式不同，数值不能横向排名。

| 工作 | 已有证据 | 学术判断与可复用资产 |
|---|---|---|
| U1U / DSR-01R | 训练集校准覆盖 4,380 条序列、123,271 条查询；修正后 300-step 两臂各评估 100 条 validation。逐序列宏 F1 差约左 +0.181 pp、右 +0.214 pp，区间跨零 | 校准和同预算比较是成果；当前 unsigned distance loss 缺少足够晋级信号。保留查询、校准、评估资产，不继续按原形式直接扩到长训练 |
| DMS / HPS 几何结构 | DMS 时间稳定性改善，但同预算结构消费者 AUC 约 0.6672 对 0.6694；多个 HPS 同池/结构变体未建立稳定增益 | “表示更稳定”不等于“对任务更有用”。停止当前轮廓/离散候选结构扩张，不推出几何整体无效 |
| CEWM-02 | 当前接触分类的 causal GRU micro F1 0.6872，对静态阈值 0.7295；误报增加 | 普通分类器没有超过简单规则。此结果不否定后来 action-conditioned future prediction |
| HOI-Dyn 迁移 | dynamics 模型有对象响应预测信号；MaMi 300-step、10-sequence screening 的 pooled F1 左右分别下降约 0.72/1.02 pp | 完成了一条可复用 dynamics 接口；当前辅助 loss 的迁移没有证明生成收益。不能据此否定原论文或所有对象动力学路线 |
| K=4 整段 seed selection | 早期 +6.90 pp 对阈值敏感；train-calibrated threshold 后转负；最终 dense v0 左 +0.947 pp、右 −0.675 pp | 最终结论为 NO-GO。历史页首 conditional GO 已被后续证据覆盖 |
| L1 短动作块几何选择 | 43 序列、78 手事件：F1 0.4131→0.4909；当前 SDF 平均 penetration 9.787→7.363 mm | 保留为局部几何修正潜力和解析基线；不是 learned WM、完整全身动作或真实抓握成功的证明 |
| Local geometry B | 9 个训练物体、4 个留出物体，3 seeds；onset AUC 0.8650→0.9125；release 0.7933→0.8431；等容量/打乱几何对照较弱 | 本周最清楚的学习表征正结果。contact F1 仅 0.8250→0.8278，未证明 downstream 选择和全身改善 |
| 对比学习 | reranker 的 C top-1 F1 0.4389，低于解析 scorer 0.4909；C/C2 均未稳定超过 B，C2 release AUC 0.8431→0.8319 | 停止当前 auxiliary 与 reranker 版本是合理决定。不需要为了保留 contrastive 名称继续调参 |
| Stage 1 learned residual | 3 seeds：H8 palm L1 0.00922→0.08876；H1 已由 0.00133→0.01205；learned scorer 相对 geom_only F1 −10.17 pp | 当前残差实现已有直接负证据，应停止本版本。长 rollout 累积可以是放大因素，但不能解释全部退化 |

原始证据入口：[DSR-01R 分析](C:/Users/何炯乐/Documents/HOI项目/dsr01-repair-20260913/analysis/two_arm_analysis.json)、[DMS 结构结果](C:/Users/何炯乐/Documents/HOI项目/dms-comparison-20260912/DMS-STRUCT-01-结构消费者结果与决策-2026-09-13.md)、[CEWM-02](C:/Users/何炯乐/Documents/HOI项目/cewm-20260914/CEWM02-causal-event-probe-2026-09-14.md)、[HOI-Dyn paired analysis](C:/Users/何炯乐/Documents/HOI项目/hoidyn-mami-ab-20260914/paired_analysis.json)、[seed selector 最终结果](C:/Users/何炯乐/Documents/HOI项目/mami-candidate-selection-result-2026-09-15.md)、[L1 合并结果](C:/Users/何炯乐/Documents/HOI项目/wm-l1-action-chunk-20260915/combined_pen20_live_metrics.json)、[B 原始汇总](C:/Users/何炯乐/Documents/HOI项目/geometry_ab_summary.json)、[contrastive reranker](C:/Users/何炯乐/Documents/HOI项目/contrastive-reranker-pilot-2026-09-15.json)、[C2](C:/Users/何炯乐/Documents/HOI项目/c2-predictive-contrastive-result-2026-09-16.md)、[Stage 1](E:/HOI/MaMi-HOI-SDFwithVLM/docs/experiments/wm-stage1-learned-residual.md)。

### 3.1 L1 正结果值得保留，但要准确命名

本次以序列为 cluster 对原始 78 个事件重抽样 20,000 次，保留原事件加权的估计目标。selected−base 仍为 **+7.7808 pp，95% 区间约 [1.77, 14.16] pp**；selected−random 为 +12.1255 pp，区间约 [6.67, 18.22] pp；平均 penetration 差为 −2.424 mm，区间约 [−4.46, −0.56] mm。因此，考虑同一序列双手事件的相关性后，主要方向仍在，不应轻易抹掉这项正结果。

它的适用范围由实验定义决定。`first_onset_window` 用 GT 接触开始定位窗口，候选评分本身不读 GT；故这是给定接触即将发生位置的局部诊断，尚未评价自主触发修正、无接触时期和释放时期。手表面通过整体平移获得候选，没有让肩肘和躯干在同一个人体模型上实现该位移。评分使用解析 rollout 和物体 SDF，当前最强结果的几何路径没有 learned residual 贡献。[实现](E:/HOI/MaMi-HOI-SDFwithVLM/scripts/run_contact_action_chunk_correction.py:283)

这里的 dense F1 是“查询手部所有顶点后，按最小距离把每帧判成接触，再计算每手事件 F1”，不是逐顶点接触图 F1。现有 50 mm 判定与负距离惩罚也不能直接解释为毫米级贴合或真实抓握。特别是已有 SDF 材料边界争议，penetration 改善应称为当前距离场下的改善；后续对真实表面贴合的主张，需要与表面距离和适用物体上的独立几何检查一致。

新增 51–70 cohort 的原始 selected F1 是 0.50560、base 0.42868，差 +7.6923 pp。13 个事件中只有一个改善，其余为零，因此它是有限的方向补证，不宜称为广泛泛化。这不推翻合并结果，只约束它的外推范围。

### 3.2 B 是有内容的正结果，尚未连接到动作改进

B 的等容量零输入与打乱几何对照有研究价值：它们帮助排除了“只是网络更大”的解释。旧截面消费者没有收益，而当前局部 SDF patch 的事件 AUC 有收益，说明几何价值依赖任务、表示和使用位置。不能把前者的失败泛化成“几何不值得做”，也不能把后者推广成“局部几何已经解决接触”。

需保留一个统计细节：现有 bootstrap 的 32 个 event groups 混合了 horizon、左右手和 onset/release；其区间不是单独 onset AUC 的区间。可准确表述为：onset/release 的汇总均值提高，各 seed 的混合事件配对比较为正。更关键的是，B 在 `residual_scale=0` 时改变 hidden/event heads，而当前 `geom_only` 评分不使用这些学习输出。**B 的事件表征收益与 L1 的动作选择收益尚未被一项受控实验连起来。**

### 3.3 负结果要用于取舍，不宜写成普遍定律

DSR-01R 的结论是当前训练目标和 300 步预算没有足够增益，不是“距离约束永远没用”。HOI-Dyn 本地使用 5 epoch dynamics 与 300-step 迁移 screening，而原论文正文报告 150 epoch dynamics 与 100,000-step 生成训练；二者证据范围不同。停止当前廉价迁移方案有依据，宣称原方法被复现否定则没有依据。[HOI-Dyn 正文 §4](https://arxiv.org/html/2507.01737v3)

同样，C/C2 和 learned residual 的具体结果已经足以决定暂时停止这些版本；还不能归因为某一种唯一训练机制，也不能预言改一个技巧就必然恢复。先把被否定的假设写清楚，再决定新设计究竟改变了什么。

## 4. 文献对当前主张的实际约束

以下按机制分组，不把所有冠以 world model 的论文当作同一种证据。文献只支持其原任务中的结果，不自动证明组件迁入 MaMi 有效。

| 文献线 | 最相关工作与实际关系 | 对本项目的要求 |
|---|---|---|
| 几何与接触引导 | CHOIS、CG-HOI、InterDiff 已分别使用接触约束、联合接触建模或物理启发修正 | 对照应包含便宜几何/接触优化，不能只比原始 MaMi |
| 局部与全身协调 | MaMi-HOI 的 GAPA+KHA、HOIDiNi 的接触蓝图到全身优化、InterMimic 的物理全身控制 | “把手部改进扩展全身”本身不足以形成贡献；必须解释新增预测如何改变决策 |
| 交互响应预测 | InterDreamer、HOI-Dyn；机器人邻域的 HAIC | 学习人体行为之后的对象/交互响应已有直接先例；需要明确新的预测目标或使用机制 |
| WM 的其他含义 | TacWAM 的预测共训练、WM-Craftnet 的 recurrent context、LOME 的动作条件视频生成 | 这些成功不能合并成“短 rollout 规划必然有效”的证据，更不能把几何距离叫 tactile |

**最直接的近邻是 InterDreamer，不能只盯机器人文献。** 它利用已有运动生成模型，以接触附近人体顶点轨迹为 action，学习对象响应，并结合 rollout 与人体/物体运动优化。相比之下，HOI-Dyn 将响应预测用于生成训练约束，HOIDiNi 则在扩散噪声空间落实接触和全身协调。它们共同说明，复用生成器、局部几何、动力学和修正的组合已有方法基础；本项目需要在这些基础之上证明一个具体的新增效果。[InterDreamer §3.3–3.4](https://arxiv.org/html/2403.19652v2)、[HOI-Dyn](https://arxiv.org/abs/2507.01737)、[HOIDiNi](https://arxiv.org/html/2506.15625v2)

MaMi-HOI 自身的 KHA 就用于让身体姿态适应局部空间目标。因此“旧方法只管手，我来管身体”不是准确定位。可研究的区别是：在同一冻结生成器上，用候选动作的未来接触信息提前调整姿态，是否优于已有的空间适配和短视修正。这是待验证的问题，不是已确认的文献空白。[MaMi-HOI §3.2–3.3](https://arxiv.org/html/2605.05756v1)

HAIC 是一个有启发的跨域例子：它从历史推断对象动态，并将预测与几何结合服务 humanoid 策略；其环境反馈和真实机器人状态，与离线生成序列有本质区别。LOME 与近期 upper-body WM 又主要针对受控视觉生成。引用它们可以说明研究趋势，不能据此宣称本项目具备真实反馈、物理稳定性或同级世界模拟能力。[HAIC，RSS 2026](https://www.roboticsproceedings.org/rss22/p013.html)、[LOME](https://arxiv.org/abs/2603.27449)、[Real-Time Human-Centric World Modeling](https://arxiv.org/abs/2607.23517)

本次检索没有建立“尚无人研究预测驱动全身修正”的优先权结论。更可靠的论文定位应来自受控实验展示的差异，而不是通过堆叠最新术语寻找一个无人使用的名称。

## 5. 对“WM 改善全身其他部分”的具体建议

**建议继续，但先从一条身体链和一种接触转换做起。** 一个值得检验的假设是：在接近、持握或释放的转换附近，短期交互预测能提示肩肘与躯干提前调整，使接触改善不以姿态突变、脚滑或物体轨迹失真为代价。第一版优先手腕—肘—肩—躯干，双脚作为约束和观察终点；待这条链有收益后，再扩展骨盆补偿或移动支撑。

这不是建议立即增加一个全身大模型。当前局部位移只作用于导出手顶点，首先需要确认它能否由同一套身体旋转与 FK/SMPL 重建实现。可以先冻结物体轨迹和生成主干，对身体参数作有界时序修正。所有方法使用同一个求解器、接触目标来源与动作预算，才能判断预测贡献，避免将 IK 或多采样的收益计入 WM。

缺少 articulated hand 不阻止这一步。它限制的是指尖闭合、精细抓握和相关主张；并不阻止用现有身体旋转检验上肢/躯干协调。真实力、摩擦或触觉也不是运动学修正的前置条件。若论文目标以后变成手指抓握或物理稳定操控，再为该目标补数据和模型，而不必现在同时开启所有分支。

### 5.1 先确认学习部分有不可替代的预测任务

当前 action 包含双掌位移和物体位姿增量。如果某个未来位姿已由动作定义和几何积分确定，网络再学习同一个增量，很容易只增加误差。若连最终候选全身轨迹和对象轨迹都已给定，则未来距离也可直接计算；此时所谓“预测优势”可能只是重新近似已知几何。

因此，WM 至少应回答一个当前几何不能直接给出的、且可评估的问题，例如候选局部修正后的接触持续/释放风险、未被 action 指定的对象响应，或身体其余部分的合理响应分布。模型不必显式恢复全部力学参数，但必须区分“动作指定了什么”和“模型预测了什么”。

观察到的一条 mocap 轨迹只提供一种实际响应；将手向内/向外移动并查询 SDF 可以验证几何敏感性，不能产生这些干预后的真实物理结果。早期可以用运动学求解器构造可实现修正数据，但论文就应称为运动学交互修正；若主张因果物理响应，则需要额外的模拟或执行反馈支撑。历史 context 有效也不等于存在 action-conditioned transition，须分别消融。

### 5.2 下一轮应解决的三个问题

**第一，局部收益能否在完整身体上实现？** 接续已经开始冻结的 analytic baseline，先用少量预先固定的序列，把候选位移落实为身体姿态修正。GT 事件可以保留作能力上界，同时单列预测触发的可运行版本。评价同一人体重建后的接触距离、事件 F1、关节修正、脚滑与时序平滑；若手顶点改善在身体求解后消失，先解决可达性或目标定义，不增加 WM 复杂度。

**第二，多步预测是否比同预算的简单方法有额外价值？** 在同一个身体求解器与候选集合上，比较解析几何短时域优化、当前状态/reactive 修正、one-step learned predictor 和 H-step predictor。再去掉 action 或打乱 history，检验哪种信息实际改变最终动作。解析对照必须也能看相同候选未来几何，不能刻意让它短视。若只有 AUC 改善而最终选择 regret、接触转换或身体质量无改善，保留表征结果，不把它升级成规划贡献。

**第三，效果能否覆盖完整事件？** 从已知 onset 窗口走向包含未接触、进入、持握和释放的连续片段，使用预测触发，不读未来 GT 标签；已生成的未来候选轨迹可以作为计划输入，但要明确这不是现实世界的未来观测。若采用滚动修正，应在执行短前缀后用重新实现的身体状态更新，而非始终回到原始参考轨迹。没有环境响应时称作离线滚动 refinement，不称真实物理闭环。

每一步都只新增一个需要判断的机制。失败时记录是目标不可达、事件触发不准、预测无额外信息、还是实现后的动作质量退化；下一步根据该结果决定，而不是预先决定必须再加一种网络。

### 5.3 主指标与研究资源应怎样使用

本阶段可把完整片段的接触转换质量与接触期间的手表面距离作为主要终点，同时检查无接触期误吸附、释放延迟、接触期间相对漂移、身体抖动和脚滑。物体轨迹固定时，只能报告不破坏参考轨迹，不能称改善对象动力学。MPJPE/JPE 可保留作辅助约束；对于多解生成，单一 GT 的距离不能独自代表自然度。动作质量还需完整人体视频和更大样本的生成评价。

50 mm 接触阈值可为历史可比性保留，但不宜独自承担精确接触结论。提前定义更严格的距离分布与阈值敏感性报告，避免在 validation 上选择最漂亮的阈值。置信区间按序列组织；有多物体时同时报告对象分层。B 的物体留出 split 与 L1 的序列 cohort 是不同证据，不能交叉借用“未见物体泛化”称号。

本周 validation 已用于多次设计选择，即使没有读取 test，也已承担开发集角色。下一次确认实验应在冻结方案后使用未参与选择的数据；不必因为某次 pilot 为正就立即消耗最终 test。canonical analytic rerun 已有 harness，本报告将其视作现有工作的收尾，不另立一个重复实验方向。

已知资源是本地同步材料和历史单卡 4090 级实验；服务器当前关闭。现有条件适合小型预测器、冻结主干和受控局部 refinement。没有依据承诺完整 humanoid RL、通用触觉模型或新手指数据管线能在同一短周期完成。用户的截止日期与每周投入未提供，本报告不编造完成周期。

## 6. 学术品位与最终评价

按研究价值看，当前最值得追求的是有效性和稳健性：真实接触质量是否改善，事件切换和未见对象上是否保持。速度、标注成本降低和跨生成器通用性目前没有实验支持，不宜提前写成贡献。方法若最终是一种小而清楚的事件驱动 refinement，也可以有研究价值；是否叫 WM 应由预测任务和使用方式决定。

论文不应承担复述所有试验的任务。SDF、截面、拓扑、对比学习和对象响应可以作为研究过程中的资产与排除依据，最终正文只保留解释主机制所需的组件和关键对照。当前最值得保存的是可靠几何接口、接触评价、明确的解析基线、局部几何 B，以及可复现的分支停止记录。

对 RQ1 的回答：本周完成了可复用实验体系，并取得了局部几何选择和事件表征两类有内容的正结果；负结果也帮助排除了几种看似自然、实际无额外收益的实现。

对 RQ2 的回答：当前 +7.78 pp 应归因解析几何 action-chunk selection；B 的事件排序改善属于学习表征；learned residual 的当前实现退化。它们尚未组成已验证的 learned WM 全身改进。

对 RQ3 的回答：值得继续检验“未来交互信息能否提前改善上肢—躯干协调”，前提是先落实身体可实现性，并公平超过同预算解析与短视对照。InterDreamer、HOI-Dyn、HOIDiNi 和 MaMi 本身是必须正面对照的近邻。当前评价为**值得继续、需要收敛问题并验证机制**；对已失败的具体 residual/contrastive 版本则维持停止。

你希望通过一步步实验把方法打磨出来，这个方向合理。下一步最有价值的进展，是让一项明确的预测在完整身体上产生可见、可测、能归因的改善。

## 参考文献与证据说明

正文优先引用一手论文。进一步的作者、年份、版本、会话与方法细节见本次审计目录中的 `history-literature.md`、`geometry-evidence.md` 和 `wm-evidence.md`。引用 2026 年预印本仅表示检索到相应方法，不意味着其结论已独立复现。

1. Hao Wang, Shiqi Wang, Qi Liu. *MaMi-HOI: Harmonizing Global Kinematics and Local Geometry for Human-Object Interaction Generation*. 2026. [论文](https://arxiv.org/abs/2605.05756)。
2. Sirui Xu, Ziyin Wang, Yu-Xiong Wang, Liang-Yan Gui. *InterDreamer: Zero-Shot Text to 3D Dynamic Human-Object Interaction*. 2024 初稿，本次查阅 2026 年 v2. [论文](https://arxiv.org/abs/2403.19652)。
3. Lin Wu, Zhixiang Chen, Jianglin Lan. *HOI-Dyn: Learning Interaction Dynamics for Human-Object Motion Diffusion*. 2025. [论文](https://arxiv.org/abs/2507.01737)。
4. Roey Ron, Guy Tevet, Haim Sawdayee, Amit H. Bermano. *HOIDiNi: Human-Object Interaction through Diffusion Noise Optimization*. 2025 初稿. [论文](https://arxiv.org/abs/2506.15625)。
5. Christian Diller, Angela Dai. *CG-HOI: Contact-Guided 3D Human-Object Interaction Generation*. CVPR 2024. [论文](https://arxiv.org/abs/2311.16097)。
6. Dongting Li et al. *HAIC: Humanoid Agile Object Interaction Control via Dynamics-Aware World Model*. RSS 2026. [会议论文页](https://www.roboticsproceedings.org/rss22/p013.html)。
7. Quankai Gao et al. *LOME: Learning Human-Object Manipulation with Action-Conditioned Egocentric World Model*. 2026. [论文](https://arxiv.org/abs/2603.27449)。
8. Chaonan Ji et al. *Real-Time Human-Centric World Modeling for Upper-Body Human-Object Interaction*. 2026. [论文](https://arxiv.org/abs/2607.23517)。
