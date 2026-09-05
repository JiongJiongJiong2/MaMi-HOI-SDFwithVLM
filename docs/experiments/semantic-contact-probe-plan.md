# U3 语义扩展：最小接触候选重排实验

日期：2026-09-05。状态：**设计方案，尚未实现或运行**。本文件补充[前期文献调查](../research/mllm-hidden-contact-novelty-2026-09-04.md)，纠正其中过强的数据判据与 prompt-swap 判读，不覆盖现有服务器任务。

## 1. 决策与任务边界

先做独立离线 probe，证明冻结 Qwen hidden feature 相对 action label 和当前 CLIP 是否提供额外接触区域信息。通过后才作为 U3 的可选语义先验；保留 diffusion 主干和现有 CLIP 路径。不要先训练完整 VLM、改 diffusion backbone 或同时加入双手角色、物体响应与截面视觉提示。

本实验的问题是：**在冻结粗预测、相同物体表面候选与相同可用状态条件下，额外语义是否改善逐手接触区域选择？** 第一版只研究 GT 标为接触的有效帧上的位置选择，不声称已经解决接触发生时间、手指接触或全身动作生成。

现有 MaMi coarse prediction 已经受 CLIP 条件影响，所以这里的 geometry-only 指“局部重排器不接收额外语义”，不代表整个生成器没有语言。离线命中率提升只是接入资格，不是最终 HOI 提升证据。

## 2. 对文献判断的修正

[VideoAfford](https://arxiv.org/abs/2602.09638) 从 HOI 视频学习，但输出是 3D actionable regions；不能仅凭输入有人的动作，就归类为 actual-contact trajectory generation。[HAMMER](https://arxiv.org/abs/2603.02329) 同样定位于 intention-driven 3D affordance grounding。两者构成机制相关工作，而不是 MaMi 全身生成的完全同任务复现。

但“affordance 不等于实际接触”本身也不足以建立新颖性：[Text2HOI](https://arxiv.org/abs/2404.00562) 已从文本和 mesh 生成 contact prior，再用于手物运动 diffusion 和接触/穿透 refinement；[JointHOI](https://arxiv.org/abs/2607.01768) 已联合生成文本条件的手物运动与动态距离 contact maps。两者是 hand-object 任务，不应再写成 MaMi 式 full-body，但其机制重叠必须认真比较。

因此本项目只能暂时提出待验证问题：“预测状态条件下，语义是否消除候选接触歧义”，不能提前主张“首次语义接触”“首次动态 contact”或“已有研究都没有解决实际接触”。

## 3. 已核查的工程基础

工作区 [README](../../../README.md) 明确把 U3 标记为 Planned；当前没有 trainable K-candidate surface refiner。统一计划的 U3 使用 `M=64` canonical anchors、逐手 predicted query、独立 surface K/V、key-specific bias，以及 human/hand-contact-only residual。统一计划第 12.1 节已经预留 VLM soft prior，见[原计划](../../../docs/future-plans/surface-contact-response-unified-plan.md)。

当前可复用的代码如下。下表是复用位置，不表示已经完成本实验。

| 当前文件 | 可复用内容与注意事项 |
|---|---|
| [trainer_control_GAPA_chois.py](../../train/trainer_control_GAPA_chois.py) | `encode_text()`：冻结 CLIP 输出 512D。当前最多 30 个内容 token，必须记录截断比例。 |
| [cano_traj_dataset.py](../../manip/data/cano_traj_dataset.py) | sequence/text、object pose、`contact_labels[T,4]`；标签没有 surface region ID。 |
| [sdf_utils.py](../../manip/model/sdf_utils.py) | `project_to_so3`、`world_to_object_points`、`build_dynamic_sdf_prediction_query`；这些 helper 已存在，以代码为准。 |
| [transformer_control_GAPA_motion_cond_diffusion.py](../../manip/model/transformer_control_GAPA_motion_cond_diffusion.py) | dynamic-SDF 路径通过 FK 取预测 palms `22:24`；不能不加区分地换成直接输出的 joint positions。 |
| [run_sectional_prior_diagnostic.py](../../scripts/run_sectional_prior_diagnostic.py) | mesh/固定随机种子表面采样、sequence split 的实现参考；不能照搬其 GT source-hand 作为新 probe 的输入。 |

## 4. 第一阶段：先建立可信的测试集

冻结 baseline checkpoint、原始 sequence split、采样 seed、guidance 配置和输入条件。训练/验证/测试都通过正常采样得到 U0 输出；不要用带 GT future 的 teacher-forced denoiser 输出替代真正的粗预测。所有对照共享同一份预测缓存。已有推理条件如观测首帧或 object waypoints 可以保留，但必须列在 manifest 中。

每条 sequence 缓存如下字段，并把 model input 与监督字段物理分开存放，避免 GT 字段被拼接进特征：

```text
metadata:
  sequence_id, original_sequence_id, window_start, object_id, split
  checkpoint_hash, seed, guidance_config, units, fps, surface_bank_hash
inputs:
  interaction_text
  pred_palm_world[T,2,3]              # 与现有 FK 定义一致
  pred_object_rotation[T,3,3], pred_object_com[T,3]
  pred_hand_contact[T,2], valid_frame[T]
  surface_points[M,3], surface_normals[M,3], surface_ids[M]
  canonical_candidate_ids[T,2,K], candidate_mask[T,2,K]
targets_only:
  gt_contact[T,2], gt_palm_world[T,2,3]
  gt_object_rotation[T,3,3], gt_object_com[T,3]
  gt_patch_mask[T,2,M], target_quality_mask[T,2]
```

`M` 是整物体 surface bank 大小，`K` 是每帧每手检索出来的候选数。第一版可从计划的 `M=64`、候选 `K=32` 开始，同时报告 `K=M` 上限；这些是实验起点，不是经过验证的最优值。先检查 bank 到原 mesh 的覆盖误差和小部件覆盖；若 64 个点表示不了真实接触区域，应增密 bank，再检查局部检索，不要把表示误差当成语义失败。

候选只由预测手、预测物体位姿和已知 mesh 得到。复用 canonical 坐标 helper，先恢复 reference rotation、物理单位和 object COM，再查询。GT 掌心通过 GT object pose 变换到同一个 canonical frame，仅用于构造 target。不要把 GT pose 用于候选检索。

优先用 mesh 上离 GT 掌心最近的连续表面区域生成多正例 patch，再映射到 bank；没有 geodesic 时可以使用欧氏近邻，但薄壁、背面与相邻部件可能混淆，需标记和抽查。当前 palms 是关节 proxy，标签应称 **GT-derived palm-proxy contact patch**，不是人工标注的真实皮肤接触区域。

如果 GT contact=1 但 GT proxy 距离表面远、pose/单位异常，记录为低质量监督；不要强行用最近点制造“可靠接触”。patch 半径用米定义并结合实际标注误差，在训练/验证阶段冻结；跨尺寸结果额外报告归一化距离，不能随测试结果调大命中半径。

第一张结果表应包含独立 sequence/object 数、同物体多动作样本数、每手接触/双手同时接触数、目标质量、bank coverage 和 candidate oracle coverage。候选覆盖定义为：

```text
coverage@K = mean[候选集合与 GT-derived positive patch 有交集]
```

不允许为提高 coverage 偷加 GT 最近点。若为了检查训练代码做 oracle-injected sanity test，必须独立命名，不能出现在正式排行榜。候选未命中的样本是检索失败；训练区域分类时单独 mask，评估总体命中时仍计失败，并额外报告 covered-only 排序结果。

数据是否包含语义信息要看 same-object、近似几何状态下的条件分布，不要求不同动作的接触距离一定大于同动作内方差。“拿杯子”和“喝水”可能使用同一个合法手握区域。缺少多动作数据时，只能说当前数据不足以验证该假设。

注意生成输出可能与唯一示范在动作时相和合法接触策略上不同。本 probe 测试对该示范的拟合，不代表其他接触都不合理；需抽查时序对齐，分别报告高质量可比较子集与全体结果，禁止按某个方法是否获益选择子集。

## 5. 第二阶段：只训练相同的小重排头

先缓存 action label、当前 CLIP 和 Qwen 特征，不训练大模型。语义提取只接收推理时可用的同一份任务文本；如需物体名，所有语义对照一致提供。action label 的归并规则从训练集确定，不能把左右手接触标签混入动作类别。

第一轮使用 text-only，明确它测试的是语言 hidden representation，而不是 VLM 是否理解 3D。不要为了“多模态”输入未来 HOI 视频、GT contact 热图或含答案的轨迹渲染。之后若比较固定 mesh render，必须增加相同图像输入的 CLIP 对照，并保持相同 mesh/frame、视角和分辨率。

Qwen 第一版只选一个固定深度和一种 masked pooling，例如最后有效任务输入 token。输入文本顺序与 token span 要记录；若后来加入图片，所选 token 应能注意到图片和完整指令。冻结 forward 的 hidden states 不等于已经执行了一段推理过程，也不要给未训练的新 `<CONTACT>` token 赋予接触语义。

一次前向可顺带缓存中层特征，但选层只能用 validation。不要第一轮同时搜索四层、三种池化、多视角和双向 cross-attention。用精确模型 ID/revision、prompt hash、layer、pooling、dtype、token mask 作为缓存元数据。

模型接口建议如下：

```text
candidate geometry: [B,T,2,K,Dg]
semantic feature:   [B,Ds]                 # 一段动作缓存一次
prediction state:  [B,T,2,Dh]
candidate mask:    [B,T,2,K]
                  ↓
small pointwise scorer → logits[B,T,2,K]
                  ↓ masked softmax over K
```

`Dg` 包含 canonical 位置、法向、候选相对预测掌心偏移与距离；`Dh` 可包含预测掌心速度、另一只预测手的位置、手 ID 和归一化时间。所有方法获得相同状态。默认不加入 GT 阶段标签和 GT 接触开关；GT contact 仅决定第一版哪些帧参与监督。

一个足够小的结构是 `geometry→128D`、`semantics→128D`，再把两者及其逐元素乘积送入两层 MLP 输出每候选分数。无额外语义组使用相同结构的常量语义输入，避免把“有无额外 MLP”误当成语言收益。匹配 head 深度、训练步数与输出维度，并报告不同输入宽度造成的 projector 参数差；若参数差可疑，再补参数量控制。

对有正例候选的有效接触帧，使用多正例目标，例如：

```text
loss = -log(sum(probability[j] for j in positive_candidates))
```

或使用固定的软 patch target 做交叉熵。不要对无正例候选的行强制 argmin 作为真值，不要在所有表面点之间用高斯距离强行分配微小“正例”掩盖候选 miss。

| 对照 | 第一轮作用 |
|---|---|
| 最近几何候选 / 无额外语义的小网络 | 几何与已有粗预测能做到多少 |
| + action label | 简单动作类别是否已足够 |
| + 冻结 CLIP text | 现有语义表示能否通过局部接入发挥作用 |
| + 冻结 Qwen pooled hidden | 大模型表示是否有额外收益 |

以上是相同几何输入的对照，不是把语义向量直接与 xyz 做 cosine。先把 no-extra/action/CLIP 跑通，再加入 Qwen；CLIP 比 geometry 差并不能逻辑上证明 Qwen 无效，但可以降低后续投入预算。

若长文本存在，额外测试未被当前 30-token 限制截断的 CLIP 原生上下文版本；否则 Qwen 的收益可能仅来自读到了更完整的任务描述。训练时随机打乱语义配对可作为容量/泄漏 sanity control；同物体内打乱用于检测动作信息，解释时要考虑不同文本可能拥有相同合法接触。

## 6. 结果怎么判读

主指标预先选 **sequence-macro Top-1 patch hit**，并同时报告 oracle coverage、covered-only Top-1、预测点到目标 patch 距离；Top-5 作为补充。双手 simultaneous hit 只在双手同时接触且标签可靠的帧上统计。不要把 padding、长序列重复窗口或大量不接触帧算成独立证据。

沿用原始 sequence-disjoint split；所有窗口必须跟随原始 sequence。seen/unseen object 分开报告，若同物体动作组太少，应明确其结果只是 pilot。bootstrap 以原始 sequence 为单位，必要时以 object 为单位；不能对高度相关的逐帧样本作独立 bootstrap。

建议的项目继续线是：相对最强便宜语义基线，预注册主指标增加至少 **3 个绝对百分点**，配对 bootstrap 95% CI 下界高于 0，3 个小头训练 seed 方向一致。数字是控制研究投入的约定，不是领域标准；实际样本不足以收窄 CI 时，结论为证据不足，而非证明无效。不要事后在多个指标中挑唯一获胜项。

如果 Qwen 与 CLIP/动作标签持平，停止这个冻结 Qwen 版本，保留更便宜的方案；这不证明所有微调 MLLM 都无效。如果所有方法都很差，先区分候选 miss、低质量监督、动作不同但接触相同，以及语义确实缺少增量。

prompt swap 仅是辅助诊断：固定预测状态和候选后换文本，可以检查语义敏感性；没有替换任务的有效接触标注，不能判定变化方向正确。对真正需要改变接触的任务对，应根据另一任务的合法区域判分，不能继续拿原 GT 判错。相同接触适用于两种动作时，输出不变是允许的。

## 7. 通过后怎样接入 U3

先验证 U3 本身相对 U2 有增益，再比较 U3、U3+局部 CLIP、U3+局部 Qwen。第一版保留原 CLIP 全局条件，不改物体轨迹，不训练 Qwen，也不把语义模块重新编号为 U4；U4 在现有计划中另有 object correction 含义。

最小接入位置是 U3 的逐 key attention logits。语义对每个 surface key 产生不同分数：

```text
semantic_logits[t,h,j] = scorer(surface_j, predicted_state[t,h], z_sem)
log_prior = log_softmax(semantic_logits, candidate_axis)
attention_logits = u3_geometry_logits + lambda_sem * log_prior
```

与原计划的静态 `pi_j(object,text)` 相比，这里允许先验被当前预测手/时相调制；它是待测扩展，不是已证明的创新。`lambda_sem` 从零基线开始在 validation 校准；先验只能在同一几何有效候选内软重排，不能强制 teleport。对所有 key 加同一语义常数会被 softmax 抵消，无效。

保留 `K>1` 个独立 K/V，后续由 attention context 和 temporal refiner 输出 human/hand-contact residual；不要在相隔很远的两个 surface 区域间直接平均 xyz，因为均值可能在物体内部。保持 FK、全身姿态及接触损失约束。attention weights 也不天然等同于 calibrated contact probability，接触图监督和 attention 解释要分开。

离线 probe 基于最终 U0 sample，而 U3 的查询来自每个去噪步的 coarse x0，二者分布不同。正式接入时必须用对应 denoising states 训练/评估小头，并报告 diffusion timestep 分组结果；不能认为离线分数可以直接无损迁移。候选由当前模型预测产生，不能切回 GT 查询。

这一阶段才评价 contact F1、onset/offset、接触位置、允许滑动与不应滑动区间的 slip、穿透、Hand JPE/MPJPE、整体运动质量和多样性。SDF 只提供几何距离/穿透约束，不保证力学稳定；surface candidates 本身 SDF 都接近 0，不能仅凭这一标量判断哪个部位语义正确。

## 8. 最小代码工作包与交付顺序

下列文件名是**建议新增项，目前不存在**；不提供会误导为可执行的命令。

| 建议文件 | 最小职责与产物 |
|---|---|
| `scripts/export_contact_probe.py` | 读取既有 split，调用冻结 U0 正常采样；写共享候选/监督缓存与 coverage/data-quality 报告。先完成此项。 |
| `scripts/cache_contact_semantics.py` | 缓存 action/CLIP/Qwen 表示，记录输入、截断率和模型版本；Qwen 为可选依赖。 |
| `manip/model/contact_candidate_scorer.py` | 纯小网络，接收已缓存 tensors；不依赖 diffusion 训练和在线 VLM。 |
| `scripts/run_contact_semantic_probe.py` | 同一 split/候选/预算训练各对照，输出逐 sequence 指标、置信区间和失败案例。 |
| `tests/test_contact_semantic_probe.py` | 预测/GT 隔离、canonical round-trip、padding/miss 行、左右手顺序、同一候选复用、无跨序列泄漏。 |

第一轮应交付三样结果：可靠的样本/候选覆盖报告、四组对照的同协议指标、少量成功与失败的 canonical 接触可视化。若数据或标签不支持继续，第一项报告本身即可结束这轮实验，避免先搭全套 VLM 模型。

## 9. 与当前 SDF/G0 的关系及本次验证状态

现有 SDF U0/U1 与 G0 截面几何两项基础实验继续按原协议。可以共用数据、split、物体坐标工具和静态 surface cache；三者不混入同一次最小训练。G0 当前允许已知一只手 GT 接触来研究另一只手的几何条件，它与本 probe 的预测状态输入不同，分数不能直接横向比较。

本 probe 不需要先证明 SDF loss 有效，也不能取代 SDF/G0。若要并行，先利用同一批数据做无训练的数据/候选审计；Qwen 训练筛选为单独任务，不自动加入当前租卡启动脚本。最终研究 SDF 与语义互补时，才固定相同 refiner 做 `无SDF/有SDF × 无额外语义/有额外语义` 的受控对照。

本次只完成代码与文献核查、实验设计。未实现以上脚本、未下载模型、未运行 Qwen/训练/真实数据评估、未查询服务器。本地所查 `data/` 顶层仅见 body-model/part-ID 目录，未获得 processed motion 数据的位置；不据此判断服务器没有数据。旧研究报告保留为历史记录，本文件是其落地判据的后续修正。
