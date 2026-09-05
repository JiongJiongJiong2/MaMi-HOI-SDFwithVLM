# MLLM hidden semantics 用于 MaMi-HOI 接触预测：保守新颖性调查与最小实验

> 调查截止：2026-09-04  
> 结论状态：文献与代码静态核查完成；尚未运行数据统计、特征提取或训练实验。

## 一句话结论

**明确判断：只值得先做一个 1–3 天的小实验，不值得把“MLLM hidden feature → 3D surface/contact”本身当主创新继续投入。** 这条机制已经被多篇论文直接覆盖；2026 年的 HAMMER 和 VideoAfford 尤其接近。仍可能形成论文的问题不是“把 Qwen feature concat 进去”，而是：

> **在同一物体、相近粗手位姿和多种几何上都合理的候选区域之间，高层交互语义能否可靠地消除人类双手真实接触的歧义，并在 SDF 物理约束下改善时序 HOI 生成？**

这个版本把研究对象从“表示组件”改成了“可证伪的 HOI 问题”。如果 MaMi-HOI 数据中不存在足够的“同物体、不同语义、不同接触区域”样本，那么连这个问题也无法在当前数据上成立，应停止 MLLM 分支或补建小型反事实评测集。

## 1. 先把五类 semantic representation 分开

下表中的差异不能在论文中混写。尤其是“生成一段文字后再编码”和“直接读取 MLLM Transformer hidden state”不是同一种方法。

| 类别 | 实际信号 | 代表形式 | 与本想法的关系 |
|---|---|---|---|
| 1. Caption / 自然语言输出 | 离散文本 | VLM 先描述交互，再由文本编码器编码 | 信息经过语言瓶颈，可解释但可能丢细节 |
| 2. CLIP embedding | 对比学习的全局图文向量 | CLIP text/image `[B,D]` | MaMi-HOI 已使用 frozen CLIP text；必须是强基线 |
| 3. VLM encoder feature | 视觉塔或多模态编码器特征 | ViT/Video encoder tokens | 不等于 LLM Transformer 内部状态 |
| 4. MLLM Transformer hidden state | LLM 各层、各 token 的连续状态 | `<AFF>` / `[CONT]` / `<SEG>` hidden token，或完整 token sequence | 用户最关心；已有直接先例 |
| 5. 推理结果再转 feature | CoT、属性、意图等文字/结构化结果 | MLLM 输出属性或 affordance label 后再融合 | GREAT 等采用；也不是原始 hidden state |

## 2. HOMIE 到底做了什么

[HOMIE: Human-object Centric Video Personalization via Multimodal Intelligent Enhancement](https://arxiv.org/abs/2607.18217)（Yiyang Cai, Nan Chen, Rongchang Xie, Junwen Pan, Chunyang Jiang, Cheng Chen, Wen Zhou, Zhenbang Sun, Wei Xue, Wenhan Luo, Yike Guo，2026，arXiv:2607.18217）是视频个性化生成，不是 3D HOI/contact 论文。它的启发价值在“冻结 MLLM、预提取连续条件、再注入生成模型”，不能作为 3D 接触新颖性的证据。

官方 [`generate_mllm_feature.py`](https://github.com/YIYANGCAI/HOMIE/blob/main/generate_mllm_feature.py) 明确显示：模型是冻结的 Qwen3-VL-2B-Thinking；前向设置 `output_hidden_states=True`；取 `outputs.hidden_states[-1]`；从 vision-start token 开始保留后续完整 token 序列，最多 `1024 × 2048`，并保存 mask。它**没有先生成文字，也没有把特征平均成一个向量**。所以：

- “HOMIE-like”严格指 token-sequence conditioning；单个 mean-pooled Qwen 向量只能叫低成本近似 probe。
- HOMIE 的结果不能证明其最后一层最适合几何，也不能证明这个特征比 CLIP/action label 更适合接触预测。

## 3. 是否已经有人做过：有，而且覆盖很直接

### 3.1 最接近的直接先例

| 论文 | hidden state 如何使用 | 输出 | 与当前设想的重合度 |
|---|---|---|---|
| [HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding](https://arxiv.org/abs/2603.02329), Lei Yao, Yong Chen, Yuejiao Su, Yi Wang, Moyun Liu, Lap-Pui Chau, CVPR 2026 / arXiv:2603.02329 | Qwen2.5-VL 最后一层全部 hidden states 与 `[CONT]` hidden；point feature 对 hidden tokens 做 cross-attention，再用多尺度几何反向丰富 intention embedding | 2,048 点的 3D affordance map | **几乎覆盖“interaction hidden semantics + point/surface feature + cross-attention + contact-like map”** |
| [VideoAfford: Grounding 3D Affordance from Human-Object-Interaction Videos via Multimodal Large Language Model](https://arxiv.org/abs/2602.09638), Hanqing Wang, Mingyu Liu, Xiaoyu Chen, Chengwei Ma, Yiming Zhong, Wenti Yin, Yuhao Liu, Zhiqing Cui, Jiahao Yuan, Lu Dai, Zhiyuan Ma, Hui Xiong, 2026, arXiv:2602.09638 | HOI video、文本与 latent action tokens 输入 Video-LLaVA；投影 `<AFF>` hidden state，与 dense point features cross-attend | 视频条件的 3D affordance mask | **直接覆盖“HOMIE 类视频 hidden prior → 3D 表面区域”** |
| [3D-AffordanceLLM: Harnessing Large Language Models for Open-Vocabulary Affordance Detection in 3D Worlds](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b364c8cbb3f229f40d5873e877391bd2-Abstract-Conference.html), Hengshuo Chu, Xiang Deng, Qi Lv, Xiaoyang Chen, Yinchuan Li, Jianye Hao, Liqiang Nie, ICLR 2025 | 取最后一层 `<AFF>` hidden embedding，MLP 投影，与 dense point features 输入 affordance decoder | per-point 3D affordance mask | hidden-token-to-3D-surface 已直接完成 |
| [SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model](https://openaccess.thecvf.com/content/CVPR2025/html/Yu_SeqAfford_Sequential_3D_Affordance_Reasoning_via_Multimodal_Large_Language_Model_CVPR_2025_paper.html), Chunlin Yu, Hanqing Wang, Ye Shi, Haoyang Luo, Sibei Yang, Jingyi Yu, Jingya Wang, CVPR 2025 | 取一个或多个最后层 `<SEG>` hidden embeddings；与 dense/sparse point features 多粒度融合 | 单步/顺序 3D affordance masks | hidden token + 3D tokens + dense grounding 已直接覆盖 |
| [Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions](https://openaccess.thecvf.com/content/CVPR2025/html/Zhu_Grounding_3D_Object_Affordance_with_Language_Instructions_Visual_Observations_and_CVPR_2025_paper.html), He Zhu, Quyu Kong, Kechun Xu, Xunlong Xia, Bing Deng, Jieping Ye, Rong Xiong, Yue Wang, CVPR 2025 | LMAffordance3D 冻结 LLaVA-v1.6-Vicuna-7B；空间特征作 query，instructional/semantic features 作 key/value | 2,048 点 3D affordance heatmap | 连“冻结 VLM + hidden semantics + point-query cross-attention”也已覆盖 |

HAMMER 是最致命的先例，因为它不仅提取 hidden state，而且明确把其称为 contact-aware intention embedding，并做双向的 semantic–geometry 融合。VideoAfford 又把输入推进到了动态 HOI 视频。因此，“把 HOMIE 的 hidden feature 迁移到 3D surface”在 2026-09 已不能作为独立方法贡献。

### 3.2 HOI、双手和邻近路线

[AffordanceLLM: Grounding Affordance from Vision Language Models](https://openaccess.thecvf.com/content/CVPR2024W/OpenSUN3D/html/Qian_AffordanceLLM_Grounding_Affordance_from_Vision_Language_Models_CVPRW_2024_paper.html)（Shengyi Qian, Weifeng Chen, Min Bai, Xiong Zhou, Zhuowen Tu, Li Erran Li，CVPR Workshop 2024）更早已经用 LLaVA 特殊 mask token 的 hidden state 驱动 2D affordance decoder。

[HandsOnVLM: Vision-Language Models for Hand-Object Interaction Prediction](https://arxiv.org/abs/2412.13187)（Chen Bao, Jiarui Xu, Xiaolong Wang, Abhinav Gupta, Homanga Bharadhwaj，2024，arXiv:2412.13187）让 VLM 同时产生文字和未来左右手 2D 轨迹，是直接的 VLM→HOI prediction，但不是 frozen hidden auxiliary，也没有三维表面接触。它报告的 2D、遮挡、快速运动和视野外目标限制说明，高层语言能力不能自动替代低层几何观测。

[2HandedAfforder: Learning Precise Actionable Bimanual Affordances from Human Videos](https://openaccess.thecvf.com/content/ICCV2025/html/Heidinger_2HandedAfforder_Learning_Precise_Actionable_Bimanual_Affordances_from_Human_Videos_ICCV_2025_paper.html)（Marvin Heidinger, Snehal Jauhri, Vignesh Prasad, Georgia Chalvatzaki，ICCV 2025）使用 LLaVA `[SEG]` hidden embedding 与 SAM 图像特征，通过两个 mask decoder 输出左右手 affordance 区域。它已覆盖“语义条件的双手区域”，但输出是 2D mask，不是时间连续的人体三维实际接触。

[Task-Aware Bimanual Affordance Prediction via VLM-Guided Semantic-Geometric Reasoning](https://arxiv.org/abs/2604.08726)（Fabian Hahne, Vignesh Prasad, Georgia Chalvatzaki, Jan Peters, Alap Kshirsagar，2026，arXiv:2604.08726）把多视角 RGB-D、6-DoF 抓取候选、VLM 筛选和 arm allocation 联合起来，并在双臂机器人上覆盖稳定、工具使用和 handover。它不是 frozen-hidden auxiliary，但已直接说明“task semantics → 接触区域 + 左右臂分配”也不是空白问题。

[BimArt](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_BimArt_A_Unified_Approach_for_the_Synthesis_of_3D_Bimanual_CVPR_2025_paper.html)（Wanyue Zhang, Rishabh Dabral, Vladislav Golyanik, Vasileios Choutas, Eduardo Alvarado, Thabo Beeler, Marc Habermann, Christian Theobalt，CVPR 2025）先生成由物体轨迹条件化的双手 distance contact maps，再生成手运动；[Stabilize to Act](https://arxiv.org/abs/2309.01087)（Jennifer Grannen, Yilin Wu, Brandon Vu, Dorsa Sadigh，CoRL 2023 / arXiv:2309.01087）显式区分 stabilizing arm 与 acting arm。这两项分别说明 paired-contact prior 和双臂角色分解已有先例，但尚未完成 MaMi-HOI 式全身时序生成下的语义反事实验证。

作为表示类别对照，[GREAT](https://openaccess.thecvf.com/content/CVPR2025/html/Shao_GREAT_Geometry-Intention_Collaborative_Inference_for_Open-Vocabulary_3D_Object_Affordance_Grounding_CVPR_2025_paper.html)（Yawen Shao, Wei Zhai, Yuhang Yang, Hongchen Luo, Yang Cao, Zheng-Jun Zha，CVPR 2025）把 MLLM 推理出的几何属性和交互意图变成 affordance knowledge 后再融合，属于“推理结果→feature”；[VGent](https://openaccess.thecvf.com/content/CVPR2026/html/Kang_VGent_Visual_Grounding_via_Modular_Design_for_Disentangling_Reasoning_and_CVPR_2026_paper.html)（Weitai Kang, Jason Kuen, Mengwei Ren, Zijun Wei, Yan Yan, Kangning Liu，CVPR 2026）则在 2D grounding 中让候选框 cross-attend 到冻结 MLLM hidden states，证明模块化“reasoning encoder + candidate selector”也不是新的通用架构。

## 4. 文献真正揭示的问题，而不是想象的问题

第一，**语义表示仍会歧义，并不会天然准确落到细小表面**。HAMMER 的补充材料报告了 bag zipper 只覆盖一小段、chair 的 armrest 被误认为可坐区域；作者明确归因于 interaction image 导出的 intention embedding 有歧义，以及意图与几何对不齐。这正支持你的“几何与语义之间仍有 gap”，但同时说明“加 hidden state”本身不是解决方案。

第二，**完全冻结的 MLLM 未必够用**。HAMMER 对同一架构做了 LoRA 与全冻结对照：seen aIOU 从 22.20 降到 19.67，unseen aIOU 从 13.71 降到 7.55。也就是说，frozen Qwen 很适合做低成本信号检测，却没有证据保证它是最终最强表示。

第三，**动态信息和空间连续性需要显式建模**。VideoAfford 报告：简单编码若干视频帧不够；latent action encoder 和 spatial loss 都有增益。采样 8 帧优于 2/4 帧，而 16 帧反而明显下降，说明“输入更多 token”会引入冗余。对你的项目而言，单个 pooled feature 很可能只能说明动作类别，未必能说明哪个时刻、哪只手、哪个表面 patch。

第四，**affordance mask 不等于物理 contact**。上述方法多预测“可操作区域”，通常一个区域包含许多潜在接触点；MaMi-HOI 需要的是由人体可达性、粗手轨迹、左右手协同、物体运动以及 SDF 可行性共同决定的实际接触。这个差异是仍可研究的空间，但必须用真实/伪真实 contact target 和生成结果证明，不能只换术语。

第五，**数据比 feature 更可能成为瓶颈**。如果同一 cup 的训练样本只出现 “pick”，或者 pick/drink/pass 的 GT 掌心落点没有显著区别，模型不可能从当前监督中学到你想证明的歧义消除。先做数据 headroom 统计，优先级高于接入 Qwen。

## 5. 新颖性分级

| 提法 | 等级 | 保守判断 |
|---|---:|---|
| MLLM hidden state 用于 affordance/contact-like surface prediction | Level 0 | HAMMER、VideoAfford、3D-AffordanceLLM、SeqAfford 已直接覆盖 |
| frozen hidden tokens 与 3D points/surface tokens 做 cross-attention | Level 0 | LMAffordance3D 已冻结 VLM；HAMMER/VGent 也覆盖同类模块化融合 |
| 在 MaMi-HOI 中把 CLIP 换为 Qwen hidden vector | Level 1 | 具体代码组合少见，但属于工程替换，不能单独支撑方法论文 |
| 在 MaMi-HOI 中离线重排粗预测附近的 surface candidates | Level 1 | 应用到实际人体 contact 的目标更窄，但重排/grounding 机制已有 |
| 同物体反事实语义下，预测 hand-specific、temporally coherent、physically feasible paired contact，并证明改善 HOI generation | Level 2 | 问题定义、控制变量和终端生成评测可能构成贡献；尚不能仅凭检索宣称无人做过 |

推荐的论文问题名是：

> **Counterfactual Semantic Disambiguation of Paired Surface Contacts for 3D Human–Object Interaction Generation**

更短的模块名可用 **Interaction-Conditioned Paired Contact Prior**。不建议把主标题写成 “Semantic-conditioned Sparse Surface Retrieval”，因为 3D semantic–surface retrieval 已相当拥挤；“Interaction-aware Geometric Contact Refinement”也太宽，容易被审稿人归入已有 affordance/contact refinement。

## 6. hidden layer selection：假设部分成立，但不能固定层号

[How Multimodal LLMs Solve Image Tasks](https://arxiv.org/abs/2508.20279)（Zhuoran Yu, Yong Jae Lee，2025，arXiv:2508.20279）在 LLaVA-1.5、LLaVA-Next-LLaMA-3 和 Qwen2-VL 上发现大致阶段：早层偏视觉 grounding，中层偏 lexical integration/semantic reasoning，末层偏输出准备；但每个阶段的具体层范围会随 base LLM 改变。[Cross-modal Information Flow in Multimodal Large Language Models](https://openaccess.thecvf.com/content/CVPR2025/html/Zhang_Cross-modal_Information_Flow_in_Multimodal_Large_Language_Models_CVPR_2025_paper.html)（Zhi Zhang, Srishti Yadav, Fengze Han, Ekaterina Shutova，CVPR 2025）得到相近的信息流结论。[Probing Multimodal Large Language Models for Global and Local Semantic Representations](https://aclanthology.org/2024.lrec-main.1142/)（Mingxu Tao, Quzhe Huang, Kun Xu, Liwei Chen, Yansong Feng, Dongyan Zhao，LREC-COLING 2024）还发现，中间层对全局语义/视觉语言蕴含可能优于最顶层。

因此，“最后层可能太贴近输出，中层可能更适合语义—几何 grounding”有间接依据，但没有论文证明它在 MaMi-HOI contact 上成立。最小 ablation 应按**归一化深度**取 `25% / 50% / 75% / last`，而不是跨模型固定 `6/12/18`。另需注意：[Seeing Before Answering](https://arxiv.org/abs/2608.16263)（Ruchen Liu, Yi Yang, Yiming Xu, Michael Ying Yang, Monika Sester, Bodo Rosenhahn，2026，arXiv:2608.16263）研究的是 vision tower 层选择，显示最佳视觉层也依任务/骨干变化；它能支持“必须选层”，但不能直接证明 LLM hidden layer 的最佳位置。

建议每层只比较三个池化方式：prompt/action token span mean、最后有效输入 token、learned attention pooling。HOMIE 式完整 token sequence 留给后续 cross-attention，不要在第一天就训练。所有层必须来自一次前向并离线缓存，避免把计算预算混入模型对比。

## 7. MaMi-HOI 代码核查：真实插入点与错误前提

当前 checkout 的 README 将项目描述为 [MaMi-HOI](https://arxiv.org/abs/2605.05756)（Hao Wang, Shiqi Wang, Qi Liu，2026；仓库标注 ICML 2026）。代码事实如下。

文本路径在 [trainer_control_GAPA_chois.py](../../train/trainer_control_GAPA_chois.py) 中加载并冻结 CLIP ViT-B/32，`encode_text` 返回 `[B,512]`；训练循环在约 816 行得到该特征。模型中的 `clip_encoder` 实际只是 `Linear(512,512)`，随后在 [transformer_control_GAPA_motion_cond_diffusion.py](../../manip/model/transformer_control_GAPA_motion_cond_diffusion.py) 约 284–286 行把 language embedding 加到 diffusion timestep embedding 上。它是全局语义条件，不是 contact-local feature。

物体路径把 `1024 × 3` BPS delta 展平，通过 `3072→512→256` MLP 得到一个全局向量，再在时间轴重复到所有帧。GAPA 在 [control_GAPA_transformer_module.py](../../manip/model/control_GAPA_transformer_module.py) 约 258 行接收 `[B,T,D]` 的 motion 与 BPS features，attention 的 key 轴是时间 `T`，不是 1,024 个 surface points。虽然 dataset 在 `use_object_keypoints` 下可重建 1,024 个近似表面点并随机取 100 个，但这些点用于 object FK loss，不是候选检索器。

手部没有独立 learned hand branch。220 维输出由 12 维物体、204 维人体和 4 维 contact flags 构成；FK loss 使用 joints 20–23，dynamic SDF 明确查询 palms 22–23。当前表示没有手指，因此任何实验只能声称 palm/wrist-level contact，不能声称 fingertip contact。

contact 监督是 `[T,4] = left hand, right hand, left foot, right foot` 的开关标签；它不包含 surface region ID。dynamic SDF 在 canonical object field 中查询预测/GT 掌心，监督 penetration、contact distance 和 trajectory ranking，但不保留“靠近的是哪个 vertex/part”。推理 guidance 对 mesh vertex 距离取 `min` 后也丢掉 vertex identity。

结论是：**主网络中没有用户描述的“已有 K 个 candidate surface regions 再 refinement”路径。** [sectional_prior.py](../../manip/geometry/sectional_prior.py) 的 `rank_surface_candidates` 是可复用的离线诊断工具，不是训练中的候选模块。因此最小实验应新建离线 reranker，而不是声称只给已有 refinement 加一个 MLP。

## 8. 最推荐的 1–3 天最小实验

### 8.1 Gate -1：先证明数据里真的有 semantic headroom

不训练任何网络。对每个 object identity，找至少两个不同 interaction text/action 的 sequence；在 GT contact frames 中，把左右掌心用 GT object pose 变换到 canonical object frame，再投到固定的 1,024 个 BPS-derived surface points。统计：同一 action 内 contact centroid/patch 的方差、不同 action 间距离、每个 object-action 的样本量以及左右手分别的分布。

只有满足以下两个条件才继续：存在足量 same-object/multi-action 组；并且 between-action surface separation 明显大于 within-action variation。如果不满足，当前 MaMi 数据不能验证“pick/drink/pass/wash 异义”，MLLM 再强也只能学 action label 或数据偏差。

### 8.2 Experiment 0：冻结特征 probe，只作 sanity check

输入采用 `interaction text + object name`；若已有同一标准视角的 mesh render，可增加一张 canonical render，否则先做 text-only，明确它验证的是语言语义而非 multimodal geometry。冻结 Qwen3-VL-2B，单次前向缓存 25%、50%、75%、last 四层 hidden states，分别做 prompt-span mean、last-input-token 与 attention pooling。

不要只用 t-SNE 看图。主 probe 是在 sequence-grouped split 上，从 hidden feature 预测**同一 object 内的 contact patch/cluster**；action classification 只能证明模型识别了动词，不能证明对 contact 有用。可记录 intra/inter-class cosine、k-NN 与线性 probe，PCA/t-SNE/UMAP 仅用于检查坏样本、prompt 泄漏和 token 选择。

### 8.3 Experiment 1：离线 K-candidate semantic reranking

这是最推荐的可决策实验，数据流为：

```text
input:
  interaction text (+ optional fixed mesh render)
  frozen U0 coarse palm prediction
  canonical object surface points/normals
        ↓
semantic extractor:
  frozen Qwen3-VL-2B, hidden layer l, masked pooling
        ↓ z_sem [D]
candidate proposal:
  K nearest / diverse surface points around coarse predicted palm
        ↓ {x_i, n_i, d_i, hand_id}, i=1..K
feature/fusion:
  q = W_s z_sem
  g_i = MLP_g[x_i, n_i, x_i-h_coarse, ||x_i-h_coarse||, hand_id]
  score_i = <LN(q), LN(g_i)> / tau + alpha * geometry_score_i
        ↓
output:
  probability distribution over K candidates for each hand
        ↓
loss:
  soft cross-entropy / KL to a Gaussian surface target around GT palm projection
```

候选必须由**冻结 U0 的预测掌心**或可用的观测前缀产生，标签才由 GT palm 产生；若用 GT palm 同时提候选和打分，会把答案泄漏给 geometry branch。第一项指标不是模型分数，而是 `candidate recall@K`：GT patch 不在候选集内时，任何 reranker 都不可能修复。建议先用 `K=32`，若 oracle recall 低则扩大到 64/128 或改用全 1,024 点。

单点最近邻标签对 mesh/BPS 误差太敏感，应用 object-scale-normalized 半径生成软标签，例如 `y_i ∝ exp(-d_surface(i, GT)^2 / 2σ²)`。如果没有 mesh geodesic，第一版用 canonical Euclidean distance，但报告中必须叫 pseudo-contact patch，不叫真实 contact annotation。

### 8.4 公平 baselines、metrics 与 split

所有方法必须共享同一候选集、同一 geometry encoder、同一投影维度和近似参数量；除小 projector/reranker 外全部冻结。推荐顺序如下。

| baseline | 要回答的问题 |
|---|---|
| Random / geometry-only | 候选先验与粗掌心本身能做到多少 |
| Action label embedding | 一个离散动作是否已经足够；这是最重要的便宜对手 |
| Frozen CLIP text | 当前 MaMi-HOI 语义表示的直接基线 |
| Caption → CLIP text | 生成文字的离散瓶颈是否有增益；有真实图像/视频时才做 |
| CLIP image 或 CLIP image+text | 提升是否只来自视觉外观 |
| Qwen 25/50/75/last pooled hidden | hidden layer 是否真的带来更好的 contact grounding |
| Qwen hidden + learned projector | 确保维度/空间不匹配不会让 raw cosine 被不公平淘汰 |

主指标使用 `Top-1/Top-5 patch recall`、MRR、预测点到 GT patch 的 object-normalized surface distance，以及左右手同时命中的 `paired success rate`。若输出软热图，再报 aIOU/SIM/MAE 便于与 affordance 文献对齐。split 必须按原始 sequence 分组，不能让同一长序列的窗口跨 train/test；核心子集还应按 same-object different-action 构造。

最关键的控制是 **counterfactual prompt swap**：固定同一 object、同一 coarse palm 与同一 candidates，只把正确 interaction text 换成该 object 的另一个 action。若输出几乎不变，所谓 semantic branch 实际没有使用语义；若变化很大但远离 GT，则是语言幻觉而不是消歧。

### 8.5 预注册 go/no-go

建议把“继续”定义为：Qwen hidden 在 same-object ambiguity 子集上同时超过 geometry-only、action label 和 CLIP text；Top-1 至少提高约 3 个绝对百分点或 paired success 提高约 5 个百分点，并且 sequence-level bootstrap 95% CI 不跨 0。阈值不是领域定律，而是防止把随机波动包装成贡献的项目决策线。

若 action label 与 Qwen 持平，结论应是“当前监督下 MLLM 没提供额外 signal”，停止 MLLM 主线；若 CLIP 与 Qwen 持平，保留便宜 CLIP；若只有 Qwen 赢，再进入 token-sequence cross-attention。Experiment 0 的聚类图再漂亮，也不能单独通过 go gate。

## 9. Experiment 2/3 何时才值得做

只有 Experiment 1 明确胜出，才把 pooled vector 升级为 token sequence：让 per-surface point tokens 作 query、Qwen masked tokens 作 key/value，或者使用双向 block。但这已非常接近 HAMMER、SeqAfford、LMAffordance3D，贡献必须来自 **human-motion-conditioned target、时序一致性和生成端效果**，不能来自 cross-attention 本身。

双手版本比单手更有论文空间。先为左右手分别输出 `P_L(i)` 与 `P_R(j)`，再加入一个轻量 pair scorer：

```text
S(i,j) = s_L(i) + s_R(j)
       + MLP_pair[g_i, g_j, x_i-x_j, normal_i, normal_j, z_sem]
```

监督由同步 GT 左右掌心 patch 得到；指标使用 both-hands-within-threshold、左右手交换错误率、pair surface distance 和 temporal switch rate。若文本/数据能可靠区分 stabilize/operate，再加 role head；否则不要由模型自我生成 role 当作真值。当前 4 维 contact flags 没有 role label，需要人工标一个小子集、从动作学弱标签，或只研究 paired geometry 而不声称角色识别。

## 10. 与 SDF 的关系：一起做论文，但不要在最小实验里混做

SDF 回答“当前位置是否在表面、是否穿透、是否满足接触距离”；semantic reranker 回答“多个都在表面且都可达的区域中，任务应该选哪一个”。二者理论上互补，但若第一轮同时加入，结果无法归因。

正确顺序是：先做独立离线 semantic reranking；若通过，再做 `base / +SDF / +semantic / +SDF+semantic` 的 2×2 factorial。最终可把 semantic heatmap 变成预测掌心到目标 patch 的加权 attraction loss，SDF 继续承担 penetration/contact feasibility。这样论文叙事是“semantic selects where; SDF enforces feasibility”，而不是把两个 loss 混成一个无法解释的提升。

## 11. 发表价值判断

**只作为辅助 ablation：值得。** 成本低，而且能直接回答 Qwen hidden 是否超过现有 CLIP；负结果也能避免继续投入。

**作为 secondary contribution：有条件值得。** 必须在 same-object ambiguity、counterfactual swaps、sequence-grouped split 下稳定超过 action/CLIP，并在最终 HOI generation 中改善接触而不恶化 motion naturalness、penetration 和多样性。仅有候选 Top-1 提升仍偏弱。

**作为 main contribution：当前版本不够。** HAMMER 和 VideoAfford 已直接覆盖核心机制，简单 concat、MLP reranking 或 cross-attention 很难支撑 CCF-C 正文。若构建一个可靠的 human-specific paired contact benchmark/protocol，并证明语义消歧、时序一致性与 SDF 可行性共同改善生成，才可能成为 workshop/较小会议的主线；再往上需要更强数据、模型与真实/物理验证。

最终决策是：**只值得做小实验。** 先跑 Gate -1 与 Experiment 1；通过才继续。当前最应该避免的，是先花一个月复刻已经由 HAMMER/VideoAfford 做过的 hidden-state-to-point cross-attention。

## 12. 当前验证边界

本报告检查了论文正文/补充材料和 HOMIE 官方提取代码，并静态审查了当前 MaMi-HOI checkout。尚未获得本地 processed dataset，因此没有验证 same-object/multi-action 样本量、候选 recall、GT 掌心投影质量，也没有运行 Qwen、训练 probe 或复现任何论文数值。代码路径判断是当前 checkout 的事实，不代表上游仓库未来版本。
