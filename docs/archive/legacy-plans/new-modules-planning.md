# MaMi-HOI 优化方案：手部局部超分 + SDF-VLM三模态对比学习

## Context

MaMi-HOI (ICML 2026) 是一个基于扩散模型的文本+几何条件HOI生成系统，输出220维/帧（12物体+204人体+4接触）。当前存在两个核心精度瓶颈：
1. **手部精度不足**：24个关节点表示无法捕捉手指级细节，手-物接触区域几何模糊
2. **语义-几何对齐弱**：CLIP文本嵌入仅通过简单加法注入timestep embedding，无显式对比约束；SDF仅作为GAPA注意力偏置，无训练时几何损失

现有实验计划（Exp 1-4）聚焦SDF查询效率和VLM关键点提取，但缺少**精度增强**和**多模态对齐**这两个维度。本方案补充这两个优化方向。

---

## A. 手部局部超分模块 (Hand Local Super-Resolution)

### A1. 问题分析

当前MaMi-HOI的人体表示是24个关节位置（每帧72维），手部仅有2个关节点（左手腕/右手腕，indices 20-21），无法表达手指姿态。虽然22个关节有6D旋转表示（包含手指关节的旋转变量），但：
- 扩散模型的L1 loss对所有维度平等对待，手部这种高频细节区域信噪比低
- GAPA的SDF注意力仅在全局特征层面调制，无法对手部局部区域做精细化

### A2. 三种策略对比分析

---

#### 策略一：联合训练 (Joint Training)

**原理**：在扩散模型训练循环中，主模型生成x_coarse后，HandSRNet立即精修手部维度，两者共享梯度反向传播。

```
训练流程:
x_noisy → Diffusion Model → x_coarse [B,T,220]
                                    ↓
                           HandSRNet(hand_dims, sdf, bps_local)
                                    ↓
                           x_refined = x_coarse + α * hand_residual
                                    ↓
loss = L1(x_refined, x_gt) + λ * sdf_penetration_loss(x_refined)
```

**优点**：
- 端到端优化，主模型和超分模块互相适应——主模型可能学会生成"更容易被超分修正"的粗结果
- 只需一次训练，无需多阶段
- 共享SDF和BPS特征，无重复计算

**缺点**：
- 修改训练循环，增加训练复杂度
- 超分模块的梯度可能干扰主扩散模型的学习（尤其在早期训练阶段）
- 需要仔细平衡主loss和手部SR loss的权重
- 主模型和SR模块耦合，无法独立ablation

**适用场景**：当你希望两个模块深度协作、追求最优整体效果时

**关键实现**：
```python
# 在trainer的forward_pass中
x_coarse = self.model(x_noisy, t, conditions)  # 主模型
if self.use_hand_sr:
    hand_residual = self.hand_sr_net(extract_hand(x_coarse), sdf_feat, bps_local)
    x_refined = x_coarse + self.hand_sr_alpha * hand_residual  # α可学习
    loss_diff = F.l1_loss(x_refined, x_gt)  # 用refined结果算diffusion loss
    loss_sdf_pen = sdf_penetration_loss(x_refined, object_sdf)
    loss_total = loss_diff + w_sdf * loss_sdf_pen
```

---

#### 策略二：后处理训练 (Post-hoc Training)

**原理**：先完整训练主扩散模型（冻结权重），然后用GT vs Coarse的残差单独训练HandSRNet。

```
阶段1: 训练主模型 (不变)
x_noisy → Diffusion Model → x_coarse
loss = L1(x_coarse, x_gt)

阶段2: 冻结主模型，训练SR网络
用训练好的主模型生成 x_coarse
hand_residual_gt = x_gt[hand_dims] - x_coarse[hand_dims]
HandSRNet(hand_dims_of_x_coarse, sdf, bps_local) → hand_residual_pred
loss_sr = L1(hand_residual_pred, hand_residual_gt) + λ * sdf_pen
```

**优点**：
- **不影响主模型**：主模型完全不变，零风险
- **灵活**：可以在不同checkpoint上分别训练SR，选择最优组合
- **可独立ablation**：超分效果纯粹来自SR模块
- **调试方便**：问题隔离，只看SR模块的residual学习情况
- **复用已有模型**：如果已有训练好的MaMi-HOI，直接在其上训练SR即可

**缺点**：
- 需要两阶段训练，总训练时间更长
- 主模型不知道有SR后处理，无法"配合"SR——可能生成SR难以修正的粗结果
- 需要先用主模型跑一遍推理生成coarse数据，数据管理开销
- x_coarse的分布随训练进展变化，阶段2需要用收敛后的模型

**适用场景**：当你希望零风险、不影响主模型、或主模型已训练好时

**关键实现**：
```python
# 阶段2：生成coarse数据
with torch.no_grad():
    for batch in dataloader:
        x_coarse = ema_model.sample(conditions)  # 冻结主模型推理
        hand_residual_gt = x_gt[hand_dims] - x_coarse[hand_dims]
        save(x_coarse, hand_residual_gt, sdf_data)

# 阶段2：训练SR网络
for batch in sr_dataloader:
    x_coarse, residual_gt, sdf_feat, bps_local = batch
    residual_pred = hand_sr_net(extract_hand(x_coarse), sdf_feat, bps_local)
    loss = F.l1_loss(residual_pred, residual_gt) + λ * sdf_pen(x_coarse + residual_pred)
```

---

#### 策略三：仅推理时SDF梯度引导 (Inference-only SDF Guidance)

**原理**：不训练任何额外网络，完全利用现有SDF数据，在DDIM/DDPM采样的最后几步，对手部关节位置做SDF梯度引导——让手"滑向"物体表面。

```
推理流程 (修改现有apply_different_guidance_loss):
for t in last_K_timesteps:
    x_t = ddim_step(x_t, t)

    # 现有: 全局hand-object interaction guidance
    loss_hoi = compute_hoi_loss(x_t, contact, object_sdf)

    # 新增: 手部局部SDF梯度引导
    hand_joints = x_t[:, :, hand_dims]  # [B, T, 6]
    sdf_values = sample_sdf_at_points(hand_joints, object_sdf)  # [B, T, 4, 1]
    sdf_gradients = compute_sdf_gradients_fd(hand_joints, object_sdf)  # [B, T, 4, 3]

    # 穿透惩罚：推手离开物体内部
    penetration_mask = (sdf_values < 0).float()
    loss_pen = (penetration_mask * sdf_values.abs()).mean()

    # 浮空惩罚：拉手靠近物体表面
    # SDF梯度指向表面外，负梯度指向表面
    contact_mask = (contact_label > 0.5).float()  # 应该接触时
    floating_loss = (contact_mask * F.relu(sdf_values - contact_threshold)).mean()

    loss_hand_sdf = loss_pen * w_pen + floating_loss * w_float
    grad = torch.autograd.grad(loss_hand_sdf, x_t)[0]
    x_t = x_t - guidance_scale * grad
```

**优点**：
- **最轻量**：不新增任何可训练参数，不修改训练流程
- **零训练成本**：完全在推理时生效
- **利用已有代码**：`apply_different_guidance_loss()`已有类似框架，只需扩展
- **`compute_sdf_gradients_fd()` 终于有了用途！**
- **可解释性强**：SDF梯度方向有明确物理含义

**缺点**：
- 效果受guidance_scale敏感，需要仔细调参
- 只能做"推/拉"式修正，无法学习复杂的残差模式
- 推理时间增加（每步额外SDF查询+梯度计算）
- 对浮空手部的引导可能不够精确（SDF梯度在远离表面时方向模糊）
- 本质上仍是"全局引导"的一种，不是真正的超分

**适用场景**：当你希望最小实现成本、快速验证SDF引导对手部精度的改善时

---

#### 🏆 三种策略推荐

| 维度 | 联合训练 | 后处理训练 | SDF梯度引导 |
|------|---------|-----------|------------|
| 实现难度 | ⭐⭐⭐ 中 | ⭐⭐ 低 | ⭐ 最低 |
| 理论上限 | ⭐⭐⭐ 最高 | ⭐⭐ 中高 | ⭐ 中 |
| 训练成本 | ⭐⭐ 中(一次) | ⭐⭐⭐ 高(两次) | ⭐ 最低(无) |
| 风险 | ⭐⭐⭐ 主模型可能受影响 | ⭐⭐ 低 | ⭐ 最低 |
| 可组合性 | ⭐⭐ 中 | ⭐⭐⭐ 好 | ⭐⭐⭐ 好 |

**建议实施路径**：
1. **先实现策略三**（SDF梯度引导）：1-2天即可完成，快速验证SDF对手部精度的改善潜力
2. 如果策略三效果明显但不足 → **升级到策略一**（联合训练），获得更好的精修能力
3. 如果不想冒险修改主模型 → **用策略二**（后处理训练），安全且灵活

---

### A3. 推荐方案的详细设计（策略一：联合训练）

以下以联合训练为例，给出详细架构设计。其他策略的实现可在上述分析基础上调整。

#### 阶段1：关节级手部超分 (Hand Joint SR)

```
输入: 粗粒度生成结果 x_coarse [B, T, 220]
      提取手部相关维度: hand_joints [B, T, 6] (joint 20,21的xyz)
                      hand_rotations [B, T, 12] (joint 20,21的6D rotation)

超分网络: HandSRNet (轻量级Transformer)
  - 输入: hand_features [B, T, 18] + SDF_query_values [B, T, 4] (4个手部关节的SDF值)
  - 条件: object_BPS_local [B, T, 128] (手部附近BPS点集的局部编码)
  - 输出: hand_residual [B, T, 18]
  - 最终: x_refined = x_coarse + alpha * hand_residual (alpha从0学习)
```

**架构选择**：
- 2层Transformer decoder，4 heads，d_model=128
- 交叉注意力：Q=hand_features, K/V=object_BPS_local + SDF_features
- 零初始化输出层，确保初始时 x_refined = x_coarse

**训练**：
- 与主扩散模型联合训练，额外loss：
  ```python
  loss_hand_sr = L1(hand_residual_gt) + lambda_sdf * sdf_penetration_loss(x_refined)
  ```
- `hand_residual_gt` = GT手部维度 - 粗粒度手部维度
- SDF穿透损失：查询refined手部位置在物体SDF中的值，惩罚负值（穿透）

#### 阶段2：顶点级手部超分 (Hand Vertex SR) — 可选扩展

如果关节级超分效果有限，可进一步扩展到SMPL-H手部顶点级（~778个顶点/手）：
- 需要FK层将关节旋转转换为顶点位置
- 使用图卷积网络(GCN)在手部mesh上做局部超分
- 训练数据需要SMPL-H顶点级GT

> **建议**：先实现阶段1，验证效果后再考虑阶段2。阶段1实现成本低，与现有代码兼容性好。

### A3. 与现有模块的交互

```
Main Diffusion Model → x_coarse [B,T,220]
                          ↓
                    HandSRNet (条件: SDF + BPS_local)
                          ↓
                    x_refined [B,T,220]
                          ↓
                    apply_different_guidance_loss() (现有推理时引导)
```

HandSRNet在训练时作为辅助模块，推理时是可选的后处理步骤。不改变主扩散模型的采样流程。

### A4. 关键文件修改

| 文件 | 修改内容 |
|------|----------|
| `manip/model/hand_sr_module.py` | **新建**：HandSRNet定义 |
| `manip/model/transformer_control_GAPA_motion_cond_diffusion.py` | 添加`--use_hand_sr`标志，forward中调用HandSRNet |
| `train/trainer_control_GAPA_chois.py` | 添加hand_sr_loss，修改训练循环 |
| `manip/data/cano_traj_dataset.py` | 添加局部BPS点集数据加载 |

---

## B. SDF-VLM三模态对比学习 (Trimodal Contrastive Learning)

### B1. 可行性分析

**创新性评估**：⭐⭐⭐⭐ 高度创新
- 现有对比学习在HOI领域几乎是空白
- MotionCLIP (2022) 仅做了motion-text双模态对齐
- BRepCLIP (2026) 做了几何-文本-图像三模态，但针对CAD，非HOI
- **三模态对比 (Text + Motion + SDF Geometry) 用于HOI生成是全新的**
- SIGGRAPH 2026的Style-SALAD证明了对比损失在motion diffusion中的有效性

**技术可行性**：⭐⭐⭐⭐ 可行
- 三个模态编码器可以复用现有模块：
  - Text encoder: 已有CLIP (frozen)
  - Motion encoder: 可用主扩散模型的transformer输出
  - Geometry encoder: 可基于SDF特征 + BPS编码
- 负样本构造有天然的数据来源（SDF穿透值、浮空距离、文本错配）

**风险点**：
- 对比学习对batch size敏感，当前batch_size=32可能偏小
- 需要精心设计负样本策略，避免trivial negatives
- 额外计算开销需要评估

### B2. 三模态编码器设计

```python
class TrimodalEncoder(nn.Module):
    """将三个模态编码到共享的d=256嵌入空间"""

    def __init__(self, d_model=512, d_embed=256):
        # Text encoder: 复用已有CLIP (frozen) + 新增projection head
        self.text_proj = nn.Sequential(
            nn.Linear(512, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_embed)
        )

        # Motion encoder: 从主transformer输出池化
        self.motion_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_embed)
        )

        # Geometry encoder: SDF + BPS融合
        self.geo_encoder = nn.Sequential(
            nn.Linear(128 + 256, d_model),  # 128 SDF feat + 256 BPS feat
            nn.GELU(),
            nn.Linear(d_model, d_embed)
        )

        # 温度参数 (可学习)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # 零初始化projection最后层
        self._zero_init_last_layers()

    def encode_text(self, clip_features):
        return F.normalize(self.text_proj(clip_features), dim=-1)

    def encode_motion(self, transformer_output):
        # transformer_output: [B, T, d_model] → 池化到 [B, d_embed]
        pooled = transformer_output.mean(dim=1)  # temporal average pooling
        return F.normalize(self.motion_proj(pooled), dim=-1)

    def encode_geometry(self, sdf_features, bps_features):
        # sdf_features: [B, T, 128] → pool → [B, 128]
        # bps_features: [B, 256]
        sdf_pooled = sdf_features.mean(dim=1)
        geo_input = torch.cat([sdf_pooled, bps_features], dim=-1)
        return F.normalize(self.geo_encoder(geo_input), dim=-1)
```

### B3. 正负样本构造策略

这是整个方案的核心创新点。负样本不是随机采样，而是**结构化构造**的：

#### 正样本 (Positive)
- **自然正样本**：训练数据中的原始配对 (text_i, motion_i, sdf_i)
  - 正确的文本描述 + 对应的GT动作 + SDF显示良好接触（穿透≈0，接触距离<阈值）

#### 负样本类型 (3种结构化负样本)

| 负样本类型 | 构造方式 | 语义含义 |
|-----------|---------|---------|
| **Text-Motion Mismatch** | 同一motion配对其他不相关text (batch内shuffle) | 语义错配：生成"举椅子"的动作配"推桌子"的文本 |
| **Penetration Negative** | 对GT motion添加随机偏移，使手部穿入物体 | 几何穿模：SDF值<0的区域增大 |
| **Floating Negative** | 对GT motion添加远离物体的偏移 | 几何浮空：手-物距离异常增大 |

**具体构造代码思路**：
```python
def construct_negatives(motion, sdf_values, object_sdf_grid, hand_joints):
    """构造三种负样本"""
    B, T, D = motion.shape

    # Type 1: Text mismatch - 在batch内shuffle text即可
    # (InfoNCE自然实现，无需显式构造)

    # Type 2: Penetration negative - 推手部关节穿入物体
    motion_pen = motion.clone()
    # 沿SDF梯度负方向偏移手部关节
    sdf_grad = compute_sdf_gradients_fd(object_sdf_grid, hand_joints)  # 已有函数！
    penetration_offset = -sdf_grad * penetration_scale  # 向物体内部偏移
    motion_pen[:, :, hand_dims] += penetration_offset

    # Type 3: Floating negative - 拉远手部与物体距离
    motion_float = motion.clone()
    floating_offset = sdf_grad * floating_scale  # 远离物体表面
    motion_float[:, :, hand_dims] += floating_offset

    return motion_pen, motion_float
```

> **关键**：`compute_sdf_gradients_fd()` 已经在 `sdf_utils.py` 中实现但从未使用！这个函数现在有了真正的用途。

### B4. 对比损失设计：三种方案对比

---

#### 方案A：全三对对比 (Full Trimodal: T↔M, T↔G, M↔G)

**原理**：三个模态两两之间都做双向InfoNCE，形成3×2=6个对比方向。

```python
def trimodal_contrastive_loss(text_emb, motion_emb, geo_emb, logit_scale):
    """全三模态对比：3对双向"""
    loss_tm = symmetric_info_nce(text_emb, motion_emb, logit_scale)    # 文本↔动作
    loss_tg = symmetric_info_nce(text_emb, geo_emb, logit_scale)      # 文本↔几何
    loss_mg = symmetric_info_nce(motion_emb, geo_emb, logit_scale)    # 动作↔几何
    return loss_tm + 0.5 * loss_tg + 0.5 * loss_mg
```

**优点**：
- **最全面的对齐**：文本-动作、文本-几何、动作-几何三对关系都显式约束
- **T↔G直接对齐**：确保"举椅子"的文本直接与椅子形状附近的SDF几何对齐，不依赖motion中转
- **理论上最强**：三对约束互相增强，特征空间结构更完善
- 创新性最高——目前没有工作做过三模态对比用于HOI

**缺点**：
- **计算开销最大**：3对InfoNCE，每对2次矩阵乘法+softmax
- **T↔G对齐可能不稳定**：文本"举椅子"和椅子SDF几何之间的语义鸿沟大（文本描述的是动作意图，几何描述的是静态形状），直接对比可能产生noisy gradient
- **需要更多调参**：3个loss权重的平衡
- batch_size要求更高才能区分有效负样本

**适用场景**：追求最强效果，计算资源充足时

---

#### 方案B：核心两对 (T↔M + M↔G，跳过T↔G)

**原理**：只做文本-动作和动作-几何两对对比。文本和几何之间通过motion间接对齐。

```python
def core_dual_contrastive_loss(text_emb, motion_emb, geo_emb, logit_scale):
    """核心两对：T↔M + M↔G"""
    loss_tm = symmetric_info_nce(text_emb, motion_emb, logit_scale)    # 文本↔动作
    loss_mg = symmetric_info_nce(motion_emb, geo_emb, logit_scale)    # 动作↔几何
    return loss_tm + loss_mg
```

**为什么T↔G可以省略**：
- 文本"举起椅子" → 与对应motion对齐 → motion已与椅子附近SDF几何对齐 → **文本间接与几何对齐**
- 类似CLIP的双塔结构：image和text通过对shared embedding space间接对齐
- T↔G的直接对比信号可能是冗余的，因为它的信息已经被T↔M和M↔G覆盖

**优点**：
- **省1/3计算量**：少一对对比
- **更稳定**：T↔M是天然对齐（CLIP已有基础），M↔G是HOI最核心的关系
- **避免T↔G的语义鸿沟问题**
- **Motion作为枢纽(hub)**：合理——motion是连接意图(文本)和物理(几何)的桥梁

**缺点**：
- T和G之间没有直接梯度信号，对齐可能不够紧
- 如果motion编码器表达力不足，间接对齐效果打折

**适用场景**：平衡效果与效率时的首选

---

#### 方案C：仅M↔G对比 (Motion↔Geometry only)

**原理**：只做动作和几何的对比学习。文本对齐完全由CLIP隐式处理。

```python
def motion_geometry_contrastive_loss(motion_emb, geo_emb, logit_scale):
    """仅动作↔几何"""
    return symmetric_info_nce(motion_emb, geo_emb, logit_scale)
```

**为什么M↔G是最核心的对比对**：
- **这是HOI的本质问题**：生成的动作是否与物体几何物理一致？
- 穿透/浮空都是M↔G不对齐的表现
- CLIP已经做了T↔M的预训练，MaMi-HOI中CLIP text embedding已通过加法注入

**优点**：
- **最轻量**：只有1对InfoNCE
- **最高信噪比**：M↔G的对比信号最直接、最物理——穿透就是负样本，良好接触就是正样本
- **负样本构造最自然**：利用SDF穿透值和浮空距离，不需要文本错配
- **与现有代码最兼容**：不需要额外处理text embedding
- **batch_size要求最低**

**缺点**：
- **文本对齐无显式约束**：如果生成"推桌子"的动作却像"举椅子"，纯M↔G对比无法惩罚
- 语义一致性完全依赖CLIP的隐式能力
- 创新性相对较低（双模态对比已有MotionCLIP等工作）

**适用场景**：快速验证、计算资源有限、或希望最小修改现有代码时

---

#### 🏆 三种方案推荐

| 维度 | 全三对 (A) | 核心两对 (B) | 仅M↔G (C) |
|------|-----------|-------------|-----------|
| 创新性 | ⭐⭐⭐ 最高 | ⭐⭐ 高 | ⭐ 中 |
| 计算开销 | ⭐⭐⭐ 最大 | ⭐⭐ 中 | ⭐ 最小 |
| 稳定性 | ⭐⭐ T↔G可能不稳定 | ⭐⭐⭐ 好 | ⭐⭐⭐ 最好 |
| 对齐强度 | ⭐⭐⭐ 最强 | ⭐⭐ 强 | ⭐ 核心 |
| 实现难度 | ⭐⭐⭐ 需调3权重 | ⭐⭐ 需调2权重 | ⭐ 最简单 |
| batch要求 | ⭐⭐⭐ 最好≥128 | ⭐⭐ ≥64 | ⭐ ≥32即可 |

**建议实施路径**：
1. **先实现方案C**（仅M↔G）：快速验证对比学习对HOI的效果，2-3天
2. 如果M↔G对比有效 → **升级到方案B**（+T↔M），增强语义一致性
3. 如果追求极限效果 → **尝试方案A**（全三对），需仔细调参

---

### B5. 推荐方案的详细设计（方案B：核心两对）

以下以核心两对方案为例给出详细设计。

**SDF几何感知增强**：在M↔G对比中，加入**几何质量加权**：
```python
def geometry_weighted_contrastive_loss(text_emb, motion_emb, geo_emb, sdf_quality, logit_scale):
    """
    sdf_quality: [B] 标量，基于SDF接触质量 (0=穿透/浮空, 1=良好接触)
    质量低的样本在对比中权重降低，避免noisy positives
    """
    logits = logit_scale.exp() * motion_emb @ geo_emb.T
    # 用sdf_quality调整label smoothing
    soft_labels = sdf_quality * 0.9 + 0.1 / B  # 质量高→hard label，质量低→soft label
    loss = F.cross_entropy(logits, soft_labels)
    return loss
```

### B6. 训练集成策略

**方案：作为辅助损失加入主训练**（非两阶段预训练）

理由：
- HOI数据集规模有限（OMOMO ~28K窗口），不适合大规模对比预训练
- 与主扩散训练联合，共享transformer特征，不增加额外forward pass
- 对比损失帮助塑造特征空间，主扩散损失负责生成质量

```python
# 在trainer的forward中：
total_loss = loss_diffusion \
           + w_feet * loss_feet \
           + w_fk * loss_fk \
           + w_obj_pts * loss_obj_pts \
           + w_contrastive * trimodal_contrastive_loss  # 新增
```

**超参数建议**：
- `w_contrastive = 0.1`（对比损失权重，初期小值）
- `logit_scale` 初始化 = `log(1/0.07) ≈ 2.66`（跟随CLIP）
- batch_size: 32 可行，但建议尝试 gradient accumulation 到有效 batch=128
- 温度参数可学习

### B7. 关键文件修改

| 文件 | 修改内容 |
|------|----------|
| `manip/model/trimodal_contrastive.py` | **新建**：TrimodalEncoder + 对比损失函数 |
| `manip/model/transformer_control_GAPA_motion_cond_diffusion.py` | 添加`--use_trimodal_contrastive`标志，返回transformer中间特征 |
| `train/trainer_control_GAPA_chois.py` | 添加对比损失计算，负样本构造 |
| `manip/model/sdf_utils.py` | `compute_sdf_gradients_fd()` 终于被使用！ |

---

## C. 整合策略与实验设计

### C1. 两个模块的关系

```
                    ┌─────────────────────┐
                    │   Text Condition     │
                    │   (CLIP, frozen)     │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Main Diffusion      │◄── Trimodal Contrastive Loss (训练时)
                    │  Model + GAPA + SDF  │    (Text↔Motion, Text↔Geo, Motion↔Geo)
                    └──────────┬──────────┘
                               │ x_coarse [B,T,220]
                    ┌──────────▼──────────┐
                    │  HandSRNet           │◄── SDF penetration loss (训练+推理时)
                    │  (局部超分)           │    (BPS_local + SDF作为条件)
                    └──────────┬──────────┘
                               │ x_refined [B,T,220]
                    ┌──────────▼──────────┐
                    │  Guidance (推理时)    │
                    │  (已有SDF引导)        │
                    └─────────────────────┘
```

- **对比学习**塑造训练时的特征空间对齐 → 帮助主模型生成语义一致的粗粒度结果
- **手部超分**在生成后精修手部细节 → 帮助局部几何精度
- 两者互补：对比学习提升"生成正确的动作"，超分提升"动作的几何精度"

### C2. 实验矩阵

在现有Exp 1-4基础上，新增Exp 5-7：

| 实验 | 标志位 | 目的 |
|------|--------|------|
| Baseline | (无新标志) | 复现原始MaMi-HOI |
| Exp 1 | `--use_local_sdf` | SDF-enhanced GAPA (已有) |
| Exp 2 | `--use_local_sdf --use_stateful_sdf` | ZipMap状态查询 (已有计划) |
| Exp 3 | `--use_vlm_keypoints` | VLM关键点 (已有计划) |
| **Exp 5** | `--use_hand_sr` | 手部局部超分 |
| **Exp 6** | `--use_trimodal_contrastive` | 三模态对比学习 |
| **Exp 7** | `--use_local_sdf --use_hand_sr --use_trimodal_contrastive` | 全组合 |

**关键评估指标**：
- 手部精度：hand MPJPE (左/右/双手)
- 交互质量：手-物穿透 (SDF metric)，接触精度/召回/F1
- 语义一致性：R-precision, Matching Score (t2m_eval)
- 分布质量：FID

### C3. 实施优先级

1. **先实现Exp 5 (手部超分)**：实现简单，与现有代码兼容好，可独立验证
2. **再实现Exp 6 (对比学习)**：创新性高，但实现复杂度较高
3. **最后Exp 7 (组合)**：验证两个模块是否互补

### C4. 风险与应对

| 风险 | 概率 | 应对策略 |
|------|------|---------|
| 手部超分在220维空间效果有限 | 中 | 降级为推理时后处理，不参与训练 |
| 对比学习batch_size不足 | 中 | 使用gradient accumulation到有效batch=128；或采用hard negative mining策略 |
| SDF梯度计算不稳定 | 低 | 已有FD实现，可用更小步长；或改用解析梯度 |
| 两个模块冲突 | 低 | 零初始化保证，独立开关控制 |
| 训练时间显著增加 | 中 | 对比损失仅需额外一次矩阵乘法，开销<5%；超分模块轻量 |

---

## D. 实施步骤 (Implementation Steps)

### Step 1: 手部局部超分 (HandSRNet)

1. 新建 `manip/model/hand_sr_module.py`
   - HandSRNet类：2层Transformer + SDF/BPS条件交叉注意力 + 零初始化
   - 输出hand_residual，学习率α

2. 修改 `transformer_control_GAPA_motion_cond_diffusion.py`
   - 添加 `--use_hand_sr` 参数
   - forward()中：x_refined = x_coarse + self.hand_sr_net(hand_features, sdf_features, bps_local)
   - 返回x_refined用于loss计算

3. 修改 `trainer_control_GAPA_chois.py`
   - 添加 `loss_hand_sr` = L1(hand_residual) + λ * sdf_penetration_loss(x_refined)
   - wandb记录手部相关指标

4. 修改 `cano_traj_dataset.py`
   - 加载局部BPS点集数据（或在线计算手部附近的BPS子集）

### Step 2: 三模态对比学习

1. 新建 `manip/model/trimodal_contrastive.py`
   - TrimodalEncoder类
   - trimodal_contrastive_loss函数
   - geometry_weighted变体

2. 修改 `transformer_control_GAPA_motion_cond_diffusion.py`
   - 添加 `--use_trimodal_contrastive` 参数
   - 添加TrimodalEncoder实例
   - forward()中返回transformer中间特征供对比损失计算

3. 修改 `trainer_control_GAPA_chois.py`
   - 实现负样本构造函数（利用compute_sdf_gradients_fd）
   - 计算trimodal_contrastive_loss并加入total_loss
   - 添加 `--loss_w_contrastive` 参数 (默认0.1)

4. 激活 `sdf_utils.py` 中的 `compute_sdf_gradients_fd()`
   - 终于让这个"写了但没用"的函数发挥作用

### Step 3: 实验与评估

1. 运行Exp 5 (hand_sr) 与 Baseline 对比
2. 运行Exp 6 (contrastive) 与 Baseline 对比
3. 运行Exp 7 (组合) 与各单独实验对比
4. 分析手部MPJPE、穿透率、R-precision变化

---

## E. 文献参考

| 论文 | 年份/会议 | 关键启示 |
|------|-----------|---------|
| TIGeR | 2025 | 文本引导的HOI精修，两阶段框架 |
| Hand-Centric Motion Refinement | AAAI 2024 | 手部中心表示 + 层次时空建模做HOI精修 |
| MOCHI | SIGGRAPH 2026 | 扩散噪声优化 + 交互目标做MHOI增强 |
| IMAGIN-4D | 2026 | 图像条件HOI生成，交互状态token分解 |
| MotionCLIP | 2022 | Motion-CLIP对齐，对比学习用于motion |
| Style-SALAD | SIGGRAPH 2026 | 监督对比损失在motion diffusion中有效 |
| DeepHandMesh | ECCV 2020 | 手部穿透避免损失 |
| BRepCLIP | 2026 | 几何-文本-图像三模态对比预训练 |
| SemanticREPA | CVPR 2026 WS | 语义表示对齐用于人体动画精修 |
