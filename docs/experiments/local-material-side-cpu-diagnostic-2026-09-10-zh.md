# 局部 material-side / penetration 最小 CPU 诊断

日期：2026-09-10。状态：**只读实现调查与最小 CPU 诊断已完成；未训练、未修改主模型、未实现 U2/U3/U4。**

## 结论摘要

当前三个对象上**不存在一种可以不加 mask 直接作为全局独立 penetration supervision 的单一 local material-side signal**。最能进入后续 U1-UP 的最小候选是：

```text
canonical raw triangle UDF
  + conservative coherent mesh 的 generalized winding number
  + 经审阅且拓扑可辩护的 unknown mask
```

这不是“winding 已修复 signed SDF”。它只是比旧 legacy SDF sign、raw face normal 和候选 pseudonormal 在本次诊断中更稳定、更透明的条件信号。

smalltable 的已审阅材料/自由点上，candidate winding 为 `9/9 material + 10/10 free`；plasticbox 已审阅自由点 `14/14`。但 trashcan 的 7 个已审阅 cavity air 点全部被 winding 判为 material，`7/7`。因此 trashcan cavity/opening 必须 mask；plasticbox 没有可信 material 正例，只能用于自由侧或保持 unknown。smalltable 是唯一既有材料正例、又有自由正例、且拓扑最干净的局部验证对象。

建议有效 penetration 激活带为 **0–10 mm**，诊断覆盖报告使用 0–20 mm；超过 20 mm 主要是 surface-contact 或自由空间证据，不应独立产生 penetration loss。GT palm 不是 surface truth，contact label 也不能被当作 inside/outside label。

**建议实现 unsigned contact + masked local penetration 对照。** 它应是一个独立的 U1-UP 诊断实验，不能解除原 signed U1、CUDA smoke 或 U1 20k 的 BLOCKED 状态。

本次没有发现需要重新阻塞 U3 的证据。U3 的 local surface query/refinement 与 unsigned UDF 条件耦合，不必等待全局 signed SDF；但当前 prediction queries 确实缺失，训练时分布的 query 证据仍未建立。

## 输入事实

| 输入 | 事实 |
|---|---|
| canonical raw mesh | `TMP/processed_data/rest_object_geo/<object>.ply`；plasticbox/trashcan/smalltable 的面数为 67,812 / 34,774 / 26,164 |
| legacy source SDF | `rest_object_sdf_256_npy_files/<object>.ply.npy` 与 `.json`，`256^3`；本机没有 `object_sdf_64` cache |
| GT hand proxy queries | 138 个 joints 22/23 canonical queries：plasticbox 48、trashcan 42、smalltable 48；来源为冻结 BPS diagnostic，`gt_palm_contact/gt_palm_noncontact` 明确分层 |
| saved model prediction queries | **MISSING，count=0**；没有用 GT 查询冒充训练时预测分布 |
| reviewed real points | `tests/fixtures/material_oracle_probes_v1.json` 的 112 点，已覆盖 cavity/opening、wall/bottom/rim unknown、tabletop/leg material、under-table air |
| fast winding | **NOT RUN / unavailable**；环境没有 libigl 或专用 fast-winding backend，因此只比较现有 direct generalized winding |

旧 SDF 采样沿用当前 `sdf_utils.sample_object_sdf_at_points`：cube domain 为 `centroid ± max(extents)/2`，negative SDF 对应 material。canonical raw triangle mesh 是 unsigned distance reference；候选 coherent mesh 仅用于 winding 和 pseudonormal 等拓扑信号，不是新的物理材料定义。

## 查询与算法

实际执行集合为 2,578 行：

- 138 个 GT palm base queries；
- 112 个 reviewed manual base probes；
- 对 GT 查询做 `±0.5 mm` 与 `±1.0 mm` 的固定轴向扰动，对 reviewed 点做 `±0.5 mm` 扰动。

每个点独立计算 raw triangle UDF 与 closest point/face，候选 coherent mesh 的 closest point 与 area-weighted pseudonormal，旧 256³ SDF 值、有效域和 central-difference gradient，以及候选 generalized winding。unknown 不并入 free，也不通过剔除难样本提高 headline accuracy。

距离分层为 0–5、5–10、10–20、20–50、>50 mm。算法 fixture 使用 closed box 与 open thick cup，两个 fixture 的 occupancy 均为 1.0，最大 distance/exit error 为 `1.11e-16 m` / `1.39e-17 m`。

## 主要结果

### Reviewed material/free accuracy

| Object | Known | legacy SDF | raw face normal | candidate pseudonormal | candidate winding |
|---|---:|---:|---:|---:|---:|
| plasticbox | 14 free | 4/14，2 abstain | 8/14 | 6/14，4 abstain | **14/14** |
| plasticbox | 0 material | NOT ASSESSABLE | NOT ASSESSABLE | NOT ASSESSABLE | NOT ASSESSABLE |
| trashcan | 14 free | 1/14，5 abstain | 5/14 | 4/14，2 abstain | 7/14 |
| trashcan | 0 material | NOT ASSESSABLE | NOT ASSESSABLE | NOT ASSESSABLE | NOT ASSESSABLE |
| smalltable | 9 material | **9/9** | **9/9** | 8/9，1 abstain | **9/9** |
| smalltable | 10 free | 4/10，6 out-of-grid | **10/10** | 4/10 | **10/10** |

“accuracy”包含 abstain/unknown 作为失败；decision accuracy 才只统计已给出 material/free 决策的样本。不能只看 decision accuracy。

### 关键自由空间的 false-positive material rate

| Region | legacy | raw normal | pseudonormal | winding |
|---|---:|---:|---:|---:|
| trashcan reviewed cavity air | 7/7 | 7/7 | 3/7，2 unknown | **7/7** |
| trashcan above opening | 0/1，1 unknown | 1/1 | 1/1 | 0/1 |
| plasticbox reviewed cavity air | 6/7 | 2/7 | **0/7，1 unknown** | **0/7** |
| plasticbox above opening | 0/1 | 0/1 | 1/1 | 0/1 |
| smalltable under-table air | 0/4 | 0/4 | 2/4 | 0/4 |

这组结果直接说明：winding 在 trashcan cavity 不是“阈值调不好”，而是 material/free 语义整体错误；pseudonormal 没有统一修复这一点。

### Distance-band coverage

只统计 138 个 base GT palm 与 112 个 reviewed base 点，不是人口覆盖。候选 winding 在所有距离带都有数值，但语义不可信区域仍不可信。

旧 legacy SDF 的有效域覆盖在 GT 0–20 mm 内为：plasticbox/trashcan 100%，smalltable 85%；>50 mm 时分别降到 37%、23%、12%。因此 legacy SDF 作为 interaction signal 的覆盖主要限定在 0–20 mm，而 **20–50 mm 和 >50 mm 不应被当作有效 penetration supervision**。

pseudonormal 在 GT 0–20 mm 的有效覆盖有较大差异：plasticbox 10–20 mm 仅 43%，trashcan 0–5 mm 81%，smalltable 10–20 mm 85%。candidate winding 则没有这类数值 unknown，更适合覆盖报告；其问题只在 open/container 语义。

### 小扰动稳定性

在 reviewed 0–5 mm 区域，固定小扰动后的 sign flip 比例为：

| Object | legacy | raw normal | pseudonormal | winding | closest face change |
|---|---:|---:|---:|---:|---:|
| plasticbox | 7.6% | 12.1% | 1.4% | 7.2% | 26.7% |
| trashcan | 5.7% | 0.0% | 5.3% | 11.1% | 14.4% |
| smalltable | 6.7% | 0.0% | 11.1% | 20.0% | 23.3% |

closest-face 切换率普遍为 14–27%，说明不能把最近 face 当作固定 oracle。候选 winding 的小扰动 sign 稳定性并不统一优于其它信号；smalltable 的 winding flip 主要来自 unknown/boundary 附近的困难样本，但 headline accuracy 仍正确。

### 薄壁短线段

smalltable 已审阅 tabletop 厚度线的 33 个样本中，四种信号均形成 material run，且 winding/raw normal/pseudonormal 为 17/33 material，legacy 为 18/33。它证明在已确认厚度的封闭薄板上，local signal 可以触发有效 penetration region，不会从一侧直接跳到另一侧而没有负区。

两个容器的 wall/rim/bottom 线本身没有可信 material interval。不同信号在这些线上互相矛盾，例如 plasticbox `wall_0_negative` 中 legacy/pseudonormal 形成 material run，而 winding 全为 free；trashcan rim 中 legacy 判 33/33 material，pseudonormal/raw normal/winding 仅约 36–52%。这只能说明现有容器边界不能独立定义 penetration region，不能从中推出壁厚。

### 局部梯度/排斥方向

对 smalltable 的 9 个已审阅 material 点，legacy SDF gradient 与 closest-surface outward direction 的 cosine 全部为正，median `1.0`。这说明一旦 sign 正确，旧 SDF 的局部排斥方向在数值上是合理的。对 trashcan 的 7 个 false-positive cavity 点，gradient 也大多指向 closest surface；问题是 sign 把 cavity 错标为 material，方向本身不能纠正语义错误。

## 最终推荐

### 最小 U1-UP

先做以下独立对照，而不是继续重建 signed SDF：

```text
A: raw triangle UDF unsigned surface-contact only
B: A + candidate winding penetration term
   - 只在 reviewed/topologically safe region 激活 material-positive
   - trashcan cavity/opening、plasticbox wall/bottom/rim 与未审阅 cavity mask 为 unknown
   - smalltable closed components 可参与材料正例
   - 0-10 mm 激活，0-20 mm 只做覆盖统计
```

当前结果支持先进入 **B 的最小 masked winding 版本**。它不需要新增重型依赖，且已经在现有代码中有独立 direct winding 与 raw UDF 实现。legacy SDF sign 不作为 penetration source；candidate pseudonormal 也不单独作为 penetration source。

### 有效 band 与 expected coverage

推荐的 penetration activation 是 **0–10 mm**。0–20 mm 是当前 GT 查询在旧 SDF cube 中仍有较高覆盖的诊断观察带；它不是“保证正确”的 band。当前观察不是数据集人口估计：

- GT 0–20 mm 旧 SDF 有效覆盖约 98%（44/45）；
- candidate winding 数值覆盖为 100%，但 trashcan cavity 等 mask 必须先行；
- smalltable reviewed known 点当前为 19/19 correct；
- plasticbox 当前只有 free-side evidence，材料正例 coverage 为 NOT ASSESSABLE。

### 必须 mask 的 mesh region

至少需要 mask：

- trashcan `reviewed_cavity_air` 与 `above_open_top`；
- trashcan wall/bottom/rim unresolved；
- plasticbox wall/bottom/rim unresolved；
- smalltable tiny reverse component 与 boundary-tolerance；
- 任何 open-boundary/nonmanifold 组件中依赖 normal 或 material-positive 的查询。

### unsigned contact + local penetration 对照

建议执行。目的不是宣称修复了 signed SDF，而是检验 **masked winding penetration signal 能否在 smalltable 上产生稳定的监督梯度，同时不在两个容器 cavity 中制造明显负效应**。两个容器可以只作为 UDF unsigned contact 和 unknown mask 的验证集。

### U3 是否需要重新阻塞

不需要。本次没有证据改变 U3 的结构必要性。U3 的 prediction-indexed local surface refinement 更接近 unsigned surface query 与 region-gated occupancy，而不是依赖全局 signed SDF。唯一仍缺的是真实模型 prediction queries；这不是重新阻塞 U3，而是保留 `prediction-query coverage: MISSING`。

## 产物与验证

- 实现：[audit_local_material_signal.py](../../scripts/audit_local_material_signal.py)
- 结果：[summary.json](../../outputs/local_material_signal_20260910_v2/summary.json)
- 细节：[queries.csv](../../outputs/local_material_signal_20260910_v2/queries.csv)
- 线段：[plasticbox](../../outputs/local_material_signal_20260910_v2/plasticbox_line_points.csv)、[trashcan](../../outputs/local_material_signal_20260910_v2/trashcan_line_points.csv)、[smalltable](../../outputs/local_material_signal_20260910_v2/smalltable_line_points.csv)

本次运行 150.80 s，输入 hash 前后一致，`query_rows=2578`，closed-box/open-cup analytic fixture 通过。未导入 trainer/dataset/CUDA，未修改主模型、README、旧 SDF 或 canonical mesh。运行中第一次 v1 仅因 free-label 编码在报告层错误而被丢弃，几何计算和输入均未改变；最终有效产物是 v2。

## README 需要同步的内容

以下只列不改：

1. U1 Gate 0 应增加本次结果：原 source-sign blocker 保持；local masked winding/UDF 只是 U1-UP 候选，不能解除全局 penetration blocker。
2. README 中“当前执行重点”与 U1 描述应区分旧 signed SDF 与 candidate winding；目前若只写“source/evaluation geometry BLOCKED”，会漏掉新获得的局部条件证据。
3. 应记录 `prediction queries MISSING`，说明当前 GT palm coverage 不能代表训练时 coarse prediction 分布。
4. U3 状态仍为 Planned，不应因本次诊断改写成 re-blocked；U1-UP 与 U3 是独立证据边界。

