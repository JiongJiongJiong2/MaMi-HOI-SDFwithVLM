# 接触条件极简截面几何：最小验证实验

更新日期：2026-09-04。状态：**代码与 synthetic smoke 已通过，真实 OMOMO validation/test 尚未运行。** AutoDL 的统一目录、shared split、环境记录和备份方式见工作区 [`服务器指南.md`](../../../服务器指南.md)。

这个实验先回答一个比“它能否提高 diffusion/VLM”更基础、也更容易证伪的问题：**已知一只手的接触位置后，由重力方向和该接触点定义的极简截面几何，能否比随机搜索或中心对跖点更好地缩小另一只手的接触区域？**

2026-09-09 证据边界：近期 material section/rim CPU 检查验证了固定截面与局部表面追踪，并识别 plasticbox 盒口开放间隙；未评估真实双手接触候选，不构成本 G0 的真实数据有效性证据。G0 仍独立于 U1 signed-SDF Gate。详见[局部检查与 G0 的关系](plasticbox-rim-connectivity-zh.md)。

第一阶段不训练网络，不修改 diffusion，也不输入 VLM。它与动态 SDF 使用同一套 OMOMO 处理数据、物体坐标系和 sequence-disjoint 划分，但保持独立运行。这样失败时可以直接否定或修改几何假设，而不会把结果混入 SDF loss、diffusion 采样或提示词效果。

## 1. 现在实现的几何

对每个双手同时接触物体的帧，将两只手掌关节（左手 22、右手 23）变换到物体规范坐标系，并投影到同一组物体表面候选点。以源手接触点 `q`、物体中心 `c` 和物体坐标系中的重力向上方向 `u` 构造：

```text
竖直截面：经过 c 和 q，且包含 u
水平截面：经过 q，法向为 u
截面弦：上述两个平面的交线
```

代码比较四个不学习的候选排序：确定性随机排序、相对物体中心的对跖点、靠近截面弦且沿物体内部方向延伸、以及在截面弦基础上偏好与源接触法向相反的表面。所有方法都屏蔽源接触附近区域，避免把同一个接触点当成预测答案。

这里的“线”不是送给模型的一串任意坐标，而是一个固定、可复现的接触坐标系。当前 G0 直接使用它的解析几何距离做诊断；只有 G0 在真实留出序列上成立，才值得在 G1 把表面截线重采样成固定数量的坐标点，或栅格化为 2D sectional-SDF/occupancy 图供 VLM 比较。

## 2. 无数据 smoke test

在仓库根目录运行：

```bash
python scripts/run_sectional_prior_diagnostic.py \
  --synthetic \
  --output_dir sectional_prior_eval/synthetic
```

合成盒子的目标点是按截面弦构造的，所以 PASS 只证明坐标变换、排序、指标和结果写出没有明显错误；它不是 HOI 有效性的证据。

## 3. 真实 OMOMO 验证集实验

项目环境需要可导入 `numpy`、`joblib` 和 `trimesh`。数据根目录至少应包含：

```text
processed_data/
  cano_test_diffusion_manip_window_120_joints24.p
  contact_labels_w_semantics_npy_files/<sequence>.npy
  rest_object_geo/<object>.ply
```

先生成一次序列互斥的验证/测试清单：

```bash
python scripts/create_experiment_split_manifest.py \
  --data_root_folder /path/to/processed_data \
  --output_path sectional_prior_eval/split_seed1.json \
  --validation_sequence_count 100 \
  --seed 1 \
  --window 120
```

然后只在 validation 上确定假设和阈值：

```bash
python scripts/run_sectional_prior_diagnostic.py \
  --data_root_folder /path/to/processed_data \
  --split_manifest sectional_prior_eval/split_seed1.json \
  --eval_split validation \
  --window 120 \
  --frame_stride 5 \
  --candidate_count 512 \
  --max_samples 2000 \
  --seed 1 \
  --output_dir sectional_prior_eval/validation_seed1
```

`--max_samples` 必须是偶数，因为每个接收帧同时评估 left-to-right 与 right-to-left，不能在截断时破坏方向配对。窗口重叠产生的重复帧按 `(sequence_name, absolute_frame)` 去重，汇总指标先按序列平均，再对序列做宏平均。

冻结所有规则后才运行 test：

```bash
python scripts/run_sectional_prior_diagnostic.py \
  --data_root_folder /path/to/processed_data \
  --split_manifest sectional_prior_eval/split_seed1.json \
  --eval_split test \
  --window 120 \
  --frame_stride 5 \
  --candidate_count 512 \
  --max_samples 2000 \
  --seed 1 \
  --output_dir sectional_prior_eval/test_seed1
```

## 4. 输出和判定

每次运行写出三个文件：`samples.csv` 保存逐方向样本、各方法 top-1 候选坐标和全部排序指标；`summary.json` 保存 sequence-macro 汇总、数据过滤计数和 go/no-go 结果；`section_frames.jsonl` 保存源点、目标点、径向轴和向上轴，为后续截线/SDF 面板生成提供固定接口。

默认 PASS 要同时满足：最佳截面方法相对 `center_antipode` 的 sequence-macro top-5 hit 至少提高 0.05，且真实目标到截面弦的归一化中位距离不超过 0.12。还应检查随机基线、top-1/5/10、MRR、法向版本是否稳定优于无方向版本，以及不同物体/序列上的失败样本，不能只看一个总分。

PASS 的含义仅是“这个极简几何包含第二接触区域的信息”；FAIL 则说明当前两平面/弦定义不足，应优先修改几何而非接入大模型。手掌关节和二值接触标签不能验证手指姿态、摩擦锥、力闭合或真正的抓取稳定性。

## 5. 与动态 SDF 的关系

建议把它们放在同一个研究计划里、共享数据版本和 split，但第一轮分别跑：动态 SDF 的 E0/E1 检验“距离场监督能否改善生成轨迹”，本实验的 G0 检验“截面先验是否含有双手接触信息”。二者均有正结果后，再实现严格的四组消融：baseline、`+dynamic SDF`、`+section prior`、`+dynamic SDF + section prior`。当前代码没有实现这四组训练，也不应把合成 PASS 解读成组合方法有效。
