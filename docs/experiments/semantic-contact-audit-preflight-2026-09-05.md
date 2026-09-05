# Semantic-contact 第一阶段：输入预检与阻塞记录

日期：2026-09-05。状态：**BLOCKED / 真实统计尚未运行**。本次完成当前仓库与 `data/` 的实际文件核查，不是 semantic headroom 或 candidate coverage 的实验结果。

## 检查范围与实际文件

检查 `E:/HOI`、两套嵌套仓库和各自的 `data/`，包括隐藏/被 Git 忽略的文件；没有扫描其他磁盘或连接服务器。`E:/HOI/data` 不存在。两套仓库的 `data/` 各有 5 个文件，没有指向外部数据的链接目录。

| 实验仓库内文件 | 大小 | 实际内容 |
|---|---:|---|
| `data/body_models/README.md` | 77 B | 要求另放 SMPL 模型的说明，未包含模型 |
| `data/part_vert_ids/left_hand_vids.npy` | 6,304 B | 一维 int64，772 个手部顶点索引 |
| `data/part_vert_ids/right_hand_vids.npy` | 6,232 B | 一维 int64，763 个手部顶点索引 |
| `data/part_vert_ids/left_foot_vids.npy` | 2,136 B | 一维 int64，251 个脚部顶点索引 |
| `data/part_vert_ids/right_foot_vids.npy` | 2,136 B | 一维 int64，251 个脚部顶点索引 |

这些 `.npy` 是静态身体部位顶点 ID，不是 `T×4` 接触标签，也不含 sequence、动作文本或三维运动。已直接读取实验仓库中的 NPY 文件头确认 dtype/shape。

工作区 `.p/.pt/.pth/.ckpt/.obj/.ply/.npz` 与 `*split*.json` 扫描仅找到两套仓库各自的 `bps.pt`、`t2m_eval/t2m_mean_std_jpos.p`。前者为几何基点，后者为 evaluator 归一化统计，不能代替 baseline checkpoint 或 processed motion。未找到实际物体 mesh、运行结果或 split 文件；不能据此推断它们不存在于其他位置。

当前 Python 为 `C:/Python314/python.exe`、3.14.4；模块发现检查未找到 NumPy、Torch、joblib、trimesh、SciPy。GPU 为 RTX 3060 Laptop、6,144 MiB。这是环境清单，不是容量测试；本次未安装依赖、下载模型或启动采样。

## 源码中确认的数据与评估接口

以[工作区 README](../../../README.md) 为导航，核对以下源码事实：

| 数据 | 已确认位置/字段 | 限制 |
|---|---|---|
| object identity | `seq_name.split('_')[1]` | 必须与实际 mesh 文件名交叉核验 |
| interaction text | `omomo_text_anno_json_data/<seq>.json` 中的 `<seq>` 字段 | `language_mapping.py` 中的模板不是样本存在的证据 |
| hand contact | `contact_labels_w_semantics_npy_files/<seq>.npy`，`T×4`，左手/右手/左脚/右脚 | 当前这些文件不可访问 |
| GT palm proxy | 处理后 window 的 `motion` 前 `24×3` 位置通道，关节 22/23 | 是手部 proxy，不是皮肤接触点 |
| GT object pose | `obj_rot_mat`、`window_obj_com_pos` | 与该 window 的人体坐标一致，不能混入其他窗口变换 |
| frame identity | `seq_name`、`start_t_idx`、`end_t_idx` | 重叠窗口需按原始 sequence/frame 去重 |
| canonical geometry | `rest_object_geo/<object>.ply`；已有 G0 面积加权确定性采样函数 | 只是离线 surface representation，不是已实现 U3 |

实现参考：[dataset](../../manip/data/cano_traj_dataset.py)、[G0 mesh adapter](../../scripts/run_sectional_prior_diagnostic.py)、[SDF 坐标函数](../../manip/model/sdf_utils.py)。

公平 U0 协议由[评估入口](../../scripts/evaluate_u0_u1.sh)与[runbook](../../../docs/current/experiment-gate0-u0-u1-runbook.md)定义：指定同一个 baseline、冻结 sequence-disjoint manifest、seed=1、window=120，主结果 guidance off；on 必须另报。现有首帧/路径条件需保持并记录，不可为提高 coverage 改为 GT palm proposal。

另有一个已确认的接口缺口：[trainer](../../train/trainer_control_GAPA_chois.py) 当前普通评估的 `res_npz_files` 只保存 `seq_name` 和 `global_jpos`。单独这一 NPZ 不包含 predicted object rotation/COM，不能直接用于本次 canonical candidate audit，更不能拿 GT object pose 补齐。相关预测 object 状态在评估内存中存在；获得数据和权重后需要一个只导出状态与 provenance 的轻量旁路，不必修改 architecture。

## 待真实输入到位后执行的方法

Semantic headroom 按原始 sequence 去重，以实际标注文本及可核验的动作归并规则分组，逐物体、逐手构造 GT-derived palm contact patch 分布。先报告各 object-action 独立 sequence 数，再用同一分布距离度量比较同动作与异动作差异，采用 sequence-level 重采样/置换检查不确定性。不同文本不自动等于不同语义，文本模板不能计为实测样本。严格按用户要求检查 between separation 是否明显大于 within variation，但不把缺少可访问数据写成生物/几何假设被否定。

Coverage 使用固定 seed 从实际 canonical mesh 采样至少 128 个、建议 1,024 个 surface points。仅用 predicted palms、projected predicted object rotation、predicted object COM 查询 KNN，依次评估 K=16/32/64/128。独立地用 GT palm/pose 得到带容忍半径的 GT-derived patch；patch 半径与物体尺度归一化方式在 validation 前冻结，并报告敏感性。禁止把 GT 最近点强行塞进候选。

对每帧每手保存候选是否覆盖 patch、oracle 最近距离、到 patch 的距离/物体尺度；按 sequence 宏平均，并分别统计左右手、接触与非接触、实际训练覆盖定义的 seen/unseen。非接触帧只报告表面邻近诊断，不能将其投影称为发生过的接触，contact recall 记 N/A。oracle Top-1 patch hit 在该定义下与候选 recall 等价，不能包装为两个独立改善证据。

先区分全 surface bank 覆盖不足、局部 K 太小/空间分布不合适、粗预测状态偏差。只有排除几何/候选瓶颈后，才讨论语义重排的机会。上述是待执行方法，未产生测量值。

## 核心结果与 Gate

| 项目 | 当前结果 | 解释 |
|---|---|---|
| 实际读取的 motion sequence | 未读取 | 缺 processed_data，不是 dataset 总数为零 |
| object-action 样本数与 between/within | N/A | 无法给 PASS/WEAK/FAIL |
| candidate recall@16 | N/A | 没有预测/mesh |
| candidate recall@32 | N/A | 没有预测/mesh |
| candidate recall@64 | N/A | 没有预测/mesh |
| candidate recall@128 | N/A | 没有预测/mesh |
| oracle / normalized distance | N/A | 未运行，不填 0 |
| 典型成功/失败样本 | 不可提供 | 没有真实样本，不用合成例子代替 |

A. Semantic headroom：**NOT_ASSESSED / BLOCKED**。B. Candidate coverage：**NOT_ASSESSED / BLOCKED**。C. 下一阶段：**HOLD**，暂不实现 geometry/action/CLIP/Qwen 重排。缺少输入时强行标 FAIL 会混淆“实验没有运行”和“数据不支持假设”。

## 产物、最小下一接口与 README

机器可读产物：[preflight.json](../../experiments/semantic_contact_audit/2026-09-05/preflight.json)。`data_directories`/`experimental_repository_data_files` 是实测目录与文件清单；`workspace_binary_scan` 是限定范围的搜索结果；`environment` 是环境发现；`gates` 是阻塞状态；`coverage` 的 null 表示未测量，绝不表示零召回。没有生成真实 headroom/coverage CSV，因为没有可分析数据。

继续所需：实际 processed_data 根目录、指定 U0 checkpoint、冻结 split 路径，以及可运行环境或服务器连接。若提供已有预测，最小导出为 `sequence/window/frame IDs + pred_palm_world[T,2,3] + pred_object_rotation[T,3,3] + pred_object_com[T,3] + checkpoint/split hash + seed/guidance/units/proxy_definition`。GT 标签另存，candidate proposal 函数仅接受预测状态与 canonical geometry。需要核对预测 proxy 与现有 FK/SMPL-H 导出定义，不默认为同一种掌心。

本次未新增运行/分析脚本，仅保存预检证据与后续接口缺口；未修改 README。README 已说明 processed_data 位于外部、U3 尚未实现、真实实验待验证，本轮没有足以升级这些状态的实验结果。数据质量技能要求将缺失输入与零值/实验失败区分，规划技能保留了阻塞与未执行阶段。
