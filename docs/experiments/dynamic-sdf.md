# 动态 SDF 实验操作手册

这是一套为“尽快得出可信结论”设计的最小实验。当前新模块只在**训练期**计算额外损失；采样和推理仍沿用原始 MaMi-HOI 路径。因此，推理时绝不能输入 GT 未来手部轨迹。

实验顺序固定为：

```text
E0：评估原始 baseline
E1：baseline + 动态 SDF 直接损失
E2：baseline + 动态 SDF 直接损失 + SDF 轨迹排序
```

E1、E2 都从同一个 baseline checkpoint 微调。只有 E1 的结果有效且为正，才运行 E2。

更完整的数据清单见 [`../../../docs/data/dynamic-sdf-data-checklist.md`](../../../docs/data/dynamic-sdf-data-checklist.md)。

## 1. 运行前准备

需要：

- 可用的 MaMi-HOI baseline checkpoint；
- 标准的 `processed_data/`、SMPL-H、BPS、评估器资源；
- 每个训练物体的 `rest_object_sdf_256_npy_files/<object>.ply.npy` 和 `<object>.ply.json`；
- Linux/AutoDL 上的项目 Conda/CUDA 环境。

不要启用旧的 `--use_local_sdf`。它属于早期、不可用于当前论文实验的实现路线。

```bash
cd /path/to/MaMi-HOI-SDFwithVLM
export DATA_ROOT=/path/to/processed_data
export BASELINE_CKPT=/path/to/baseline/weights/model-9.pt
```

## 2. 生成每物体 64³ SDF 缓存

```bash
python scripts/prepare_object_sdf64.py \
  --data_root_folder "${DATA_ROOT}" \
  --resolution 64
```

应生成：

```text
${DATA_ROOT}/object_sdf_64/
  largetable.pt
  woodchair.pt
  ...
```

每个文件只含 `sdf_grid`、`centroid`、`extents`。这里的“动态”来自预测的手掌/物体姿态在每个训练步产生不同查询点，不是为每个序列生成不同 SDF。

## 3. 预检和 smoke test

在正式训练前必须执行：

```bash
python tests/test_dynamic_sdf.py
bash -n scripts/train_dynamic_sdf.sh scripts/train_dynamic_sdf_contrastive.sh
```

然后用 E1 脚本先跑 200--500 step。检查：

- `Dynamic SDF Loss` 和总损失均为有限数值；
- 没有 SDF 文件缺失、NaN 或明显的坐标越界错误；
- 动态 SDF 损失乘上其权重，初期通常约占原训练损失的 5%--10%。

若比例明显不对，只将 `LOSS_W_SDF` 乘以 10 或除以 10 后重跑一次 E1；不要做大规模权重搜索。

## 4. E0：baseline 评估

无需训练。用同一评估脚本、同一 `seed`、同一场景/物体划分、同一 guidance 开关评估 baseline checkpoint，并把结果写入结果表的 E0 行。

```bash
python train/trainer_control_GAPA_chois.py \
  --window=120 \
  --batch_size=32 \
  --data_root_folder="${DATA_ROOT}" \
  --pretrained_model="${BASELINE_CKPT}" \
  --save_res_folder=./dynamic_sdf_eval/E0_seed1 \
  --seed=1 \
  --input_first_human_pose \
  --use_random_frame_bps \
  --add_language_condition \
  --use_object_keypoints \
  --add_semantic_contact_labels \
  --loss_w_feet=1 --loss_w_fk=0.5 --loss_w_obj_pts=1 \
  --test_sample_res \
  --use_long_planned_path \
  --test_object_name=all \
  --test_scene_name=frl_apartment_4 \
  --use_guidance_in_denoising \
  --compute_metrics
```

若测试 unseen object，则对 E0/E1/E2 同时加入 `--test_unseen_objects`。

## 5. E1：动态 SDF 直接损失

```bash
DATA_ROOT="${DATA_ROOT}" \
BASELINE_CKPT="${BASELINE_CKPT}" \
PROJECT=./dynamic_sdf_experiments \
EXP_NAME=E1_dynamic_sdf_seed1 \
TRAIN_STEPS=20000 \
SAVE_EVERY=20000 \
LOSS_W_SDF=1.0 \
SEED=1 \
bash scripts/train_dynamic_sdf.sh
```

完成后，最终权重应类似：

```text
dynamic_sdf_experiments/E1_dynamic_sdf_seed1/weights/model-final-20000.pt
```

将 E0 的评估命令中的 `--pretrained_model` 和 `--save_res_folder` 改为 E1 对应路径后，重新评估。其余参数不得改变。

## 6. E2：SDF 轨迹排序（轻量对比学习）

E2 的“对比”不使用 VLM、三塔或 InfoNCE；它只要求预测 SDF 轨迹更接近 GT 轨迹，并远离穿透/浮空扰动轨迹。它必须从 `BASELINE_CKPT` 出发，不能接着 E1 训练。

```bash
DATA_ROOT="${DATA_ROOT}" \
BASELINE_CKPT="${BASELINE_CKPT}" \
PROJECT=./dynamic_sdf_experiments \
EXP_NAME=E2_dynamic_sdf_ranking_seed1 \
TRAIN_STEPS=20000 \
SAVE_EVERY=20000 \
LOSS_W_SDF=1.0 \
LOSS_W_RANKING=1.0 \
SEED=1 \
bash scripts/train_dynamic_sdf_contrastive.sh
```

使用与 E0/E1 完全相同的评估设置评估 E2。

## 7. 每次实验必须保存什么

每个 `(实验, seed, 测试划分)` 一行，填写 [`../../experiments/dynamic_sdf_results_template.csv`](../../experiments/dynamic_sdf_results_template.csv)。同时保留：

- `opt.yaml`、代码 commit、完整启动命令、`DATA_ROOT` 和 baseline checkpoint 路径；
- 最终 checkpoint，以及如果使用验证集选择的最佳 checkpoint；
- total、diffusion、FK、object-points、dynamic-SDF、SDF-ranking 的训练曲线；
- 原始逐序列 metric JSON、生成 `.npz`、至少 3 个成功和 3 个失败的渲染视频；
- 吞吐量、峰值显存、每条序列推理时间；
- guidance 是否开启、场景/物体选择、是否 unseen-object 测试。

主要比较指标是 Contact Precision/Recall/F1、GT 接触帧的 `D_hand`、手-物穿透、Hand JPE、MPJPE、foot sliding、物体 COM/旋转误差。标准 evaluator 可用时再补 Matching Score、R-precision 和 FID。

## 8. 20k 筛选的决策规则

- 仅当 E1 在穿透或 `D_hand` 上约有 5% 改善，且 Contact-F1/Hand-JPE 没有明显退化，才继续 E2。
- 仅当 E2 在接触--穿透权衡上优于 E1，才保留 ranking loss。
- 若 E1 无改善或退化，先检查数据覆盖和坐标变换；不要立刻叠加 VLM、ZipMap 或更多模块。
- 筛选得到正结果后，再用至少 3 个 seed 重跑 E0/E1/E2；置信区间按原始 sequence 聚合，不能把滑窗当独立样本。
