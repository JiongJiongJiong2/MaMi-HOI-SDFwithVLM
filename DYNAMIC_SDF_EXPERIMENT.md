# Dynamic SDF experiments: how to run and what to record

# Dynamic SDF 实验：如何运行以及需要记录什么

This is the minimal experiment sequence for the new implementation. The new
SDF is a **training-time loss only**. Sampling remains the original MaMi-HOI
sampling path, so do not pass GT future-hand query points at inference.

这是新实现所需的最小实验流程。新的 SDF 只是一个**训练阶段的损失项**。采样仍然使用原始 MaMi-HOI 的采样路径，因此推理时不要传入 GT 未来手部查询点。

## 0. Required inputs

## 0. 必需输入

You need:

你需要准备：

- a working MaMi-HOI baseline checkpoint, saved by this repository;
  - 一个可用的 MaMi-HOI baseline checkpoint，并且该 checkpoint 由本仓库保存；
- `processed_data/rest_object_sdf_256_npy_files/`, containing one
  `<object>.ply.npy` and `<object>.ply.json` pair per training object;
  - `processed_data/rest_object_sdf_256_npy_files/`，其中每个训练物体都包含一组 `<object>.ply.npy` 和 `<object>.ply.json` 文件；
- the usual MaMi-HOI processed data, SMPL-H models, BPS file and evaluator
  assets described in `README.md`.
  - `README.md` 中描述的常规 MaMi-HOI 处理后数据、SMPL-H 模型、BPS 文件和 evaluator 资源。

Set these once in your Linux/AutoDL shell:

在你的 Linux/AutoDL shell 中一次性设置以下变量：

```bash
cd /path/to/MaMi-HOI-SDFwithVLM
export DATA_ROOT=/path/to/processed_data
export BASELINE_CKPT=/path/to/baseline/weights/model-9.pt
```

Do not enable the old `--use_local_sdf` flag in any experiment below.

下面任何实验都不要启用旧的 `--use_local_sdf` 标志。

## 1. Prepare compact per-object SDF files

## 1. 准备紧凑的逐物体 SDF 文件

This uses geometry only; it does not use motion, contact labels or hand
positions.

这一步只使用几何信息；它不会使用运动、接触标签或手部位置。

```bash
python scripts/prepare_object_sdf64.py \
  --data_root_folder "${DATA_ROOT}" \
  --resolution 64
```

Expected result:

预期结果：

```text
${DATA_ROOT}/object_sdf_64/
  largetable.pt
  woodchair.pt
  ...
```

Each file must contain `sdf_grid`, `centroid` and `extents`.

每个文件都必须包含 `sdf_grid`、`centroid` 和 `extents`。

## 2. Run the unit test before any GPU training

## 2. 在任何 GPU 训练前运行单元测试

```bash
python tests/test_dynamic_sdf.py
```

It checks SDF sign/scale, world-to-object conversion, differentiability,
contact loss and ranking-loss ordering. Do not proceed if it fails.

该测试会检查 SDF 的符号/尺度、世界坐标到物体坐标的转换、可微性、接触损失以及 ranking loss 的排序关系。如果测试失败，不要继续后续步骤。

## 3. Experiments E0, E1 and E2

## 3. 实验 E0、E1 和 E2

Use the same baseline checkpoint, `TRAIN_STEPS`, `SEED`, evaluation prompts and
sampling seed for every comparison.

每组对比都要使用相同的 baseline checkpoint、`TRAIN_STEPS`、`SEED`、evaluation prompts 和 sampling seed。

| ID | What it tests | Command |
|---|---|---|
| E0 | Existing baseline checkpoint | Evaluate the baseline checkpoint directly. |
| E1 | Dynamic SDF direct loss | `bash scripts/train_dynamic_sdf.sh` |
| E2 | Dynamic SDF loss + ranking | `bash scripts/train_dynamic_sdf_contrastive.sh` |

| ID | 测试内容 | 命令 |
|---|---|---|
| E0 | 现有 baseline checkpoint | 直接评估 baseline checkpoint。 |
| E1 | Dynamic SDF 直接损失 | `bash scripts/train_dynamic_sdf.sh` |
| E2 | Dynamic SDF 损失 + ranking | `bash scripts/train_dynamic_sdf_contrastive.sh` |

### E1: direct SDF loss

### E1：直接 SDF 损失

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

### E2: SDF ranking loss

### E2：SDF ranking loss

E2 must start from `BASELINE_CKPT`, not from the E1 checkpoint.

E2 必须从 `BASELINE_CKPT` 开始，而不是从 E1 的 checkpoint 开始。

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

Both scripts always write a final checkpoint such as
`weights/model-final-20000.pt`, even when `SAVE_EVERY` equals `TRAIN_STEPS`.

两个脚本都会写出最终 checkpoint，例如 `weights/model-final-20000.pt`，即使 `SAVE_EVERY` 等于 `TRAIN_STEPS` 也是如此。

The first 500 steps are only for scale checking. In the logs,
`loss_w_sdf × Dynamic SDF Loss` should normally be roughly 5%-10% of the
original training loss. If it is much smaller/larger, change only
`LOSS_W_SDF` by ×10 or ÷10 and rerun E1 once. Do not grid-search many weights.

前 500 step 只用于检查损失尺度。在日志中，`loss_w_sdf × Dynamic SDF Loss` 通常应大约是原始训练损失的 5%-10%。如果它明显更小或更大，只调整 `LOSS_W_SDF`，按 ×10 或 ÷10 改一次，然后重新运行一次 E1。不要对许多权重做网格搜索。

## 4. Evaluation command pattern

## 4. 评估命令模板

Use the existing sampling/evaluation settings for E0, E1 and E2. For example,
adapt the paths, object/scene split and output directory in this command:

E0、E1 和 E2 都使用现有的采样/评估设置。例如，在下面命令中根据需要修改路径、物体/场景划分以及输出目录：

```bash
python train/trainer_control_GAPA_chois.py \
  --window=120 \
  --batch_size=32 \
  --data_root_folder="${DATA_ROOT}" \
  --pretrained_model=/path/to/checkpoint.pt \
  --save_res_folder=./dynamic_sdf_eval/E1_seed1 \
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

For unseen-object evaluation, add `--test_unseen_objects` and keep that flag
identical across E0/E1/E2. The SDF loss is not used during sampling, so no
dynamic-SDF flag is needed in this command.

对于 unseen-object 评估，添加 `--test_unseen_objects`，并确保 E0/E1/E2 中该标志保持完全一致。SDF loss 不会在采样阶段使用，因此这条命令中不需要任何 dynamic-SDF 标志。

## 5. What data to collect

## 5. 需要收集的数据

Create one row per `(experiment, seed, split)` in
`experiments/dynamic_sdf_results_template.csv`.

在 `experiments/dynamic_sdf_results_template.csv` 中，为每个 `(experiment, seed, split)` 创建一行记录。

### Always save

### 始终保存

- `opt.yaml`, git commit hash, baseline checkpoint path, seed and environment;
  - `opt.yaml`、git commit hash、baseline checkpoint 路径、seed 和运行环境；
- the final checkpoint path and any periodic checkpoint you select as best on the fixed validation set;
  - 最终 checkpoint 路径，以及你在固定验证集上选出的任意最佳周期性 checkpoint；
- training curves: total, diffusion, FK, object-points, dynamic-SDF and SDF-ranking losses;
  - 训练曲线：total、diffusion、FK、object-points、dynamic-SDF 和 SDF-ranking losses；
- exact evaluation command, test split, scene/object selection and guidance on/off;
  - 完整评估命令、test split、scene/object 选择以及 guidance 是否开启；
- raw per-sequence metric JSON files, generated `.npz` outputs and at least a
  few rendered success/failure videos.
  - 原始的逐序列 metric JSON 文件、生成的 `.npz` 输出，以及至少几个渲染出的成功/失败视频。

### Main quantitative columns

### 主要定量指标列

- Contact Precision, Recall and F1;
  - Contact Precision、Recall 和 F1；
- `D_hand` on GT contact frames;
  - GT contact frames 上的 `D_hand`；
- hand-object penetration score/depth;
  - 手-物体穿透分数/深度；
- Hand JPE, MPJPE and foot sliding;
  - Hand JPE、MPJPE 和 foot sliding；
- object COM/rotation and condition-following errors;
  - 物体 COM/rotation 误差以及 condition-following 误差；
- Matching Score, R-precision and FID when the standard evaluator is available;
  - 当标准 evaluator 可用时，记录 Matching Score、R-precision 和 FID；
- training throughput, peak GPU memory and inference time.
  - 训练吞吐量、峰值 GPU 显存和推理时间。

### Decision rule after 20k-step screening

### 20k-step 筛选后的决策规则

Move from E1 to E2 only when E1 improves either penetration or `D_hand` by
about 5% without a meaningful Contact-F1/Hand-JPE regression. Keep E2 only if
it beats E1 on the contact-penetration trade-off. Otherwise report dynamic SDF
supervision alone and do not add VLM yet.

只有当 E1 在 penetration 或 `D_hand` 上带来约 5% 的提升，并且没有明显的 Contact-F1/Hand-JPE 退化时，才从 E1 进入 E2。只有当 E2 在 contact-penetration 权衡上优于 E1 时，才保留 E2。否则只报告 dynamic SDF supervision，不要暂时加入 VLM。

## 6. Before a paper table

## 6. 生成论文表格前

After a positive 20k-step result, retrain E0/E1/E2 from scratch (or from the
same declared initialization) with at least three seeds. Use sequence-level,
not sliding-window-level, aggregation for confidence intervals.

在 20k-step 筛选得到正向结果后，使用至少三个 seed 从头重新训练 E0/E1/E2（或从同一个声明过的初始化开始训练）。置信区间应使用 sequence-level 聚合，而不是 sliding-window-level 聚合。
