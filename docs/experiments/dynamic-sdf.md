# Dynamic SDF 有效性实验：U0/U1 操作手册

更新日期：2026-09-04

本实验只回答一个问题：从同一个 MaMi-HOI baseline checkpoint 开始，加入由当前预测手部与物体姿态驱动的 dynamic-SDF loss，是否改善生成结果的接触—穿透权衡，并且不损害 Hand JPE、Contact-F1 和整体动作质量。

当前只运行 U0/U1：

```text
U0：冻结 baseline，只评估
U1：同一 baseline 严格加载 model/EMA，fresh optimizer，从 step 0 微调
```

旧 E2/U6 fixed-target ranking 已有代码但不属于这轮基础验证；`--use_local_sdf`、VLM、ZipMap、G0 截面特征和 U2–U4 全部关闭。G0 是另一条独立的无训练实验，见 [`sectional-prior.md`](sectional-prior.md)。

## 1. 所需输入

设仓库为 `REPO_ROOT`、处理后数据为 `DATA_ROOT`。必须具备：

```text
${REPO_ROOT}/bps.pt
${BASELINE_CKPT}
${DATA_ROOT}/cano_train_diffusion_manip_window_120_joints24.p
${DATA_ROOT}/cano_test_diffusion_manip_window_120_joints24.p
${DATA_ROOT}/cano_min_max_mean_std_data_window_120_joints24.p
${DATA_ROOT}/rest_object_sdf_256_npy_files/<object>.ply.npy
${DATA_ROOT}/rest_object_sdf_256_npy_files/<object>.ply.json
${DATA_ROOT}/rest_object_geo/<object>.ply
${DATA_ROOT}/contact_labels_w_semantics_npy_files/<sequence>.npy
${DATA_ROOT}/smpl_all_models/smplh_amass/male/model.npz
```

完整来源和验收见工作区的 [`dynamic-sdf-data-checklist.md`](../../../docs/data/dynamic-sdf-data-checklist.md)。

## 2. 固定变量与 split

```bash
export REPO_ROOT=/root/autodl-tmp/hoi/src/MaMi-HOI-SDFwithVLM
export DATA_ROOT=/root/autodl-tmp/hoi/data/processed_data
export BASELINE_CKPT=/root/autodl-tmp/hoi/checkpoints/baseline/model-9.pt
export OUTPUT_ROOT=/root/autodl-tmp/hoi/outputs/sdf_gate0
export SPLIT_MANIFEST=/root/autodl-tmp/hoi/outputs/protocol/split_seed1.json
export SMPLH_PATH="${DATA_ROOT}/smpl_all_models/smplh_amass"
export SEED=1
export WANDB_MODE=offline

cd "${REPO_ROOT}"
mkdir -p "$(dirname "${SPLIT_MANIFEST}")" "${OUTPUT_ROOT}"

python scripts/create_experiment_split_manifest.py \
  --data_root_folder="${DATA_ROOT}" \
  --output_path="${SPLIT_MANIFEST}" \
  --validation_sequence_count=100 \
  --seed="${SEED}" \
  --window=120
```

这个 manifest 只生成一次并记录 SHA-256；U0、U1 和独立 G0 都复用它，不能在看到 test 后重新划分。

## 3. Stage A：开跑前必须通过

若 `object_sdf_64` 完全不存在，生成一次且不使用 `--overwrite`：

```bash
test -d "${DATA_ROOT}/object_sdf_64" || python scripts/prepare_object_sdf64.py \
  --data_root_folder="${DATA_ROOT}" \
  --resolution=64
```

随后运行真实 64³/256³ 数值审计、关节审计和单元测试：

```bash
python scripts/audit_sdf64_vs_256.py \
  --data_root_folder="${DATA_ROOT}" \
  --output_folder="${OUTPUT_ROOT}/stage_a/sdf_resolution" \
  --samples_per_object=100000 \
  --seed="${SEED}"

python scripts/audit_joint_ordering.py \
  --data_root_folder="${DATA_ROOT}" \
  --output_folder="${OUTPUT_ROOT}/stage_a/joint_ordering" \
  --split=test \
  --window=120

python -m unittest tests.test_dynamic_sdf -v
bash -n scripts/train_dynamic_sdf.sh scripts/evaluate_u0_u1.sh
```

人工确认 20/22 为同一左侧手腕—手链、21/23 为同一右侧链、无左右交换。真实 SDF sign/unit/axis、64³ 误差、OOB 与 rotation diagnostics 未通过前，不运行 20k。

## 4. U0 冻结基线

```bash
EVAL_CKPT="${BASELINE_CKPT}" ROLE=U0 EVAL_SPLIT=validation GUIDANCE=off \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${BASELINE_CKPT}" ROLE=U0 EVAL_SPLIT=test GUIDANCE=off \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${BASELINE_CKPT}" ROLE=U0 EVAL_SPLIT=test GUIDANCE=on \
  bash scripts/evaluate_u0_u1.sh
```

U0 不训练。guidance-off 是主结果，guidance-on 单独报告。

## 5. U1 300-step CUDA smoke

```bash
TRAIN_STEPS=300 \
SAVE_EVERY=300 \
SMOKE_TEST=1 \
EXP_NAME=U1_smoke_300 \
bash scripts/train_dynamic_sdf.sh
```

只有 `${OUTPUT_ROOT}/U1_smoke_300/smoke_test_report.json` 的 `status` 为 `PASS`，并且 total/dynamic-SDF loss、三类 query gradient、OOB/rotation、throughput、peak CUDA memory、final checkpoint strict reload 均通过，才运行 20k。若 RTX 4090 在冻结的 batch=32 上 OOM，应保存证据并换 48 GB 卡，不要无记录地改变正式协议。

## 6. U1 20k 与固定评估

```bash
TRAIN_STEPS=20000 \
SAVE_EVERY=5000 \
SMOKE_TEST=0 \
EXP_NAME=U1_dynamic_sdf_20k \
bash scripts/train_dynamic_sdf.sh
```

最终 checkpoint：

```text
${OUTPUT_ROOT}/U1_dynamic_sdf_20k/weights/model-final-20000.pt
```

```bash
export U1_CKPT="${OUTPUT_ROOT}/U1_dynamic_sdf_20k/weights/model-final-20000.pt"

EVAL_CKPT="${U1_CKPT}" ROLE=U1 EVAL_SPLIT=validation GUIDANCE=off ENABLE_DYNAMIC_SDF_DIAGNOSTICS=1 \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${U1_CKPT}" ROLE=U1 EVAL_SPLIT=test GUIDANCE=off ENABLE_DYNAMIC_SDF_DIAGNOSTICS=1 \
  bash scripts/evaluate_u0_u1.sh

EVAL_CKPT="${U1_CKPT}" ROLE=U1 EVAL_SPLIT=test GUIDANCE=on ENABLE_DYNAMIC_SDF_DIAGNOSTICS=1 \
  bash scripts/evaluate_u0_u1.sh
```

5k/10k/15k 只能在 validation 上作为诊断；预声明的最终比较 checkpoint 是 20k，不能按 test 选模型。

## 7. 输出与判定

每个 SDF run 自动保存 `run_manifest.json`、`opt.yaml`、`console.log`、逐序列与 aggregate metrics、生成 `.npz`、diagnostics、checkpoint provenance、throughput 和 peak memory。smoke 另有 `smoke_test_report.json`，训练目录下保存 `weights/model-*.pt` 与 `model-final-*.pt`。

必须区分两个结论：smoke/Stage A 全通过只是 **RUN_GATE_PASS**；只有 U1 在冻结的 guidance-off test 上相对 U0 使 `D_hand` 或 legacy mean negative-SDF penetration score 约有 5% 改善，且 Contact-F1、Hand JPE 无明显退化，才记为 **EFFICACY_PASS**。不能把 legacy penetration score 改名为尚未实现的 penetration ratio 或 conditional depth。

更严格的命令与指标边界以工作区 [`experiment-gate0-u0-u1-runbook.md`](../../../docs/current/experiment-gate0-u0-u1-runbook.md) 为准，完整 AutoDL 文件布局与备份见 [`服务器指南.md`](../../../服务器指南.md)。
