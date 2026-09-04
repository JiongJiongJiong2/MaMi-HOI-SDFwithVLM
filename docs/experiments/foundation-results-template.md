# 两个基础实验结果记录

> 将本文件复制到服务器 `outputs/RESULTS.md` 后填写。`RUN_GATE_PASS` 只表示运行可信；只有冻结 test 指标达到预声明门槛才能写 `EFFICACY_PASS`。

## 会话身份

| 字段 | 值 |
|---|---|
| 日期 | |
| 代码 commit | |
| git 是否 dirty；若是，diff 存放位置 | |
| split manifest SHA-256 | |
| baseline checkpoint SHA-256 | |
| GPU / 显存 | |
| Python / PyTorch / CUDA / PyTorch3D | |
| `pip-freeze.txt` | `outputs/provenance/pip-freeze.txt` |

## 运行台账

| 实验 | split | guidance | seed | 状态 | 运行时间 | 峰值显存 | 结果目录 | 一句决策 |
|---|---|---|---:|---|---:|---:|---|---|
| G0 synthetic | synthetic | n/a | 1 | | | n/a | `g0_sectional/synthetic` | |
| G0 | validation | n/a | 1 | | | n/a | `g0_sectional/validation_seed1` | |
| G0 | test | n/a | 1 | | | n/a | `g0_sectional/test_seed1` | |
| U0 | validation | off | 1 | | | | `sdf_gate0/U0_validation_guidance-off` | |
| U0 | test | off | 1 | | | | `sdf_gate0/U0_test_guidance-off` | |
| U0 | test | on | 1 | | | | `sdf_gate0/U0_test_guidance-on` | |
| U1 smoke 300 | validation | off | 1 | | | | `sdf_gate0/U1_smoke_300` | |
| U1 20k | train | n/a | 1 | | | | `sdf_gate0/U1_dynamic_sdf_20k` | |
| U1 | validation | off | 1 | | | | `sdf_gate0/U1_validation_guidance-off` | |
| U1 | test | off | 1 | | | | `sdf_gate0/U1_test_guidance-off` | |
| U1 | test | on | 1 | | | | `sdf_gate0/U1_test_guidance-on` | |

允许的状态：`NOT_RUN`、`BLOCKED`、`RUN_GATE_PASS`、`EFFICACY_PASS`、`EFFICACY_FAIL`。不要用 `DONE` 代替科学结论。

## G0 冻结指标

| split | best chord top-5 | center-antipode top-5 | 差值 | median target-to-chord norm | gate | `summary.json` |
|---|---:|---:|---:|---:|---|---|
| validation | | | | | | |
| test | | | | | | |

validation 规则是否在 test 前冻结：`是 / 否`  
主要失败物体/序列：  
G0 结论：`PASS / FAIL / BLOCKED`  
下一步：

## SDF U0/U1 冻结指标

主结果只填 guidance-off test；guidance-on 另表报告。

| 指标 | U0 | U1 20k | 相对变化 | 是否支持假设 |
|---|---:|---:|---:|---|
| Contact Precision | | | | |
| Contact Recall | | | | |
| Contact F1 | | | | |
| D_hand (mm) | | | | |
| legacy mean negative-SDF penetration score | | | | |
| Hand JPE | | | | |
| root-relative MPJPE | | | | |
| foot sliding | | | | |
| object COM error | | | | |
| object rotation error | | | | |

Stage A：`PASS / FAIL / BLOCKED`  
300-step smoke：`PASS / FAIL / BLOCKED`  
SDF 有效性：`PASS / FAIL / BLOCKED`  
下一步：

## 最终边界

SDF 与 G0 是否分别独立为正：  
是否允许设计未来四组组合消融：`是 / 否`  
不得扩大声称的限制：

## 备份

| 文件 | SHA-256 | 已复制到本地/文件存储 |
|---|---|---|
| evidence archive | | |
| U1 final checkpoint | | |
| split manifest | | |
