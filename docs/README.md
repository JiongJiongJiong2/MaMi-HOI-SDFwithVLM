# MaMi-HOI-SDFwithVLM 文档导航

## 当前主路线

- [下一阶段实验路线](research/next-stage-experiment-roadmap-2026-09-16.md)：仓库内唯一主路线图，记录保留配置、实验依赖、冲突规则、遗漏项和停止条件。
- [Forward-inverse consistency protocol](research/forward-inverse-consistency-protocol-2026-09-16.md)：尚未实现的时间一致性前置诊断协议。
- [本周研究成果评价](research/weekly-research-assessment-2026-09-16.md)：2026-09-16 的完整证据审计和下一步取舍。
- [Analytic contact baseline](experiments/analytic-contact-baseline.md)：canonical rerun 和晋级条件。
- [Contact episode baseline](experiments/contact-episode-baseline-2026-09-18.md)：stable contact、onset delay 和 false-contact 诊断结果。
- [Forward-inverse consistency](experiments/forward-inverse-consistency-2026-09-18.md)：E2 时间一致性诊断；当前表示 NO-GO。
- [Sequence-level contact dataset](experiments/contactopt-sequence-dataset-2026-09-19.md)：404 个序列不重叠连续窗口的冻结清单与覆盖限制。
- [Sequence ContactOpt and smoothing](experiments/contactopt-sequence-contact-and-temporal-2026-09-19.md)：376 个 train/dev 窗口的原始与五 tap 平滑结果；固定平滑仍为 NO-GO。
- [E3 hand data gate](experiments/e3-hand-data-gate-2026-09-18.md)：CPU-only finger supervision 和 MANO/SMPL-X 资产审计。
- [WM Stage 1 learned residual](experiments/wm-stage1-learned-residual.md)：当前 residual 分支的 NO-GO 决策。

## 基础实验与边界

- [Dynamic SDF U0/U1 操作手册](experiments/dynamic-sdf.md)：同一 baseline 下验证 dynamic-SDF loss 是否改善生成接触；真实 CUDA Gate 尚未通过。
- [G0 接触条件极简截面几何](experiments/sectional-prior.md)：不训练模型，验证截面弦是否包含第二只手的接触区域信息；合成 smoke 已通过，真实数据待验证。
- [基础实验结果记录模板](experiments/foundation-results-template.md)：复制为服务器 `outputs/RESULTS.md`，统一记录 provenance、状态、主指标和备份 hash。
- [表示能力边界](research/representation-scope.md)：解释为何当前工作只讨论手部区域/接触，而不声称完整手指姿态生成。
- [历史方案](archive/legacy-plans/)：以前的模块规划和旧 Experiment 1 修改记录，仅供追溯。

SDF U1 不读取 G0 几何，G0 也不读取 SDF 或 checkpoint。两者可以共享数据版本、split manifest 和 seed，但结果分别归档。旧 E2/U6 ranking 虽已有代码，却不属于当前基础验证；VLM、ZipMap、U2–U4、HandSR、联合 diffusion 和旧 `--use_local_sdf` 均后置。
