# MaMi-HOI-SDFwithVLM 文档导航

当前近期工作只有两个平行、独立的基础实验。

- [Dynamic SDF U0/U1 操作手册](experiments/dynamic-sdf.md)：同一 baseline 下验证 dynamic-SDF loss 是否改善生成接触；真实 CUDA Gate 尚未通过。
- [G0 接触条件极简截面几何](experiments/sectional-prior.md)：不训练模型，验证截面弦是否包含第二只手的接触区域信息；合成 smoke 已通过，真实数据待验证。
- [基础实验结果记录模板](experiments/foundation-results-template.md)：复制为服务器 `outputs/RESULTS.md`，统一记录 provenance、状态、主指标和备份 hash。
- [表示能力边界](research/representation-scope.md)：解释为何当前工作只讨论手部区域/接触，而不声称完整手指姿态生成。
- [历史方案](archive/legacy-plans/)：以前的模块规划和旧 Experiment 1 修改记录，仅供追溯。

SDF U1 不读取 G0 几何，G0 也不读取 SDF 或 checkpoint。两者可以共享数据版本、split manifest 和 seed，但结果分别归档。旧 E2/U6 ranking 虽已有代码，却不属于当前基础验证；VLM、ZipMap、U2–U4、HandSR、联合 diffusion 和旧 `--use_local_sdf` 均后置。
