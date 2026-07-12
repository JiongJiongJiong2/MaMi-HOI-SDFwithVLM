# MaMi-HOI-SDFwithVLM 文档导航

本目录是改进分支的中文文档入口。

- [动态 SDF 实验操作手册](experiments/dynamic-sdf.md)：按命令完成数据预处理、E0/E1/E2、评估和结果归档。
- [表示能力边界](research/representation-scope.md)：解释为何当前工作只讨论手部区域/接触，而不声称完整手指姿态生成。
- [历史方案](archive/legacy-plans/)：以前的模块规划和旧 Experiment 1 修改记录，仅供追溯。

当前论文路线只有：**动态 SDF 直接监督 → SDF 轨迹排序**。VLM、ZipMap 和旧 `--use_local_sdf` 路径暂不作为本轮实验的一部分。
