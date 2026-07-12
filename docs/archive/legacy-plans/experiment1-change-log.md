# Experiment 1 修改记录：Original MaMi-HOI + Basic SDF Query

## 概述

本次修改实现了实验 1（SDF-enhanced GAPA），通过 `--use_local_sdf` flag 控制。所有新增可学习模块均**零初始化**，确保模型初始行为与原始 MaMi-HOI 完全一致；flag 关闭时行为与原代码 bit-identical。

## 新建文件

### `manip/model/sdf_utils.py`

三个纯函数（无 `nn.Module`）：

- **`sample_sdf_at_points(sdf_grid, query_points, origin, voxel_size)`**
  - 用 `F.grid_sample` 在局部 SDF 网格上批量查询 SDF 值
  - 输入：`sdf_grid [B,1,D,H,W]`, `query_points [B,N,3]`, `origin [B,3]`, `voxel_size scalar/[B,1]`
  - 输出：`sdf_values [B,N,1]`
  - 参考现有 `compute_signed_distances()` (trainer:66-90) 的归一化和轴交换逻辑

- **`compute_sdf_gradients_fd(sdf_grid, query_points, origin, voxel_size, eps=1e-3)`**
  - 通过中心有限差分计算 SDF 梯度：`(sdf(p+εe_i) - sdf(p-εe_i)) / 2ε`
  - 输出：`gradients [B,N,3]`

- **`extract_contact_points(jpos, hand_joints=[20,21,22,23])`**
  - 从 24 关节位置中提取手腕/手部关节作为 SDF 查询点
  - 输入：`jpos [B,T,24,3]` → 输出：`contact_points [B,T,4,3]`

## 修改文件

### `manip/data/cano_traj_dataset.py`

**`__init__`**：
- 新增参数 `use_local_sdf=False`
- 新增 `self.use_local_sdf = use_local_sdf`
- 当 `use_local_sdf=True` 时，设置 `self.local_sdf_folder = os.path.join(data_root_folder, "local_sdf_patches")`

**`__getitem__`**（在 `data_input_dict` 构建末尾，`use_object_keypoints` 块之后）：
- 当 `self.use_local_sdf` 为 True 时，从 `{local_sdf_folder}/{seq_name}.pt` 加载：
  - `local_sdf_grid` [1, 64, 64, 64]
  - `local_sdf_origin` [3]
  - `local_sdf_voxel_size` scalar
  - `hand_query_points` [T, 4, 3]（不足 window 长度时 zero-padding）
- 文件不存在时 fallback 为零张量

---

### `manip/model/control_GAPA_transformer_module.py`

**`GeometryAwareProximityAdapter.__init__`**（在 `self.fc` / `self.layer_norm` 零初始化之后）：

```python
# SDF attention bias: SDF值 → 标量偏置 (零初始化)
self.sdf_bias_mlp = nn.Sequential(
    nn.Linear(1, 32), nn.ReLU(), nn.Linear(32, 1))
nn.init.zeros_(self.sdf_bias_mlp[-1].weight)
nn.init.zeros_(self.sdf_bias_mlp[-1].bias)

# SDF feature projection: SDF特征 → 值向量增强 (零初始化)
self.sdf_feat_proj = nn.Linear(128, n_head * d_v)
nn.init.zeros_(self.sdf_feat_proj.weight)
nn.init.zeros_(self.sdf_feat_proj.bias)
```

**`GeometryAwareProximityAdapter.forward`**：

- 签名改为：`forward(self, motion_feat, bps_feat, sdf_feat=None, sdf_values=None, sdf_gradients=None)`

- 在 `attn_score = content_score + dist_bias` 之后添加 SDF 注意力偏置：
  ```python
  if sdf_values is not None:
      sdf_bias = self.sdf_bias_mlp(sdf_values)  # [B,T,1]
      attn_score = attn_score + sdf_bias[:, None, :, :]  # broadcast to [B,H,T,T]
  ```

- 在 `v_final = v_expanded + rel_geo_feat` 之后添加 SDF 值增强：
  ```python
  if sdf_feat is not None:
      sdf_v = self.sdf_feat_proj(sdf_feat).view(bs, n_q, self.n_head, self.d_v)
      sdf_v = sdf_v.transpose(1, 2).unsqueeze(2)  # [B,H,1,T,d_v]
      v_final = v_final + sdf_v
  ```

---

### `manip/model/transformer_control_GAPA_motion_cond_diffusion.py`

**新增导入**：
```python
from manip.model.sdf_utils import sample_sdf_at_points, compute_sdf_gradients_fd
```

**`TransformerDiffusionModel.__init__`**（在 `self.bps_proj` 之后）：
```python
self.sdf_value_proj = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 128))
nn.init.zeros_(self.sdf_value_proj[-1].weight)
nn.init.zeros_(self.sdf_value_proj[-1].bias)
```

**`TransformerDiffusionModel.forward`**：
- 签名新增：`local_sdf_grid=None, local_sdf_origin=None, local_sdf_voxel_size=None, hand_query_points=None`
- 在 GAPA 调用前插入 SDF 查询逻辑：
  ```python
  sdf_feat, sdf_values_avg = None, None
  if local_sdf_grid is not None and hand_query_points is not None:
      with torch.cuda.amp.autocast(enabled=False):
          B, T, K, _ = hand_query_points.shape
          flat_query = hand_query_points.reshape(B, T * K, 3)
          sdf_values_flat = sample_sdf_at_points(
              local_sdf_grid.float(), flat_query.float(),
              local_sdf_origin, local_sdf_voxel_size)
          sdf_values = sdf_values_flat.reshape(B, T, K, 1)
          sdf_values_avg = sdf_values.mean(dim=2)  # [B,T,1]
      sdf_feat = self.sdf_value_proj(sdf_values_avg)  # [B,T,128]
  gaa_residual = self.gaa_block(motion_feats, bps_emb, sdf_feat=sdf_feat,
                                 sdf_values=sdf_values_avg, sdf_gradients=None)
  ```

**`ObjectCondGaussianDiffusion` 方法链 SDF 参数透传**：

以下所有方法均新增 4 个可选参数 `local_sdf_grid=None, local_sdf_origin=None, local_sdf_voxel_size=None, hand_query_points=None`，并在内部调用中透传：

| 方法 | 修改内容 |
|------|---------|
| `p_mean_variance()` | 新增参数，传给 `denoise_fn()` |
| `p_mean_variance_reconstruction_guidance()` | 新增参数，传给 `denoise_fn()` |
| `p_sample_guided_reconstruction_guidance()` | 新增参数，传给 `p_mean_variance_reconstruction_guidance()` 和 `p_mean_variance()` |
| `p_sample_loop_guided()` | 新增参数，传给 `p_sample_guided_reconstruction_guidance()` 和 `p_sample()` |
| `p_sample()` | 新增参数，传给 `p_mean_variance()` |
| `p_sample_loop()` | 新增参数，传给 `p_sample()` |
| `sample_sliding_window_w_canonical()` | 新增参数，传给 `p_sample_loop_sliding_window_w_canonical()` |
| `p_sample_loop_sliding_window_w_canonical()` | 新增参数，传给内部 `p_sample()` 和 `p_sample_guided_reconstruction_guidance()` 调用 |
| `sample()` | 新增参数，传给 `p_sample_loop()` 和 `p_sample_loop_guided()` |
| `ddim_sample()` | 新增参数，传给内部 `denoise_fn()` 调用 |
| `p_losses()` | 新增参数，传给 `denoise_fn()` |
| `forward()` | 新增参数，传给 `p_losses()` |

---

### `train/trainer_control_GAPA_chois.py`

**`parse_opt()`**：
```python
parser.add_argument("--use_local_sdf", action="store_true", default=False,
                    help="Use local SDF volumes for SDF-enhanced GAPA attention")
```

**`Trainer.__init__`**：
```python
self.use_local_sdf = self.opt.use_local_sdf
```

**`Trainer.prep_dataloader`**：
- `CanoObjectTrajDataset` 构造新增 `use_local_sdf=self.use_local_sdf`（训练集和验证集）

**`Trainer.train`**（训练循环）：
- 在 `rest_human_offsets` 提取后，新增 SDF 数据提取：
  ```python
  if self.use_local_sdf:
      local_sdf_grid = data_dict['local_sdf_grid'].cuda()
      local_sdf_origin = data_dict['local_sdf_origin'].cuda()
      local_sdf_voxel_size = data_dict['local_sdf_voxel_size'].cuda()
      hand_query_points = data_dict['hand_query_points'].cuda()
  else:
      local_sdf_grid = local_sdf_origin = local_sdf_voxel_size = hand_query_points = None
  ```
- `self.model()` 调用新增 4 个 SDF 参数

**验证循环**（`train()` 内部的 val 采样）：
- 同样提取 SDF 数据并传给 `self.model()` 和 `self.ema.ema_model.sample()`

**`Trainer.cond_sample_res`**：
- 提取 SDF 数据并传给 `self.ema.ema_model.sample()`

**`Trainer.cond_sample_res_key_finding`**：
- 提取 SDF 数据并传给 `self.ema.ema_model.sample()`

**`Trainer.cond_sample_res_w_long_planned_path`**：
- 提取 SDF 数据并传给 `self.ema.ema_model.sample_sliding_window_w_canonical()`

---

## 数据流

```
Dataset.__getitem__() → data_dict
  ├─ local_sdf_grid      [1, 64, 64, 64]
  ├─ local_sdf_origin    [3]
  ├─ local_sdf_voxel_size scalar
  └─ hand_query_points   [T, 4, 3]

Trainer.train() → 提取 SDF 字段 → model(..., local_sdf_grid, local_sdf_origin, local_sdf_voxel_size, hand_query_points)
  → ObjectCondGaussianDiffusion.forward() → p_losses() → denoise_fn()
    → TransformerDiffusionModel.forward():
        1. SDF 查询: sdf_values = sample_sdf_at_points(local_sdf_grid, hand_query_points, ...)
        2. SDF 特征投影: sdf_feat = sdf_value_proj(sdf_values_avg)  [B, T, 128]
        3. GAPA 调用: gaa_block(motion_feats, bps_emb, sdf_feat, sdf_values_avg, sdf_gradients=None)
        4. GAPA 内部:
           - attn_score += sdf_bias_mlp(sdf_values)  [SDF 注意力偏置]
           - v_final += sdf_feat_proj(sdf_feat)       [SDF 值增强]
```

## 零初始化保证

| 模块 | 初始化方式 | 初始输出 |
|------|-----------|---------|
| `sdf_bias_mlp` (最后一层) | `nn.init.zeros_` | 0 → attn_score 不变 |
| `sdf_feat_proj` | `nn.init.zeros_` | 0 → v_final 不变 |
| `sdf_value_proj` (最后一层) | `nn.init.zeros_` | 0 → sdf_feat=0 传给 GAPA，但 GAPA 的 sdf_feat_proj 也是零，双重保险 |

## 实验切换方式

| 实验 | 命令行参数 |
|------|-----------|
| Baseline | （无新参数） |
| 实验 1 | `--use_local_sdf` |
| 实验 2 | `--use_local_sdf --use_stateful_sdf` (待实现) |
| 实验 3 | `--use_vlm_keypoints` (待实现) |
| 实验 4 | `--use_local_sdf --use_stateful_sdf --use_vlm_keypoints` (待实现) |

## 待做

- **预处理脚本 `scripts/precompute_local_sdf.py`**：生成 64³ 局部 SDF 数据（暂缓实现）
- **实验 2**：`--use_stateful_sdf` ZipMap 有状态查询
- **实验 3**：`--use_vlm_keypoints` VLM 语义关键点
- **实验 4**：全部组合
