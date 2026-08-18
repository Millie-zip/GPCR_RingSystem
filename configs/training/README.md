# Stage 3 训练配置

使用 `scripts/train.py --config <yaml>` 启动训练。配置至少需要提供：

- `data.path`：真实数据物化后的 `stage3_training_record_v1` JSONL；
- `data.feature_manifest`：与该 JSONL 同批生成的特征清单；
- `training`：epoch、batch、切分比例、输出目录；
- `device` 和 `seed`；
- `diffusion`、`chemical_vocabulary`、`model`；
- `field_conditioning.mode: none`。

训练数据记录必须包含已有 3D 坐标、固定拓扑、原子/键类别和已确认的特征值。`ring_atom_mask`、`noise_mask` 可以保存在记录中用于测试审计，但训练 collate 不会把它们放进 loss。

训练生成的 `last.ckpt` / `best.ckpt` 可以直接作为现有 Stage 3 推理管线的 checkpoint；推理配置也应使用相同的模型维度、类别词表和 `field_conditioning.mode: none`。
