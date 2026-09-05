# Stage 3 训练配置

使用 `scripts/train.py --config <yaml>` 启动训练。配置至少需要提供：

- `data.path`：真实数据物化后的 `stage3_training_record_v1` JSONL；
- `data.feature_manifest`：与该 JSONL 同批生成的特征清单；
- `training`：epoch、batch、切分比例、输出目录；
- `device` 和 `seed`；
- `diffusion`、`chemical_vocabulary`、`model`；
- `field_conditioning.mode: none`。

训练数据记录必须包含已有 3D 坐标、固定拓扑、原子/键类别和已确认的特征值。`ring_atom_mask`、`noise_mask` 可以保存在记录中用于测试审计，但训练 collate 不会把它们放进 loss。

联合化学与坐标扩散使用 `configs/training/stage3_joint_real.yaml`。该模式要求
`stage3_joint_feature_manifest_v1`，训练器会分别对坐标、原子类别和固定边键类别
计算 masked loss；拓扑本身不扩散，也不存在 `NO_BOND` 类别。

训练生成的 `last.ckpt` / `best.ckpt` 可以直接作为现有 Stage 3 推理管线的 checkpoint；推理配置也应使用相同的模型维度、类别词表和 `field_conditioning.mode: none`。

独立 RA 奖励模型使用 `scripts/build_rascore_labels.py` 生成标签缓存，再用
`scripts/train_ra_reward.py --config configs/training/ra_reward.yaml` 训练。推理时
可参考 `configs/training/stage3_joint_rascore_inference.yaml`，在 `ra_reward.checkpoint`
填入 RA reward 的 `best.ckpt`。该引导只修正 atom/bond logits，不覆盖生成模型
checkpoint，也不会修改固定拓扑。
