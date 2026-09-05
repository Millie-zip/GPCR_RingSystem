# Stage 3 真实数据特征清单

训练入口要求 `data.feature_manifest` 指向一个与 JSONL 资产库同步生成的 YAML 文件。代码不会根据列宽猜测特征含义，也不会把当前 smoke 配置中的维度当作正式科学特征。

清单的最小结构如下；其中名称必须替换为已经确认的真实特征语义：

```yaml
schema_version: stage3_feature_manifest_v1
features:
  atom: [confirmed_atom_feature_1]
  edge: [confirmed_edge_feature_1]
  descriptor: []
  pocket: []
```

当前训练阶段要求 `descriptor` 和 `pocket` 为空，`field_conditioning.mode` 必须为 `none`。如果后续确认了描述符或 pocket 特征，需要同步修改清单、记录生成器和模型配置后再训练。

联合化学扩散使用 `stage3_joint_feature_manifest_v1`。该清单只允许固定拓扑
和明确声明的固定条件进入特征；原子元素、原子序数、键类别、芳香性和共轭性
等 clean chemical label 不得重复放入输入特征。原子类别和已有边的键类别由
`atom_types`、`bond_types` 单独作为扩散目标传入训练器。
