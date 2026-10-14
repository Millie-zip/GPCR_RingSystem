# GPCR_RingSystem：GPCR 环系统分子生成研究代码

本项目提供 GPCR 环系统相关的数据准备、第三阶段分子扩散训练与推理、独立 RAscore 奖励网络，以及部分环核心与可变侧链生成接口。Python 包名为 `gpcr-ringgen`，导入名为 `gpcr_ringgen`。

截至 2026-10-13，固定拓扑的坐标扩散和原子类型、键类型、坐标联合扩散可以完成数据物化、小型训练、检查点保存与推理。记录训练当前不接入结合口袋或口袋场，要求 `field_conditioning.mode: none` 和 `model.descriptor_dim: null`；独立 JSON 推理支持结合口袋预处理及口袋场条件。

**训练前务必检查并按实验方案修改网络深度。配置中的 3 层是示例值，尚未证明是正式数据的最佳深度。** 当前仍有未修复问题，尤其是噪声预测输出幅度上限 N5。流程跑通和测试通过不能证明模型已收敛，或生成分子具有化学、GPCR 活性及实验合成有效性。

## 1. 训练人员须先检查网络深度

训练脚本直接读取 `--config` 指定的完整 YAML。**请修改实际传入的训练配置；只修改 `configs/model/equivariant_denoiser.yaml`，不会自动覆盖 `configs/training/` 中的参数。**

| 训练对象 | 示例配置 | 深度字段 | 当前示例值 |
| --- | --- | --- | ---: |
| 坐标扩散生成模型 | [stage3_real.yaml](configs/training/stage3_real.yaml) | `model.num_layers` | 3 |
| 联合扩散生成模型 | [stage3_joint_real.yaml](configs/training/stage3_joint_real.yaml) | `model.num_layers` | 3 |
| 独立 RAscore 奖励网络 | [ra_reward.yaml](configs/training/ra_reward.yaml) | `ra_reward_model.num_layers` | 3 |

生成模型配置中的相关字段：

```yaml
model:
  num_layers: 3          # EGNN 消息传递层数；按实验方案修改，3 仅为示例
  hidden_dim: 64        # 隐藏层宽度，与层数分别设置
  coordinate_scale: 0.1 # 内部坐标更新尺度
```

奖励模型使用独立配置，不会继承生成模型的层数：

```yaml
ra_reward_model:
  num_layers: 3   # 奖励网络消息传递层数；按奖励模型的实验方案修改
  hidden_dim: 128
```

`num_layers` 是网络深度，`diffusion.num_timesteps` 是扩散步数，`training.epochs` 是训练轮数。修改后两者不会增加网络层数。深度和宽度应根据任务、数据量及显存做对比实验，保存每次实际使用的 YAML；项目未给出经过验证的最优层数。

### 1.1 不同数据量级的起始配置

配置建议更新于 2026-10-14。**下表是本项目建议的实验起点和候选范围，尚未在正式 GPCR 数据上验证，不是“数据量达到某个阈值就必须用某个深度”的结论。** 每行先运行起始配置，再用固定验证集比较候选；存在明确欠拟合且增加深度带来验证收益时，才采用更深网络。

这里的 `N` 是完成质量控制、去重和训练/验证划分后，**训练部分的有效独立分子数**。同一分子的多个环记录、构象或重复实例不应累加为独立样本。生成模型按自己的训练分子数选行，奖励网络按实际具有有效教师模型标签的训练分子数选行，不能借用生成数据总量。

| 有效训练分子数 N | 生成模型起始配置：层数 / 隐藏层宽度 | 生成深度候选 | 奖励网络起始配置：层数 / 隐藏层宽度 | 奖励深度候选 |
| --- | --- | --- | --- | --- |
| N < 1,000 | 2 / 32 | 2、3 | 2 / 32 | 2、3 |
| 1,000 ≤ N < 10,000 | 3 / 64 | 2、3、4 | 2 / 64 | 2、3 |
| 10,000 ≤ N < 100,000 | 4 / 128 | 3、4、6 | 3 / 128 | 2、3、4 |
| 100,000 ≤ N < 1,000,000 | 6 / 128 | 4、6、8 | 3 / 128 | 2、3、4 |
| N ≥ 1,000,000 | 6 / 128 | 4、6、8 | 4 / 128 | 3、4、6 |

百万级以上先扩大数据覆盖和训练预算，不自动增加到 10 层或更多。分子原子数、图连接方式、骨架多样性、标签质量及显存都会影响配置选择。大分子或路线 A 候选图的成本需要单独评估；上表主要针对本文固定拓扑训练入口。

例如，生成模型有 30 万个有效训练分子，可先覆盖实际训练 YAML 中的以下字段，其他数据、特征和扩散配置保持其合法定义：

```yaml
model:
  num_layers: 6
  hidden_dim: 128
```

如果奖励网络只有约 2,000 个有标签训练分子，应单独从小数据配置开始，不能因为生成模型有 30 万分子而给奖励网络套用相同深度：

```yaml
ra_reward_model:
  num_layers: 2
  hidden_dim: 64
```

历史先导实验的 1,399 条可用记录还需要划分训练/验证部分，其逻辑回归本身也没有 `num_layers` 参数；此表不能用来“修改”旧 joblib 分类器。

选择配置时固定分组划分、特征、词表、扩散目标和评估规则，先单独比较深度，再比较宽度，避免把两者的效果混在一起。记录各分支验证损失、训练与验证差距、显存和吞吐；在相同评估条件下比较生成质量，保留较浅基线和多个随机种子。测试集不参与深度选择。只有训练误差改善而验证表现下降时，不继续据此加深网络。

作为方法参照，[EDM 作者的官方实现](https://github.com/ehoogeboom/e3_diffusion_for_molecules#training-the-edm)在 QM9 示例中采用 9 层、256 隐藏维度，在 GEOM-Drugs 示例中采用 4 层、256 隐藏维度。它们的图构建和网络实现与本仓库不同，不能直接迁移为本项目的最优值。上表的数据量阈值和起始配置是本项目待验证的实验建议，不是该论文给出的样本量—深度对应关系。

改变深度会改变权重结构，不能直接用旧深度的检查点恢复训练。新实验使用新的输出目录；推理时的层数、宽度、词表和扩散参数须与检查点一致。

**N5 仍暂缓：** 噪声预测目前来自受约束的累计坐标位移，无固定锚点时的保守幅度上限为 `2 × num_layers × coordinate_scale`；3 层、坐标更新尺度为 0.1 时，上限为 0.6。增加层数会改变上限，但没有消除结构上的幅度限制，不能据此认定 N5 已修复。新噪声预测输出头未实现，正式训练前需评估这一限制。

### 1.2 N5 的解决方案：独立等变噪声预测输出头

**状态：方案待实施，继续保留用户要求的暂缓决定。本节没有改变现有网络或训练配置。** 上面的量级配置可以用于工程试跑和对照实验，但 N5 会影响当前生成模型的表达能力；在 N5 未解决前，不能把这些配置写成已通过正式训练验收的推荐结构。

当前每层坐标更新经过双曲正切函数和坐标更新尺度缩放，并按节点度数取平均，噪声预测又直接取累计位移。因此，无锚点的输出范数有 `2 × 层数 × scale` 的保守上限；有固定锚点时上限为 `层数 × scale`。训练目标是高斯噪声，结构上不应由这个固定上限截断其可表达范围。

建议保留内部有界坐标更新，另外新增一个等变向量读出头：

| 实施部分 | 具体方案 |
| --- | --- |
| 内部 EGNN 层 | 保留现有双曲正切函数和坐标更新尺度，用于稳定内部坐标与隐藏状态传播 |
| 独立噪声预测头 | 用最终隐藏状态、边特征、距离平方等不变量计算标量系数，再与相对方向组合并按节点聚合；标量多层感知机的最后一层为线性层，不加双曲正切函数或硬幅度裁剪 |
| 最终输出 | 用新读出结果替代 `updated_coordinates - initial_coordinates`，输出范围不再由 `num_layers × coordinate_scale` 决定；初始化幅度通过小权重初始化控制，而不是永久限制输出 |
| 掩码与质心 | 填充节点和固定坐标锚点的噪声预测为 0；无固定锚点时按现有噪声目标投影到零质心子空间，有锚点时保留相应参考系；零有效更新节点的归一化显式保护 |
| 配置及检查点 | 记录读出类型/版本并纳入 N3 的语义校验；新头新增参数，旧检查点不能直接当作新结构的完整权重加载 |

该设计以标量不变量控制相对向量的线性组合，保留旋转/反射等变性和共同平移下的一致性。它移除的是当前固定幅度上限，不保证独立解决图结构、孤立节点或其他模型容量问题。

建议实现位置为新的 `models/equivariant/coordinate_readout.py`，并修改 `models/equivariant/backbone.py`、模型配置类和检查点契约。这是待实现的文件与字段设计，目前没有可开启新头的配置项；不要只在 YAML 中添加一个未经实现的字段就认为 N5 已解决。

实现后应依次完成以下验收：

1. 构造受控权重，证明输出范数可以超过旧的 `2 × 层数 × scale` 上限，并在匹配条件下表达较大噪声目标。
2. 验证旋转、反射、平移和节点置换规则，覆盖 FP32/FP64、填充节点、固定锚点、无更新节点及无边/孤立节点的明确输出策略。
3. 验证梯度有限、输出数值稳定，完成坐标模式和联合模式的小型训练、检查点恢复与两条推理入口回归。
4. 固定数据和验证集，对新旧读出结构做训练对照，报告各项损失、输出/目标幅度和生成质量；仅移除上限或通过单元测试不代表正式训练已经收敛。

新输出头需要训练。若计划复用旧主干网络，需另行定义并记录显式迁移与继续训练流程，不能跳过旧/新参数差异或把随机初始化的新头当成已训练权重。增加层数、坐标更新尺度或训练轮数不作为关闭 N5 的依据。

## 2. 环境准备

推荐 Python 3.10 和 Conda 环境 `GPCR`，最低 Python 版本为 3.10。核心依赖为 PyTorch、NumPy 和 PyYAML，构象及部分预处理需要 RDKit。项目依赖记录在 [requirements.txt](requirements.txt) 和 [pyproject.toml](pyproject.toml)。

首次创建环境，在仓库根目录执行：

```powershell
conda create -n GPCR python=3.10
conda activate GPCR
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/validate_environment.py
```

已有 `GPCR` 环境时直接激活并检查，无需重复创建。GPU 训练前确认 PyTorch 与机器条件兼容；本轮验证使用 CPU，CUDA/FP16 未验证。添加项目依赖时同步维护依赖清单。

## 3. 数据准备

主要流程为：源 SQLite 数据库 → 独立 3D 资产数据库及质量控制记录 → JSONL 训练记录 → 扩散模型训练 → 检查点推理。

先选择训练模式，数据特征清单与模型维度必须相符：

| 模式 | 特征清单 | 原子/边特征维度 | 生成内容 |
| --- | --- | --- | --- |
| `coordinate_only` | [stage3_feature_manifest.yaml](configs/data/stage3_feature_manifest.yaml) | 19 / 8 | 固定原子和键标签，生成坐标 |
| `joint_chemical` | [stage3_joint_feature_manifest.yaml](configs/data/stage3_joint_feature_manifest.yaml) | 3 / 1 | 原子类型、固定边上的键类型及坐标 |

联合模式的输入特征须避免包含目标中未加噪的元素或键阶指标。N6 的自动准入检查尚未完成，训练人员须核对实际特征清单；仅维度或版本名匹配还不够。

下面以联合模式为例，源数据库和输出路径需替换。源数据库须满足预处理接口的表结构；示例路径不代表仓库附带生产数据库：

```powershell
python scripts/preprocess.py --source-db "data/raw/source.db" --output-db "data/interim/stage3_assets.db" --training-jsonl "data/processed/stage3_joint_training.jsonl" --manifest configs/data/stage3_joint_feature_manifest.yaml --progress "outputs/stage3_progress.jsonl"
```

已有完成质量控制的资产数据库时，可单独物化 JSONL：

```powershell
python scripts/materialize_training.py --asset-db "data/interim/stage3_assets.db" --output-jsonl "data/processed/stage3_joint_training.jsonl" --manifest configs/data/stage3_joint_feature_manifest.yaml
```

JSONL 保存节点/边特征、坐标、类别 ID、拓扑和分组元数据。固定原子 ID 顺序为 `C, N, O, S, P, F, Cl, Br, I, H, B, Si`，键顺序为 `SINGLE, DOUBLE, TRIPLE, AROMATIC`。使用当前物化器时保持这套映射；自定义映射需要同步数据构建、训练和推理。

训练/验证依据 `metadata.split_group` 分组，缺失时回退到样本标识（sample_id）。应提前确认同一分子的相关记录不会被拆到不同集合。当前 JSONL 批处理组装流程不把记录中的环掩码和噪声掩码接入训练约束，默认对所有有效原子设置坐标扩散掩码；不能仅因记录保存了环掩码就认为训练已固定环坐标。

N7 的旧 `.idx` 更新问题仍存在。准备完成后固定 JSONL 内容及其索引，追加或替换记录后核对实际读取数量。大型数据库可放在外部数据目录；上游命令见 [预处理说明](scripts/preprocessing/README.md) 和 [数据流程文档](docs/preprocessing.md)。

## 4. 生成模型训练

选择模式并将示例另存为实验配置，例如 `configs/training/stage3_joint_local.yaml`。部分示例仍有历史机器路径，必须替换。训练前核对：

| 配置项 | 要确认的内容 |
| --- | --- |
| `data.path`、`data.feature_manifest` | JSONL 路径和与其一致的特征清单 |
| `model.num_layers`、`model.hidden_dim` | 按实验方案修改并记录的深度、宽度 |
| `model.atom_feature_dim`、`model.edge_feature_dim` | 与特征清单一致的输入维度 |
| `chemical_vocabulary` 和类别数量 | 联合模式的数据、模型和扩散使用同一映射 |
| `training.output_dir` | 本次实验专用目录，保留已有产物 |
| `training.epochs`、`batch_size`、`learning_rate` | 按实验与资源条件设置 |
| `field_conditioning.mode`、`model.descriptor_dim` | 当前记录训练使用 `none`、`null` |
| `device.type`、`device.dtype`、`seed` | 设备、FP32/FP64 和可追溯的随机种子 |

先用小样本确认完整流程，再开展正式实验：

```powershell
python scripts/train.py --config configs/training/stage3_joint_local.yaml
```

输出包括 `last.ckpt`、`best.ckpt`、`metrics.jsonl` 和 `summary.json`。新检查点为版本 2，记录模型、词表、特征、扩散/口袋场配置和训练数据来源。

恢复训练时设置 `training.resume_from`，保留原网络结构、数据版本、分组比例、随机种子及科学语义。N8 尚未修复：恢复会重置历史最佳损失，较差模型可能覆盖 `best.ckpt`，恢复前保留已有最佳产物。N9 也未修复：零损失权重不会自动跳过对应分支，空有效边仍可能触发错误。

## 5. 推理与检查点兼容性

### 从 JSONL 记录推理

在与训练一致的配置中补充 `inference`，另存为推理 YAML：

```yaml
inference:
  checkpoint: runs/stage3_joint_real/best.ckpt
  record_index: 0
  output: outputs/stage3_record_output.json
```

```powershell
python scripts/infer.py --config configs/training/stage3_joint_inference_local.yaml
```

推理可使用新记录，但特征、类别映射、模型层数及扩散参数必须与检查点一致。不要只修改推理 YAML 的深度或词表来尝试旧权重。

### 独立结合口袋与分子骨架 JSON 推理

[stage3_smoke.json](examples/inputs/stage3_smoke.json) 展示单样本结合口袋、分子骨架、描述符和元数据输入。下面的配置使用随机权重，仅用于结构冒烟检查：

```powershell
python scripts/run_stage3_inference.py --config configs/pipeline/stage3_smoke.yaml
```

输出中的 `random_untrained`、`pipeline_smoke_test_only` 表示未训练状态。加载正式检查点时，设置 `pipeline.checkpoint_path`，并提供 `data.feature_manifest` 或顶层内联 `feature_manifest`；模型、口袋场等科学配置要与训练契约一致。JSON 的显式 `diffusion_mask` 可保留固定坐标锚点；联合模式的原子/键类别仍按有效拓扑生成。

旧 9 类坐标模型需显式设置 `model.atom_type_vocab_size: 9` 并保留原尺寸，仍只支持 ID 0–8。旧生成检查点只有配置和特征证据足够时才能兼容；缺少训练数据版本证据时不能恢复训练。裸权重和语义不完整的文件被拒绝加载，不会自动补造历史元数据。详见 [接口与契约](docs/interfaces.md)。

## 6. 独立 RAscore 奖励网络

奖励网络使用第三阶段 JSONL 和外部教师模型生成的 SQLite 标签缓存。需提供可调用的教师模型函数及明确版本；下面的模块名和路径是待替换的示例：

```powershell
python scripts/build_rascore_labels.py --records "data/processed/stage3_joint_training.jsonl" --output "data/processed/rascore_labels.db" --score-function "your_teacher_module:score" --teacher-version "your_teacher_version"
```

构建后检查成功写入和失败数量，缺失标签不能自动当作负例。复制 [ra_reward.yaml](configs/training/ra_reward.yaml) 为本地配置，修改 JSONL、标签缓存、特征清单、输出目录，并单独检查 `ra_reward_model.num_layers` 和 `hidden_dim`：

```powershell
python scripts/train_ra_reward.py --config configs/training/ra_reward_local.yaml
```

奖励检查点保存特征、词表、教师模型版本和数据指纹。缓存逻辑指纹包含按 SMILES 排序的评分与教师模型版本；空缓存或混合教师模型版本会被拒绝。仅有尺寸、缺少语义及来源信息的旧奖励文件不能兼容加载。

接入生成推理时参考 [stage3_joint_rascore_inference.yaml](configs/training/stage3_joint_rascore_inference.yaml)，将奖励训练 `summary.json` 的真实值写入：

```yaml
ra_reward:
  expected_provenance:
    teacher_version: REPLACE_WITH_TRAINING_TEACHER_VERSION
    labels_sha256: REPLACE_WITH_TRAINING_LABELS_SHA256
    records_sha256: REPLACE_WITH_TRAINING_RECORDS_SHA256
```

占位值必须替换，奖励特征和类别顺序也须与生成模型相同。`guidance_scale`、温度等可作为推理实验参数。RAscore 的求解成功标签反映指定搜索策略、原料库和搜索预算下的结果，不能直接解释为实验合成成功概率。

## 7. rascore_pilot_v1 身份隔离与历史产物

`rascore_pilot_v1/` 保存历史 ECFP4 逻辑回归、教师模型表、交叉验证、验证及导出包。它与独立奖励模型（RARewardModel）使用不同流程；先导实验 CSV 不能直接替代第三阶段 JSONL 或标签数据库。

本轮校验 1,400 条训练记录：1,399 条通过规则，1 条明确身份冲突被隔离。其中 999 条保存的完整 InChIKey 一致，400 条缺键但输入包含立体化学信息的 SMILES 与教师模型目标分子完全一致；这些身份字段尚未独立重算。

本地结果在 `outputs/rascore_identity_v1/`，包括 `training_eligible.csv`、`quarantined.csv`、`audit.json` 和 `README.md`。该目录被 Git 忽略，不随克隆自动获得。复现时使用新的目录：

```powershell
python scripts/audit_rascore_identity.py --source rascore_pilot_v1/rascore_teacher_combined_1400.csv --output-dir outputs/rascore_identity_v2
```

原始表、模型、交叉验证和导出包保留。**N10 仅完成数据阶段，旧模型和指标尚未基于清洗数据重新生成。** 原训练/特征提取/交叉验证/评估流程仍缺失，需补齐后重建；不能从旧交叉验证删去一行后沿用原指标。历史采样及 AiZynthFinder 配置还有路径和导入副作用问题。详见 [身份质量状态](rascore_pilot_v1/IDENTITY_STATUS.md)。

## 8. 路线 A 与其他接口

路线 A 的接口和测试已实现：按环特征、描述符生成外部原子数量，动态扩展候选图，生成原子类型、带 `NO_BOND` 的键状态和 3D 坐标。环核心的节点/边存在性固定；环原子类型、已有环键类型及环坐标可生成，环形式电荷和手性保持条件值。候选图按本批实际原子数扩展，`runtime_atom_limit` 用于分配保护。

相关配置：[ring_sidechain.yaml](configs/diffusion/ring_sidechain.yaml)、[ring_sidechain_denoiser.yaml](configs/model/ring_sidechain_denoiser.yaml)。完整训练衔接仍需核对；本文 `scripts/train.py` 的 JSONL 流程使用固定拓扑 DDPM，不能据此宣称已完成路线 A 正式训练。

第三阶段使用 `atom_mask`、`edge_mask`、`diffusion_mask` 处理有效节点、边和坐标更新区域。口袋场推理支持连续的占据值、最近距离查询与坐标还原。细节见 [interfaces.md](docs/interfaces.md) 和 [Stage 3 流程解析](docs/Stage3代码流程详细解析_固定拓扑联合扩散.md)。

## 9. 测试与遗留问题

本轮最后一次完整测试为 **226 项通过、2 项跳过**，跳过 CUDA 与缺少 RDKit 的构象测试。修改代码已编译检查；正式数据长期训练、外部教师模型、RDKit/InChI 身份重算、真实对接及 CUDA/FP16 未验证。

在项目根目录执行：

```powershell
python scripts/validate_environment.py
python -m pytest -q
```

系统临时目录不可写时，为 pytest 指定专用的新目录；本轮小型张量测试使用单线程：

```powershell
python -s -c "import torch, pytest; torch.set_num_threads(1); raise SystemExit(pytest.main(['-q', '--tb=short', '--basetemp=outputs/pytest_readme_check']))"
```

| 审查问题 | 当前状态 |
| --- | --- |
| N1–N4：推理、词表、检查点语义和分类后验 | 已完成本地修复及回归测试 |
| N5：噪声预测输出幅度硬上限 | 用户要求暂缓，问题仍存在 |
| N10：RAscore 身份冲突 | 数据隔离完成，模型/交叉验证重建待处理 |
| N6–N9、N11–N16 | 10 项 P2 未修复 |

完整范围及兼容性变化见 [本次修改说明](docs/本次修改说明.md) 和 [代码审查](docs/代码审查.md)。近期修改目前仍在本地工作树，尚未提交或推送；后续修复继续逐项说明方案并取得确认。

## 10. 仓库目录

| 目录 | 内容 |
| --- | --- |
| `src/gpcr_ringgen/` | 数据契约、预处理、模型、扩散、训练、采样和检查点校验 |
| `scripts/` | 物化、训练、推理、教师模型缓存和身份审计入口 |
| `configs/` | 特征清单、模型、扩散及推理配置 |
| `tests/` | 单元测试和训练到推理的集成回归 |
| `examples/` | 标准输入示例 |
| `rascore_pilot_v1/` | 历史先导实验数据、模型及质量状态 |
| `docs/` | 接口、流程、审查和修改说明 |
| `data/`、`runs/`、`outputs/` | 本地数据、训练及审计输出，大部分内容被 Git 忽略 |
