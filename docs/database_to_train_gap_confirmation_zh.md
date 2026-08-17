# 5-HT2C 分子库到 `train` 输入的现状与缺口确认

> 文档性质：实施前确认稿，不是转换程序。
>
> 本文只依据现有数据库检查报告、构建记录和当前 `train` 分支代码得出结论。本文不会修改原数据库，也不把 smoke test 的占位维度当成正式特征定义。

## 1. 本次确认的对象

- 原始数据库：`/share/home/weiliao02/lyx/zinc_project/5ht2c_3d_diffusion_library_v1/5ht2c_3d_diffusion_library_v1.db`
- 检查报告生成时间：2026-08-16
- 报告记录的数据库文件大小：437,522,702,336 bytes，约 437.5 GB（十进制），即约 407.5 GiB
- 当前模型代码：GitHub `train` 分支对应的本地代码
- 当前阶段目标：先构建不依赖 pocket 的三维坐标训练资产；pocket 坐标系约束留到以后生成或 pocket-conditioned 阶段处理

## 2. 现有数据库里已经有什么

### 2.1 已确认的数据规模

| 内容 | 已确认数量 | 说明 |
|---|---:|---|
| 通过 pocket 预筛的 chemical scaffolds | 314,659 | PASS 299,633；BORDERLINE 15,026 |
| 完整分子 `selected_molecules` | 15,601,969 | 已从上游分片库回收；缺失记录为 0 |
| molecule–scaffold 关系 | 17,530,765 | 一个分子可对应一个或多个 scaffold |
| 环系统 `selected_ring_systems` | 17,554,723 | 构建元数据中的完成统计 |
| scaffold variants | 86,222,700 | 构建元数据中的完成统计 |
| attachment sites | 30,333,882 | 构建元数据中的完成统计 |
| topologies | 32,399 | 构建元数据中的完成统计 |

注意：15,601,969 是完整分子数，17,554,723 是环系统记录数。因此最终训练样本若按“分子 + 环系统”定义，同一个分子的 `x0` 可能对应多个不同的环掩码。

### 2.2 完整分子和二维图信息

`selected_molecules` 已保存：

- 唯一来源定位：`source_part`、`molecule_uid`、`source_rowid`、`source_record_id`；
- `original_smiles`；
- `canonical_isomeric_smiles` 和 `canonical_nonisomeric_smiles`；
- 分子式、原子数、重原子数、键数、形式电荷、分子量；
- `atom_facts_json`：按 `atom_index` 排列的原子属性；
- `bond_facts_json`：按 `bond_index` 排列的键及其两个端点；
- degree、bond-order sum、formal-charge 等序列。

从样例和现有导出代码可确认：

- `atom_facts_json` 提供原子序号、元素、原子序数、形式电荷、芳香性、同位素、手性标签、度、氢数、价态等；
- `bond_facts_json` 提供键序号、起止原子序号、键类型、键级、芳香性、共轭性、是否在环中和立体信息；
- 当前记录的是完整分子的重原子图，样例中 `atom_count == heavy_atom_count`。

这些数据已经足够重建模型需要的二维分子图，不需要重新做 fused-ring 筛选、scaffold 去重或完整分子回映射。

### 2.3 环、连接位点和拓扑信息

`selected_ring_systems` 已保存：

- `ring_system_uid` 以及其所属 `molecule_uid`；
- `ring_atom_indices_json` 和 `ring_bond_indices_json`；
- basis rings、共享原子、共享键、环大小、cycle rank 和 ring-system type。

因此可以直接按原数据库索引生成：

- `ring_atom_mask[i] = true`：原子 `i` 属于当前环系统；
- `noise_mask[i] = true`：原子 `i` 在当前环系统之外，需要参与扩散加噪；
- 当前 `train` 代码中的名称是 `diffusion_mask`，语义与这里的 `noise_mask` 相同。

`selected_attachment_sites` 还保存环原子到环外原子的连接关系，但当前模型已确认不需要 `attachment_index` 或 `attachment_mask`，所以它可用于审计，不是当前训练输入必填项。

### 2.4 已有但本阶段不使用的信息

数据库中还保留了 pocket prefilter 结果及 scaffold–molecule 映射。它们说明分子来源和筛选历史，但本阶段不要求把生成的三维坐标放进 8DPF pocket 坐标系，也不要求再次进行 docking。

## 3. 当前数据库明确没有什么

现有数据库不是可直接训练的 3D tensor 数据集。当前没有确认存在以下内容：

1. `x0 [N,3]`：与数据库原子顺序严格对齐的真实/构象三维坐标；
2. 3D 生成程序使用的 RDKit 原子序号到源库 `atom_index` 的显式双射映射；
3. 每条 3D 记录的生成状态、失败原因和质控结果；
4. 可直接被 PyTorch Dataset/DataLoader 读取的正式训练样本表或分片格式；
5. 正式冻结的 `atom_features`、`edge_features` 和 descriptor 数值定义；
6. train/validation/test split 以及防止同 scaffold 或同拓扑泄漏的划分结果。

## 4. 当前 `train` 分支真正要求什么

### 4.1 图和坐标张量

当前 `MolecularGraph` / `Stage3DiffusionState` 的结构要求：

| 模型字段 | 形状 | 当前状态 | 可从哪里获得 |
|---|---|---|---|
| `atom_types` | `[B,N]` | 必需 | `atom_facts_json.symbol` 转固定类别编号 |
| `edge_index` | `[B,2,E]` | 必需 | `bond_facts_json` 的 begin/end index |
| `bond_types` | `[B,E]` | 必需 | `bond_facts_json.bond_type` 转固定类别编号 |
| `atom_features` | `[B,N,F_atom]` | 接口必需，但正式列语义未冻结 | 不能把 smoke test 的 4 列当正式定义 |
| `edge_features` | `[B,E,F_edge]` | 接口必需，但正式列语义未冻结 | 不能把 smoke test 的 2 列当正式定义 |
| `coordinates` / `x0` | `[B,N,3]` | 训练必需，原库缺失 | 需要新增 3D 构象生成 |
| `atom_mask` | `[B,N]` | 必需 | batching 时由真实原子数产生 |
| `edge_mask` | `[B,E]` | 必需 | batching 时由真实键数产生 |
| `diffusion_mask` | `[B,N]` | Stage3 必需 | 由 `ring_atom_indices_json` 生成 |

固定类别词表已经明确：

- 原子：`C=0, N=1, O=2, S=3, P=4, F=5, Cl=6, Br=7, I=8`；
- 键：`SINGLE=0, DOUBLE=1, TRIPLE=2, AROMATIC=3`；
- 没有 unknown 类，词表外元素或键类型必须失败并记录，不能随意映射。

每条无向化学键在数据中保存一次即可；模型代码的图处理负责后续消息传递。不能为了“看起来双向”擅自重复数据库键。

### 4.2 环内固定、环外加噪

当前前向 DDPM 和损失函数已经按 `diffusion_mask` 工作：

- 环内原子：`diffusion_mask=false`，保留 `x0`，不加噪，也不进入噪声预测损失；
- 环外原子：`diffusion_mask=true`，参与前向加噪和噪声预测损失；
- padding：`atom_mask=false`，不参与任何计算；
- 每条训练样本必须至少有一个 `diffusion_mask=true` 的原子，否则当前校验会拒绝该样本。

虽然模型直接使用的是 `diffusion_mask`，输出数据库仍建议同时保存 `ring_atom_mask` 和 `noise_mask`，并强制验证二者在真实原子范围内互补。

### 4.3 Descriptor 和 pocket 的真实状态

- smoke 配置中的 `F_atom=4`、`F_edge=2`、`D=3`、`F_pocket=2` 只是测试占位值，不是正式 schema；
- descriptor 已确认是可选条件，当前可以关闭；但 `Stage3Input` 类型目前仍要求一个非空 `DescriptorBatch`，现有导出器采用“全零值 + 全 false mask”占位；
- pocket field conditioning 可以在模型配置中使用 `mode=none` 关闭；
- 但是当前 `Stage3Input` 边界仍形式上要求非空 `PocketBatch` 和 `pocket_mask`；
- 因而“不使用 pocket 训练”在科学目标上可行，但在正式训练 Dataset 落地时，需要让训练输入路径真正支持 pocket-disabled，而不是把虚构 pocket 数据写入三维数据库。

### 4.4 训练代码还未完成

当前 `scripts/train.py` 最终调用的 `run_training()` 会直接抛出 `NotImplementedError`；`MolecularDataset` 也只是抽象接口。因此：

- 当前分支已经定义了张量边界、扩散过程、掩码语义和损失；
- 但正式训练循环、数据库 Dataset、collate 和 split 读取流程仍需后续实现；
- 本次先构建规范清楚、可审计的 3D 资产库是合理的，但不能把“3D 库完成”表述成“已经可以启动完整训练”。

## 5. 这一步必须补什么

### 5.1 必须写入新三维资产库的内容

建议以 `(source_part, molecule_uid, ring_system_uid)` 作为训练样本唯一键，并至少保存：

1. 来源键：`source_part`、`molecule_uid`、`ring_system_uid`；
2. 生成使用的 SMILES 及其选择依据；
3. `x0_json` 或二进制数组：形状严格为 `[N,3]`；
4. `ring_atom_mask_json`：长度 `N`；
5. `noise_mask_json`：长度 `N`，与模型 `diffusion_mask` 同义；
6. `source_to_rdkit_atom_map_json` 和反向映射；
7. 生成方法、固定随机种子、软件版本和参数；
8. `status`、`error_code`、`error_text`；
9. `qc_json`：原子数、元素、键拓扑、映射、坐标有限性和几何检查结果。

同一个 molecule 有多个 ring system 时，只需生成一次分子构象；各 ring-system 样本复用同一个 `x0` 和原子映射，但使用各自的 `ring_atom_mask/noise_mask`。实际数据库可采用“分子构象表 + 环系统样本表”两表设计，避免重复保存大量坐标。

### 5.2 必须执行的质控

最低限度应检查：

- SMILES 可被 RDKit 解析；
- 元素和键类型全部在冻结词表中，否则拒绝并记录；
- 源库 atom/bond index 连续且引用合法；
- 3D 生成成功并确实存在一个 conformer；
- 源原子与 RDKit 原子建立完整、唯一、可逆的映射；
- 映射后元素、形式电荷、键连接和键类型与源库一致；
- `x0.shape == [N,3]` 且所有坐标为有限数；
- ring index 全部位于 `[0,N)`；
- 两个 mask 长度均为 `N`，且在真实原子位置严格互补；
- 至少存在一个环外原子可加噪；
- 键长和非键碰撞没有明显异常；异常阈值只用于 PASS/FAIL，不擅自改变分子化学结构；
- 每次失败必须保留 molecule ID、阶段和原因，不能静默跳过。

## 6. 拟采用的 3D 生成方法

根据已经审阅的 `kssrikar4/SMILES-to-2D-3D-Molecule` 仓库，本阶段只把它作为三维构象生成核心，而不是模型：

1. 用 RDKit 从 SMILES 创建分子；
2. `Chem.AddHs` 添加生成 3D 所需的氢；
3. 使用仓库中的 `AllChem.ETKDGv3()`；
4. 使用固定 `randomSeed = 0xf00d`；
5. 调用 `EmbedMolecule`，保留 embedding failure 信息；
6. 主嵌入失败时，仅使用该 notebook 已展示的 `useRandomCoords=True` 路径作为 fallback；
7. 只抽取并映射回源数据库中的重原子坐标，新增氢不改变训练图的 `N`；
8. 不自行增加该仓库没有实际执行的 docking、MMFF/UFF 优化或其他化学算法。

说明：原 notebook 中存在多次重复调用 `EmbedMolecule` 的交互式演示。批处理实现会保留其有效算法含义，即“ETKDGv3 主尝试 + notebook 已有的随机坐标 fallback”，但不会机械重复会覆盖 conformer 的演示单元格。

## 7. 建议的实施顺序

### 阶段 A：只读小样本验证

- 原库以 SQLite `mode=ro` 打开；
- 先固定抽取 100 个分子，不改原库；
- 生成独立的测试 DB、JSON 汇总和可实时查看的 progress JSON；
- 输出逐条成功/失败与原因；
- 人工抽查若干 molecule 的原子映射、坐标和掩码。

### 阶段 B：Golden sample 接口验证

- 从通过 QC 的记录中导出少量真实训练样本；
- 写入 `train` 数据适配器，验证张量形状、词表、mask 和一次前向 DDPM；
- 对 pocket 使用正式的 disabled 路径，不把假 pocket 当真实训练数据；
- 与模型同学确认 `atom_features/edge_features` 是明确置零、删除，还是冻结正式列定义。

### 阶段 C：扩大规模

- 先跑 1,000，再跑 10,000；
- 统计成功率、失败类型、耗时和输出体积；
- 只有在成功率和映射正确性稳定后，才决定是否处理全部 15,601,969 个分子；
- 支持断点续跑、原子化提交、进度监控和失败重试，避免长任务中断后重做。

### 阶段 D：正式训练准备

- 按 scaffold/topology 进行 train/validation/test 分组，防止结构泄漏；
- 实现正式 Dataset、collate 和 training loop；
- 若以后启用 pocket-conditioned 模型，再新增 pocket 坐标系对齐与 pocket field 数据，不回头污染本阶段的无 pocket 三维资产。

## 8. 当前结论和待确认事项

### 已可确定

- 原数据库的二维图、环系统和来源关系很完整；
- 不需要修改这 400+ GB 原数据库；
- 本阶段核心缺口是 `x0 + 显式原子映射 + mask + QC`；
- `ring_atom_mask/noise_mask` 可直接由已有环原子索引生成；
- pocket 对齐可以暂缓；
- 应生成独立的新资产库，并保留失败报告和进度文件；
- 应先做 100 条只读试验，不能直接对 1,560 万分子全量运行。

### 在正式全量处理前仍需确认

1. 一个 molecule 对应多个 ring system 时，训练样本是否确定按 `(molecule, ring_system)` 展开；
2. 3D 坐标是只保留一次固定种子构象，还是以后每个 molecule 保留多个构象；当前建议先保留一次确定性构象；
3. `atom_features/edge_features` 是否明确采用零占位，还是由模型同学冻结正式 schema；
4. pocket-disabled 的训练输入接口如何调整；
5. 正式 split 的分组单位和目标训练集规模。

在上述确认完成前，不应启动全库 3D 转换。

## 9. 本文依据

- `database_report.txt`：SQLite schema、样例记录、数据库大小和构建元数据；
- 数据库构建记录：15,601,969 个完整分子、17,530,765 条 molecule–scaffold 关系、缺失分子记录为 0；
- `src/gpcr_ringgen/core/types.py`：图、坐标、mask、descriptor 和 pocket 张量约束；
- `src/gpcr_ringgen/core/vocab.py`：冻结的原子/键类别词表；
- `src/gpcr_ringgen/models/diffusion/transitions.py`：环外加噪、环内坐标固定；
- `src/gpcr_ringgen/training/losses.py`：损失只作用于 `diffusion_mask=true` 的原子；
- `src/gpcr_ringgen/training/trainer.py` 和 `data/datasets.py`：正式训练循环与 Dataset 尚未实现；
- `kssrikar4/SMILES-to-2D-3D-Molecule` 的 `rdkit.ipynb`：ETKDGv3、固定种子和 random-coordinate fallback。
