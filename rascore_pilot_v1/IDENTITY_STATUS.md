# RAscore pilot 身份质量状态

2026-10-13 已完成 N10 的数据校验与隔离阶段。**现有模型、CV 与导出包仍是历史产物，未基于清洗后的数据重新生成，N10 尚未完全关闭。**

`rascore_teacher_combined_1400.csv` 的 1,400 条记录中，本次规则接收 1,399 条、隔离 1 条。隔离记录为 `MOL_00000249675062`：输入与 teacher 的 Full InChIKey 分别为 `FDFWGNXXYGAEBO-CLGOVRMQSA-N` 和 `FDFWGNXXYGAEBO-BXRZMJOUSA-N`，且 `teacher_stereo_identity_match=False`。未把这个 teacher 结果重新指定为输入分子的正确标签。

可用记录的证据分为两类：999 条保存的 Full InChIKey 一致；400 条缺少完整键对，但 `canonical_isomeric_smiles` 与 `teacher_target` 完全一致。后者标记为字符串一致，不宣称已通过 RDKit/InChI 独立复核。原始字符串不同时，即使非立体 SMILES 相同或旧标记为 True，也需要进一步核对；明确 False 或完整键冲突会优先隔离。

本地结果在项目根目录的 `outputs/rascore_identity_v1/`：

| 文件 | 用途 |
| --- | --- |
| `training_eligible.csv` | 1,399 条通过当前身份与标签一致性规则的训练输入 |
| `quarantined.csv` | 隔离记录、原始字段、来源行号与原因 |
| `audit.json` | 原表和输出文件指纹、计数、历史产物清单与状态 |
| `README.md` | 本次结果及校验边界 |

输出保留原表所有列的值，追加 `audit_*` 字段；原表、原始 teacher 搜索结果、joblib 模型、CV、验证预测及导出包均未覆盖。数据输出位于已忽略的 `outputs/`，不会自动进入 Git 提交。记录原产物指纹时只读取文件字节，不加载 joblib/pickle 对象。

可以在项目根目录复现，使用新的输出目录：

```powershell
python -s scripts/audit_rascore_identity.py --source rascore_pilot_v1/rascore_teacher_combined_1400.csv --output-dir outputs/rascore_identity_v2 --historical-artifact rascore_pilot_v1/rascore_models/rascore_v2_logreg_ecfp4_1400.joblib --historical-artifact rascore_pilot_v1/rascore_models/rascore_v2_cv_predictions_1400.csv
```

`--historical-artifact` 可以重复，记录模型、CV 表、验证预测或导出包。已有输出目录会被拒绝，源文件在运行中变化时不会发布结果。脚本使用 Python 标准库，不依赖 RDKit、AiZynthFinder 或 PyTorch，也不执行外部搜索。

后续训练应使用校验后的输入，并维护相同身份规则。旧 `rascore_models/` 中的三个模型、两份 CV，以及现有验证预测和 `rascore_export_v1` 都不能被称为清洗后产物；不能从旧 CV 删去一个记录后沿用原指标。原 classifier/featurizer、拆分和 CV 流程尚未归档，需补齐这些信息后训练新版本、重新做 CV 和独立验证，再生成新的导出包。此次没有改变立体化学监督定义，也没有重训或重新打标。
