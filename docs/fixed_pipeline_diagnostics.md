# 固定骨架闭环诊断 v1

本轮保留已有图扩散检查点，关闭操作补全。先定位瓶颈，不同时替换扩散时间机制。
历史 pointer 检查点中的数值头仍参与原模型内部计算；新参数模型的输入白名单会清除其数值、候选指针、原始参数文本。这个版本不是已经重新训练的“纯离散图扩散”模型。

## 运行

在原来的 py310 训练环境、仓库根目录运行，无需安装新服务器依赖：

```bash
python -m pytest -q tests
python -u scripts/run_fixed_pipeline.py --limit 32 --parameter-train-records 256 --batch-size 2 --parameter-max-length 2048
```

默认使用：
- `outputs/checkpoints/openexp_small_hash_pointer_v2.pt`
- `outputs/checkpoints/openexp_small_hash_v1_seq2seq_skeleton.pt`
- `outputs/prepared_splits/openexp/scale_small/train.jsonl`
- `outputs/prepared_splits/openexp/scale_small/val.jsonl`
- 可用时读取 `outputs/skeleton/openexp_small_hash_v1_seq2seq_skeleton_val.jsonl`；不存在则重新推理。缺失部分 ID 会报错，不会静默退回图模型骨架。

上述路径可通过 `--help` 中的参数修改。已有模型和旧结果不覆盖。新文件保存在唯一的 `outputs/fixed_pipeline/<时间>/`；同名目录会拒绝覆盖。

首轮自动从**训练集**抽取 256 条记录，使用其全部可解析数量槽位，从原始专业预训练模型 `laituan245/molt5-base` 初始化独立数值模型并训练 1 epoch（不加载骨架微调权重，也不是随机初始化）。该小训练用于检验接口与学习信号，不是最终性能结论。训练按每 20 个 batch 打印进度。

默认仅从本机 Hugging Face 缓存读取预训练权重；若未缓存则明确报错，不自动退回骨架或随机权重。可用 `--parameter-base-model /本地原始MolT5目录` 指定原始模型。骨架阶段继续使用已有骨架检查点或缓存。本次对照保持 256 条训练记录、1 epoch、32 条验证记录及其他设置一致。

后续可增大 `--parameter-train-records` 与 `--parameter-epochs`；或指定 `--parameter-model outputs/fixed_pipeline/<时间>/parameter_model` 复用数值模型。参数模型记录训练 ID、输入策略和数据哈希，复用时检查与验证集交叉、原始模型及初始化类型。早期从骨架权重初始化的数值模型会被拒绝复用；请重新训练。

使用 `--regenerate-skeleton` 可明确重新运行第一阶段；缓存模式无法追溯历史 prompt 的所有细节，报告会标注此限制。

## 输入与数据边界

默认输入策略为 `source_free_with_condition_maps`：输入反应物、产物、催化剂、溶剂、分子映射和温度/时间映射，**不传 actions、_features、_quality、_buckets 或 source**。温度与时间依然使用现有显式条件引用，不在本轮重新生成。

`--include-source` 显式开启 source-assisted 条件。原始 source 可能已经包含目标过程信息；此设置的结果不能与 source-free 方法混表。默认新数值模型可能面临数量无法从输入唯一确定的问题，这需要反映在报告中，不能用目标文本补证据。

legacy_pointer 是在本次选定输入策略、骨架及采样配置下重新采样的对照，**不是原历史指标的逐项复现**，特别是旧候选池使用 source 而本次禁用时。每个报告保存输入策略和检查点配置。

完整读取训练/验证 ID 检查交叉；不自动声称反应等价类无泄漏。验证集按种子随机抽样，不使用前缀做默认小评测；同一批记录用于全部对照。测试集被拒绝用于本诊断脚本。

## 输出与归因

| 行 | 用途 |
|---|---|
| legacy_pointer | 当前输入策略下的旧数值证据指针对照 |
| new_before_gate | 固定骨架→图槽位→独立 MolT5 数值填充→编译 |
| new_after_gate | 门控拒绝的过程按空预测计入全部样本指标 |
| gold_skeleton_ORACLE | 替换正确骨架，图与参数仍预测 |
| gold_discrete_ORACLE | 替换正确离散槽位，参数仍预测 |
| gold_numeric_ORACLE | 只替换操作对齐、数量位置和单位匹配的数值 |
| compiler_structured_ORACLE | 正确结构与参数经过规范化编译，移除原文参数字段 |
| compiler_surface_ORACLE | 保留正确原文参数字段的编译往返检查 |

ORACLE 仅使用参考答案做诊断，不能作为模型生成结果。差值代表受控干预下的改进空间，存在模块交互与分布变化，不是可相加的因果贡献。正确骨架仍然表现差时，应继续检查图/参数与编译；正确数值改善有限时，不能只优化数值模型。

`report.json` 给出完整指标、骨架混淆、门控覆盖率、接受样本条件指标、按骨架正确性分组的结果，以及参考干预差值。`skeleton_diagnostics.jsonl` 包含每条的漏操作、多操作、替换、顺序错误提示和可用的原始骨架文本。`summary.md` 便于阅读。预测按记录增量保存；训练前已保存原始图槽位和 partial_report.json。

## 指标定义与 Thoth 的关系

来源：[Thoth 原文 Appendix F / Section 3.3](https://arxiv.org/html/2510.15600v2#A6)。实现为独立适配，不复制第三方代码。版本 `thoth_openexp_v1`。

- `thoth_step_m`：预测与参考步骤数严格相等。
- `thoth_order_s`：操作序列严格相同；不同于奖励中允许子序列的 Order_strict。
- `thoth_order_lcs`：2×LCS长度/(预测长度+参考长度)。
- `thoth_order_tau_monotone`：论文单调操作锚点的 Kendall 形式，有至少两锚点时恒为 1，否则为 0。因此它不适合独立衡量乱序。
- `occurrence_order_tau`：额外的诊断，按同类操作出现次数匹配位置，然后计算 Kendall 相关；不足两次匹配时为 0，另报 defined 比例。它不是原文指标。
- `thoth_semantic_a_adapted`：操作 LCS 对齐后，按材料引用集合算 IoU；材料 IoU≥0.5 才给参数匹配分。参数为规范化数值/单位及条件引用集合，使用位置衰减和子序列项；范围 0–2.5。与 SciRecipe 的对象抽取、子词补偿不同，不可直接比较论文分数。
- `aligned_object_iou`、`aligned_parameter_iou`：对齐分数按两序列较长长度归一化，让漏步骤/多步骤影响分数。
- 非空参考下空输出计失败；双空样本不奖励 Step-M/Order-S。
- 继续输出全部旧文本与自定义语义指标。

同类操作之间数值交换会影响新的逐步参数指标；但单步内多个材料的精确数量绑定仍未完全表达。不能把这些指标解释为实际实验成功率。

任意已有预测文件也可重新评测：
```bash
python scripts/evaluate_predictions.py --predictions <预测.jsonl> --output <新的指标.json>
```

## 门控范围

当前检查数值解析、有限值、已知单位、数值类型、材料/条件引用、非温度负值、绝对零度下限。无证据数值单独统计，不自动声称安全；`chemical_safety_verified` 始终为 false。

`--rules <文件.json>` 可接入**有来源、明确适用操作和单位**的范围约束列表，每条需要 id/source/operation/unit 及 min 或 max。默认不编造化学安全阈值。不自动重试到通过，不静默丢弃失败槽位，不把拒绝样本移出总分分母。

操作补全开关保留，但 `--allow-operation-completion` 会明确报尚未实现；本轮所有实验均关闭。


## v2 提示词修复（首次服务器试跑之后）

首次 32 条试跑发现 350 次参数提示截断；82 个数量槽位的原始输出中，47 次为 `0.`，10 次含非数值文本。这些是 smoke 结果，不代表最终模型能力。

现在使用紧凑的完整步骤表（操作、材料、数量单位、条件引用），移除重复的数量字典和分子名称映射；保留反应输入、材料 SMILES 映射、全部操作步骤与条件映射。训练和推理均检查真实 token 数，**超过预算报错，不再静默截断**。默认预算 1024，可显式用 `--parameter-max-length 2048` 提高。

提示版本为 `compact_discrete_graph_v2`，不允许直接复用 v1 的参数检查点。按原命令重新小训练即可；旧骨架和图检查点继续复用。训练输出长度统计；报告新增参数原始输出分布、零值数量、有效数字比例所需计数、单位分布和预测/参考数量总数。数量总数之比不是匹配召回率。

用户从服务器拿回的结果默认放在本地 `E:\iclr\ubuntu`，分析时先查看该目录。它是结果收集目录，不属于 Git 源码。
