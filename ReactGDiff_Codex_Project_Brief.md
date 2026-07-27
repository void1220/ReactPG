# ReactGDiff 本地项目说明文件（供 Codex 理解）

**Working title:** `ReactGDiff: Verifiable Generation of Chemical Experimental Procedures via Joint Graph Diffusion`

本文件用于帮助 Codex 理解本地实验项目的研究目标、AAAI 投稿导向下的最小实现范围，以及需要优先实现的评价指标。本项目应被理解为一个 **AI procedure-generation benchmark and model-building project**，而不是现实湿实验执行指南。

---

## 1. 研究定位

### 1.1 核心研究问题

本项目针对的核心任务是 **experimental procedure prediction / generation**：

> Given a chemical reaction representation, generate a structured experimental procedure.

已有强基线方法，如 **ReactXT** 和 **ReactGPT**，主要将该任务建模为 **reaction-to-action-sequence** 或 **reaction-to-text generation**。这些方法生成的文本或 action sequence 在 BLEU、ROUGE、Levenshtein 等文本相似性指标上可以接近参考步骤，但实验流程中的规则和约束大多是以 token 序列中的隐式模式被模型学习。

本项目提出的 **ReactGDiff** 将任务重新表述为：

> reaction input → heterogeneous process graph → deterministic structured procedure text/action sequence

本项目的核心主张不是简单追求输出文本与参考文本更相似，而是希望生成的实验步骤在以下方面更可靠：

- 结构更完整；
- 引用关系更一致；
- 类型化属性更合法；
- 对安全合规规则更敏感；
- 可以通过中间图表示追溯最终步骤来源。

### 1.2 与 ReactXT / ReactGPT 的主要区别

ReactXT / ReactGPT 风格的方法大致是：

```text
reaction input → language/sequence model → action sequence or procedure text
```

ReactGDiff 的流程是：

```text
reaction input + optional constraints
    → joint graph diffusion
    → heterogeneous process graph
    → graph-level validation
    → deterministic graph-to-sequence compilation
    → structured experimental procedure
```

中间异构图是本项目的关键。它需要显式表示 operation nodes、material nodes、container/state nodes、typed attributes，以及 input、output、reference、precedence、constraint links 等图关系。

---

## 2. 面向 AAAI 的最小实现范围

考虑到 AAAI 主文篇幅较短，本项目实现时应优先支持一个紧凑、有说服力的核心实验，而不是扩展成过于庞杂的系统。

### 2.1 支撑论文所需的最小实验模块

建议实现并报告以下四个实验模块：

1. **Main OpenExp comparison**
   - 将 ReactGDiff 与强 sequence/text baselines 对比。
   - 使用紧凑的核心指标集合。

2. **Ablation study**
   - 展示 graph constraints、continuous attributes、discrete-continuous coupling，以及 graph generation 相比 sequence diffusion 的作用。

3. **Complexity-stratified analysis**
   - 分析 short / medium / long procedures，以及 condition-heavy 或 multi-reference 子集。
   - 目标是证明 graph-based generation 在结构复杂的实验步骤中更稳健。

4. **One case study**
   - 展示生成图如何映射到最终编译后的 procedure。
   - 展示可追溯性和局部规则检查。
   - 不要包含操作敏感的湿实验执行细节；必要时使用 schematic 或 redacted outputs。

### 2.2 主结果表推荐指标

主表建议使用以下紧凑指标：

| Metric | Direction | Meaning |
|---|---:|---|
| `BLEU-4` | ↑ | 标准文本 / action-sequence 生成质量。 |
| `75%LEV` | ↑ | normalized Levenshtein similarity 高于 75% 的预测比例。 |
| `RC` | ↑ | Reference Consistency，即引用一致性。 |
| `SC` | ↑ | Structural Completeness，即结构完整度。 |
| `AttrValid` | ↑ | Attribute Validity，即属性合法性。 |
| `SVR` | ↓ | Safety-rule Violation Rate，即安全规则违规率。 |

主表中不要放过多传统 NLP 指标。如果已经实现，`ROUGE-L`、`Validity`、`CSR` 等可以放在 appendix 或 supplementary analysis 中。

---

## 3. 数据集策略

### 3.1 主数据集：OpenExp

使用 **OpenExp** 作为主要 benchmark。原因是 OpenExp 是 experimental procedure prediction 方向的重要公开基准，便于与 ReactXT / ReactGPT-style methods 直接对比。

OpenExp 提供 reaction-level information 和 structured experimental action sequences。本地项目中可用字段可能包括：

```text
index
REACTANT
PRODUCT
CATALYST
SOLVENT
actions
source
extracted_molecules
extracted_temperature
extracted_duration
molecules
score
```

OpenExp 应用于以下工作：

- train / validation / test splitting；
- action sequence prediction evaluation；
- 构建目标 heterogeneous process graphs；
- 派生 action schema 和 operation-slot expectations；
- 估计 typed attributes 的经验分布。

### 3.2 OpenExp 不提供什么

OpenExp **不提供标准化安全标签**。因此：

- `SVR` 不能被描述为 OpenExp-native label。
- `SVR` 必须实现为一种 **rule-based safety-compliance metric**，需要结合外部 hazard annotations 和 graph-level rules。
- OpenExp 提供触发安全合规检查所需的 materials / actions / procedure structure，但不提供 safety ground truth 本身。

### 3.3 可选外部安全合规标注

对于 `SVR`，可在 material nodes 上补充外部高层 hazard annotations，例如：

- GHS/SDS-style hazard class；
- hazard pictogram category；
- signal word；
- coarse risk category，如 flammable、corrosive、toxic、oxidizing、environmental hazard。

这些标注只能用于高层安全合规检查。不要实现、打印或生成具体危险操作指令。

---

## 4. 核心表示：Heterogeneous Process Graph

### 4.1 图对象

每个实验步骤表示为一个 heterogeneous graph：

```python
G = (V, E, A)
```

其中：

- `V`：typed nodes；
- `E`：typed directed edges；
- `A`：attached to nodes/edges 的 typed attributes。

### 4.2 节点类型

推荐的 node types 如下：

| Node type | Description |
|---|---|
| `operation` | 实验操作节点，即 procedure action node，例如通用实验动作类别。 |
| `material` | 反应物、产物、催化剂、溶剂、试剂或其他物质提及。 |
| `container` | 容器、位置或 container placeholder；如果数据中缺失，可进行合理推断。 |
| `state` | 中间混合物、溶液、残留物、产物状态等。 |
| `condition` | 非数值型或类别型条件标记。 |
| `safety_control` | 高层安全合规标记，不是具体操作指令。 |

MVP 阶段可以简化节点集合，但 `operation`、`material` 和 `state` 是必须保留的核心节点类型。

### 4.3 边类型

推荐的 edge types 如下：

| Edge type | Meaning |
|---|---|
| `input_to` | material / state 作为某个 operation 的输入。 |
| `output_from` | operation 产生 material / state。 |
| `precede` | operation-level precedence relation，即操作先后关系。 |
| `refer_to` | 后续节点引用前面出现的 material / state / container。 |
| `located_in` | operation 或 state 与 container / location placeholder 相关联。 |
| `has_condition` | operation 与 condition / attribute node 关联。 |
| `requires_control` | 存在风险的 material / operation 与高层 safety-control node 关联。 |

### 4.4 类型化属性

推荐的 typed attributes 如下：

| Attribute | Attached to | Evaluated by |
|---|---|---|
| `temperature` | operation / condition | `AttrValid` |
| `duration` | operation / condition | `AttrValid` |
| `volume` | material / operation | `AttrValid` |
| `amount` | material / operation | `AttrValid` |
| `equivalent` | material / operation | `AttrValid` |
| `unit` | attribute object | `AttrValid` |
| `hazard_tag` | material | `SVR` trigger，不属于 `AttrValid` |
| `safety_control_tag` | safety_control node | `SVR` |

重要边界：

- 数值单位、数值范围和属性绑定属于 `AttrValid`。
- hazard-aware process-compliance checks 属于 `SVR`。

---

## 5. 模型：Joint Graph Diffusion

### 5.1 完整方法名称

使用以下方法名：

```text
ReactGDiff = Reaction-oriented Joint Graph Diffusion
```

完整论文题目：

```text
ReactGDiff: Verifiable Generation of Chemical Experimental Procedures via Joint Graph Diffusion
```

### 5.2 模型分解

模型应包含两个耦合的生成组件：

1. **Discrete graph diffusion chain**
   - 生成 operation skeleton、node types、edge types 和 topological dependencies。

2. **Continuous attribute diffusion chain**
   - 生成 duration、temperature、volume、amount、equivalent 等 typed continuous attributes。

两条链应在 denoising 过程中交换信息，使 procedural structure 与 attributes 能够一致生成。

### 5.3 为什么使用 joint discrete-continuous diffusion

实验步骤生成同时包含两类决策：

- discrete structural decisions：operation type、operation order、material-flow relation、reference relation；
- continuous typed decisions：temperature、duration、volume、amount、equivalent。

文本模型通常把这两类信息都作为 tokens 处理。ReactGDiff 应显式建模二者的差异。

### 5.4 MVP 模型变体

面向 AAAI 的 MVP 阶段，应实现完整模型和以下紧凑消融：

| Variant | Purpose |
|---|---|
| `ReactGDiff-full` | 完整方法。 |
| `w/o graph constraints` | 检验 graph-level constraints 是否重要。 |
| `w/o continuous attributes` | 检验 typed attribute modeling 是否重要。 |
| `w/o discrete-continuous coupling` | 检验 structure-attribute coupling 是否重要。 |
| `sequence diffusion` | 检验提升是否来自 graph generation，而不是 diffusion alone。 |

---

## 6. Deterministic Graph-to-Sequence Compilation

### 6.1 Compiler 的作用

compiler 将生成图转换为结构化 action sequence：

```python
y = compile_graph_to_sequence(G)
```

这一步用于：

- 与 ReactXT / ReactGPT-style action sequence outputs 进行公平对比；
- 计算标准 BLEU-4 / 75%LEV；
- 从生成文本回溯到 graph nodes 和 graph edges。

### 6.2 Compiler 要求

compiler 应该完成：

1. 对 operation nodes 进行 topological order；
2. 读取每个 operation 的 input materials / states；
3. 读取 output states / materials；
4. 挂接合法 typed attributes；
5. 将每个 operation 序列化为项目定义的 action syntax；
6. 记录 output tokens / actions 与 graph nodes 的 alignment。

### 6.3 可追溯输出

对每个编译后的 step，保存如下 metadata：

```json
{
  "step_id": 3,
  "compiled_action": "<redacted_or_structured_action>",
  "source_operation_node": "op_3",
  "input_nodes": ["mat_1", "state_2"],
  "output_nodes": ["state_3"],
  "attribute_nodes": ["attr_time_3"],
  "checked_rules": ["slot_complete", "reference_valid", "safety_order_valid"]
}
```

论文 case study 中不要暴露操作敏感的湿实验细节。

---

## 7. 需要实现的评价指标

### 7.1 标准序列指标

#### BLEU-4

在 compiled action sequences 上使用标准 corpus-level BLEU-4。

#### 75%LEV

计算 predicted action sequence 与 reference action sequence 之间的 normalized Levenshtein similarity：

```python
lev_sim = 1 - edit_distance(pred, ref) / max(len(pred), len(ref), 1)
```

然后计算：

```python
75%LEV = mean(lev_sim >= 0.75)
```

---

## 7.2 RC: Reference Consistency

### 定义

`RC` 用于评价生成 procedure 中的引用关系是否有效。

```python
RC = number_of_valid_references / number_of_all_references
```

### 有效引用包括

- 后续 operation 不引用从未引入的 material / state；
- 作为输入使用的 state 已经在前文产生或初始化；
- container / state reference 与图顺序不冲突；
- 不存在 use-before-create 依赖。

### 实现说明

- 对 ReactGDiff，直接在 generated graph 上计算。
- 对 text / sequence baselines，先用相同 parser 将 generated sequence 解析为 temporary graph，再计算 RC。
- 如果解析失败，将 unresolved references 计为 invalid references。

---

## 7.3 SC: Structural Completeness

### 定义

`SC` 用于评价 required structural slots 是否存在。

```python
SC = number_of_filled_required_slots / number_of_all_required_slots
```

### 必需槽位

required slots 应由 operation schema 定义。对每类 operation type，定义 required fields 和 optional fields。

示例 schema 形式：

```json
{
  "operation_type": "GENERIC_OPERATION_TYPE",
  "required_slots": ["input", "output", "precedence"],
  "optional_slots": ["condition", "container", "duration", "temperature"]
}
```

不要只依赖临时字符串匹配，应优先解析为 typed action / graph objects。

### 实现说明

- 缺失 required material / state references 会降低 SC。
- 缺失 required operation output 会降低 SC。
- 缺失 required precedence relation 会降低 SC。
- 数值范围合法性不在 SC 中计算，应归入 `AttrValid`。

---

## 7.4 AttrValid: Attribute Validity

### 定义

`AttrValid` 用于评价 numerical and typed-attribute validity。

```python
AttrValid = number_of_valid_attributes / number_of_all_generated_attributes
```

### 合法性检查

一个 attribute 被认为有效，需要满足：

1. 能够被解析；
2. 单位可识别；
3. 单位类型与 attribute type 匹配；
4. attribute 被挂接到兼容的 operation / node type；
5. 如果启用 empirical-range checking，则该 attribute 落在训练集估计出的 operation-specific empirical range 内。

### 重要边界

`AttrValid` 不应包含 high-level hazard 或 safety-control checks。这些属于 `SVR`。

### 经验范围估计

对每个 `(operation_type, attribute_type)` 组合，计算训练集分位数，例如：

```python
lower = q01
upper = q99
```

也可以使用基于 median 和 interquartile range 的 robust interval。

除非来自经过验证的外部规则来源，否则不要硬编码具体化学危险阈值。

---

## 7.5 SVR: Safety-rule Violation Rate

### 定义

`SVR` 用于评价 generated process graphs 上的 high-level safety-compliance rule violations。

```python
SVR = number_of_safety_rule_violations / number_of_applicable_safety_rule_checks
```

数值越低越好。

### 重要表述边界

SVR **不能证明现实湿实验安全性**。它只能衡量生成 procedure 是否违反预定义的 high-level safety-compliance rules。

不要仅根据 SVR 声称某个生成 procedure 可以安全执行于真实实验室。

### SVR 不应与 AttrValid 重叠

SVR **不能检查**：

- temperature 是否落在训练集分布中；
- duration 是否合理；
- volume unit 是否可解析；
- amount / equivalent 是否数值合法。

这些内容属于 `AttrValid`。

### SVR 规则来源

OpenExp 不提供 safety labels。SVR 使用三类输入：

1. **OpenExp reaction/procedure structure**
   - 提供触发检查所需的 materials、actions 和 graph positions。

2. **External hazard annotations**
   - 可选的 GHS/SDS/PubChem-style hazard categories，用于 material nodes。
   - 只使用 coarse hazard labels。

3. **High-level process-graph safety-compliance rules**
   - 规则定义在 graph structure 上，不是详细湿实验操作指令。

### 推荐的 SVR 规则类别

| Category | Check | Notes |
|---|---|---|
| Hazard annotation completeness | 风险物质节点应携带可获得的 hazard tags。 | 使用外部 hazard annotation。 |
| Safety-control binding | 风险 material / operation nodes 在适用时应连接到高层 control marker。 | 不指定实际操作细节。 |
| Incompatibility pattern | 检测粗粒度 hazard-category 与 operation-category 的不兼容组合。 | 保持抽象和规则化。 |
| Safety-order consistency | hazard / control marker 应出现在相关 risky operation node 之前或同一位置。 | 图顺序检查。 |
| Terminal handling completeness | 风险相关 procedure branches 应具有 end / handling state marker。 | 高层流程完整性检查。 |

### 抽象规则对象示例

```json
{
  "rule_id": "SVR_CONTROL_BINDING_001",
  "trigger": {
    "material_hazard_tag_in": ["flammable", "corrosive", "toxic", "oxidizing"]
  },
  "required_graph_pattern": {
    "edge_type": "requires_control",
    "target_node_type": "safety_control"
  },
  "violation_if_missing": true,
  "severity": "high_level_compliance"
}
```

该规则属于高层 compliance rule，不是具体 procedure instruction。

---

## 8. Baselines

### 8.1 最小 baseline 集合

面向 7 页 AAAI 论文，使用紧凑 baseline 集合：

| Baseline | Purpose |
|---|---|
| `NearestNeighbor` or fingerprint retrieval | 检验 similarity retrieval 是否已经足够。 |
| `T5/BART/MolT5` | 标准 sequence generation baseline。 |
| `ReactXT` | 强 OpenExp-oriented reaction-text baseline。 |
| `ReactGPT` | 强 LLM / in-context tuning baseline。 |
| `ReactGPT + validator/retry` | 检验 post-hoc validation 是否能接近 graph-native generation。 |
| `sequence diffusion` | 检验没有 graph representation 的 diffusion 表现。 |
| `ReactGDiff` | 本项目完整方法。 |

### 8.2 关于 QFANG

QFANG 是与 organic synthesis procedure generation 高度相关的近期工作，使用大规模 reaction-action dataset 和 reasoning-based training。对于紧凑 AAAI 版本：

- 在 related work 中讨论 QFANG；
- 如果代码和数据在时间允许范围内可复现，可以直接比较；
- 否则，仅在可行时实现一个 **QFANG-style reasoning LLM baseline**；
- 不要过度扩展主实验表。

核心信息是：ReactGDiff 不是通过 dataset scale 或 chain-of-thought prompting 取胜，而是评价 **explicit process-graph generation** 是否能在 sequence generation 之外提升 structural reliability。

---

## 9. Complexity-Stratified Evaluation

从 OpenExp 构造简单测试子集。

### 9.1 建议子集

| Subset | Construction idea |
|---|---|
| `short` | action count 较低的 procedures。 |
| `medium` | action count 中等的 procedures。 |
| `long` | action count 较高的 procedures。 |
| `condition_heavy` | 包含多个 extracted attributes，如 duration / temperature / amount 的 procedures。 |
| `multi_reference` | 包含较多 material / state references 或重复对象的 procedures。 |

### 9.2 分层分析指标

图中只使用 2–3 个指标：

- `RC`；
- `SC`；
- 可选 `SVR` 或 `AttrValid`。

预期研究假设：

> Sequence models 可能保持较好的平均文本相似性，但 graph-native generation 应在 long、condition-heavy 和 multi-reference procedures 上下降更慢。

---

## 10. 建议代码仓库结构

```text
reactgdiff/
  README.md
  configs/
    dataset_openexp.yaml
    model_reactgdiff.yaml
    eval_metrics.yaml
  data/
    raw/
    processed/
    graphs/
    hazard_annotations/
  reactgdiff/
    data/
      openexp_loader.py
      action_parser.py
      graph_builder.py
      graph_schema.py
    models/
      discrete_graph_diffusion.py
      continuous_attribute_diffusion.py
      coupling_module.py
      sequence_diffusion_baseline.py
    constraints/
      structural_rules.py
      reference_rules.py
      attribute_rules.py
      safety_rules.py
      rule_engine.py
    compile/
      graph_to_sequence.py
      traceability.py
    eval/
      bleu.py
      lev.py
      rc.py
      sc.py
      attrvalid.py
      svr.py
      stratified_eval.py
    baselines/
      nearest_neighbor.py
      seq2seq_runner.py
      reactxt_runner.py
      reactgpt_runner.py
      validator_retry.py
    utils/
      logging.py
      seed.py
      io.py
  scripts/
    build_graph_dataset.py
    train_reactgdiff.py
    sample_reactgdiff.py
    evaluate_predictions.py
    run_ablation.py
    run_stratified_eval.py
  outputs/
    predictions/
    metrics/
    figures/
    case_studies/
```

---

## 11. 实现优先级

### Priority 1: data and graph construction

- 加载 OpenExp。
- 解析 action sequences。
- 构建 deterministic target graphs。
- 验证 graph schema。

### Priority 2: metrics

在优化模型前，优先实现指标：

- BLEU-4；
- 75%LEV；
- RC；
- SC；
- AttrValid；
- SVR。

这些指标必须同时适用于：

- ReactGDiff 的 graph-native outputs；
- sequence / LLM baselines 经过解析后的 outputs。

### Priority 3: deterministic compiler

- 将 graph 编译为 action sequence。
- 保存 traceability maps。
- 确保 compiled outputs 可以使用标准 sequence metrics 评价。

### Priority 4: model and ablations

- 实现完整 ReactGDiff。
- 实现紧凑 ablations。
- 实现 sequence diffusion baseline。

### Priority 5: paper outputs

生成：

- main result table；
- ablation table；
- stratified evaluation figure；
- one traceability case study。

---

## 12. 论文需要支撑的主张

实验应支撑以下主张：

1. **Comparable sequence generation quality**
   - ReactGDiff 在 BLEU-4 和 75%LEV 上应与 ReactXT / ReactGPT 具有竞争力。

2. **Better structural reliability**
   - ReactGDiff 应提升 RC 和 SC，尤其是在更长或更复杂的 procedures 上。

3. **Better typed-attribute validity**
   - continuous attribute chain 应相较 token / sequence generation 提升 AttrValid。

4. **Lower high-level safety-compliance rule violations**
   - 由于 hazard-aware compliance rules 在图上表示并检查，SVR 应更低。

5. **Traceability**
   - 最终生成的 actions 可以回溯到 operation / material / state / attribute nodes。

---

## 13. 需要避免的表述

不要声称：

- 系统保证真实湿实验安全；
- 生成步骤无需专家审查即可直接由人或机器人执行；
- OpenExp 提供原生 safety labels；
- SVR 是真实 laboratory safety certification；
- 模型发现了新的化学知识；
- 模型应被用于 hazardous synthesis execution。

应使用更稳妥的表述：

- “safety-compliance rule violation rate”；
- “high-level hazard-aware graph checks”；
- “structural reliability”；
- “verifiable intermediate representation”；
- “procedure generation benchmark evaluation”。

---

## 14. 论文摘要的最小逻辑

论文应遵循以下故事线：

1. Experimental procedure prediction 连接 reaction understanding 与 automated synthesis planning。
2. 现有 ReactXT / ReactGPT-style methods 生成 linear action/text sequences。
3. Linear sequence generation 将 procedural dependencies、references、typed attributes 和 safety-compliance constraints 隐藏在 tokens 中。
4. ReactGDiff 将该任务重构为 compilable heterogeneous process graph generation。
5. Joint graph diffusion 以耦合方式生成 discrete topology 和 continuous attributes。
6. Graph validation 和 deterministic compilation 使结构化 procedure generation 具备 traceable 和 verifiable 特征。
7. 在 OpenExp 上，ReactGDiff 应在 BLEU-4 / 75%LEV 上具有竞争力，并在 RC、SC、AttrValid 和 SVR 上更强。

---

## 15. 一句话核心贡献

> ReactGDiff shifts chemical experimental procedure generation from token-level sequence prediction to verifiable heterogeneous process graph generation, enabling explicit modeling of procedural dependencies, typed attributes, reference consistency, and high-level safety-compliance rules before deterministic compilation into structured steps.
