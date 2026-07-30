# ReactPG

**ReactPG: Verifiable Generation of Chemical Experimental Procedures via Skeleton-Conditioned Graph Diffusion**

ReactPG 将 OpenExp 实验步骤预测拆成两个阶段：

1. 自回归序列模型根据反应输入预测操作骨架；
2. 离散图扩散模型在固定骨架上补全材料指针、条件、数量需求、单位和数值证据指针，再通过确定性编译器生成 OpenExp 风格步骤文本。

本项目面向 procedure-generation benchmark 和模型研究，不是现实湿实验执行指南。

## 当前算法

```text
OpenExp reaction fields
        │
        ├── reactants / products / catalysts / solvents
        ├── molecule placeholders and SMILES
        └── temperature / duration / numeric evidence
        ↓
Stage 1: MolT5 autoregressive skeleton predictor
        ↓
ADD → STIR → FILTER → YIELD ...
        ↓
Stage 2: skeleton-conditioned categorical graph diffusion
        ├── material slots → record-local material pointers
        ├── condition slots → discrete condition classes
        ├── quantity gates → whether a numeric slot is required
        ├── unit slots → discrete unit classes
        └── numeric slots → record-local numeric evidence pointers
        ↓
Generated ProcessGraph
        ↓
Lossless deterministic renderer
        ↓
OpenExp-style action sequence + render trace
```

### 第一阶段：AR 操作骨架

默认使用 `laituan245/molt5-base`，输入为反应物、产物、催化剂、溶剂和抽取到的条件证据，输出只保留有序操作类型。当前默认设置为：

- 自然语言操作目标，而不是不透明特殊 token；
- 最大输入长度 384，最大目标长度 96；
- batch size 8，梯度累积 2；
- BF16；
- 200 epochs，每 10 epochs 在验证集选择最佳骨架检查点；
- beam size 4，并保留 top-4 候选信息。

第一阶段输出三类文件：

- `outputs/checkpoints/<run>_seq2seq_skeleton.pt`
- `outputs/skeleton/<run>_seq2seq_skeleton_val.jsonl`
- `outputs/metrics/<run>_seq2seq_skeleton_val.json`

### 第二阶段：槽位图扩散

操作骨架在图扩散阶段作为固定条件，扩散模型主要补全非骨架槽位。默认图设置为：

- 最大 32 个操作步骤；
- 每步最多 4 个材料/数量槽位；
- 记录内最多 16 个材料候选和 64 个数值候选；
- 材料不自由生成，而是分类指向当前记录的材料候选；
- 数值不进行无约束连续回归，而是分类指向从输入证据和 `source` 中抽取的数值候选；
- 数值候选使用动态候选特征，而不是统一的候选 ID 词向量；
- DiT hidden size 256、4 层、8 个注意力头；
- 32 步 cosine categorical diffusion；
- posterior reverse sampler，中间采样、最后一步 argmax；
- balanced sampling 和 operation class weighting；
- batch size 64，800 epochs；
- 以 `semantic_score` 选择最佳检查点。

图扩散当前默认使用 `hash` 条件编码，适合显存受限的验证实验。也可以启用 `shared_molt5`：第二阶段从第一阶段检查点复制 MolT5 encoder，并只在图模型中更新这个副本。第一阶段的 AR 检查点不会被改变，完整 AR decoder 也不会在第二阶段继续保留。

### 确定性解码

`GraphTargetCodec` 将预测槽位绑定到实际材料和数值候选，然后构建 `ProcessGraph`。当前 lossless renderer 按操作顺序消费同一操作下的材料、条件和数量，保留：

- 同一种材料的重复出现；
- 同一数值候选的多次合法使用；
- 每个预测槽位恰好一次的渲染覆盖；
- `YIELD $-1$ (q1, q2, ...)` 的统一数量分组；
- 可审计的确定性渲染轨迹。

## 环境与依赖

当前环境验证版本：

- Python 3.10.15
- PyTorch 2.5.1
- CUDA 12.1
- Transformers 4.48.2
- SentencePiece 0.2.0

创建环境：

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU 训练时，应先根据服务器驱动安装匹配 CUDA 的 PyTorch 2.5.1，再安装其余依赖。`requirements.txt` 不安装 CUDA 驱动或 CUDA Toolkit。

MolT5 首次使用时需要从 Hugging Face 下载模型。已有本地模型时，可以传入本地目录并增加 `--skeleton-local-files-only`。

## 数据准备

预期原始文件为：

```text
data/raw/OpenExp.json
```

生成清洗数据、确定性 train/validation/test 划分和复杂度分桶：

```bash
python scripts/prepare_openexp_dataset.py \
  --input data/raw/OpenExp.json \
  --output-dir data/processed/openexp
```

当前完整数据统计：

| Split | Records |
|---|---:|
| Clean | 274,438 |
| Train | 219,480 |
| Validation | 27,454 |
| Test | 27,504 |

两阶段启动器默认选择 `scale_small`，并从正式 split 中生成：

```text
outputs/prepared_splits/openexp/scale_small/train.jsonl
outputs/prepared_splits/openexp/scale_small/val.jsonl
```

不要直接把 `buckets/scale_small.jsonl` 同时当作训练集和验证集；启动器生成的缓存保留了原始 split 边界。

## 训练

### 1. 查看将要执行的完整子命令

`--dry-run` 不训练模型，也不创建输出文件：

```bash
python scripts/train_ar_then_graphdiff.py \
  --run-name openexp_small_hash_pointer_v3 \
  --dry-run
```

### 2. 当前显存友好的完整两阶段实验

除实验名外均使用当前脚本默认值：

```bash
python scripts/train_ar_then_graphdiff.py \
  --run-name openexp_small_hash_pointer_v3
```

该命令会依次训练 MolT5 骨架模型和 hash-conditioned 图扩散模型。默认实验规模为 `scale_small`，AR 为 200 epochs，图扩散为 800 epochs。

### 3. 复用已有 AR，只重新训练图扩散

骨架缓存必须与当前 validation split 一一对应：

```bash
python scripts/train_ar_then_graphdiff.py \
  --run-name openexp_small_hash_pointer_v3 \
  --skip-ar \
  --ar-checkpoint outputs/checkpoints/openexp_small_hash_v1_seq2seq_skeleton.pt \
  --ar-predictions outputs/skeleton/openexp_small_hash_v1_seq2seq_skeleton_val.jsonl
```

hash 图编码不会读取 AR checkpoint 的 encoder，但保留 `--ar-checkpoint` 有利于记录完整实验来源。

### 4. 大显存服务器上的共享 MolT5 编码实验

```bash
python scripts/train_ar_then_graphdiff.py \
  --run-name openexp_small_shared_molt5_pointer_v1 \
  --graph-condition-encoder shared_molt5 \
  --shared-encoder-mode finetune
```

此模式会在第二阶段复制第一阶段训练完成的 MolT5 encoder，并以较小学习率联合更新 encoder 副本和图扩散网络。若需要复用已有 AR，同样增加 `--skip-ar`、`--ar-checkpoint` 和 `--ar-predictions`。

### 5. 小规模冒烟测试

```bash
python scripts/train_ar_then_graphdiff.py \
  --run-name tmp_pipeline_smoke \
  --train-limit 512 \
  --val-limit 128 \
  --seq2seq-epochs 1 \
  --graph-epochs 1
```

## 推理后分析

### 文本和语义评估

训练入口会自动写出图槽位、文本和语义指标。也可以对预测文件重新计算：

```bash
python scripts/evaluate_predictions.py \
  --predictions outputs/predictions/<run>_val.jsonl \
  --output outputs/metrics/<run>_reeval_val.json
```

### 固定扩散输出，只扫描确定性解码参数

该脚本只运行一次反向扩散，然后在相同预测张量上扫描数量门限和候选复用惩罚：

```bash
python scripts/sweep_graph_decoding.py \
  --checkpoint outputs/checkpoints/<run>.pt \
  --input outputs/prepared_splits/openexp/scale_small/val.jsonl \
  --skeleton-cache outputs/skeleton/<run>_seq2seq_skeleton_val.jsonl \
  --existing-predictions outputs/predictions/<run>_val.jsonl \
  --sweep-output outputs/analysis/<run>_decode_sweep.json \
  --best-predictions outputs/predictions/<run>_decode_best_val.jsonl \
  --best-metrics outputs/metrics/<run>_decode_best_val.json
```

### 使用最新确定性模板重渲染历史槽位

该步骤不会重新训练或重新采样模型：

```bash
python scripts/recompile_graph_prediction_templates.py \
  --checkpoint outputs/checkpoints/<run>.pt \
  --input outputs/prepared_splits/openexp/scale_small/val.jsonl \
  --predictions outputs/predictions/<run>_decode_best_val.jsonl \
  --output outputs/predictions/<run>_decode_lossless_val.jsonl \
  --metrics outputs/metrics/<run>_decode_lossless_val.json
```

## 指标

项目同时报告四层指标：

| 层级 | 主要指标 | 用途 |
|---|---|---|
| 表面文本 | BLEU-2、ROUGE-1、LEV90/75/50、Exact | 与 ReactXT 风格结果直接对比 |
| 数值归一文本 | number-normalized LEV | 减少数值表面格式差异的影响 |
| 模板无关语义 | operation similarity/exact、material/condition/numeric F1、semantic score | 判断同义模板下的实际语义正确性 |
| 图槽位与渲染 | material/numeric pointer accuracy、slot accuracy、graph exact、occurrence coverage | 定位图预测和确定性解码错误 |

LEV90、LEV75 和 LEV50 表示字符级 Levenshtein similarity 分别不低于 0.90、0.75 和 0.50 的样本比例，越高越好。

当前最新的可复现实验产物是 hash-conditioned `openexp_small_hash_pointer_v2`。使用 lossless renderer 重编译后的验证集结果为：

| Metric | Value |
|---|---:|
| BLEU-2 | 0.6077 |
| ROUGE-1 | 0.7758 |
| LEV90 | 0.0195 |
| LEV75 | 0.1419 |
| LEV50 | 0.7001 |
| Number-normalized LEV75 | 0.1743 |
| Exact | 0.0048 |
| Semantic score | 0.7117 |

这些数值来自 `outputs/metrics/openexp_small_hash_pointer_v2_decode_lossless_val.json`，用于记录当前开发状态，不代表完整规模最终对比结论。

## 代码结构

```text
reactgdiff/
├── data/       # OpenExp读取、清洗、prompt、数值证据和轻量图构建
├── compile/    # sequence ↔ graph 编译与可追溯性
├── models/     # MolT5条件编码、图codec、DiT图扩散及早期模型
├── eval/       # 文本、语义、槽位和确定性渲染指标
├── baselines/  # 尚未完整实现的基线接口
├── constraints/# 尚未完整实现的图规则接口
└── utils/      # JSONL读写等工具

scripts/
├── train_ar_then_graphdiff.py             # 当前两阶段总入口
├── train_skeleton_seq2seq.py              # MolT5骨架训练
├── train_reactgdiff.py                    # 图扩散和旧后端通用入口
├── sample_reactgdiff.py                   # 检查点采样
├── sweep_graph_decoding.py                # 解码阈值扫描
├── recompile_graph_prediction_templates.py# 确定性重渲染
└── diagnose_*.py                          # 解码上界、反向扩散和检索诊断
```

`configs/*.yaml` 主要保留早期实验设计和参数说明。当前训练以 Python 入口的 `argparse` 默认值为准，YAML 文件不会被两阶段启动器自动加载。

## 测试

```bash
python -m pytest -q
```

当前测试集中于数值候选、材料/数值指针、语义模板不变性、重复槽位保留和 lossless renderer。完整训练回归、正式 baseline runner、RC/SC/AttrValid/SVR 仍待补充。

## 重要实现入口

- `reactgdiff/models/graph_codec.py`：图目标编码、候选指针落地和确定性渲染。
- `reactgdiff/models/procedure_graph_diffusion.py`：槽位条件离散图扩散。
- `reactgdiff/models/shared_text_conditioning.py`：MolT5 encoder 复制与候选上下文化。
- `reactgdiff/data/numeric_evidence.py`：数值证据候选池。
- `reactgdiff/eval/semantic.py`：模板无关语义指标。
- `reactgdiff/eval/rendering.py`：确定性渲染覆盖与重复保留指标。

研究范围和后续论文指标规划见 `ReactGDiff_Codex_Project_Brief.md`，槽位化图补全的设计演进见 `继续改进说明.md`。
