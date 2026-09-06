# 材料绑定 V3 与万级数值训练

本版仍是使用真实图的数值模块实验，不是完整闭环成绩。assisted 使用原文，必须与 source-free 分开报告。

V3 按 `$材料引用$ (数量)` 的明确相邻结构恢复训练标签，同一个括号里的质量/物质的量绑定到同一材料。模型输入只保留引用和结构，清除目标数值。请求包含材料标识、同单位槽位位置和前后操作。不明确的绑定保持 unresolved，不按数量序号强行匹配材料。旧图扩散没有材料-数量绑定头；旧预测图只能使用单材料推断或保留 unresolved，多材料真实绑定只用于训练及 oracle 诊断。

使用完整训练集（当前服务器此前报告53828条）：

```bash
python -u scripts/run_numeric_fit_diagnostic.py \
  --base-model /home/void/models/molt5-base \
  --arm assisted --train-records 0 \
  --validation-records 512 --train-eval-records 32 \
  --epochs 3 --batch-size 2 --accumulation 8 --lr 0.0001 \
  --max-length 4096 --overlength-policy skip --save-every-epoch
```

`--train-records 0` 使用整个现有 train 文件；也可明确用 `--train-records 10000`。从原始模型开始便于形成独立实验；要利用第60轮权重，额外指定 `--warm-start-run outputs/numeric_fit/<第60轮结果目录>`。后者会核对历史训练ID不与验证集重叠，但属于更换数据/提示后的新实验，不是原配置续训。新提示禁止通过 --resume-run 静默复用旧提示实验。

训练指标只覆盖固定32条训练记录产生的探针槽位，报告中保留ID和数量，不能解释为全训练集准确率。验证仅在训练前和最终轮评估，固定轮数不按验证结果选 checkpoint；验证集也不得当成最终测试集。每轮保存独立checkpoint和诊断来源（不保存优化器状态），最后另保存 model。

万级输入预检分批分词，避免一次性构建全部token数组。显式 skip 策略按输入长度排除超4096 token的样本，绝不截断；train/validation的排除清单和计数随结果保存。这会改变实际评测子集，比较时必须使用相同保留ID及槽位。严格不排除可改为 --overlength-policy error。报告另外记录有数值槽位的训练记录数；并非每条反应均产生可训练数值。graph codec 仍有原有步骤/槽位容量限制。

两个条件默认不会都跑，本命令只跑assisted。单独跑free需要 --arm free，并给出同规模训练预算。运行时间与实际槽位数量和文本长度有关，不能按32条的总时间直接保证全量完成时间。

回传report.json、summary.md、最后一轮train/validation预测及排除清单即可，模型权重保留服务器。输入样本文件较大，无需每次全部下载。
