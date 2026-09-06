# 数值学习独立诊断

目的：区分数字格式学习、训练集拟合和验证泛化，不用于正式 benchmark。
两组从同一原始模型分别初始化，使用相同 32 条训练记录和 32 条验证记录、真实图、10 epochs、学习率 1e-4、batch 2、累积 1。
- source_free：保持原无 source 输入。
- source_assisted_DIAGNOSTIC：附加记录原始 source，可能含目标过程，仅用于抽取能力诊断。不是给无证据输入凭空补目标答案，也不是 source-free 成绩。

```bash
python -m pytest -q tests
python -u scripts/run_numeric_fit_diagnostic.py --base-model /home/void/models/molt5-base
```

每轮打印训练合法数值率、数值准确率、累计更新数；训练前及最后一轮评估验证集，固定轮数且不按验证表现选 checkpoint。保存两组最终模型及每轮训练预测。
输出到唯一 outputs/numeric_fit/<时间>/。回传整个结果目录到 iclr/ubuntu 下的新子目录，保留历史结果。

先检查 report.json 的 conflicting_training_prompts；相同输入对应不同标签时，训练集精确拟合也可能有上限。numeric_exact_rate 容差 rel=1e-6、abs=1e-8；另报 10% 相对误差内比例，均把非法输出计为错误。输入证据匹配仅检查值与单位，不能证明对象/步骤绑定正确。

解读：训练集数字格式仍失败，检查编码、优化和解码；格式成功而训练精确率低，检查训练充分性、标签和输入冲突；训练成功而验证失败，检查泛化及信息不足。有 source 组明显更好，只支持证据有帮助，不能证明无 source 的每条记录都不可预测。

此诊断相对旧冒烟测试改变训练规模、学习率与更新预算，因此不是仅改变某一超参数的消融。只比较本次两组的输入条件。无图模型推理，真实图仅通过已有 checkpoint 的 codec 构造。source 提示较长，默认预算4096且绝不静默截断；超预算会明确报错。显存不足时可用 --batch-size 1 --accumulation 2 保持每次更新的样本数基本一致。

## 继续训练

`--resume-run outputs/numeric_fit/<上一轮时间> --arm assisted --epochs 50` 从该组最后保存的权重再训练50轮，例如第10轮到第60轮。保留上一轮数据、抽样种子、学习率及批量配置，校验数据哈希与样本ID；输出新目录，不覆盖旧模型。旧版本没有保存优化器状态，因此重建优化器，不是完整训练状态恢复。报告记录续训来源，轮数累计，optimizer_steps 是本次新增更新数。继续固定轮数，不按验证表现挑 checkpoint。
