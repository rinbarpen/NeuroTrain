# 混合指标方向使用指南

## 📋 功能说明

`data_to_latex.py` 工具支持为不同的指标列设置不同的优化方向，这在学术论文中非常常见：

- **越高越好** (True): accuracy, precision, recall, f1_score, AUC等
- **越低越好** (False): loss, error_rate, inference_time, memory_usage等

## 🎯 核心参数

### --metric-columns
指定要进行高亮的指标列名列表

### --higher-is-better  
为每个指标列指定优化方向（True/False）

**重要**: `--higher-is-better` 的顺序必须与 `--metric-columns` 的顺序一一对应！

## 📊 使用示例

### 示例1: 标准机器学习指标

**场景**: 评估分类模型性能

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy precision recall f1_score \
  --higher-is-better True True True True \
  --our-model "OurModel"
```

**数据示例**:
```csv
model,accuracy,precision,recall,f1_score
ModelA,0.95,0.94,0.96,0.95
ModelB,0.93,0.92,0.94,0.93
OurModel,0.97,0.96,0.98,0.97
```

**解释**: 所有指标都是越高越好

---

### 示例2: 性能与效率权衡

**场景**: 在精度和速度之间权衡

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy inference_time memory_usage \
  --higher-is-better True False False \
  --our-model "OurModel"
```

**数据示例**:
```csv
model,accuracy,inference_time,memory_usage
SlowModel,0.97,200,1024
FastModel,0.91,45,256
OurModel,0.95,80,512
```

**解释**: 
- accuracy越高越好
- inference_time越低越好（越快）
- memory_usage越低越好（占用更少）

---

### 示例3: 损失函数与指标

**场景**: 训练过程中的多个指标

```bash
python tools/data_to_latex.py \
  -i training_results.csv \
  -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy loss val_accuracy val_loss \
  --higher-is-better True False True False \
  --our-model "OurModel"
```

**数据示例**:
```csv
model,accuracy,loss,val_accuracy,val_loss
ModelA,0.95,0.12,0.93,0.15
ModelB,0.93,0.15,0.91,0.18
OurModel,0.97,0.08,0.96,0.10
```

**解释**: 
- accuracy/val_accuracy: 越高越好
- loss/val_loss: 越低越好

---

### 示例4: 完整的模型评估

**场景**: 全面评估模型性能

```bash
python tools/data_to_latex.py \
  -i comprehensive_results.csv \
  -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns accuracy loss error_rate inference_time params \
  --higher-is-better True False False False False \
  --our-model "OurModel"
```

**数据示例**:
```csv
model,accuracy,loss,error_rate,inference_time,params
ResNet50,0.9523,0.125,0.0477,120.5,25.6M
VGG16,0.9234,0.156,0.0766,145.2,138.4M
OurModel,0.9678,0.089,0.0322,95.3,5.3M
MobileNet,0.9012,0.198,0.0988,78.4,4.2M
EfficientNet,0.9556,0.098,0.0444,85.6,7.8M
```

**解释**: 
- accuracy: 越高越好 ↑
- loss: 越低越好 ↓
- error_rate: 越低越好 ↓
- inference_time: 越低越好 ↓（更快）
- params: 越低越好 ↓（更轻量）

**输出结果**:
```latex
\begin{tabular}{llllll}
\toprule
model & accuracy & loss & error_rate & inference_time & params \\
\midrule
ResNet50 & 0.9523 & 0.125 & 0.0477 & 120.5 & 25.6M \\
VGG16 & 0.9234 & 0.156 & 0.0766 & 145.2 & 138.4M \\
\underline{OurModel} & \textbf{0.9678} & \textbf{0.089} & \textbf{0.0322} & 95.3 & \textbf{5.3M} \\
MobileNet & 0.9012 & 0.198 & 0.0988 & \textbf{78.4} & \textit{4.2M} \\
EfficientNet & \textit{0.9556} & \textit{0.098} & \textit{0.0444} & \textit{85.6} & 7.8M \\
\bottomrule
\end{tabular}
```

**分析**:
- OurModel在accuracy, loss, error_rate, params上都是最佳 ⭐
- MobileNet在inference_time上最快
- EfficientNet在多个指标上是次佳

---

### 示例5: 多任务学习

**场景**: 不同任务有不同的指标

```bash
python tools/data_to_latex.py \
  -i multitask_results.csv \
  -t table --style booktabs \
  --highlight-best --highlight-second \
  --metric-columns cls_acc det_map seg_iou total_loss \
  --higher-is-better True True True False \
  --our-model "MultiTaskModel" \
  --group-column task
```

---

## 🔧 常见指标方向速查表

### 越高越好 (True)

| 指标类别 | 指标名称 | 说明 |
|---------|---------|------|
| **准确率** | accuracy, precision, recall | 分类准确性 |
| **F值** | f1_score, f2_score | 综合指标 |
| **AUC** | auc, auc_roc, auc_pr | ROC/PR曲线下面积 |
| **IoU** | iou, dice_coefficient | 分割重叠度 |
| **mAP** | map, map50, map75 | 检测平均精度 |
| **相关性** | correlation, r2_score | 回归相关性 |
| **BLEU** | bleu_score | 机器翻译质量 |
| **准确匹配** | exact_match | 问答准确度 |

### 越低越好 (False)

| 指标类别 | 指标名称 | 说明 |
|---------|---------|------|
| **损失** | loss, cross_entropy, mse | 损失函数值 |
| **错误率** | error_rate, eer | 错误百分比 |
| **时间** | inference_time, training_time | 执行时间 |
| **资源** | memory_usage, params, flops | 资源占用 |
| **距离** | distance, mae, rmse | 误差距离 |
| **困惑度** | perplexity | 语言模型质量 |
| **WER** | word_error_rate | 语音识别错误率 |

---

## ⚠️ 注意事项

### 1. 参数顺序要匹配

❌ **错误示例**:
```bash
# metric-columns有4个，但higher-is-better只有3个
--metric-columns acc loss f1 time \
--higher-is-better True False True
```

✅ **正确示例**:
```bash
# 4个指标，4个True/False
--metric-columns acc loss f1 time \
--higher-is-better True False True False
```

### 2. True/False区分大小写

❌ **错误**:
```bash
--higher-is-better true false TRUE FALSE
```

✅ **正确**:
```bash
--higher-is-better True False True False
```

### 3. 确保列名完全匹配

先用 `--show-info` 查看列名：
```bash
python tools/data_to_latex.py -i data.csv --show-info
```

然后使用正确的列名：
```bash
--metric-columns accuracy f1_score inference_time
# 不是: acc f1 time
```

---

## 💡 实用技巧

### 技巧1: 只高亮关键指标

不必高亮所有列，只选择最重要的：
```bash
# 只高亮accuracy和loss，不高亮params
--metric-columns accuracy loss \
--higher-is-better True False
```

### 技巧2: 与分组功能结合

在多数据集场景下，每组内分别计算最佳/次佳：
```bash
--metric-columns accuracy loss \
--higher-is-better True False \
--group-column dataset
```

### 技巧3: 只高亮最佳值

如果不想显示次佳，去掉 `--highlight-second`:
```bash
--highlight-best \
--metric-columns accuracy loss \
--higher-is-better True False
```

### 技巧4: 数值列右对齐

让表格更美观：
```bash
--column-align "lrrrr"  # 第一列左对齐，数值列右对齐
```

---

## 📝 完整命令模板

### 模板1: 标准评估
```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table \
  --style booktabs \
  --caption "Model Performance Comparison" \
  --label "tab:performance" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy precision recall f1_score \
  --higher-is-better True True True True \
  --our-model "ProposedMethod" \
  -o paper/table1.tex
```

### 模板2: 效率分析
```bash
python tools/data_to_latex.py \
  -i efficiency.csv \
  -t table \
  --style booktabs \
  --caption "Efficiency Analysis" \
  --label "tab:efficiency" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy inference_time memory_usage params \
  --higher-is-better True False False False \
  --our-model "EfficientModel" \
  --column-align "lrrrr" \
  -o paper/table2.tex
```

### 模板3: 训练过程
```bash
python tools/data_to_latex.py \
  -i training.csv \
  -t table \
  --style booktabs \
  --caption "Training Results" \
  --label "tab:training" \
  --highlight-best \
  --metric-columns train_acc train_loss val_acc val_loss \
  --higher-is-better True False True False \
  -c epoch train_acc train_loss val_acc val_loss \
  -o paper/table3.tex
```

---

## 🎓 学术论文示例

### 论文表格标准配置

```bash
conda run -n ntrain python tools/data_to_latex.py \
  -i paper_results.csv \
  -t table \
  --style booktabs \
  --caption "Comparison with State-of-the-Art Methods on Benchmark Datasets" \
  --label "tab:sota_comparison" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score inference_time \
  --higher-is-better True True False \
  --our-model "Ours" \
  --column-align "lcccc" \
  -c method accuracy f1_score params inference_time \
  -o paper_tables/comparison.tex
```

**LaTeX文档中使用**:
```latex
\documentclass{article}
\usepackage{booktabs}

\begin{document}

\section{Experimental Results}

Table~\ref{tab:sota_comparison} shows the comparison with state-of-the-art methods.
Our method achieves the best accuracy and F1-score while maintaining competitive inference time.

\input{paper_tables/comparison.tex}

\end{document}
```

---

## 🔗 相关文档

- **高级功能指南**: `tools/ADVANCED_FEATURES_GUIDE.md`
- **样式指南**: `tools/TABLE_STYLES_GUIDE.md`
- **快速参考**: `tools/data_to_latex_quickref.md`

---

## 📞 问题排查

### Q: 为什么高亮结果不对？
A: 检查 `--higher-is-better` 的顺序是否与 `--metric-columns` 匹配

### Q: 可以不指定 `--higher-is-better` 吗？
A: 可以，默认所有指标都是越高越好（True）

### Q: 如何验证设置是否正确？
A: 使用 `--show-info` 先查看数据，确认列名和数值

---

## 🎯 最佳实践

1. **清楚标注方向**: 在论文中说明哪些指标越高越好
2. **选择关键指标**: 不要高亮太多列，保持表格简洁
3. **一致性**: 同类论文中使用相同的指标方向
4. **验证结果**: 生成后检查最佳/次佳值是否正确
5. **文档说明**: 在caption或文中说明 "**Bold**: best, *Italic*: second best"

