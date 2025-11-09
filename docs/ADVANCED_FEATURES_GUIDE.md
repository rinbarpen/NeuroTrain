# LaTeX表格高级功能指南

## 📋 概述

`data_to_latex.py` 工具现在支持学术论文中常用的高级表格功能：
- ✅ **自动高亮最佳值**（粗体）
- ✅ **自动高亮次佳值**（斜体）
- ✅ **标注自己的模型**（下划线）
- ✅ **多数据集/多任务比较**（分组显示）
- ✅ **灵活的指标方向**（越高越好/越低越好）

## 🎯 核心功能

### 1. 高亮最佳和次佳值

自动识别并高亮表格中的最佳和次佳值，让读者一眼看出最好的结果。

#### 基本用法

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True
```

#### 输出示例

```latex
\begin{tabular}{llll}
\toprule
model & accuracy & f1_score & params \\
\midrule
ResNet50 & 0.9523 & 0.9412 & 25.6M \\
VGG16 & 0.9234 & 0.9145 & 138.4M \\
OurModel & \textbf{0.9678} & \textbf{0.9589} & 5.3M \\  % 最佳（粗体）
MobileNet & 0.9012 & 0.8934 & 4.2M \\
EfficientNet & \textit{0.9556} & \textit{0.9478} & 7.8M \\  % 次佳（斜体）
\bottomrule
\end{tabular}
```

### 2. 标注我们的模型

使用下划线突出显示你的模型，让它在表格中更显眼。

#### 基本用法

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table \
  --style booktabs \
  --our-model "OurModel"
```

#### 输出示例

```latex
\begin{tabular}{llll}
\toprule
model & accuracy & f1_score & params \\
\midrule
ResNet50 & 0.9523 & 0.9412 & 25.6M \\
\underline{OurModel} & 0.9678 & 0.9589 & 5.3M \\  % 我们的模型（下划线）
VGG16 & 0.9234 & 0.9145 & 138.4M \\
\bottomrule
\end{tabular}
```

### 3. 组合功能

同时使用高亮和模型标注功能：

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "OurModel"
```

#### 输出示例

```latex
\begin{tabular}{llll}
\toprule
model & accuracy & f1_score & params \\
\midrule
ResNet50 & 0.9523 & 0.9412 & 25.6M \\
VGG16 & 0.9234 & 0.9145 & 138.4M \\
\underline{OurModel} & \textbf{0.9678} & \textbf{0.9589} & 5.3M \\  % 我们的模型 + 最佳
MobileNet & 0.9012 & 0.8934 & 4.2M \\
EfficientNet & \textit{0.9556} & \textit{0.9478} & 7.8M \\  % 次佳
\bottomrule
\end{tabular}
```

### 4. 指标方向设置

不同的指标有不同的优化方向：
- **Accuracy, F1-Score**: 越高越好 (True)
- **Loss, Error Rate**: 越低越好 (False)

#### 示例：混合指标方向

```bash
python tools/data_to_latex.py \
  -i results.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --metric-columns accuracy loss error_rate \
  --higher-is-better True False False
```

数据示例：
```csv
model,accuracy,loss,error_rate
ModelA,0.95,0.12,0.05
ModelB,0.93,0.08,0.07
ModelC,0.97,0.15,0.03
```

输出：
- **accuracy**: ModelC (0.97) 最佳
- **loss**: ModelB (0.08) 最佳（越低越好）
- **error_rate**: ModelC (0.03) 最佳（越低越好）

### 5. 多数据集/多任务比较

使用分组功能在一个表格中比较多个数据集或任务的结果。

#### 基本用法

```bash
python tools/data_to_latex.py \
  -i multi_dataset.csv \
  -t table \
  --style booktabs \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "OurModel" \
  --group-column dataset
```

#### 数据格式

```csv
dataset,model,accuracy,f1_score
CIFAR10,ResNet50,0.9523,0.9412
CIFAR10,VGG16,0.9234,0.9145
CIFAR10,OurModel,0.9678,0.9589
CIFAR100,ResNet50,0.7812,0.7645
CIFAR100,VGG16,0.7534,0.7412
CIFAR100,OurModel,0.8234,0.8156
ImageNet,ResNet50,0.7634,0.7523
ImageNet,VGG16,0.7156,0.7045
ImageNet,OurModel,0.7945,0.7834
```

#### 输出示例

```latex
\begin{tabular}{llll}
\toprule
dataset & model & accuracy & f1_score \\
\midrule
CIFAR10 & ResNet50 & \textit{0.9523} & \textit{0.9412} \\
CIFAR10 & VGG16 & 0.9234 & 0.9145 \\
CIFAR10 & \underline{OurModel} & \textbf{0.9678} & \textbf{0.9589} \\
\midrule  % 自动分组分隔
CIFAR100 & ResNet50 & \textit{0.7812} & \textit{0.7645} \\
CIFAR100 & VGG16 & 0.7534 & 0.7412 \\
CIFAR100 & \underline{OurModel} & \textbf{0.8234} & \textbf{0.8156} \\
\midrule
ImageNet & ResNet50 & \textit{0.7634} & \textit{0.7523} \\
ImageNet & VGG16 & 0.7156 & 0.7045 \\
ImageNet & \underline{OurModel} & \textbf{0.7945} & \textbf{0.7834} \\
\bottomrule
\end{tabular}
```

**特点**:
- 每个数据集内部分别计算最佳/次佳
- OurModel在每个组都被标注
- 组之间自动添加分隔线

## 📊 实际应用场景

### 场景1: 单一数据集模型比较

**数据**: `single_dataset_results.csv`
```csv
model,accuracy,precision,recall,f1_score
Baseline,0.8234,0.8123,0.8345,0.8232
ResNet50,0.9123,0.9045,0.9201,0.9122
BERT,0.9345,0.9256,0.9434,0.9344
OurModel,0.9567,0.9478,0.9656,0.9566
Transformer,0.9234,0.9145,0.9323,0.9233
```

**命令**:
```bash
python tools/data_to_latex.py \
  -i single_dataset_results.csv \
  -t table \
  --style booktabs \
  --caption "Model Comparison on Dataset X" \
  --label "tab:single_comparison" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy precision recall f1_score \
  --higher-is-better True True True True \
  --our-model "OurModel" \
  -o paper_tables/single_comparison.tex
```

### 场景2: 多数据集泛化性能比较

**数据**: `generalization_results.csv`
```csv
dataset,model,accuracy,f1_score,inference_time
MNIST,CNN,0.9912,0.9910,12.3
MNIST,OurModel,0.9945,0.9943,8.7
MNIST,MLP,0.9856,0.9854,15.2
CIFAR10,CNN,0.8567,0.8534,45.6
CIFAR10,OurModel,0.8823,0.8801,32.1
CIFAR10,MLP,0.7234,0.7201,58.9
SVHN,CNN,0.9234,0.9212,38.4
SVHN,OurModel,0.9456,0.9434,25.8
SVHN,MLP,0.8123,0.8101,51.2
```

**命令**:
```bash
python tools/data_to_latex.py \
  -i generalization_results.csv \
  -t table \
  --style booktabs \
  --caption "Generalization Performance Across Datasets" \
  --label "tab:generalization" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score inference_time \
  --higher-is-better True True False \
  --our-model "OurModel" \
  --group-column dataset \
  -o paper_tables/generalization.tex
```

### 场景3: 消融实验

**数据**: `ablation_study.csv`
```csv
component,model_variant,accuracy,params,training_time
Full Model,OurModel-Full,0.9567,12.3M,120
w/o Attention,OurModel-NoAttn,0.9234,10.1M,95
w/o Residual,OurModel-NoRes,0.9123,11.8M,110
w/o Both,OurModel-Minimal,0.8856,9.2M,85
```

**命令**:
```bash
python tools/data_to_latex.py \
  -i ablation_study.csv \
  -t table \
  --style booktabs \
  --caption "Ablation Study Results" \
  --label "tab:ablation" \
  --highlight-best \
  --metric-columns accuracy training_time \
  --higher-is-better True False \
  --our-model "OurModel-Full" \
  -c model_variant accuracy params training_time \
  -o paper_tables/ablation.tex
```

### 场景4: 多任务学习

**数据**: `multitask_results.csv`
```csv
task,model,accuracy,f1_score
Classification,Single-Task,0.9234,0.9212
Classification,Multi-Task,0.9456,0.9434
Classification,OurModel,0.9567,0.9545
Detection,Single-Task,0.8567,0.8534
Detection,Multi-Task,0.8723,0.8701
Detection,OurModel,0.8856,0.8834
Segmentation,Single-Task,0.7856,0.7823
Segmentation,Multi-Task,0.8123,0.8101
Segmentation,OurModel,0.8345,0.8312
```

**命令**:
```bash
python tools/data_to_latex.py \
  -i multitask_results.csv \
  -t table \
  --style booktabs \
  --caption "Multi-Task Learning Performance" \
  --label "tab:multitask" \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "OurModel" \
  --group-column task \
  -o paper_tables/multitask.tex
```

## 🔧 参数详解

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `-i, --input` | 输入数据文件 | `-i results.csv` |

### 高亮相关参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--highlight-best` | 启用最佳值高亮（粗体） | `--highlight-best` |
| `--highlight-second` | 启用次佳值高亮（斜体） | `--highlight-second` |
| `--metric-columns` | 指定要高亮的指标列 | `--metric-columns acc f1` |
| `--higher-is-better` | 指定每列的优化方向 | `--higher-is-better True False` |

### 模型标注参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--our-model` | 指定我们的模型名称 | `--our-model "OurModel"` |

### 分组参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--group-column` | 指定分组列 | `--group-column dataset` |

## 💡 使用技巧

### 1. 选择合适的列

只选择重要的列展示：
```bash
-c model accuracy f1_score params
```

### 2. 自定义对齐

数值列右对齐更美观：
```bash
--column-align "lrrrr"  # 第一列左对齐，其余右对齐
```

### 3. 组合样式

使用booktabs获得最佳效果：
```bash
--style booktabs
```

### 4. 只高亮最佳值

如果表格较小，只高亮最佳值：
```bash
--highlight-best  # 不加--highlight-second
```

### 5. 检查数据

转换前先查看数据：
```bash
--show-info
```

## ⚠️ 注意事项

### 1. 指标列名要精确匹配

```bash
# 正确：列名必须完全匹配
--metric-columns accuracy f1_score

# 错误：列名不匹配（实际是'f1_score'，写成了'f1'）
--metric-columns accuracy f1
```

### 2. higher-is-better数量要匹配

```bash
# 正确：3个指标列，3个True/False
--metric-columns acc loss f1 \
--higher-is-better True False True

# 错误：3个指标列，只有2个True/False
--metric-columns acc loss f1 \
--higher-is-better True False
```

### 3. 分组列必须存在

确保指定的分组列存在于数据中：
```bash
# 先用--show-info检查列名
python tools/data_to_latex.py -i data.csv --show-info

# 然后使用正确的列名
--group-column dataset
```

### 4. 数值列类型

确保指标列包含可比较的数值：
```bash
# 好：纯数值
0.9234, 0.9567, 0.8856

# 坏：混合格式
"95.6%", "0.9234", "N/A"
```

## 📝 完整示例命令

### 示例1：完整配置

```bash
python tools/data_to_latex.py \
  -i experiment_results.csv \
  -o paper/table1.tex \
  -t table \
  --style booktabs \
  --caption "Comparison of State-of-the-Art Methods" \
  --label "tab:comparison" \
  --column-align "lcccc" \
  -c model accuracy f1_score params \
  --highlight-best \
  --highlight-second \
  --metric-columns accuracy f1_score \
  --higher-is-better True True \
  --our-model "ProposedMethod"
```

### 示例2：多数据集比较

```bash
python tools/data_to_latex.py \
  -i multi_dataset_results.csv \
  -o paper/table2.tex \
  -t table \
  --style booktabs \
  --caption "Cross-Dataset Evaluation" \
  --label "tab:crossdataset" \
  --highlight-best \
  --metric-columns accuracy precision recall \
  --higher-is-better True True True \
  --our-model "OurApproach" \
  --group-column dataset
```

## 🎓 学术论文最佳实践

### 1. 使用booktabs样式

```bash
--style booktabs
```

### 2. 添加有意义的标题和标签

```bash
--caption "Performance Comparison on Benchmark Datasets" \
--label "tab:benchmark_comparison"
```

### 3. 高亮最重要的指标

不要高亮所有列，只高亮主要指标：
```bash
--metric-columns accuracy f1_score  # 不包括params等辅助列
```

### 4. 明确指标方向

```bash
--higher-is-better True False True  # 明确每个指标的优化方向
```

### 5. 使用分组展示多场景

```bash
--group-column dataset  # 或 task, domain 等
```

## 📚 更多资源

- **基础教程**: `tools/data_to_latex_examples.md`
- **样式指南**: `tools/TABLE_STYLES_GUIDE.md`
- **快速参考**: `tools/data_to_latex_quickref.md`
- **模板功能**: `tools/TEMPLATE_FEATURE_SUMMARY.md`

