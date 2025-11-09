# Metrics 模块文档

## 概述

Metrics模块提供了丰富的评估指标，用于评估各种深度学习任务的模型性能，包括分割、检测、分类和多模态任务。

## 分割任务指标

### Dice Coefficient

Dice系数是医学图像分割中最常用的指标，范围0-1，值越大越好。

```python
from src.metrics import dice, dice_coefficient

# 计算Dice系数
dice_score = dice(predictions, targets)
# or
dice_score = dice_coefficient(predictions, targets)

print(f"Dice: {dice_score:.4f}")
```

### IoU (Intersection over Union)

交并比，也称为Jaccard指数。

```python
from src.metrics import iou_seg

iou_score = iou_seg(predictions, targets)
print(f"IoU: {iou_score:.4f}")
```

### Normalized Surface Dice (NSD)

归一化表面距离，更关注边界的准确性。

```python
from src.metrics import nsd, normalized_surface_dice

nsd_score = nsd(predictions, targets, tolerance=2.0)
print(f"NSD: {nsd_score:.4f}")
```

## 分类任务指标

### 基础指标

```python
from src.metrics import accuracy, precision, recall, f1, auc

# 准确率
acc = accuracy(predictions, targets)

# 精确率
prec = precision(predictions, targets)

# 召回率
rec = recall(predictions, targets)

# F1分数
f1_score = f1(predictions, targets)

# AUC
auc_score = auc(predictions, targets)
```

### Top-K指标

用于多类别分类任务，评估前K个预测中是否包含正确类别。

```python
from src.metrics import (
    top1_accuracy,
    top3_accuracy,
    top5_accuracy,
    topk_accuracy
)

# Top-1准确率
top1_acc = top1_accuracy(predictions, targets)

# Top-5准确率
top5_acc = top5_accuracy(predictions, targets)

# 自定义Top-K
topk_acc = topk_accuracy(predictions, targets, k=10)
```

## 检测任务指标

### Bounding Box IoU

```python
from src.metrics import iou_bbox

# 边界框格式：[x1, y1, x2, y2]
pred_boxes = [[10, 10, 50, 50], [60, 60, 100, 100]]
target_boxes = [[12, 12, 48, 48], [62, 62, 98, 98]]

iou = iou_bbox(pred_boxes, target_boxes)
```

### mAP (mean Average Precision)

```python
from src.metrics import mAP_at_iou_bbox

map_score = mAP_at_iou_bbox(
    predictions,
    targets,
    iou_thresholds=[0.5, 0.75, 0.95]
)
print(f"mAP: {map_score:.4f}")
```

## 多模态任务指标

### CLIP指标

用于图像-文本对比学习模型的评估。

```python
from src.metrics import (
    clip_accuracy,
    image_retrieval_recall_at_1,
    text_retrieval_recall_at_1,
    image_text_similarity
)

# CLIP准确率
clip_acc = clip_accuracy(image_features, text_features, labels)

# 图像检索召回率@1
img_recall = image_retrieval_recall_at_1(image_features, text_features)

# 文本检索召回率@1
txt_recall = text_retrieval_recall_at_1(text_features, image_features)

# 图像-文本相似度
similarity = image_text_similarity(image_features, text_features)
```

### BLEU分数

用于文本生成任务。

```python
from src.metrics import bleu_1, bleu_4, corpus_bleu

# BLEU-1
bleu1 = bleu_1(predictions, references)

# BLEU-4
bleu4 = bleu_4(predictions, references)

# Corpus BLEU
corpus_score = corpus_bleu(predictions, references)
```

## 阈值相关指标

### 指定阈值的指标

```python
from src.metrics import (
    at_threshold,
    at_accuracy_threshold,
    at_iou_threshold_seg
)

# 在特定阈值下的准确率
acc_at_thresh = at_accuracy_threshold(
    predictions,
    targets,
    threshold=0.5
)

# 在特定IoU阈值下的指标
metrics_at_iou = at_iou_threshold_seg(
    predictions,
    targets,
    iou_threshold=0.5
)
```

## 批量计算指标

### many_metrics

一次计算多个指标。

```python
from src.metrics import many_metrics

metric_fns = ['dice', 'iou_seg', 'accuracy', 'precision', 'recall']

results = many_metrics(
    predictions,
    targets,
    metric_names=metric_fns
)

for name, value in results.items():
    print(f"{name}: {value:.4f}")
```

## 预定义指标集

### 分割任务指标集

```python
from src.metrics import seg_metrics

# 包含：dice, accuracy, recall, f1, precision, auc, iou_seg, mAP, mF1
for metric_fn in seg_metrics:
    score = metric_fn(predictions, targets)
    print(f"{metric_fn.__name__}: {score:.4f}")
```

### 检测任务指标集

```python
from src.metrics import detection_metrics

# 包含：iou_bbox, mAP_at_iou_bbox, mF1_at_iou_bbox
for metric_fn in detection_metrics:
    score = metric_fn(pred_boxes, target_boxes)
    print(f"{metric_fn.__name__}: {score:.4f}")
```

### 分类任务指标集

```python
from src.metrics import classification_topk_metrics

# 包含：top1, top3, top5, top10 accuracy
for metric_fn in classification_topk_metrics:
    score = metric_fn(predictions, targets)
    print(f"{metric_fn.__name__}: {score:.4f}")
```

## 使用示例

### 训练时计算指标

```python
from src.metrics import dice, iou_seg, accuracy
import torch

model.train()
for images, masks in train_loader:
    images, masks = images.to(device), masks.to(device)
    
    # 前向传播
    outputs = model(images)
    loss = criterion(outputs, masks)
    
    # 计算指标（不影响梯度）
    with torch.no_grad():
        dice_score = dice(outputs, masks)
        iou_score = iou_seg(outputs, masks)
        acc_score = accuracy(outputs, masks)
    
    # 反向传播
    loss.backward()
    optimizer.step()
    
    # 记录指标
    print(f"Loss: {loss:.4f}, Dice: {dice_score:.4f}, "
          f"IoU: {iou_score:.4f}, Acc: {acc_score:.4f}")
```

### 测试时评估

```python
from src.metrics import many_metrics

model.eval()
all_predictions = []
all_targets = []

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        outputs = model(images)
        
        all_predictions.append(outputs.cpu())
        all_targets.append(masks)

# 合并所有批次
predictions = torch.cat(all_predictions)
targets = torch.cat(all_targets)

# 计算所有指标
metrics = many_metrics(
    predictions,
    targets,
    metric_names=['dice', 'iou_seg', 'accuracy', 'precision', 'recall']
)

print("Test Results:")
for name, value in metrics.items():
    print(f"  {name}: {value:.4f}")
```

### 多类别指标

```python
num_classes = 3
class_metrics = {}

for class_id in range(num_classes):
    # 为每个类别单独计算指标
    class_pred = (predictions == class_id).float()
    class_target = (targets == class_id).float()
    
    class_metrics[f'class_{class_id}'] = {
        'dice': dice(class_pred, class_target),
        'iou': iou_seg(class_pred, class_target),
        'accuracy': accuracy(class_pred, class_target)
    }

# 打印结果
for class_name, metrics in class_metrics.items():
    print(f"\n{class_name}:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

# 计算平均指标
mean_dice = np.mean([m['dice'] for m in class_metrics.values()])
print(f"\nMean Dice: {mean_dice:.4f}")
```

## 自定义指标

### 创建自定义指标

```python
import torch

def custom_metric(predictions: torch.Tensor, 
                 targets: torch.Tensor) -> float:
    """
    自定义指标函数
    
    Args:
        predictions: 预测结果，shape (B, C, H, W) 或 (B, C)
        targets: 目标标签，shape (B, H, W) 或 (B,)
    
    Returns:
        float: 指标值
    """
    # 实现自定义指标计算
    # ...
    return metric_value

# 使用自定义指标
score = custom_metric(predictions, targets)
```

### 指标工厂函数

```python
from functools import partial

def make_iou_at_threshold(threshold=0.5):
    """创建指定阈值的IoU指标"""
    def iou_at_threshold(predictions, targets):
        binary_pred = (predictions > threshold).float()
        return iou_seg(binary_pred, targets)
    
    iou_at_threshold.__name__ = f'iou_at_{threshold}'
    return iou_at_threshold

# 创建不同阈值的IoU指标
iou_05 = make_iou_at_threshold(0.5)
iou_07 = make_iou_at_threshold(0.7)

print(f"IoU@0.5: {iou_05(predictions, targets):.4f}")
print(f"IoU@0.7: {iou_07(predictions, targets):.4f}")
```

## 最佳实践

1. **选择合适的指标**: 根据任务类型选择最相关的指标
   - 分割：Dice, IoU
   - 分类：Accuracy, F1
   - 检测：mAP, IoU
   
2. **多指标评估**: 使用多个指标全面评估模型
   ```python
   metrics = ['dice', 'iou_seg', 'accuracy', 'precision', 'recall']
   ```

3. **类别级指标**: 对于多类别任务，为每个类别分别计算指标

4. **阈值选择**: 对于概率输出，选择合适的阈值
   ```python
   thresholds = [0.3, 0.5, 0.7]
   for t in thresholds:
       score = iou_at_threshold(predictions, targets, t)
   ```

5. **批量计算**: 在整个测试集上计算指标，而不是单个batch

## 参考资料

- [Dice Coefficient](https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
- [IoU/Jaccard Index](https://en.wikipedia.org/wiki/Jaccard_index)
- [mAP for Object Detection](https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173)
- [BLEU Score](https://en.wikipedia.org/wiki/BLEU)

---

更多示例请查看 `examples/` 目录。

