from src.metrics.x_threshold import *
import numpy as np

shape = (2, 3, 32, 32)
# 生成二值数据 (0 或 1)，避免全0数组导致的sklearn警告
np.random.seed(42)  # 设置随机种子确保结果可重现

# 生成更真实的测试数据：有一定相关性但不完全相同
base_pattern = np.random.rand(*shape)
pred = (base_pattern > 0.4).astype(int)  # 基于基础模式生成预测
# 在目标中添加一些噪声，模拟真实场景
noise = np.random.rand(*shape) * 0.3  # 30%的噪声
target = ((base_pattern + noise) > 0.5).astype(int)

from pprint import pp
pp({
    "accuracy": accuracy_at_threshold(pred, target, thresholds=[0.5, 0.75, 0.9]),
    # "recall": recall_at_threshold(pred, target),
    # "f1": f1_at_threshold(pred, target),
    # "precision": precision_at_threshold(pred, target),
    # "auc": auc_at_threshold(pred, target),
    # "miou_seg": miou_at_threshold_seg(pred, target),
    # "miou_bbox": miou_at_threshold_bbox(pred, target),
})
