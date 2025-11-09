# Engine 模块文档

## 概述

Engine模块是NeuroTrain的训练引擎，负责模型的训练、测试和推理。提供了标准训练器、分布式训练器、测试器和预测器，支持混合精度训练、检查点保存、早停机制等功能。

## 主要组件

### 1. Trainer - 标准训练器

标准训练器提供完整的训练流程管理，包括训练循环、验证、检查点保存等。

#### 基本使用

```python
from src.engine import Trainer
from pathlib import Path

# 创建训练器
trainer = Trainer(
    output_dir=Path("runs/experiment_001/train"),
    model=model,
    is_continue_mode=False  # 是否从检查点继续训练
)

# 开始训练
trainer.train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=100,
    device='cuda'
)
```

#### 主要功能

**1. 混合精度训练**

自动使用PyTorch的Automatic Mixed Precision (AMP)加速训练。

```python
# 配置文件中启用
[training]
use_amp = true
amp_dtype = "bfloat16"  # 或 "float16"
```

**2. 检查点保存**

自动保存最佳模型和最新模型检查点。

```python
# 检查点包含：
# - model_state_dict: 模型权重
# - optimizer_state_dict: 优化器状态
# - scheduler_state_dict: 学习率调度器状态
# - epoch: 当前轮次
# - best_metric: 最佳指标值
# - metrics_history: 历史指标

# 保存在以下位置：
# runs/{run_id}/train/checkpoints/best.pth
# runs/{run_id}/train/checkpoints/last.pth
```

**3. 早停机制**

根据验证指标自动停止训练，避免过拟合。

```python
from src.utils import EarlyStopping

early_stopping = EarlyStopping(
    patience=10,      # 等待轮数
    min_delta=1e-4,   # 最小改善量
    mode='min'        # 'min'或'max'
)

# 在训练循环中使用
if early_stopping(val_loss):
    print("Early stopping triggered!")
    break
```

**4. 训练恢复**

支持从中断处继续训练。

```python
# 创建恢复训练器
trainer = Trainer(
    output_dir=output_dir,
    model=model,
    is_continue_mode=True  # 启用恢复模式
)

# 训练器会自动加载最新的检查点和历史记录
```

**5. 指标记录**

自动记录训练和验证指标。

```python
# 指标记录器
from src.recorder import MeterRecorder

class_labels = ['background', 'vessel']
metric_labels = ['dice', 'iou', 'accuracy']

recorder = MeterRecorder(
    class_labels=class_labels,
    metric_labels=metric_labels,
    logger=logger,
    saver=data_saver,
    prefix="train_"
)

# 更新指标
recorder.update(metric_name='dice', class_name='vessel', value=0.85)

# 获取平均值
mean_dice = recorder.get_mean('dice', 'vessel')

# 打印摘要
recorder.print_summary()
```

#### 训练器配置

```toml
[training]
epochs = 100
batch_size = 8
learning_rate = 0.001
optimizer = "adam"
scheduler = "cosine"
weight_decay = 1e-4

# 混合精度
use_amp = true
amp_dtype = "bfloat16"

# 梯度处理
clip_grad_norm = 1.0
gradient_accumulation_steps = 1

# 检查点
save_every_n_epochs = 10
keep_last_n_checkpoints = 3

# 早停
early_stopping = true
patience = 10
min_delta = 1e-4

# 验证
validation_frequency = 1  # 每N个epoch验证一次
```

### 2. DeepSpeedTrainer - 分布式训练器

DeepSpeed训练器支持大规模模型训练，提供ZeRO优化、梯度累积等功能。

#### 基本使用

```python
from src.engine import DeepSpeedTrainer
import deepspeed

# DeepSpeed配置
ds_config = {
    "train_batch_size": 32,
    "gradient_accumulation_steps": 2,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 1e-4,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 1e-4,
            "warmup_num_steps": 1000
        }
    },
    "fp16": {
        "enabled": true
    },
    "zero_optimization": {
        "stage": 2
    }
}

# 创建训练器
trainer = DeepSpeedTrainer(
    output_dir=output_dir,
    model=model,
    deepspeed_config=ds_config
)

# 训练
trainer.train(train_loader, valid_loader, num_epochs=100)
```

#### DeepSpeed特性

**1. ZeRO优化**

分片优化器状态、梯度和参数，大幅减少内存使用。

```python
# Stage 1: 优化器状态分片
"zero_optimization": {
    "stage": 1
}

# Stage 2: 优化器状态 + 梯度分片
"zero_optimization": {
    "stage": 2
}

# Stage 3: 优化器状态 + 梯度 + 参数分片
"zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
        "device": "cpu"  # 卸载到CPU
    }
}
```

**2. 梯度累积**

在小batch size下模拟大batch训练。

```python
"gradient_accumulation_steps": 4  # 累积4个step后更新
```

**3. 混合精度**

```python
# FP16
"fp16": {
    "enabled": true,
    "loss_scale": 0,
    "initial_scale_power": 16
}

# BF16
"bf16": {
    "enabled": true
}
```

#### 启动分布式训练

```bash
# 单机多卡
deepspeed --num_gpus=4 main_deepspeed.py --config config.toml

# 多机多卡
deepspeed --num_gpus=4 --num_nodes=2 \
          --master_addr=192.168.1.1 \
          --master_port=29500 \
          main_deepspeed.py --config config.toml
```

### 3. Tester - 测试器

测试器用于评估模型性能，生成详细的测试报告。

#### 基本使用

```python
from src.engine import Tester

# 创建测试器
tester = Tester(
    output_dir=Path("runs/experiment_001/test"),
    model=model
)

# 运行测试
results = tester.test(
    test_loader=test_loader,
    criterion=criterion,
    device='cuda'
)

# 结果包含：
# - 各类别指标
# - 混淆矩阵
# - 预测可视化
# - 统计报告
```

#### 测试输出

测试结果保存在以下位置：

```
runs/{run_id}/test/
├── metrics.json           # 测试指标（JSON格式）
├── metrics.csv            # 测试指标（CSV格式）
├── confusion_matrix.png   # 混淆矩阵
├── predictions/           # 预测结果
│   ├── sample_001.png
│   ├── sample_002.png
│   └── ...
└── report.txt            # 详细报告
```

#### 指标计算

```python
# 分割任务指标
from src.metrics import dice, iou_seg, accuracy

# 在测试循环中计算
for images, masks in test_loader:
    predictions = model(images)
    
    # 计算指标
    dice_score = dice(predictions, masks)
    iou_score = iou_seg(predictions, masks)
    acc_score = accuracy(predictions, masks)
```

### 4. Predictor - 预测器

预测器用于对新数据进行推理预测。

#### 基本使用

```python
from src.engine import Predictor

# 创建预测器
predictor = Predictor(
    model=model,
    device='cuda',
    output_dir=Path("runs/predictions")
)

# 单样本预测
image = Image.open("test_image.jpg")
prediction = predictor.predict_single(image)

# 批量预测
predictions = predictor.predict_batch(test_loader)

# 保存预测结果
predictor.save_predictions(predictions, output_dir)
```

#### 预测后处理

```python
from src.utils.postprocess import threshold, morphology_ops

# 阈值化
binary_mask = threshold(prediction, threshold=0.5)

# 形态学操作
cleaned_mask = morphology_ops(
    binary_mask,
    operation='opening',
    kernel_size=3
)
```

## 训练流程详解

### 完整训练流程

```python
from src.engine import Trainer
from src.models import get_model
from src.dataset import get_train_valid_test_dataloader
from src.utils.criterion import get_criterion
from src.metrics import get_metric_fns
import torch.optim as optim
from pathlib import Path

# 1. 加载配置
config = get_config()

# 2. 准备数据
train_loader, valid_loader, test_loader = get_train_valid_test_dataloader(config)

# 3. 创建模型
model = get_model(config['model']['name'], config['model'])
model = model.to(device)

# 4. 定义损失函数
criterion = get_criterion(config['training']['loss'])

# 5. 定义优化器
optimizer = optim.Adam(
    model.parameters(),
    lr=config['training']['learning_rate'],
    weight_decay=config['training']['weight_decay']
)

# 6. 学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config['training']['epochs']
)

# 7. 创建训练器
output_dir = Path(f"runs/{config['basic']['run_id']}/train")
trainer = Trainer(output_dir, model)

# 8. 训练
trainer.train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=config['training']['epochs'],
    device=device
)
```

### 训练循环伪代码

```python
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    for batch in train_loader:
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 记录指标
        metrics = compute_metrics(outputs, targets)
        recorder.update(metrics)
    
    # 验证阶段
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            metrics = compute_metrics(outputs, targets)
            recorder.update(metrics)
    
    # 学习率调度
    scheduler.step()
    
    # 保存检查点
    if is_best_model:
        save_checkpoint('best.pth')
    save_checkpoint('last.pth')
    
    # 早停检查
    if early_stopping(val_loss):
        break
```

## 高级功能

### 1. 自定义训练循环

```python
class CustomTrainer(Trainer):
    def train_epoch(self, train_loader, criterion, optimizer, device):
        """自定义训练一个epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, (images, targets) in enumerate(train_loader):
            images, targets = images.to(device), targets.to(device)
            
            # 自定义前向传播
            outputs = self.custom_forward(images)
            loss = self.custom_loss(outputs, targets)
            
            # 自定义反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 自定义梯度处理
            self.custom_grad_clip()
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
```

### 2. 多任务训练

```python
# 多任务损失
class MultiTaskLoss(nn.Module):
    def __init__(self, task_weights):
        super().__init__()
        self.task_weights = task_weights
        self.seg_loss = nn.BCEWithLogitsLoss()
        self.cls_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, targets):
        seg_out, cls_out = outputs
        seg_target, cls_target = targets
        
        seg_loss = self.seg_loss(seg_out, seg_target)
        cls_loss = self.cls_loss(cls_out, cls_target)
        
        total_loss = (
            self.task_weights['seg'] * seg_loss +
            self.task_weights['cls'] * cls_loss
        )
        
        return total_loss, {'seg': seg_loss, 'cls': cls_loss}
```

### 3. 对抗训练

```python
class AdversarialTrainer:
    def __init__(self, model, discriminator):
        self.model = model
        self.discriminator = discriminator
    
    def train_step(self, real_images, fake_images):
        # 训练判别器
        d_loss = self.train_discriminator(real_images, fake_images)
        
        # 训练生成器
        g_loss = self.train_generator(fake_images)
        
        return d_loss, g_loss
```

### 4. 课程学习

```python
class CurriculumTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty_schedule = self.create_schedule()
    
    def create_schedule(self):
        """定义难度递增计划"""
        return {
            0: 'easy',      # epoch 0-20: 简单样本
            20: 'medium',   # epoch 20-50: 中等样本
            50: 'hard'      # epoch 50+: 困难样本
        }
    
    def get_dataloader(self, epoch):
        """根据epoch返回对应难度的数据"""
        difficulty = self.get_difficulty(epoch)
        return self.dataloaders[difficulty]
```

## 性能优化

### 1. 数据加载优化

```python
# 使用多进程和pin_memory
train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2
)
```

### 2. 混合精度训练

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for batch in train_loader:
    with autocast(dtype=torch.bfloat16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 3. 梯度累积

```python
accumulation_steps = 4

for i, batch in enumerate(train_loader):
    outputs = model(inputs)
    loss = criterion(outputs, targets) / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 4. 模型编译（PyTorch 2.0+）

```python
# 编译模型以获得更快的推理
model = torch.compile(model, mode='reduce-overhead')
```

## 监控和调试

### 1. 训练监控

```python
# 使用WandB
import wandb

wandb.init(project="neurotrain", name="experiment_001")

# 在训练循环中记录
wandb.log({
    'train_loss': train_loss,
    'val_loss': val_loss,
    'learning_rate': optimizer.param_groups[0]['lr'],
    'epoch': epoch
})
```

### 2. 可视化训练过程

```python
from src.visualizer import Plot

plot = Plot(output_dir / 'plots')

# 绘制训练曲线
plot.plot_training_curves(
    train_losses=train_losses,
    valid_losses=valid_losses,
    save_path='training_curves.png'
)

# 绘制学习率变化
plot.plot_lr_schedule(
    learning_rates=lrs,
    save_path='lr_schedule.png'
)
```

### 3. 梯度和激活监控

```python
# 监控梯度
def monitor_gradients(model):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            print(f"{name}: grad_norm={grad_norm:.4f}")

# 监控激活
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

model.layer1.register_forward_hook(get_activation('layer1'))
```

## 最佳实践

1. **使用验证集**: 定期在验证集上评估，避免过拟合
2. **保存检查点**: 定期保存模型，防止训练中断
3. **混合精度**: 使用AMP加速训练，节省内存
4. **学习率调度**: 使用学习率衰减提高收敛
5. **早停机制**: 避免不必要的长时间训练
6. **梯度裁剪**: 防止梯度爆炸
7. **数据增强**: 提高模型泛化能力
8. **批归一化**: 加速训练，提高稳定性

## 参考资料

- [PyTorch训练教程](https://pytorch.org/tutorials/)
- [DeepSpeed文档](https://www.deepspeed.ai/)
- [混合精度训练](https://pytorch.org/docs/stable/amp.html)
- [分布式训练](https://pytorch.org/tutorials/beginner/dist_overview.html)

---

更多示例请查看 `examples/` 目录。

