# DDP训练说明

## 概述

`main.py` 现在支持DDP（Distributed Data Parallel）分布式训练，相比DeepSpeed具有以下特点：

1. **原生PyTorch支持**：使用PyTorch内置的DDP，无需额外依赖
2. **简单易用**：配置简单，启动方便
3. **性能稳定**：经过充分测试的分布式训练方案
4. **兼容性好**：与标准训练模式完全兼容

## 使用方法

### 1. 使用torchrun命令（推荐）

```bash
# 单机多GPU训练
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.yaml --task train

# 多机多GPU训练
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 main.py --config configs/ddp_example.yaml --task train
```

### 2. 使用python -m torch.distributed.launch（旧版本）

```bash
# 单机多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 main.py --config configs/ddp_example.yaml --task train

# 多机多GPU训练
python -m torch.distributed.launch --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 main.py --config configs/ddp_example.yaml --task train
```

### 3. 使用提供的启动脚本

```bash
# 修改run_ddp.sh中的配置
./run_ddp.sh
```

## 配置要求

### 1. 确保配置文件包含DDP配置

```yaml
ddp:
  enabled: true
  log_level: "INFO"
```

### 2. 环境变量设置

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 设置可见的GPU
export NCCL_DEBUG=INFO  # 可选：启用NCCL调试信息
```

## 主要特性

### 1. 自动DDP初始化
- 自动初始化分布式环境
- 自动设置CUDA设备
- 自动设置日志级别

### 2. 智能数据加载
- 自动创建分布式采样器
- 支持验证集和测试集
- 自动处理数据分布

### 3. 模型管理
- 支持预训练模型加载
- 支持检查点恢复
- 自动模型信息记录

### 4. 训练模式支持
- 训练模式：使用DDP包装的标准训练器
- 测试模式：加载训练好的模型进行测试
- 预测模式：使用训练好的模型进行预测

## 注意事项

1. **确保PyTorch版本**：需要PyTorch 1.6.0+
2. **GPU内存**：DDP会在每个GPU上复制模型，确保有足够的GPU内存
3. **网络配置**：多机训练需要正确配置网络和防火墙
4. **日志文件**：只有主进程会输出日志，避免重复信息

## 故障排除

### 1. CUDA版本兼容性
确保CUDA版本与PyTorch兼容：
- 检查CUDA版本：`nvidia-smi`
- 检查PyTorch版本：`python -c "import torch; print(torch.__version__)"`

### 2. 内存不足
- 减少批次大小：调整 `batch_size`
- 使用梯度累积：调整 `grad_accumulation_steps`
- 使用混合精度训练：在训练器中启用

### 3. 网络问题
- 检查防火墙设置
- 确保所有节点可以相互通信
- 使用正确的master_addr和master_port

### 4. 数据加载问题
- 调整 `num_workers` 参数
- 检查数据路径是否正确
- 确保数据集大小足够

## 性能优化建议

1. **使用混合精度**：在训练器中启用FP16或BF16
2. **优化数据加载**：调整 `num_workers` 和 `pin_memory`
3. **使用梯度累积**：当GPU内存不足时使用 `grad_accumulation_steps`
4. **调整批次大小**：根据GPU内存和模型大小调整

## 与DeepSpeed的区别

| 特性 | DDP | DeepSpeed |
|------|-----|-----------|
| 依赖 | PyTorch内置 | 需要安装DeepSpeed |
| 内存优化 | 基础 | 高级（ZeRO等） |
| 配置复杂度 | 简单 | 复杂 |
| 性能 | 稳定 | 更高（大模型） |
| 兼容性 | 好 | 一般 |

## 示例命令

```bash
# 单机4GPU训练
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.yaml --task train

# 单机4GPU测试
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.yaml --task test

# 单机4GPU预测
torchrun --nproc_per_node=4 main.py --config configs/ddp_example.yaml --task predict
```
