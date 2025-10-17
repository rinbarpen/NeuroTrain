# DeepSpeed训练说明

## 概述

`main_deepspeed.py` 是专门用于DeepSpeed分布式训练的主文件，相比 `main.py` 具有以下优势：

1. **简化的代码结构**：专注于DeepSpeed功能，移除了标准训练的复杂逻辑
2. **更好的性能**：针对DeepSpeed优化，减少不必要的条件判断
3. **清晰的日志**：只在主进程输出日志，避免重复信息
4. **专门的启动脚本**：提供便捷的启动方式

## 使用方法

### 1. 直接使用deepspeed命令

```bash
# 单机多GPU训练
deepspeed --num_gpus=4 main_deepspeed.py --config configs/your_config.yaml --task train

# 多机多GPU训练
deepspeed --num_gpus=4 --num_nodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 main_deepspeed.py --config configs/your_config.yaml --task train
```

### 2. 使用torchrun命令（推荐）

```bash
# 单机多GPU训练
torchrun --nproc_per_node=4 main_deepspeed.py --config configs/your_config.yaml --task train

# 多机多GPU训练
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 --master_addr="192.168.1.100" --master_port=29500 main_deepspeed.py --config configs/your_config.yaml --task train
```

### 3. 使用提供的启动脚本

```bash
# 修改run_deepspeed.sh中的配置
./run_deepspeed.sh
```

## 配置要求

### 1. 确保配置文件包含DeepSpeed配置

```yaml
deepspeed:
  enabled: true
  zero_stage: 2
  fp16: true
  cpu_offload: false
  log_level: "INFO"
  config: null  # 使用默认配置，或指定配置文件路径
```

### 2. 环境变量设置

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 设置可见的GPU
export NCCL_DEBUG=INFO  # 可选：启用NCCL调试信息
```

## 主要特性

### 1. 自动DeepSpeed初始化
- 自动检查DeepSpeed可用性
- 自动初始化分布式环境
- 自动设置日志级别

### 2. 智能数据加载
- 自动创建DeepSpeed数据加载器
- 支持验证集和测试集
- 自动处理数据分布

### 3. 模型管理
- 支持预训练模型加载
- 支持检查点恢复
- 自动模型信息记录

### 4. 训练模式支持
- 训练模式：使用DeepSpeedTrainer
- 测试模式：加载训练好的模型进行测试
- 预测模式：使用训练好的模型进行预测

## 注意事项

1. **确保DeepSpeed已安装**：`pip install deepspeed`
2. **GPU内存**：DeepSpeed会优化内存使用，但仍需确保有足够的GPU内存
3. **网络配置**：多机训练需要正确配置网络和防火墙
4. **日志文件**：只有主进程会输出日志，避免重复信息

## 故障排除

### 1. DeepSpeed安装问题
```bash
pip install deepspeed
# 或者从源码安装
pip install deepspeed --no-build-isolation
```

### 2. CUDA版本兼容性
确保CUDA版本与DeepSpeed兼容：
- DeepSpeed 0.9.0+ 支持 CUDA 11.8+
- 检查CUDA版本：`nvidia-smi`

### 3. 内存不足
- 启用CPU卸载：`cpu_offload: true`
- 减少批次大小：调整 `batch_size`
- 使用更高级的ZeRO阶段：`zero_stage: 3`

### 4. 网络问题
- 检查防火墙设置
- 确保所有节点可以相互通信
- 使用正确的master_addr和master_port

## 性能优化建议

1. **使用ZeRO-3**：对于大模型，使用 `zero_stage: 3`
2. **启用混合精度**：设置 `fp16: true` 或 `bf16: true`
3. **优化数据加载**：调整 `num_workers` 和 `pin_memory`
4. **使用CPU卸载**：当GPU内存不足时启用 `cpu_offload: true`
