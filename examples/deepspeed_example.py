#!/usr/bin/env python3
"""
DeepSpeed 训练示例

本示例演示如何使用 NeuroTrain 的 DeepSpeed 支持进行分布式训练。

运行前请确保：
1. 已安装 DeepSpeed: pip install deepspeed
2. 有多个 GPU 可用
3. 已安装必要的依赖包

使用方法：
1. 单机多GPU训练：
   deepspeed --num_gpus=2 examples/deepspeed_example.py --config configs/single/train-deepspeed.toml

2. 多机多GPU训练：
   deepspeed --num_gpus=4 --num_nodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 examples/deepspeed_example.py --config configs/single/train-deepspeed.toml
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.deepspeed_utils import (
    is_deepspeed_available,
    init_deepspeed_distributed,
    create_deepspeed_config,
    get_deepspeed_rank_info,
    is_main_process,
    log_deepspeed_memory_usage
)


def example_deepspeed_config():
    """演示如何创建 DeepSpeed 配置"""
    print("=== DeepSpeed 配置示例 ===")
    
    # 检查 DeepSpeed 是否可用
    if not is_deepspeed_available():
        print("❌ DeepSpeed 不可用，请安装: pip install deepspeed")
        return
    
    print("✅ DeepSpeed 可用")
    
    # 创建不同 ZeRO 阶段的配置
    configs = {
        "ZeRO Stage 1": create_deepspeed_config(
            zero_stage=1,
            train_batch_size=32,
            micro_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            fp16=True
        ),
        "ZeRO Stage 2": create_deepspeed_config(
            zero_stage=2,
            train_batch_size=64,
            micro_batch_size=16,
            gradient_accumulation_steps=4,
            learning_rate=1e-4,
            weight_decay=0.01,
            fp16=True,
            cpu_offload=False
        ),
        "ZeRO Stage 3": create_deepspeed_config(
            zero_stage=3,
            train_batch_size=128,
            micro_batch_size=8,
            gradient_accumulation_steps=16,
            learning_rate=1e-4,
            weight_decay=0.01,
            fp16=True,
            cpu_offload=True
        )
    }
    
    for name, config in configs.items():
        print(f"\n{name} 配置:")
        print(f"  - 训练批次大小: {config['train_batch_size']}")
        print(f"  - 微批次大小: {config['train_micro_batch_size_per_gpu']}")
        print(f"  - 梯度累积步数: {config['gradient_accumulation_steps']}")
        print(f"  - ZeRO 阶段: {config['zero_optimization']['stage']}")
        print(f"  - CPU 卸载: {config['zero_optimization'].get('cpu_offload', False)}")
        print(f"  - 混合精度: {'FP16' if config.get('fp16', {}).get('enabled', False) else 'None'}")


def example_deepspeed_training():
    """演示 DeepSpeed 训练流程"""
    print("\n=== DeepSpeed 训练示例 ===")
    
    if not is_deepspeed_available():
        print("❌ DeepSpeed 不可用，跳过训练示例")
        return
    
    # 初始化分布式环境
    try:
        rank_info = init_deepspeed_distributed()
        print(f"✅ 分布式环境初始化成功")
        print(f"  - 进程排名: {rank_info['rank']}")
        print(f"  - 本地排名: {rank_info['local_rank']}")
        print(f"  - 世界大小: {rank_info['world_size']}")
        
        # 记录内存使用情况
        log_deepspeed_memory_usage(logging.getLogger(__name__), "初始化后")
        
        # 创建 DeepSpeed 配置
        config = create_deepspeed_config(
            zero_stage=2,
            train_batch_size=32,
            micro_batch_size=16,
            gradient_accumulation_steps=2,
            learning_rate=1e-4,
            weight_decay=0.01,
            fp16=True
        )
        
        print(f"✅ DeepSpeed 配置创建成功")
        print(f"  - ZeRO 阶段: {config['zero_optimization']['stage']}")
        print(f"  - 训练批次大小: {config['train_batch_size']}")
        
        if is_main_process():
            print("🎯 这是主进程，可以执行训练逻辑")
        else:
            print(f"🔄 这是工作进程 {rank_info['rank']}")
            
    except Exception as e:
        print(f"❌ DeepSpeed 初始化失败: {e}")


def example_command_line_usage():
    """演示命令行使用方法"""
    print("\n=== 命令行使用示例 ===")
    
    print("""
    1. 单机多GPU训练（2个GPU）：
    deepspeed --num_gpus=2 main.py --config configs/single/train-deepspeed.toml

    2. 单机多GPU训练（4个GPU）：
    deepspeed --num_gpus=4 main.py --config configs/single/train-deepspeed.toml

    3. 多机多GPU训练（2个节点，每个节点2个GPU）：
    # 在第一个节点上：
    deepspeed --num_gpus=2 --num_nodes=2 --node_rank=0 --master_addr="192.168.1.1" --master_port=29500 main.py --config configs/single/train-deepspeed.toml
    
    # 在第二个节点上：
    deepspeed --num_gpus=2 --num_nodes=2 --node_rank=1 --master_addr="192.168.1.1" --master_port=29500 main.py --config configs/single/train-deepspeed.toml

    4. 使用自定义 DeepSpeed 配置：
    deepspeed --num_gpus=2 main.py --config configs/single/train-deepspeed.toml --deepspeed.config configs/deepspeed/deepspeed_zero3.json

    5. 启用 CPU 卸载（节省 GPU 内存）：
    deepspeed --num_gpus=2 main.py --config configs/single/train-deepspeed.toml --deepspeed.cpu_offload true

    6. 使用 BF16 混合精度：
    deepspeed --num_gpus=2 main.py --config configs/single/train-deepspeed.toml --deepspeed.bf16 true
    """)


def example_configuration_options():
    """演示配置选项"""
    print("\n=== 配置选项说明 ===")
    
    print("""
    DeepSpeed 配置选项（在 TOML 配置文件中）：
    
    [deepspeed]
    enabled = true                    # 启用 DeepSpeed
    zero_stage = 2                   # ZeRO 优化阶段 (1, 2, 3)
    cpu_offload = false              # 是否启用 CPU 卸载
    fp16 = true                      # 是否启用 FP16 混合精度
    bf16 = false                     # 是否启用 BF16 混合精度
    log_level = "INFO"               # 日志级别
    config = "path/to/config.json"   # 自定义 DeepSpeed 配置文件路径
    
    ZeRO 阶段说明：
    - Stage 1: 优化器状态分片
    - Stage 2: 优化器状态 + 梯度分片
    - Stage 3: 优化器状态 + 梯度 + 参数分片
    
    内存优化建议：
    - 小模型（< 1B 参数）：使用 ZeRO Stage 1
    - 中等模型（1B - 10B 参数）：使用 ZeRO Stage 2
    - 大模型（> 10B 参数）：使用 ZeRO Stage 3 + CPU 卸载
    """)


def main():
    """主函数"""
    print("DeepSpeed 训练示例")
    print("=" * 50)
    
    # 运行示例
    example_deepspeed_config()
    example_deepspeed_training()
    example_command_line_usage()
    example_configuration_options()
    
    print("\n=== 示例完成 ===")
    print("请查看以下文件获取更多信息：")
    print("- configs/single/train-deepspeed.toml: DeepSpeed 训练配置示例")
    print("- configs/deepspeed/: DeepSpeed 配置文件模板")
    print("- src/engine/deepspeed_trainer.py: DeepSpeed 训练器实现")
    print("- src/utils/deepspeed_utils.py: DeepSpeed 工具函数")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
