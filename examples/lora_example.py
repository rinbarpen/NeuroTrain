#!/usr/bin/env python3
"""
LoRA 工具使用示例

本示例演示如何使用 NeuroTrain 的 LoRA 分析器进行：
1. LoRA 适配器合并
2. LoRA 权重分析
3. 多个适配器比较
4. 可视化生成

运行前请确保：
1. 已安装必要的依赖包
2. 有可用的 LoRA 适配器文件
3. 有足够的内存和存储空间
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tools.analyzers import LoRAAnalyzer, analyze_lora_weights, merge_lora_adapters, compare_lora_adapters


def example_lora_analysis():
    """LoRA 分析示例"""
    print("=== LoRA 分析示例 ===")
    
    # 创建 LoRA 分析器
    analyzer = LoRAAnalyzer(
        output_dir="runs/lora_example",
        device="auto",
        use_proxy=True,  # 如果需要下载模型
        trust_remote_code=False,
        local_files_only=False
    )
    
    # 示例：分析单个 LoRA 适配器
    print("\n1. 分析 LoRA 权重...")
    try:
        # 假设有一个 LoRA 适配器
        adapter_path = "path/to/your/lora/adapter"
        if os.path.exists(adapter_path):
            analysis_results = analyzer.analyze_lora_weights(adapter_path)
            print(f"分析完成！结果保存在: {analyzer.analysis_dir}")
            print(f"参数统计: {analysis_results['weight_statistics']}")
        else:
            print(f"适配器路径不存在: {adapter_path}")
            print("请提供有效的 LoRA 适配器路径")
    except Exception as e:
        print(f"分析失败: {e}")
    
    # 示例：合并多个 LoRA 适配器
    print("\n2. 合并 LoRA 适配器...")
    try:
        base_model = "THUDM/chatglm3-6b"  # 或者本地模型路径
        adapters = [
            "path/to/lora/adapter1",
            "path/to/lora/adapter2"
        ]
        
        # 检查适配器是否存在
        existing_adapters = [adapter for adapter in adapters if os.path.exists(adapter)]
        if existing_adapters:
            merge_info = analyzer.merge_adapters(
                base_model=base_model,
                adapters=existing_adapters,
                output_name="merged_model",
                merge_strategy="weighted",
                weights=[0.7, 0.3],
                merge_dtype=None,  # 自动选择
                save_tokenizer=True,
                save_safetensors=True
            )
            print(f"合并完成！模型保存在: {merge_info['output_path']}")
            print(f"合并时间: {merge_info['merge_time']:.2f} 秒")
            print(f"模型大小: {merge_info['model_size_mb']:.2f} MB")
        else:
            print("没有找到有效的适配器路径")
            print("请提供有效的 LoRA 适配器路径")
    except Exception as e:
        print(f"合并失败: {e}")
    
    # 示例：比较多个适配器
    print("\n3. 比较多个 LoRA 适配器...")
    try:
        adapters_to_compare = [
            "path/to/lora/adapter1",
            "path/to/lora/adapter2",
            "path/to/lora/adapter3"
        ]
        
        existing_adapters = [adapter for adapter in adapters_to_compare if os.path.exists(adapter)]
        if len(existing_adapters) >= 2:
            comparison_results = analyzer.compare_adapters(existing_adapters)
            print(f"比较完成！结果保存在: {analyzer.analysis_dir}")
            print(f"比较了 {len(existing_adapters)} 个适配器")
        else:
            print("需要至少2个适配器进行比较")
    except Exception as e:
        print(f"比较失败: {e}")


def example_merge_strategies():
    """演示不同的合并策略"""
    print("\n=== 合并策略示例 ===")
    
    base_model = "THUDM/chatglm3-6b"
    adapters = ["path/to/adapter1", "path/to/adapter2"]
    
    # 检查适配器是否存在
    existing_adapters = [adapter for adapter in adapters if os.path.exists(adapter)]
    if not existing_adapters:
        print("没有找到有效的适配器，跳过合并策略示例")
        return
    
    print(f"使用适配器: {existing_adapters}")
    
    # 1. 顺序合并
    print("\n1. 顺序合并...")
    try:
        merge_info = merge_lora_adapters(
            base_model=base_model,
            adapters=existing_adapters,
            output_name="sequential_merged",
            merge_strategy="sequential",
            output_dir="runs/lora_example"
        )
        print(f"顺序合并完成: {merge_info['output_path']}")
    except Exception as e:
        print(f"顺序合并失败: {e}")
    
    # 2. 加权合并
    print("\n2. 加权合并...")
    try:
        merge_info = merge_lora_adapters(
            base_model=base_model,
            adapters=existing_adapters,
            output_name="weighted_merged",
            merge_strategy="weighted",
            weights=[0.6, 0.4],
            output_dir="runs/lora_example"
        )
        print(f"加权合并完成: {merge_info['output_path']}")
    except Exception as e:
        print(f"加权合并失败: {e}")
    
    # 3. 平均合并
    print("\n3. 平均合并...")
    try:
        merge_info = merge_lora_adapters(
            base_model=base_model,
            adapters=existing_adapters,
            output_name="average_merged",
            merge_strategy="average",
            output_dir="runs/lora_example"
        )
        print(f"平均合并完成: {merge_info['output_path']}")
    except Exception as e:
        print(f"平均合并失败: {e}")


def example_command_line_usage():
    """演示命令行使用方法"""
    print("\n=== 命令行使用示例 ===")
    
    print("""
    1. 基本顺序合并：
    python tools/lora_merge.py \\
        --base THUDM/chatglm3-6b \\
        --adapters ./output/lora_run1 ./output/lora_run2 \\
        --output ./merged_model \\
        --merge-dtype float16 \\
        --trust-remote-code \\
        --save-tokenizer \\
        --use-proxy

    2. 加权合并：
    python tools/lora_merge.py \\
        --base THUDM/chatglm3-6b \\
        --adapters ./output/lora_run1 ./output/lora_run2 \\
        --output ./merged_model \\
        --merge-strategy weighted \\
        --weights 0.7 0.3 \\
        --merge-dtype float16

    3. 平均合并：
    python tools/lora_merge.py \\
        --base THUDM/chatglm3-6b \\
        --adapters ./output/lora_run1 ./output/lora_run2 \\
        --output ./merged_model \\
        --merge-strategy average

    4. 仅使用本地文件：
    python tools/lora_merge.py \\
        --base ./local_model \\
        --adapters ./lora_out \\
        --output ./merged_model \\
        --local-files-only
    """)


def main():
    """主函数"""
    print("LoRA 工具使用示例")
    print("=" * 50)
    
    # 创建输出目录
    os.makedirs("runs/lora_example", exist_ok=True)
    
    # 运行示例
    example_lora_analysis()
    example_merge_strategies()
    example_command_line_usage()
    
    print("\n=== 示例完成 ===")
    print("请查看 runs/lora_example/ 目录中的结果文件")
    print("包括：")
    print("- merged_models/: 合并后的模型")
    print("- analysis/: 分析结果JSON文件")
    print("- visualizations/: 可视化图表")
    print("- analysis_report.md: 分析报告")


if __name__ == "__main__":
    main()
