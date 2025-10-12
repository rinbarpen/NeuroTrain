#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文本生成示例：展示如何像调用llama一样使用ProbNet

该示例演示了：
1. 如何初始化ProbNet模型进行文本生成
2. 如何输入文本并获得文本响应
3. 如何调整生成参数
4. 如何处理错误情况
"""

import torch
import sys
import os
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

from prob_net import ProbNet
from ...llm.transformers import LLAMA3_MODEL_ID_8B_INSTRUCT

def test_text_generation():
    """
    测试文本生成功能
    """
    print("=== ProbNet 文本生成测试 ===")
    
    # 初始化模型
    print("\n正在初始化ProbNet模型...")
    try:
        model = ProbNet(
            model_id=LLAMA3_MODEL_ID_8B_INSTRUCT,
            extract_layers=[2, 6, 10],  # 提取第2、6、10层特征
            hidden_dim=4096,
            num_heads=8,
            dropout=0.1
        )
        print("✓ 模型初始化成功")
    except Exception as e:
        print(f"✗ 模型初始化失败: {e}")
        return
    
    # 测试用例
    test_cases = [
        "你好，请介绍一下人工智能的发展历史。",
        "What is the capital of France?",
        "请解释一下深度学习中的注意力机制。",
        "How does machine learning work?",
        "请写一首关于春天的诗。"
    ]
    
    print("\n=== 开始文本生成测试 ===")
    
    for i, input_text in enumerate(test_cases, 1):
        print(f"\n--- 测试用例 {i} ---")
        print(f"输入: {input_text}")
        
        try:
            # 方式1: 使用generate方法
            response1 = model.generate(
                text=input_text,
                max_length=256,
                temperature=0.7
            )
            print(f"generate方法输出: {response1}")
            
            # 方式2: 直接调用（像函数一样）
            response2 = model(input_text, max_length=256, temperature=0.8)
            print(f"直接调用输出: {response2}")
            
        except Exception as e:
            print(f"✗ 生成失败: {e}")
        
        print("-" * 50)

def test_different_parameters():
    """
    测试不同的生成参数
    """
    print("\n=== 测试不同生成参数 ===")
    
    try:
        model = ProbNet()
        input_text = "请解释什么是机器学习？"
        
        # 测试不同温度
        temperatures = [0.1, 0.5, 0.9, 1.2]
        
        for temp in temperatures:
            print(f"\n温度 {temp}:")
            response = model(input_text, temperature=temp, max_length=128)
            print(f"输出: {response}")
            
    except Exception as e:
        print(f"参数测试失败: {e}")

def interactive_chat():
    """
    交互式聊天模式
    """
    print("\n=== 交互式聊天模式 ===")
    print("输入 'quit' 或 'exit' 退出")
    
    try:
        model = ProbNet()
        
        while True:
            user_input = input("\n你: ").strip()
            
            if user_input.lower() in ['quit', 'exit', '退出']:
                print("再见！")
                break
            
            if not user_input:
                continue
            
            print("ProbNet正在思考...")
            response = model(user_input, temperature=0.7, max_length=256)
            print(f"ProbNet: {response}")
            
    except KeyboardInterrupt:
        print("\n\n聊天已中断")
    except Exception as e:
        print(f"聊天模式错误: {e}")

def benchmark_performance():
    """
    性能基准测试
    """
    print("\n=== 性能基准测试 ===")
    
    try:
        model = ProbNet()
        test_text = "请简要介绍深度学习的基本概念。"
        
        # 预热
        print("预热中...")
        _ = model(test_text, max_length=64)
        
        # 性能测试
        num_runs = 5
        total_time = 0
        
        print(f"\n开始 {num_runs} 次性能测试...")
        
        for i in range(num_runs):
            start_time = datetime.now()
            response = model(test_text, max_length=128)
            end_time = datetime.now()
            
            duration = (end_time - start_time).total_seconds()
            total_time += duration
            
            print(f"第 {i+1} 次: {duration:.2f}秒")
            print(f"输出长度: {len(response)} 字符")
        
        avg_time = total_time / num_runs
        print(f"\n平均生成时间: {avg_time:.2f}秒")
        
    except Exception as e:
        print(f"性能测试失败: {e}")

def main():
    """
    主函数
    """
    print("ProbNet 文本生成演示")
    print("=" * 50)
    
    # 设置随机种子
    torch.manual_seed(42)
    
    # 运行测试
    test_text_generation()
    
    # 测试不同参数
    test_different_parameters()
    
    # 性能测试
    benchmark_performance()
    
    # 交互式聊天（可选）
    user_choice = input("\n是否进入交互式聊天模式？(y/n): ").strip().lower()
    if user_choice in ['y', 'yes', '是']:
        interactive_chat()
    
    print("\n演示完成！")

if __name__ == "__main__":
    main()