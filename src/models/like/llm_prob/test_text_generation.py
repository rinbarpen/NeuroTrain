#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单测试脚本：验证ProbNet文本生成功能

这是一个轻量级的测试脚本，用于快速验证ProbNet是否能够：
1. 正确初始化
2. 接受文本输入
3. 返回文本输出
4. 像调用llama一样工作
"""

import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

def test_basic_functionality():
    """
    基础功能测试
    """
    print("开始基础功能测试...")
    
    try:
        # 导入ProbNet
        from .prob_net import ProbNet
        print("✓ ProbNet导入成功")
        
        # 初始化模型（使用默认参数）
        print("正在初始化模型...")
        model = ProbNet()
        print("✓ 模型初始化成功")
        
        # 测试文本输入输出
        test_input = "Hello, how are you?"
        print(f"\n测试输入: {test_input}")
        
        # 方法1: 使用generate方法
        print("使用generate方法...")
        output1 = model.generate(test_input, max_length=50)
        print(f"generate输出: {output1}")
        
        # 方法2: 直接调用
        print("\n使用直接调用...")
        output2 = model(test_input, max_length=50)
        print(f"直接调用输出: {output2}")
        
        print("\n✓ 基础功能测试通过！")
        return True
        
    except ImportError as e:
        print(f"✗ 导入错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 测试失败: {e}")
        return False

def test_chinese_input():
    """
    测试中文输入
    """
    print("\n开始中文输入测试...")
    
    try:
        from .prob_net import ProbNet
        
        model = ProbNet()
        
        # 中文测试
        chinese_input = "你好，请介绍一下自己。"
        print(f"中文输入: {chinese_input}")
        
        output = model(chinese_input, max_length=100)
        print(f"中文输出: {output}")
        
        print("✓ 中文输入测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 中文测试失败: {e}")
        return False

def test_error_handling():
    """
    测试错误处理
    """
    print("\n开始错误处理测试...")
    
    try:
        from .prob_net import ProbNet
        
        model = ProbNet()
        
        # 测试空输入
        try:
            output = model("", max_length=50)
            print("空输入处理正常")
        except Exception as e:
            print(f"空输入错误处理: {e}")
        
        # 测试超长输入
        try:
            long_input = "This is a very long input. " * 100
            output = model(long_input, max_length=50)
            print("超长输入处理正常")
        except Exception as e:
            print(f"超长输入错误处理: {e}")
        
        print("✓ 错误处理测试完成！")
        return True
        
    except Exception as e:
        print(f"✗ 错误处理测试失败: {e}")
        return False

def main():
    """
    主测试函数
    """
    print("ProbNet 文本生成功能测试")
    print("=" * 40)
    
    # 运行所有测试
    tests = [
        ("基础功能", test_basic_functionality),
        ("中文输入", test_chinese_input),
        ("错误处理", test_error_handling)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name}测试 {'='*20}")
        if test_func():
            passed += 1
    
    # 总结
    print("\n" + "="*50)
    print(f"测试总结: {passed}/{total} 项测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！ProbNet可以像llama一样使用了！")
    else:
        print("⚠️  部分测试失败，请检查实现")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)