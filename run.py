#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

def run_command(command, unknown_args=None):
    """运行命令并转发未知参数"""
    if unknown_args:
        command.extend(unknown_args)
    
    print(f"🚀 Running: {' '.join(command)}")
    try:
        # 使用 sys.executable 确保使用相同的 Python 解释器
        result = subprocess.run([sys.executable] + command, check=True)
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed with return code {e.returncode}")
        return e.returncode
    except KeyboardInterrupt:
        print("\n👋 Interrupted by user")
        return 130

def main():
    parser = argparse.ArgumentParser(
        description="NeuroTrain 统一入口 - 简化训练、推理和预处理流程",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python run.py train -c configs/model.yaml
  python run.py train --deepspeed -c configs/model.yaml
  python run.py predict -i data/input.jpg -c configs/model.yaml
  python run.py preprocess cache list
  python run.py clean --log
        """
    )
    
    subparsers = parser.add_subparsers(dest="mode", help="运行模式")

    # --- Train/Test/Predict ---
    train_parser = subparsers.add_parser("train", help="启动训练阶段")
    train_parser.add_argument("--deepspeed", action="store_true", help="使用 DeepSpeed 启动训练")

    test_parser = subparsers.add_parser("test", help="启动测试阶段")
    predict_parser = subparsers.add_parser("predict", help="启动推理/预测阶段")

    # --- Preprocess ---
    preprocess_parser = subparsers.add_parser("preprocess", help="数据预处理相关工具")
    preprocess_subparsers = preprocess_parser.add_subparsers(dest="tool", help="预处理工具")
    
    # cache 子命令
    cache_parser = preprocess_subparsers.add_parser("cache", help="数据集缓存管理工具")
    # parquet 子命令
    parquet_parser = preprocess_subparsers.add_parser("parquet", help="转换为 Parquet 索引工具")

    # --- Monitor ---
    subparsers.add_parser("monitor", help="启动 Web 监控面板")

    # --- Clean ---
    subparsers.add_parser("clean", help="清理日志、输出或运行目录")

    # --- Export ---
    subparsers.add_parser("export", help="导出模型 (如 ONNX)")

    # 解析已知参数，将剩余参数转发给底层脚本
    args, unknown = parser.parse_known_args()

    if not args.mode:
        parser.print_help()
        sys.exit(0)

    # 逻辑分发
    if args.mode == "train":
        if args.deepspeed:
            cmd = ["main_deepspeed.py"]
        else:
            cmd = ["main.py", "--train"]
        sys.exit(run_command(cmd, unknown))

    elif args.mode == "test":
        cmd = ["main.py", "--test"]
        sys.exit(run_command(cmd, unknown))

    elif args.mode == "predict":
        cmd = ["main.py", "--predict"]
        sys.exit(run_command(cmd, unknown))

    elif args.mode == "preprocess":
        if args.tool == "cache":
            cmd = ["tools/dataset_cache_tool.py"]
            sys.exit(run_command(cmd, unknown))
        elif args.tool == "parquet":
            cmd = ["tools/to_parquet.py"]
            sys.exit(run_command(cmd, unknown))
        else:
            preprocess_parser.print_help()
            sys.exit(1)

    elif args.mode == "monitor":
        cmd = ["start_web_monitor.py"]
        sys.exit(run_command(cmd, unknown))

    elif args.mode == "clean":
        cmd = ["tools/cleanup.py"]
        sys.exit(run_command(cmd, unknown))

    elif args.mode == "export":
        cmd = ["tools/onnx_export.py"]
        sys.exit(run_command(cmd, unknown))

    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()

