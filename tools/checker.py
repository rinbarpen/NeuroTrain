from argparse import ArgumentParser
import torch
import psutil
import os
import platform

def main():
    args = parse_args()
    if args.sys:
        check_system_info()
    if args.cuda:
        check_cuda()
    if args.mem:
        check_memory()
    if args.cpu:
        check_cpu()

def check_system_info():
    """检查系统基本信息"""
    print("\n=== 系统信息 ===")
    print(f"操作系统: {platform.system()} {platform.version()}")
    print(f"Python版本: {platform.python_version()}")
    print(f"PyTorch版本: {torch.__version__}")

def check_cuda():
    """检查CUDA设备信息"""
    print("\n=== CUDA信息 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"CUDA设备{i}名称: {torch.cuda.get_device_name(i)}")
        print(f"CUDA后端版本: {torch.backends.cudnn.version()}")
        # 获取CUDA显存信息
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"总显存: {total_mem / 1024**3:.2f}GB")
        print(f"可用显存: {free_mem / 1024**3:.2f}GB")
        print(f"已用显存: {(total_mem - free_mem) / 1024**3:.2f}GB")
    else:
        print("CUDA不可用")

def check_memory():
    """检查系统内存使用情况"""
    print("\n=== 内存信息 ===")
    mem = psutil.virtual_memory()
    print(f"总内存: {mem.total / 1024**3:.2f}GB")
    print(f"可用内存: {mem.available / 1024**3:.2f}GB")
    print(f"内存使用率: {mem.percent}%")

def check_cpu():
    """检查CPU信息"""
    print("\n=== CPU信息 ===")
    print(f"CPU核心数: {os.cpu_count()}")
    print(f"CPU使用率: {psutil.cpu_percent()}%")
    print(f"CPU频率: {psutil.cpu_freq().current:.2f}MHz")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True, help='check cuda')
    parser.add_argument('--mem', action='store_true', default=True, help='check memory')
    parser.add_argument('--cpu', action='store_true', default=True, help='check cpu')
    parser.add_argument('--sys', action='store_true', default=True, help='check system info')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()
