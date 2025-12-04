#!/usr/bin/env python3
"""
数据集转Parquet工具

将数据集按照split划分，转换为parquet格式。
如果一个split的数据过多，会自动分多个文件管理。

使用方法:
    python tools/dataset_to_parquet.py --dataset_name <dataset_name> --root_dir <root_dir> --output_dir <output_dir> [options]
"""

import argparse
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from src.dataset.dataset import get_train_dataset, get_valid_dataset, get_test_dataset
from src.dataset.custom_dataset import CustomDataset

# 默认日志配置：不干扰训练/推理输出
# 只有在明确启用DEBUG模式时才输出调试信息
_DEBUG_MODE = False
_VERBOSE_MODE = False

def setup_logger(debug: bool = False, verbose: bool = False):
    """
    设置日志配置
    
    Args:
        debug: 是否启用DEBUG模式，启用后会输出所有调试信息
        verbose: 是否启用详细模式，启用后会输出INFO级别信息
    """
    global _DEBUG_MODE, _VERBOSE_MODE
    _DEBUG_MODE = debug
    _VERBOSE_MODE = verbose
    
    # 获取当前模块的logger，避免影响全局日志配置
    logger = logging.getLogger(__name__)
    
    # 清除已有的handlers，避免重复添加
    logger.handlers.clear()
    logger.propagate = False  # 不向上传播，避免影响其他模块
    
    # 根据模式设置日志级别
    if debug:
        logger.setLevel(logging.DEBUG)
        console_level = logging.DEBUG
    elif verbose:
        logger.setLevel(logging.INFO)
        console_level = logging.INFO
    else:
        # 默认模式：只输出WARNING及以上级别，不干扰训练/推理输出
        logger.setLevel(logging.WARNING)
        console_level = logging.WARNING
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

# 初始化logger（默认不输出调试信息）
logger = setup_logger(debug=False, verbose=False)


def serialize_tensor_or_array(data: Union[Any, np.ndarray]) -> bytes:
    """将tensor或numpy数组序列化为字节"""
    if TORCH_AVAILABLE and isinstance(data, torch.Tensor):
        data = data.cpu().detach().numpy()
    if isinstance(data, np.ndarray):
        # 使用pickle序列化numpy数组
        return pickle.dumps(data)
    else:
        # 其他类型也尝试pickle
        return pickle.dumps(data)


def deserialize_tensor_or_array(data: bytes) -> np.ndarray:
    """从字节反序列化tensor或numpy数组"""
    return pickle.loads(data)


def convert_sample_to_dict(sample: Any, index: int) -> Dict[str, Any]:
    """将数据集样本转换为字典格式，便于存储到parquet"""
    result = {"index": index}
    
    if isinstance(sample, dict):
        # 处理字典格式的样本
        for key, value in sample.items():
            is_tensor = isinstance(value, torch.Tensor)
            is_array = isinstance(value, np.ndarray)
            if is_tensor or is_array:
                # 序列化tensor/array为字节
                result[f"{key}_data"] = serialize_tensor_or_array(value)
                result[f"{key}_shape"] = list(value.shape) if hasattr(value, 'shape') else None
                result[f"{key}_dtype"] = str(value.dtype) if hasattr(value, 'dtype') else None
            elif isinstance(value, (int, float, str, bool, type(None))):
                # 直接存储简单类型
                result[key] = value
            elif isinstance(value, (list, tuple)):
                # 处理列表和元组
                if len(value) > 0 and isinstance(value[0], (int, float, str, bool)):
                    result[key] = value
                else:
                    # 复杂类型序列化
                    result[f"{key}_data"] = pickle.dumps(value)
            elif isinstance(value, dict):
                # 嵌套字典，展平存储
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (int, float, str, bool, type(None))):
                        result[f"{key}_{nested_key}"] = nested_value
                    else:
                        result[f"{key}_{nested_key}_data"] = pickle.dumps(nested_value)
            else:
                # 其他类型序列化
                result[f"{key}_data"] = pickle.dumps(value)
    elif isinstance(sample, (tuple, list)):
        # 处理元组/列表格式的样本
        for i, item in enumerate(sample):
            is_tensor = isinstance(item, torch.Tensor)
            is_array = isinstance(item, np.ndarray)
            if is_tensor or is_array:
                result[f"item_{i}_data"] = serialize_tensor_or_array(item)
                result[f"item_{i}_shape"] = list(item.shape) if hasattr(item, 'shape') else None
                result[f"item_{i}_dtype"] = str(item.dtype) if hasattr(item, 'dtype') else None
            else:
                result[f"item_{i}"] = item
    else:
        # 其他类型，直接序列化
        result["data"] = pickle.dumps(sample)
    
    return result


def dataset_to_parquet(
    dataset: CustomDataset,
    split: str,
    output_dir: Path,
    max_samples_per_file: int = 10000,
    **kwargs
) -> List[Path]:
    """
    将数据集转换为parquet文件
    
    Args:
        dataset: 数据集对象
        split: 数据集划分名称 (train/valid/test)
        output_dir: 输出目录
        max_samples_per_file: 每个parquet文件的最大样本数
        **kwargs: 其他参数
    
    Returns:
        生成的parquet文件路径列表
    """
    if dataset is None:
        logger.warning(f"数据集 {split} 为空，跳过")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = dataset.name() if hasattr(dataset, 'name') and callable(dataset.name) else "unknown"
    dataset_name = dataset_name() if callable(dataset_name) else str(dataset_name)
    
    total_samples = len(dataset)
    logger.info(f"开始转换数据集 {dataset_name} ({split}): {total_samples} 个样本")
    
    # 计算需要多少个文件
    num_files = (total_samples + max_samples_per_file - 1) // max_samples_per_file
    
    logger.debug(f"数据集 {dataset_name} ({split}) 将分为 {num_files} 个文件，每文件最多 {max_samples_per_file} 个样本")
    
    output_files = []
    
    for file_idx in range(num_files):
        start_idx = file_idx * max_samples_per_file
        end_idx = min((file_idx + 1) * max_samples_per_file, total_samples)
        
        logger.info(f"处理文件 {file_idx + 1}/{num_files}: 样本 {start_idx} 到 {end_idx - 1}")
        logger.debug(f"文件 {file_idx + 1} 索引范围: [{start_idx}, {end_idx})")
        
        # 收集数据
        rows = []
        # 只在verbose或debug模式下显示进度条
        disable_tqdm = not (_VERBOSE_MODE or _DEBUG_MODE)
        for idx in tqdm(range(start_idx, end_idx), desc=f"处理 {split} 样本", disable=disable_tqdm):
            try:
                sample = dataset[idx]
                row = convert_sample_to_dict(sample, idx)
                rows.append(row)
                logger.debug(f"成功处理样本 {idx}: shape={row.get('image_shape', 'N/A')}")
            except Exception as e:
                logger.error(f"处理样本 {idx} 时出错: {e}")
                if _DEBUG_MODE:
                    import traceback
                    logger.debug(traceback.format_exc())
                continue
        
        if not rows:
            logger.warning(f"文件 {file_idx + 1} 没有有效数据，跳过")
            continue
        
        # 转换为DataFrame
        logger.debug(f"将 {len(rows)} 行数据转换为DataFrame")
        df = pd.DataFrame(rows)
        logger.debug(f"DataFrame形状: {df.shape}, 列: {list(df.columns)}")
        
        # 生成文件名
        if num_files == 1:
            filename = f"{dataset_name}_{split}.parquet"
        else:
            filename = f"{dataset_name}_{split}_part{file_idx + 1:04d}_of_{num_files:04d}.parquet"
        
        output_path = output_dir / filename
        
        # 保存为parquet
        try:
            logger.debug(f"开始保存parquet文件: {output_path}")
            # 尝试使用pyarrow，如果失败则使用fastparquet
            try:
                df.to_parquet(output_path, engine='pyarrow', compression='snappy', index=False)
                logger.debug("使用pyarrow引擎保存成功")
            except (ImportError, ValueError) as e:
                logger.warning(f"pyarrow不可用，尝试使用fastparquet: {e}")
                try:
                    import fastparquet
                    fastparquet.write(output_path, df, compression='snappy')
                    logger.debug("使用fastparquet引擎保存成功")
                except ImportError:
                    # 如果fastparquet也不可用，尝试不指定engine
                    df.to_parquet(output_path, compression='snappy', index=False)
                    logger.debug("使用默认引擎保存成功")
            
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"已保存: {output_path} ({file_size_mb:.2f} MB, {len(rows)} 样本)")
            logger.debug(f"文件详细信息: 路径={output_path}, 大小={file_size_mb:.2f} MB, 样本数={len(rows)}")
            output_files.append(output_path)
        except Exception as e:
            logger.error(f"保存文件 {output_path} 时出错: {e}")
            if _DEBUG_MODE:
                import traceback
                logger.debug(traceback.format_exc())
    
    logger.info(f"完成转换 {split}: 共生成 {len(output_files)} 个文件")
    return output_files


def convert_dataset_splits_to_parquet(
    dataset_name: str,
    root_dir: Union[str, Path],
    output_dir: Union[str, Path],
    splits: Optional[List[str]] = None,
    max_samples_per_file: int = 10000,
    **dataset_kwargs
) -> Dict[str, List[Path]]:
    """
    将数据集的所有split转换为parquet文件
    
    Args:
        dataset_name: 数据集名称
        root_dir: 数据集根目录
        output_dir: 输出目录
        splits: 要转换的split列表，如果为None则转换所有可用的split
        max_samples_per_file: 每个parquet文件的最大样本数
        **dataset_kwargs: 传递给数据集构造函数的其他参数
    
    Returns:
        字典，键为split名称，值为生成的parquet文件路径列表
    """
    root_dir = Path(root_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if splits is None:
        splits = ["train", "valid", "test"]
    
    result = {}
    
    for split in splits:
        logger.info(f"\n{'='*60}")
        logger.info(f"处理 split: {split}")
        logger.info(f"{'='*60}")
        
        try:
            # 获取对应的数据集
            if split == "train":
                dataset = get_train_dataset(dataset_name, root_dir, **dataset_kwargs)
            elif split in ["valid", "val"]:
                dataset = get_valid_dataset(dataset_name, root_dir, **dataset_kwargs)
            elif split == "test":
                dataset = get_test_dataset(dataset_name, root_dir, **dataset_kwargs)
            else:
                logger.warning(f"未知的split: {split}，跳过")
                continue
            
            if dataset is None:
                logger.warning(f"无法获取数据集 {dataset_name} 的 {split} split，跳过")
                continue
            
            # 转换为parquet
            files = dataset_to_parquet(
                dataset=dataset,
                split=split,
                output_dir=output_dir,
                max_samples_per_file=max_samples_per_file,
                **dataset_kwargs
            )
            
            result[split] = files
            
        except Exception as e:
            logger.error(f"处理 {split} split 时出错: {e}", exc_info=True)
            result[split] = []
    
    return result


def main():
    parser = argparse.ArgumentParser(
        description="将数据集转换为parquet格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 转换MNIST数据集的所有split
    python tools/dataset_to_parquet.py --dataset_name mnist --root_dir data/mnist --output_dir data/mnist_parquet
    
    # 只转换train和test split，每个文件最多5000个样本
    python tools/dataset_to_parquet.py --dataset_name cifar10 --root_dir data/cifar10 --output_dir data/cifar10_parquet --splits train test --max_samples_per_file 5000
        """
    )
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="数据集名称 (如: mnist, cifar10, coco等)"
    )
    
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="数据集根目录路径"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="输出parquet文件的目录"
    )
    
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=None,
        choices=["train", "valid", "val", "test"],
        help="要转换的split列表 (默认: 所有可用的split)"
    )
    
    parser.add_argument(
        "--max_samples_per_file",
        type=int,
        default=10000,
        help="每个parquet文件的最大样本数 (默认: 10000)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示详细日志 (INFO级别)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用DEBUG模式，显示所有调试信息（可能干扰训练/推理输出）"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="静默模式，只显示错误和警告（默认模式）"
    )
    
    args, unknown = parser.parse_known_args()
    
    # 设置日志级别
    # 优先级: debug > verbose > quiet (默认)
    if args.debug:
        logger = setup_logger(debug=True, verbose=True)
    elif args.verbose:
        logger = setup_logger(debug=False, verbose=True)
    elif args.quiet:
        logger = setup_logger(debug=False, verbose=False)
    else:
        # 默认：不干扰训练/推理输出
        logger = setup_logger(debug=False, verbose=False)
    
    # 解析额外的数据集参数 (格式: key=value)
    dataset_kwargs = {}
    for arg in unknown:
        if '=' in arg:
            key, value = arg.split('=', 1)
            # 尝试转换类型
            try:
                if value.lower() == 'true':
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.replace('.', '', 1).isdigit():
                    value = float(value) if '.' in value else int(value)
            except:
                pass
            dataset_kwargs[key] = value
    
    logger.info(f"数据集名称: {args.dataset_name}")
    logger.info(f"根目录: {args.root_dir}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"Splits: {args.splits or '所有'}")
    logger.info(f"每文件最大样本数: {args.max_samples_per_file}")
    if dataset_kwargs:
        logger.info(f"数据集参数: {dataset_kwargs}")
    
    # 执行转换
    result = convert_dataset_splits_to_parquet(
        dataset_name=args.dataset_name,
        root_dir=args.root_dir,
        output_dir=args.output_dir,
        splits=args.splits,
        max_samples_per_file=args.max_samples_per_file,
        **dataset_kwargs
    )
    
    # 打印总结
    logger.info(f"\n{'='*60}")
    logger.info("转换完成总结")
    logger.info(f"{'='*60}")
    for split, files in result.items():
        logger.info(f"{split}: {len(files)} 个文件")
        for f in files:
            logger.info(f"  - {f.name}")


if __name__ == "__main__":
    main()


