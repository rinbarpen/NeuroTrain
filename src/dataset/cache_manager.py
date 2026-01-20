"""
数据集缓存管理器

提供数据集的缓存功能，支持多种缓存格式和策略。
缓存文件存储在 cache/{dataset_name}/ 目录下。
"""

import pickle
import json
import hashlib
import logging
from pathlib import Path
from typing import Any, Optional, Dict, Callable, List
import torch
from datetime import datetime
import shutil

logger = logging.getLogger(__name__)


class DatasetCacheManager:
    """数据集缓存管理器
    
    功能：
    - 支持多种缓存格式（pickle, torch, json）
    - 自动生成缓存键值
    - 缓存版本管理
    - 缓存有效性检查
    """
    
    def __init__(
        self,
        dataset_name: str,
        cache_root: Optional[Path] = None,
        version: str = "v1",
        enable_cache: bool = True
    ):
        """
        初始化缓存管理器
        
        Args:
            dataset_name: 数据集名称
            cache_root: 缓存根目录，默认为 ./cache
            version: 缓存版本号
            enable_cache: 是否启用缓存
        """
        self.dataset_name = dataset_name
        self.version = version
        self.enable_cache = enable_cache
        
        # 设置缓存目录
        if cache_root is None:
            cache_root = Path.cwd() / "cache"
        self.cache_root = Path(cache_root)
        
        # 数据集特定的缓存目录: cache/datasets/{dataset_name}/{version}
        self.cache_dir = self.cache_root / "datasets" / dataset_name / version
        
        if self.enable_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"缓存目录: {self.cache_dir}")
    
    def _generate_cache_key(self, split: str, config: Optional[Dict] = None) -> str:
        """生成缓存键值
        
        Args:
            split: 数据集划分（train/valid/test）
            config: 数据集配置参数
        
        Returns:
            缓存键值（哈希字符串）
        """
        key_parts = [self.dataset_name, split, self.version]
        
        if config:
            # 对配置进行排序并序列化，确保相同配置生成相同的键
            config_str = json.dumps(config, sort_keys=True)
            key_parts.append(config_str)
        
        # 生成MD5哈希
        key_string = "_".join(str(p) for p in key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(
        self,
        split: str,
        config: Optional[Dict] = None,
        format: str = "pkl"
    ) -> Path:
        """获取缓存文件路径
        
        Args:
            split: 数据集划分
            config: 数据集配置
            format: 缓存格式（pkl/pt/json）
        
        Returns:
            缓存文件路径
        """
        cache_key = self._generate_cache_key(split, config)
        filename = f"{split}_{cache_key}.{format}"
        return self.cache_dir / filename
    
    def _get_metadata_path(self, cache_path: Path) -> Path:
        """获取元数据文件路径"""
        return cache_path.with_suffix('.meta.json')
    
    def exists(
        self,
        split: str,
        config: Optional[Dict] = None,
        format: str = "pkl"
    ) -> bool:
        """检查缓存是否存在
        
        Args:
            split: 数据集划分
            config: 数据集配置
            format: 缓存格式
        
        Returns:
            缓存是否存在
        """
        if not self.enable_cache:
            return False
        
        cache_path = self._get_cache_path(split, config, format)
        return cache_path.exists()
    
    def save(
        self,
        data: Any,
        split: str,
        config: Optional[Dict] = None,
        format: str = "pkl",
        metadata: Optional[Dict] = None
    ) -> bool:
        """保存数据到缓存
        
        Args:
            data: 要缓存的数据
            split: 数据集划分
            config: 数据集配置
            format: 缓存格式（pkl/pt/json）
            metadata: 额外的元数据信息
        
        Returns:
            是否保存成功
        """
        if not self.enable_cache:
            logger.debug("缓存已禁用，跳过保存")
            return False
        
        try:
            cache_path = self._get_cache_path(split, config, format)
            
            # 保存数据
            if format == "pkl":
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif format == "pt":
                torch.save(data, cache_path)
            elif format == "json":
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"不支持的缓存格式: {format}")
            
            # 保存元数据
            meta = {
                "dataset_name": self.dataset_name,
                "split": split,
                "version": self.version,
                "config": config,
                "format": format,
                "created_at": datetime.now().isoformat(),
                "file_size": cache_path.stat().st_size
            }
            if metadata:
                meta.update(metadata)
            
            meta_path = self._get_metadata_path(cache_path)
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            
            logger.info(f"已保存缓存: {cache_path}")
            logger.info(f"缓存大小: {meta['file_size'] / 1024 / 1024:.2f} MB")
            return True
            
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
            return False

    def save_chunked(
        self,
        data_list: List[Any],
        split: str,
        chunk_size: int,
        config: Optional[Dict] = None,
        format: str = "pkl",
        metadata: Optional[Dict] = None
    ) -> bool:
        """分片保存大数据集缓存
        
        Args:
            data_list: 数据列表
            split: 数据集划分
            chunk_size: 每个分片的样本数量
            config: 数据集配置
            format: 缓存格式
            metadata: 额外的元数据
        """
        if not self.enable_cache:
            return False
            
        num_chunks = (len(data_list) + chunk_size - 1) // chunk_size
        logger.info(f"开始分片保存缓存: 共 {len(data_list)} 条数据, 分为 {num_chunks} 个分片")
        
        success_count = 0
        base_config = config or {}
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(data_list))
            chunk_data = data_list[start_idx:end_idx]
            
            chunk_config = base_config.copy()
            chunk_config["_chunk_index"] = i
            chunk_config["_chunk_range"] = (start_idx, end_idx)
            
            chunk_meta = metadata.copy() if metadata else {}
            chunk_meta.update({
                "chunk_index": i,
                "chunk_total": num_chunks,
                "range": (start_idx, end_idx)
            })
            
            if self.save(chunk_data, split, chunk_config, format, metadata=chunk_meta):
                success_count += 1
                
        # 保存主元数据文件，记录分片信息
        master_config = base_config.copy()
        master_config["_is_master"] = True
        master_meta = metadata.copy() if metadata else {}
        master_meta.update({
            "is_chunked": True,
            "num_chunks": num_chunks,
            "chunk_size": chunk_size,
            "total_samples": len(data_list)
        })
        self.save({"chunk_info": master_meta}, split, master_config, "json", metadata=master_meta)
        
        return success_count == num_chunks

    def load_chunked(
        self,
        split: str,
        config: Optional[Dict] = None,
        format: str = "pkl"
    ) -> Optional[List[Any]]:
        """从分片缓存加载数据"""
        if not self.enable_cache:
            return None
            
        base_config = config or {}
        master_config = base_config.copy()
        master_config["_is_master"] = True
        
        master_data = self.load(split, master_config, "json")
        if not master_data or "chunk_info" not in master_data:
            logger.debug("未找到分片缓存主文件")
            return None
            
        chunk_info = master_data["chunk_info"]
        num_chunks = chunk_info["num_chunks"]
        
        all_data = []
        for i in range(num_chunks):
            chunk_config = base_config.copy()
            chunk_config["_chunk_index"] = i
            # 我们不严格检查 _chunk_range，因为哈希键主要由 _chunk_index 决定
            
            chunk_data = self.load(split, chunk_config, format)
            if chunk_data is None:
                logger.error(f"加载缓存分片 {i} 失败")
                return None
            all_data.extend(chunk_data)
            
        return all_data
    
    def load(
        self,
        split: str,
        config: Optional[Dict] = None,
        format: str = "pkl",
        check_validity: bool = True
    ) -> Optional[Any]:
        """从缓存加载数据
        
        Args:
            split: 数据集划分
            config: 数据集配置
            format: 缓存格式
            check_validity: 是否检查缓存有效性
        
        Returns:
            缓存的数据，如果不存在或无效则返回None
        """
        if not self.enable_cache:
            logger.debug("缓存已禁用，跳过加载")
            return None
        
        try:
            cache_path = self._get_cache_path(split, config, format)
            
            if not cache_path.exists():
                logger.debug(f"缓存不存在: {cache_path}")
                return None
            
            # 检查有效性
            if check_validity:
                if not self._check_validity(cache_path):
                    logger.warning(f"缓存无效，已删除: {cache_path}")
                    return None
            
            # 加载数据
            if format == "pkl":
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
            elif format == "pt":
                data = torch.load(cache_path)
            elif format == "json":
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                raise ValueError(f"不支持的缓存格式: {format}")
            
            logger.info(f"已加载缓存: {cache_path}")
            return data
            
        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return None
    
    def _check_validity(self, cache_path: Path) -> bool:
        """检查缓存有效性
        
        Args:
            cache_path: 缓存文件路径
        
        Returns:
            缓存是否有效
        """
        try:
            # 检查文件是否可读
            if not cache_path.exists():
                return False
            
            # 检查文件大小
            if cache_path.stat().st_size == 0:
                logger.warning(f"缓存文件为空: {cache_path}")
                return False
            
            # 检查元数据
            meta_path = self._get_metadata_path(cache_path)
            if not meta_path.exists():
                logger.warning(f"元数据文件不存在: {meta_path}")
                return True  # 没有元数据也认为有效，只是缺少元信息
            
            with open(meta_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            # 检查版本
            if metadata.get('version') != self.version:
                logger.warning(f"缓存版本不匹配: 期望 {self.version}, 实际 {metadata.get('version')}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"检查缓存有效性失败: {e}")
            return False
    
    def clear(
        self,
        split: Optional[str] = None,
        config: Optional[Dict] = None,
        format: Optional[str] = None
    ) -> int:
        """清除缓存
        
        Args:
            split: 数据集划分，None表示清除所有
            config: 数据集配置，None表示清除所有
            format: 缓存格式，None表示清除所有格式
        
        Returns:
            清除的文件数量
        """
        if not self.enable_cache:
            return 0
        
        count = 0
        
        try:
            if split is None:
                # 清除整个数据集的缓存目录
                if self.cache_dir.exists():
                    shutil.rmtree(self.cache_dir)
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    logger.info(f"已清除所有缓存: {self.cache_dir}")
                    return 1
            else:
                # 清除特定的缓存文件
                if format is None:
                    formats = ["pkl", "pt", "json"]
                else:
                    formats = [format]
                
                for fmt in formats:
                    cache_path = self._get_cache_path(split, config, fmt)
                    if cache_path.exists():
                        cache_path.unlink()
                        count += 1
                        logger.info(f"已删除缓存: {cache_path}")
                    
                    # 删除元数据
                    meta_path = self._get_metadata_path(cache_path)
                    if meta_path.exists():
                        meta_path.unlink()
                        count += 1
            
            if count > 0:
                logger.info(f"共清除 {count} 个缓存文件")
            
            return count
            
        except Exception as e:
            logger.error(f"清除缓存失败: {e}")
            return 0
    
    def get_cache_info(self, split: Optional[str] = None) -> Dict[str, Any]:
        """获取缓存信息
        
        Args:
            split: 数据集划分，None表示获取所有信息
        
        Returns:
            缓存信息字典
        """
        info = {
            "dataset_name": self.dataset_name,
            "version": self.version,
            "cache_dir": str(self.cache_dir),
            "enabled": self.enable_cache,
            "files": []
        }
        
        if not self.enable_cache or not self.cache_dir.exists():
            return info
        
        try:
            total_size = 0
            for cache_file in self.cache_dir.glob("*.*"):
                if cache_file.suffix == ".json" and cache_file.stem.endswith('.meta'):
                    continue  # 跳过元数据文件
                
                file_info = {
                    "filename": cache_file.name,
                    "size": cache_file.stat().st_size,
                    "modified": datetime.fromtimestamp(
                        cache_file.stat().st_mtime
                    ).isoformat()
                }
                
                # 读取元数据
                meta_path = self._get_metadata_path(cache_file)
                if meta_path.exists():
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    file_info["metadata"] = metadata
                
                # 如果指定了split，只返回匹配的文件
                if split is None or (
                    file_info.get("metadata", {}).get("split") == split
                ):
                    info["files"].append(file_info)
                    total_size += file_info["size"]
            
            info["total_size"] = total_size
            info["total_files"] = len(info["files"])
            info["total_size_mb"] = total_size / 1024 / 1024
            
        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")
        
        return info
    
    @staticmethod
    def cache_dataset(
        dataset_cls: type,
        cache_manager: "DatasetCacheManager",
        split: str,
        root_dir: Path,
        processor_fn: Optional[Callable] = None,
        format: str = "pkl",
        force_rebuild: bool = False,
        **kwargs
    ) -> Optional[Any]:
        """通用的数据集缓存包装器
        
        Args:
            dataset_cls: 数据集类
            cache_manager: 缓存管理器实例
            split: 数据集划分
            root_dir: 数据集根目录
            processor_fn: 数据处理函数（可选）
            format: 缓存格式
            force_rebuild: 是否强制重建缓存
            **kwargs: 传递给数据集构造函数的参数
        
        Returns:
            数据集实例或缓存的数据
        """
        # 生成配置用于缓存键
        cache_config = {
            "root_dir": str(root_dir),
            **kwargs
        }
        
        # 尝试加载缓存
        if not force_rebuild and cache_manager.exists(split, cache_config, format):
            logger.info(f"尝试加载缓存数据集: {split}")
            cached_data = cache_manager.load(split, cache_config, format)
            if cached_data is not None:
                return cached_data
        
        # 创建数据集
        logger.info(f"创建新的数据集: {split}")
        dataset = dataset_cls(root_dir, split, **kwargs)
        
        # 应用处理函数
        if processor_fn is not None:
            logger.info("应用数据处理函数")
            data_to_cache = processor_fn(dataset)
        else:
            data_to_cache = dataset
        
        # 保存缓存
        metadata = {
            "num_samples": len(dataset),
            "dataset_class": dataset_cls.__name__
        }
        cache_manager.save(
            data_to_cache,
            split,
            cache_config,
            format,
            metadata=metadata
        )
        
        return dataset

