import torch
from pathlib import Path
from typing import Any, Dict
from PIL import Image
import logging
import fastparquet

from .custom_dataset import LazyCustomDataset

logger = logging.getLogger(__name__)

class ParquetLazyDataset(LazyCustomDataset):
    """基于 Parquet 索引的延迟加载数据集
    
    适用于元数据非常大（例如有数千万个样本路径）的情况。
    索引文件应包含图像路径和标签。
    """
    
    def __init__(
        self, 
        root_dir: Path, 
        split: str, 
        index_file: str,
        image_column: str = "image_path",
        label_column: str = "label",
        transform=None,
        **kwargs
    ):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集划分
            index_file: Parquet 索引文件路径（相对于 root_dir）
            image_column: 包含图像路径的列名
            label_column: 包含标签的列名
            transform: 图像变换
        """
        self.index_file_rel = index_file
        self.image_column = image_column
        self.label_column = label_column
        self.transform = transform
        
        super().__init__(root_dir, split, index_path=root_dir / index_file, **kwargs)

    def _setup_index(self):
        """初始化 Parquet 句柄并获取数据集大小"""
        if self.index_path is None:
            raise ValueError("未指定索引文件路径 (index_path)")
            
        if not self.index_path.exists():
            raise FileNotFoundError(f"索引文件不存在: {self.index_path}")
            
        # 仅打开文件句柄，不加载数据到内存
        self.parquet_file = fastparquet.ParquetFile(str(self.index_path))
        
        # 获取总行数
        if self.n == 0:
            self.n = self.parquet_file.count
            
        logger.info(f"成功加载 Parquet 索引 (fastparquet): {self.index_path}, 总样本数: {self.n}")
        self._index_ready = True

    def _get_metadata(self, index: int) -> Dict[str, Any]:
        """从 Parquet 文件中读取特定行"""
        # 寻找索引所在的 row group
        row_group_idx = -1
        local_index = -1
        cumulative_rows = 0
        for i, rg in enumerate(self.parquet_file.row_groups):
            num_rows = rg.num_rows
            if index < cumulative_rows + num_rows:
                row_group_idx = i
                local_index = index - cumulative_rows
                break
            cumulative_rows += num_rows
            
        if row_group_idx == -1:
            raise IndexError(f"Index {index} not found in parquet row groups")
        
        # 使用 fastparquet 读取该 row group 的该行
        # to_pandas() 会返回该 row group 的完整 DataFrame（仅包含请求的列）
        df = self.parquet_file.row_groups[row_group_idx].to_pandas(columns=[self.image_column, self.label_column])
        row = df.iloc[local_index]
        
        return {
            "image_path": self.root_dir / row[self.image_column],
            "label": row[self.label_column]
        }

    def load_sample(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """根据元数据加载图像"""
        img_path = metadata["image_path"]
        label = metadata["label"]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            logger.error(f"加载图像失败 {img_path}: {e}")
            image = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            image = self.transform(image)
        else:
            import numpy as np
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            
        return {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "path": str(img_path)
        }

    @staticmethod
    def name() -> str:
        return "ParquetLazyDataset"

    @staticmethod
    def get_train_dataset(root_dir: Path, **kwargs) -> "ParquetLazyDataset":
        return ParquetLazyDataset(root_dir, "train", **kwargs)

    @staticmethod
    def get_valid_dataset(root_dir: Path, **kwargs) -> "ParquetLazyDataset":
        return ParquetLazyDataset(root_dir, "valid", **kwargs)

    @staticmethod
    def get_test_dataset(root_dir: Path, **kwargs) -> "ParquetLazyDataset":
        return ParquetLazyDataset(root_dir, "test", **kwargs)

