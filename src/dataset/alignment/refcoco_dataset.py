"""
RefCOCO数据集适配器，用于EMOE Region-Text对齐任务
将RefCOCO数据集适配为支持区域裁剪和对齐的格式
"""
import torch
from pathlib import Path
from typing import List, Dict, Literal, Union
from PIL import Image
import numpy as np
import logging

from ..refcoco_dataset import BaseRefCOCODataset
from ..custom_dataset import CustomDataset

logger = logging.getLogger(__name__)


class RefCOCOAlignmentDataset(CustomDataset):
    """
    RefCOCO对齐数据集适配器
    
    将RefCOCO数据集转换为适合EMOE对齐任务的格式：
    - 从图像中根据bbox裁剪区域
    - 返回区域图像和对应文本
    """
    
    def __init__(
        self,
        root_dir: Union[str, Path],
        split: Literal['train', 'val', 'test', 'valid'],
        dataset_name: str = 'refcoco',
        image_dir: str = 'images',
        region_crop_size: int = 224,
        region_padding: float = 0.1,
        transform=None,
        **kwargs
        ):
        """
        Args:
            root_dir: 数据集根目录
            split: 数据集划分
            dataset_name: 数据集名称 (refcoco, refcoco+, refcocog)
            image_dir: 图像目录
            region_crop_size: 区域裁剪尺寸
            region_padding: 区域边界padding比例
            transform: 图像变换
        """
        # 先调用父类初始化
        super(RefCOCOAlignmentDataset, self).__init__(root_dir, split, **kwargs)
        
        # 使用BaseRefCOCODataset加载数据
        self.base_dataset = BaseRefCOCODataset(
            root_dir=root_dir,
            split=split,
            dataset_name=dataset_name,
            image_dir=image_dir,
            transform=None,  # 我们在这里处理变换
            **kwargs
        )
        
        self.region_crop_size = region_crop_size
        self.region_padding = region_padding
        self.transform = transform
        
        # 设置数据集大小
        self.n = len(self.base_dataset)
        
        logger.info(f"RefCOCOAlignmentDataset initialized: {self.n} samples")
    
    def __len__(self):
        return self.n
    
    def _crop_region(self, image: Image.Image, bbox: torch.Tensor, padding: float = 0.1) -> Image.Image:
        """
        根据bbox裁剪图像区域
        
        Args:
            image: PIL图像
            bbox: [x1, y1, x2, y2] 或 [x, y, w, h]
        
        Returns:
            裁剪后的区域图像
        """
        img_width, img_height = image.size
        
        # 处理bbox格式
        if len(bbox) == 4:
            if bbox[2] > img_width or bbox[3] > img_height:
                # 可能是COCO格式 [x, y, w, h]
                x, y, w, h = bbox.tolist()
                x1, y1 = x, y
                x2, y2 = x + w, y + h
            else:
                # 已经是 [x1, y1, x2, y2] 格式
                x1, y1, x2, y2 = bbox.tolist()
        else:
            raise ValueError(f"Invalid bbox format: {bbox}")
        
        # 添加padding
        region_width = x2 - x1
        region_height = y2 - y1
        
        pad_x = region_width * padding
        pad_y = region_height * padding
        
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(img_width, x2 + pad_x)
        y2 = min(img_height, y2 + pad_y)
        
        # 裁剪区域
        region = image.crop((x1, y1, x2, y2))
        
        # 调整大小
        region = region.resize((self.region_crop_size, self.region_crop_size), Image.Resampling.BILINEAR)
        
        return region
    
    def __getitem__(self, index: int) -> Dict:
        """
        获取数据样本
        
        Returns:
            {
                'region': (C, H, W) 裁剪后的区域图像
                'text': str 文本描述
                'bbox': [x1, y1, x2, y2] 边界框
                'metadata': 元数据
            }
        """
        # 从基础数据集获取数据
        sample = self.base_dataset[index]
        
        image = sample['image']
        text = sample['text']
        bbox = sample.get('bbox')
        
        # 如果image是tensor，需要转换回PIL Image进行裁剪
        if isinstance(image, torch.Tensor):
            # 转换为PIL Image
            if image.dim() == 3:  # (C, H, W)
                image_np = image.permute(1, 2, 0).numpy()
                if image_np.max() <= 1.0:
                    image_np = (image_np * 255).astype(np.uint8)
                image = Image.fromarray(image_np)
            else:
                raise ValueError(f"Unexpected image tensor shape: {image.shape}")
        
        # 裁剪区域
        if bbox is not None:
            region = self._crop_region(image, bbox, self.region_padding)
        else:
            # 如果没有bbox，使用整张图像
            region = image.resize((self.region_crop_size, self.region_crop_size), Image.Resampling.BILINEAR)
            logger.warning(f"Sample {index} has no bbox, using full image")
        
        # 应用变换
        if self.transform is not None:
            region = self.transform(region)
        else:
            # 默认转换为tensor
            region = torch.from_numpy(np.array(region)).permute(2, 0, 1).float() / 255.0
        
        return {
            'region': region,  # (C, H, W)
            'text': text,
            'bbox': bbox if bbox is not None else None,
            'metadata': sample.get('metadata', {})
        }
    
    @staticmethod
    def name() -> str:
        return "RefCOCOAlignment"
    
    @staticmethod
    def get_train_dataset(root_dir: Union[str, Path], **kwargs):
        return RefCOCOAlignmentDataset(root_dir, 'train', **kwargs)
    
    @staticmethod
    def get_valid_dataset(root_dir: Union[str, Path], **kwargs):
        return RefCOCOAlignmentDataset(root_dir, 'val', **kwargs)
    
    @staticmethod
    def get_test_dataset(root_dir: Union[str, Path], **kwargs):
        return RefCOCOAlignmentDataset(root_dir, 'test', **kwargs)


    def get_collate_fn(self):
        """返回collate函数"""
        max_regions = getattr(self, 'max_regions_per_image', 32)
        return lambda batch: collate_refcoco_alignment(batch, max_regions)


def collate_refcoco_alignment(batch: List[Dict], max_regions_per_image: int = 32):
    """
    Collate function for RefCOCO alignment dataset
    
    Args:
        batch: List of samples, each containing 'region', 'text', 'bbox', 'metadata'
        max_regions_per_image: 每个batch最大区域数
    
    Returns:
        批处理后的数据字典
    """
    # 收集所有区域和文本
    regions = [item['region'] for item in batch]  # List of (C, H, W)
    texts = [item['text'] for item in batch]
    
    # 转换为tensor并添加batch维度
    regions = torch.stack(regions, dim=0)  # (B, C, H, W)
    
    # 为每个区域添加对象维度：从 (B, C, H, W) -> (B, 1, C, H, W)
    # 为了匹配EMOE的输入格式 (B, N_OBJ, C, H, W)
    regions = regions.unsqueeze(1)  # (B, 1, C, H, W)
    
    # 如果需要，可以padding到max_regions_per_image
    # 当前每个样本只有一个区域，所以N_OBJ=1
    
    return {
        'regions': regions,  # (B, N_OBJ, C, H, W)
        'texts': texts,  # List[str]
        'metadata': [item['metadata'] for item in batch]
    }

