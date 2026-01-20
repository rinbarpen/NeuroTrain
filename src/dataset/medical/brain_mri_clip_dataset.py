import json
import torch
from PIL import Image
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union
from torchvision import transforms as mtf

from ..custom_dataset import CustomDataset


class BrainMRIClipDataset(CustomDataset):
    """
    Brain MRI CLIP数据集类
    支持图像-文本对的CLIP训练
    
    数据集结构:
    base_dir/
    ├── train/
    │   ├── images/
    │   ├── captions.json
    │   └── annotations.json
    ├── valid/
    │   ├── images/
    │   ├── captions.json
    │   └── annotations.json
    └── test/
        ├── images/
        ├── captions.json
        └── annotations.json
    """
    
    def __init__(self, root_dir: Union[str, Path], split: str = 'train', transform=None, **kwargs):
        super().__init__(root_dir, split, transform=transform)
        
        self.split = split
        self.root_dir = Path(root_dir)
        self.split_dir = self.root_dir / split
        self.images_dir = self.split_dir / 'images'
        
        # 加载标注文件
        self.captions_file = self.split_dir / 'captions.json'
        self.annotations_file = self.split_dir / 'annotations.json'
        
        # 初始化数据
        self._load_data()

        self.n = len(self.samples)
        
    def _load_data(self):
        """加载数据集"""
        # 检查必要的文件是否存在
        if not self.images_dir.exists():
            print(f"Warning: Images directory {self.images_dir} does not exist")
            return
            
        if not self.captions_file.exists():
            print(f"Warning: Captions file {self.captions_file} does not exist")
            # 如果没有标注文件，创建示例数据
            self._create_sample_data()
            return
            
        # 加载标注数据
        try:
            with open(self.captions_file, 'r', encoding='utf-8') as f:
                captions_data = json.load(f)
                
            # 处理标注数据格式
            if isinstance(captions_data, dict):
                for image_name, caption in captions_data.items():
                    image_path = self.images_dir / image_name
                    if image_path.exists():
                        self.samples.append({
                            'image_path': image_path,
                            'caption': caption,
                            'image_name': image_name
                        })
            elif isinstance(captions_data, list):
                for item in captions_data:
                    image_path = self.images_dir / item['image']
                    if image_path.exists():
                        self.samples.append({
                            'image_path': image_path,
                            'caption': item['caption'],
                            'image_name': item['image']
                        })
                        
        except Exception as e:
            print(f"Error loading captions: {e}")
            self._create_sample_data()
    
    def _create_sample_data(self):
        """创建示例数据（当标注文件不存在时）"""
        print(f"Creating sample data for {self.split} split...")
        
        # 获取所有图像文件
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        image_files = []
        
        if self.images_dir.exists():
            for ext in image_extensions:
                image_files.extend(list(self.images_dir.glob(f'*{ext}')))
                image_files.extend(list(self.images_dir.glob(f'*{ext.upper()}')))
        
        # 示例标题模板
        caption_templates = [
            "A brain MRI scan showing normal brain tissue",
            "Brain MRI image with clear anatomical structures",
            "Medical brain scan displaying brain anatomy",
            "MRI brain slice with visible brain regions",
            "Brain imaging showing neurological structures"
        ]
        
        # 为每个图像创建示例标注
        for i, image_path in enumerate(image_files):
            caption = caption_templates[i % len(caption_templates)]
            self.samples.append({
                'image_path': image_path,
                'caption': caption,
                'image_name': image_path.name
            })
        
        # 保存示例标注文件
        if self.samples:
            self._save_sample_captions()
    
    def _save_sample_captions(self):
        """保存示例标注文件"""
        try:
            # 确保目录存在
            self.split_dir.mkdir(parents=True, exist_ok=True)
            
            # 创建标注数据
            captions_data = {}
            annotations_data = []
            
            for sample in self.samples:
                captions_data[sample['image_name']] = sample['caption']
                annotations_data.append({
                    'image': sample['image_name'],
                    'caption': sample['caption'],
                    'category': 'brain_mri'
                })
            
            # 保存标注文件
            with open(self.captions_file, 'w', encoding='utf-8') as f:
                json.dump(captions_data, f, indent=2, ensure_ascii=False)
                
            with open(self.annotations_file, 'w', encoding='utf-8') as f:
                json.dump(annotations_data, f, indent=2, ensure_ascii=False)
                
            print(f"Saved {len(self.samples)} sample annotations to {self.captions_file}")
            
        except Exception as e:
            print(f"Error saving sample captions: {e}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        if index >= len(self.samples):
            raise IndexError(f"Index {index} out of range for dataset of size {len(self.samples)}")
            
        sample = self.samples[index]
        
        try:
            # 加载图像
            image = Image.open(sample['image_path']).convert('RGB')
            
            # 应用变换
            if self.transform:
                image = self.transform(image)
            
            return {
                'image': image,
                'text': sample['caption'],
                'image_name': sample['image_name']
            }
            
        except Exception as e:
            print(f"Error loading sample {index}: {e}")
            # 返回一个默认的样本
            dummy_image = Image.new('RGB', (224, 224), color='black')
            if self.transform:
                dummy_image = self.transform(dummy_image)
            
            return {
                'image': dummy_image,
                'text': "Error loading image",
                'image_name': f"error_{index}.jpg"
            }
    
    @classmethod
    def get_train_dataset(cls, root_dir: Union[str, Path], **kwargs):
        """获取训练数据集"""
        return cls(root_dir, split='train', **kwargs)
    
    @classmethod
    def get_valid_dataset(cls, root_dir: Union[str, Path], **kwargs):
        """获取验证数据集"""
        return cls(root_dir, split='valid', **kwargs)
    
    @classmethod
    def get_test_dataset(cls, root_dir: Union[str, Path], **kwargs):
        """获取测试数据集"""
        return cls(root_dir, split='test', **kwargs)
