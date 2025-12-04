"""
BraTS2020 Brain Tumor Segmentation Dataset

This module provides a PyTorch Dataset class for loading and preprocessing
BraTS2020 brain tumor MRI data with T1, T1ce, T2, and FLAIR modalities.

Dataset Structure:
- Training: 371 samples with segmentation masks
- Validation: 125 samples without segmentation masks
- Each sample contains 4 MRI modalities: T1, T1ce, T2, FLAIR
"""

import os
import glob
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from pathlib import Path

from .custom_dataset import CustomDataset

logger = logging.getLogger(__name__)


class BraTS2020Dataset(CustomDataset):
    """
    BraTS2020 Brain Tumor Segmentation Dataset
    
    Args:
        data_root (str): Root directory of BraTS2020 dataset
        split (str): 'train' or 'val' for training or validation split
        modalities (List[str]): List of modalities to load ['t1', 't1ce', 't2', 'flair']
        transform (callable, optional): Optional transform to be applied on a sample
        load_seg (bool): Whether to load segmentation masks (only available for training)
        cache_data (bool): Whether to cache loaded data in memory
        normalize (bool): Whether to normalize intensity values to [0, 1]
    """
    
    # BraTS2020 tumor class labels
    TUMOR_CLASSES = {
        0: 'background',
        1: 'necrotic_core',      # NCR - Necrotic and Non-Enhancing Tumor Core
        2: 'peritumoral_edema',  # ED - Peritumoral Edema  
        4: 'enhancing_tumor'     # ET - GD-Enhancing Tumor
    }
    
    # Standard modalities
    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    # Mapping between BraTS2020 modalitie names and file suffixes
    MODALITY_SUFFIXES = {
        't1': '_t1.nii.gz',
        't1ce': '_t1ce.nii.gz',
        't2': '_t2.nii.gz',
        'flair': '_flair.nii.gz'
    }
    # Mapping
    MAPPING = {
        'train': ('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/BraTS2020_*.nii.gz',
                 'BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/BraTS2020_*.nii.gz'),
        'valid': ('BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/BraTS2020_*.nii.gz',
                 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/BraTS2020_*.nii.gz'),
        'test':  ('BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/BraTS2020_*.nii.gz',
                 'BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/BraTS2020_*.nii.gz'),
    }
    
    def __init__(
        self,
        root_dir: Path,
        split: str = 'train',
        modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
        transform: Optional[Callable] = None,
        load_seg: bool = True,
        cache_data: bool = False,
        normalize: bool = True,
        **kwargs
    ):
        # 调用父类构造函数
        super().__init__(root_dir, split, **kwargs)
        
        # BraTS2020特定参数
        self.modalities = modalities or self.MODALITIES
        self.transform = transform or mtf.Compose()
        self.load_seg = load_seg and (split == 'train')  # Only training has segmentation
        self.cache_data = cache_data
        self.normalize = normalize
        
        # 验证输入参数
        self._validate_inputs()
        
        # 获取数据路径
        self.data_paths = self._get_data_paths()
        
        # 统计样本数量
        self.n = len(self.data_paths)
        
        # 缓存
        self._cache = {} if cache_data else None
        
        logger.info(f"Initialized BraTS2020Dataset: {self.n} samples, "
                   f"split={split}, modalities={self.modalities}")
    
    def _validate_inputs(self):
        """验证输入参数"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data root not found: {self.root_dir}")
        
        if self.split not in ['train', 'valid', 'test']:
            raise ValueError(f"Split must be 'train', 'valid', or 'test', got {self.split}")
        
        for modality in self.modalities:
            if modality not in self.MODALITIES:
                raise ValueError(f"Invalid modality: {modality}. "
                               f"Must be one of {self.MODALITIES}")
    
    def _get_data_paths(self) -> List[Dict[str, str]]:
        """获取指定分割的所有数据文件路径"""
        if self.split == 'train':
            data_dir = self.root_dir / self.MAPPING[self.split][0]
        elif self.split == 'valid':
            data_dir = self.root_dir / self.MAPPING[self.split][1]
        else:  # test
            data_dir = self.root_dir / self.MAPPING[self.split][1]
        
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        # 获取所有主题目录
        subject_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
        
        data_paths = []
        for subject_dir in subject_dirs:
            subject_id = subject_dir.name
            
            # Build paths for each modality
            paths = {'subject_id': subject_id}
            
            for modality in self.modalities:
                modality_file = subject_dir / f"{subject_id}_{modality}.nii"
                if not modality_file.exists():
                    # Try .nii.gz extension
                    modality_file = subject_dir / f"{subject_id}_{modality}.nii.gz"
                
                if modality_file.exists():
                    paths[modality] = str(modality_file)
                else:
                    logger.warning(f"Missing {modality} file for {subject_id}")
                    continue
            
            # Add segmentation path if available and requested
            if self.load_seg:
                seg_file = subject_dir / f"{subject_id}_seg.nii"
                if not seg_file.exists():
                    seg_file = subject_dir / f"{subject_id}_seg.nii.gz"
                
                if seg_file.exists():
                    paths['seg'] = str(seg_file)
                else:
                    logger.warning(f"Missing segmentation file for {subject_id}")
            
            # Only add if all required modalities are present
            if all(mod in paths for mod in self.modalities):
                data_paths.append(paths)
        
        if not data_paths:
            raise RuntimeError(f"No valid data found in {data_dir}")
        
        return data_paths
    
    def _load_nifti(self, file_path: str) -> np.ndarray:
        """Load NIfTI file and return numpy array"""
        try:
            nii_img = nib.load(file_path)
            data = nii_img.get_fdata().astype(np.float32)
            return data
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            raise
    
    def _normalize_intensity(self, data: np.ndarray) -> np.ndarray:
        """Normalize intensity values to [0, 1] range"""
        # Remove background (zero values) from normalization
        mask = data > 0
        if mask.sum() > 0:
            data_masked = data[mask]
            # Normalize to [0, 1] using min-max normalization
            data_min, data_max = data_masked.min(), data_masked.max()
            if data_max > data_min:
                data[mask] = (data_masked - data_min) / (data_max - data_min)
        return data
    
    def _preprocess_volume(self, data: np.ndarray) -> np.ndarray:
        """Apply preprocessing to volume data"""
        if self.normalize:
            data = self._normalize_intensity(data)
        return data
    
    def __len__(self) -> int:
        return self.n
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str]]:
        """
        Get a sample from the dataset
        
        Returns:
            Dict containing:
                - 'image': Tensor of shape (C, H, W, D) where C is number of modalities
                - 'seg': Segmentation mask (if load_seg=True)
                - 'subject_id': Subject identifier
                - 'modalities': List of loaded modalities
        """
        if idx >= len(self.data_paths):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        paths = self.data_paths[idx]
        subject_id = paths['subject_id']
        
        # Check cache first
        if self._cache is not None and idx in self._cache:
            return self._cache[idx]
        
        # Load modality data
        modality_data = []
        for modality in self.modalities:
            if modality in paths:
                data = self._load_nifti(paths[modality])
                data = self._preprocess_volume(data)
                modality_data.append(data)
            else:
                raise RuntimeError(f"Missing {modality} data for {subject_id}")
        
        # Stack modalities along channel dimension
        image = np.stack(modality_data, axis=0)  # Shape: (C, H, W, D)
        image = torch.from_numpy(image).float()
        
        # Prepare sample
        sample = {
            'image': image,
            'subject_id': subject_id,
            'modalities': self.modalities.copy()
        }
        
        # Load segmentation if available
        if self.load_seg and 'seg' in paths:
            seg_data = self._load_nifti(paths['seg'])
            seg_data = torch.from_numpy(seg_data).long()
            sample['seg'] = seg_data
        
        # Apply transforms
        if self.transform:
            sample = self.transform(sample)
        
        # Cache if enabled
        if self._cache is not None:
            self._cache[idx] = sample
        
        return sample
    
    def name(self) -> str:
        """返回数据集名称"""
        return f"BraTS2020_{self.split}"
    
    def get_train_dataset(self) -> 'BraTS2020Dataset':
        """获取训练数据集"""
        return BraTS2020Dataset(
            root_dir=self.root_dir,
            split='train',
            modalities=self.modalities,
            transform=self.transform,
            load_seg=True
        )
    
    def get_valid_dataset(self) -> 'BraTS2020Dataset':
        """获取验证数据集"""
        return BraTS2020Dataset(
            root_dir=self.root_dir,
            split='valid',
            modalities=self.modalities,
            transform=self.transform,
            load_seg=True
        )
    
    def get_test_dataset(self) -> 'BraTS2020Dataset':
        """获取测试数据集"""
        return BraTS2020Dataset(
            root_dir=self.root_dir,
            split='test',
            modalities=self.modalities,
            transform=self.transform,
            load_seg=True
        )
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced segmentation classes"""
        if not self.load_seg:
            raise RuntimeError("Class weights only available when segmentation is loaded")
        
        # Initialize class counts for all possible classes (0, 1, 2, 4)
        class_counts = torch.zeros(5)  # Index 0-4, where index 4 corresponds to class 4
        
        for idx in range(len(self)):
            sample = self[idx]
            
            # Skip samples without segmentation
            if 'seg' not in sample:
                continue
                
            seg = sample['seg']
            
            for class_id in self.TUMOR_CLASSES.keys():
                class_counts[class_id] += (seg == class_id).sum().item()
        
        # Calculate inverse frequency weights, avoiding division by zero
        total_voxels = class_counts.sum()
        weights = total_voxels / (class_counts + 1e-8)
        
        # Normalize weights
        weights = weights / weights.sum() * len(self.TUMOR_CLASSES)
        
        # Return only the weights for the actual classes used
        class_weights = torch.zeros(len(self.TUMOR_CLASSES))
        for i, class_id in enumerate(self.TUMOR_CLASSES.keys()):
            class_weights[i] = weights[class_id]
        
        return class_weights
    
    def get_statistics(self) -> Dict[str, float]:
        """Get dataset statistics"""
        stats = {
            'num_samples': len(self),
            'modalities': self.modalities,
            'has_segmentation': self.load_seg
        }
        
        if len(self) > 0:
            sample = self[0]
            image_shape = sample['image'].shape
            stats.update({
                'image_shape': image_shape,
                'num_modalities': image_shape[0],
                'spatial_dims': image_shape[1:]
            })
        
        return stats



if __name__ == "__main__":
    # 示例用法
    base_dir = Path("/media/rczx/Data/data/BraTS2020")
    
    # 创建数据集实例
    dataset = BraTS2020Dataset(
        base_dir=base_dir,
        dataset_type='train',
        modalities=['t1', 't1ce', 't2', 'flair'],
        load_seg=True,
        normalize=True
    )
    
    print(f"Dataset name: {dataset.name}")
    print(f"Dataset statistics: {dataset.get_statistics()}")
    
    # 获取样本
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    if 'seg' in sample:
        print(f"Segmentation shape: {sample['seg'].shape}")
        print(f"Unique labels: {torch.unique(sample['seg'])}")