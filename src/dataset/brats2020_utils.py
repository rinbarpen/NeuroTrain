"""
BraTS2020 Dataset Utilities

This module provides utility functions for working with the BraTS2020 dataset,
including data preprocessing, augmentation, visualization, and analysis tools.
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import torch
import torch.nn.functional as F
from pathlib import Path
import json
import warnings

# Suppress NiBabel warnings
warnings.filterwarnings("ignore", category=UserWarning, module="nibabel")


def dice_coefficient(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient between prediction and target
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        smooth: Smoothing factor to avoid division by zero
        
    Returns:
        Dice coefficient
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum()
    
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return float(dice)


class BraTS2020Preprocessor:
    """Preprocessing utilities for BraTS2020 dataset"""
    
    def __init__(self, 
                 target_size: Tuple[int, int, int] = (128, 128, 128),
                 normalize_method: str = 'z_score',
                 clip_percentiles: Tuple[float, float] = (1.0, 99.0)):
        """
        Initialize preprocessor
        
        Args:
            target_size: Target spatial dimensions for resizing
            normalize_method: Normalization method ('z_score', 'min_max', 'percentile')
            clip_percentiles: Percentiles for clipping outliers
        """
        self.target_size = target_size
        self.normalize_method = normalize_method
        self.clip_percentiles = clip_percentiles
    
    def resize_volume(self, volume: np.ndarray, target_size: Optional[Tuple[int, int, int]] = None) -> np.ndarray:
        """
        Resize 3D volume to target size
        
        Args:
            volume: Input volume of shape (H, W, D)
            target_size: Target size, defaults to self.target_size
            
        Returns:
            Resized volume
        """
        if target_size is None:
            target_size = self.target_size
        
        # Convert to tensor for interpolation
        volume_tensor = torch.from_numpy(volume).float().unsqueeze(0).unsqueeze(0)
        
        # Resize using trilinear interpolation
        resized = F.interpolate(
            volume_tensor, 
            size=target_size, 
            mode='trilinear', 
            align_corners=False
        )
        
        return resized.squeeze().numpy()
    
    def normalize_volume(self, volume: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Normalize volume intensity
        
        Args:
            volume: Input volume
            mask: Optional brain mask for normalization
            
        Returns:
            Normalized volume
        """
        if mask is not None:
            # Use only brain region for normalization
            brain_voxels = volume[mask > 0]
        else:
            # Use non-zero voxels
            brain_voxels = volume[volume > 0]
        
        if len(brain_voxels) == 0:
            return volume
        
        if self.normalize_method == 'z_score':
            mean_val = np.mean(brain_voxels)
            std_val = np.std(brain_voxels)
            if std_val > 0:
                volume = (volume - mean_val) / std_val
        
        elif self.normalize_method == 'min_max':
            min_val = np.min(brain_voxels)
            max_val = np.max(brain_voxels)
            if max_val > min_val:
                volume = (volume - min_val) / (max_val - min_val)
        
        elif self.normalize_method == 'percentile':
            p_low, p_high = self.clip_percentiles
            low_val = np.percentile(brain_voxels, p_low)
            high_val = np.percentile(brain_voxels, p_high)
            
            # Clip and normalize
            volume = np.clip(volume, low_val, high_val)
            if high_val > low_val:
                volume = (volume - low_val) / (high_val - low_val)
        
        return volume
    
    def create_brain_mask(self, volumes: List[np.ndarray], threshold: float = 0.01) -> np.ndarray:
        """
        Create brain mask from multiple modalities
        
        Args:
            volumes: List of modality volumes
            threshold: Threshold for brain detection
            
        Returns:
            Binary brain mask
        """
        # Combine all modalities
        combined = np.stack(volumes, axis=0)
        
        # Create mask where any modality has signal
        mask = np.any(combined > threshold, axis=0)
        
        return mask.astype(np.uint8)


class BraTS2020Augmentor:
    """Data augmentation utilities for BraTS2020"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 translation_range: float = 0.1,
                 scaling_range: float = 0.1,
                 noise_std: float = 0.05):
        """
        Initialize augmentor
        
        Args:
            rotation_range: Maximum rotation angle in degrees
            translation_range: Maximum translation as fraction of image size
            scaling_range: Maximum scaling factor deviation
            noise_std: Standard deviation for Gaussian noise
        """
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.scaling_range = scaling_range
        self.noise_std = noise_std
    
    def random_rotation(self, volume: np.ndarray, angle: Optional[float] = None) -> np.ndarray:
        """Apply random rotation around z-axis"""
        if angle is None:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Convert to tensor for rotation
        volume_tensor = torch.from_numpy(volume).float()
        
        # Create rotation matrix (simplified 2D rotation in axial plane)
        angle_rad = np.radians(angle)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Apply rotation using affine transformation
        # This is a simplified implementation - for production use torchio or similar
        return volume  # Placeholder - implement proper 3D rotation
    
    def add_gaussian_noise(self, volume: np.ndarray, std: Optional[float] = None) -> np.ndarray:
        """Add Gaussian noise to volume"""
        if std is None:
            std = self.noise_std
        
        noise = np.random.normal(0, std, volume.shape)
        return volume + noise
    
    def random_intensity_shift(self, volume: np.ndarray, shift_range: float = 0.1) -> np.ndarray:
        """Apply random intensity shift"""
        shift = np.random.uniform(-shift_range, shift_range)
        return volume + shift


class BraTS2020Visualizer:
    """Visualization utilities for BraTS2020 dataset"""
    
    @staticmethod
    def plot_sample(image: np.ndarray, 
                   segmentation: Optional[np.ndarray] = None,
                   modalities: List[str] = ['T1', 'T1ce', 'T2', 'FLAIR'],
                   slice_idx: Optional[int] = None,
                   figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot a sample with all modalities and segmentation
        
        Args:
            image: Image tensor of shape (C, H, W, D)
            segmentation: Segmentation mask of shape (H, W, D)
            modalities: List of modality names
            slice_idx: Slice index to display (middle slice if None)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if slice_idx is None:
            slice_idx = image.shape[-1] // 2
        
        n_modalities = image.shape[0]
        n_cols = n_modalities + (1 if segmentation is not None else 0)
        
        fig, axes = plt.subplots(1, n_cols, figsize=figsize)
        if n_cols == 1:
            axes = [axes]
        
        # Plot modalities
        for i in range(n_modalities):
            ax = axes[i]
            im = ax.imshow(image[i, :, :, slice_idx], cmap='gray')
            ax.set_title(f'{modalities[i] if i < len(modalities) else f"Modality {i}"}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Plot segmentation
        if segmentation is not None:
            ax = axes[-1]
            # Create colored segmentation
            seg_colored = np.zeros((*segmentation.shape[:2], 3))
            seg_slice = segmentation[:, :, slice_idx]
            
            # Color mapping for BraTS labels
            seg_colored[seg_slice == 1] = [1, 0, 0]  # Red for necrotic core
            seg_colored[seg_slice == 2] = [0, 1, 0]  # Green for edema
            seg_colored[seg_slice == 4] = [0, 0, 1]  # Blue for enhancing tumor
            
            ax.imshow(image[0, :, :, slice_idx], cmap='gray', alpha=0.7)
            ax.imshow(seg_colored, alpha=0.5)
            ax.set_title('Segmentation Overlay')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    @staticmethod
    def plot_volume_slices(volume: np.ndarray, 
                          n_slices: int = 9,
                          figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
        """
        Plot multiple slices of a 3D volume
        
        Args:
            volume: 3D volume of shape (H, W, D)
            n_slices: Number of slices to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        depth = volume.shape[-1]
        slice_indices = np.linspace(0, depth-1, n_slices, dtype=int)
        
        cols = int(np.ceil(np.sqrt(n_slices)))
        rows = int(np.ceil(n_slices / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = axes.flatten() if n_slices > 1 else [axes]
        
        for i, slice_idx in enumerate(slice_indices):
            if i < len(axes):
                axes[i].imshow(volume[:, :, slice_idx], cmap='gray')
                axes[i].set_title(f'Slice {slice_idx}')
                axes[i].axis('off')
        
        # Hide unused subplots
        for i in range(len(slice_indices), len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        return fig


class BraTS2020Analyzer:
    """Analysis utilities for BraTS2020 dataset"""
    
    @staticmethod
    def compute_tumor_statistics(segmentation: np.ndarray) -> Dict[str, float]:
        """
        Compute tumor statistics from segmentation
        
        Args:
            segmentation: Segmentation mask
            
        Returns:
            Dictionary with tumor statistics
        """
        stats = {}
        
        # Tumor regions
        necrotic_core = (segmentation == 1).sum()
        edema = (segmentation == 2).sum()
        enhancing_tumor = (segmentation == 4).sum()
        
        # Combined regions
        whole_tumor = necrotic_core + edema + enhancing_tumor
        tumor_core = necrotic_core + enhancing_tumor
        
        stats['necrotic_core_voxels'] = int(necrotic_core)
        stats['edema_voxels'] = int(edema)
        stats['enhancing_tumor_voxels'] = int(enhancing_tumor)
        stats['whole_tumor_voxels'] = int(whole_tumor)
        stats['tumor_core_voxels'] = int(tumor_core)
        
        # Ratios
        total_voxels = segmentation.size
        stats['whole_tumor_ratio'] = float(whole_tumor / total_voxels)
        stats['tumor_core_ratio'] = float(tumor_core / total_voxels)
        
        return stats
    
    @staticmethod
    def compute_dice_scores(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
        """
        Compute Dice scores for different tumor regions
        
        Args:
            pred: Predicted segmentation
            target: Ground truth segmentation
            
        Returns:
            Dictionary with Dice scores
        """
        dice_scores = {}
        
        # Whole tumor (all non-zero labels)
        pred_wt = (pred > 0).astype(int)
        target_wt = (target > 0).astype(int)
        dice_scores['whole_tumor'] = dice_coefficient(pred_wt, target_wt)
        
        # Tumor core (labels 1 and 4)
        pred_tc = ((pred == 1) | (pred == 4)).astype(int)
        target_tc = ((target == 1) | (target == 4)).astype(int)
        dice_scores['tumor_core'] = dice_coefficient(pred_tc, target_tc)
        
        # Enhancing tumor (label 4)
        pred_et = (pred == 4).astype(int)
        target_et = (target == 4).astype(int)
        dice_scores['enhancing_tumor'] = dice_coefficient(pred_et, target_et)
        
        return dice_scores
    
    @staticmethod
    def analyze_dataset_statistics(dataset_path: str) -> Dict:
        """
        Analyze statistics of the entire dataset
        
        Args:
            dataset_path: Path to dataset root
            
        Returns:
            Dictionary with dataset statistics
        """
        from src.dataset.brats2020_dataset import BraTS2020Dataset
        
        stats = {
            'train': {'samples': 0, 'tumor_stats': []},
            'val': {'samples': 0, 'tumor_stats': []}
        }
        
        for split in ['train', 'val']:
            try:
                dataset = BraTS2020Dataset(
                    data_root=dataset_path,
                    split=split,
                    modalities=['t1'],  # Just one modality for speed
                    load_seg=(split == 'train'),
                    normalize=False
                )
                
                stats[split]['samples'] = len(dataset)
                
                if split == 'train':
                    # Analyze tumor statistics for training samples
                    for i in range(min(50, len(dataset))):  # Sample first 50
                        try:
                            sample = dataset[i]
                            if 'seg' in sample:
                                tumor_stats = BraTS2020Analyzer.compute_tumor_statistics(
                                    sample['seg'].numpy()
                                )
                                stats[split]['tumor_stats'].append(tumor_stats)
                        except Exception as e:
                            print(f"Error analyzing sample {i}: {e}")
                
            except Exception as e:
                print(f"Error analyzing {split} split: {e}")
        
        return stats


def save_nifti(data: np.ndarray, 
               output_path: str, 
               affine: Optional[np.ndarray] = None,
               header: Optional[nib.Nifti1Header] = None) -> None:
    """
    Save numpy array as NIfTI file
    
    Args:
        data: 3D numpy array
        output_path: Output file path
        affine: Affine transformation matrix
        header: NIfTI header
    """
    if affine is None:
        affine = np.eye(4)
    
    img = nib.Nifti1Image(data, affine, header)
    nib.save(img, output_path)


def load_brats_sample(sample_dir: str, 
                     modalities: List[str] = ['t1', 't1ce', 't2', 'flair'],
                     load_seg: bool = True) -> Dict[str, np.ndarray]:
    """
    Load a complete BraTS sample
    
    Args:
        sample_dir: Path to sample directory
        modalities: List of modalities to load
        load_seg: Whether to load segmentation
        
    Returns:
        Dictionary with loaded data
    """
    sample_dir = Path(sample_dir)
    sample_id = sample_dir.name
    
    data = {}
    
    # Load modalities
    for modality in modalities:
        modality_file = sample_dir / f"{sample_id}_{modality}.nii"
        if not modality_file.exists():
            modality_file = sample_dir / f"{sample_id}_{modality}.nii.gz"
        
        if modality_file.exists():
            img = nib.load(str(modality_file))
            data[modality] = img.get_fdata()
        else:
            raise FileNotFoundError(f"Modality file not found: {modality_file}")
    
    # Load segmentation
    if load_seg:
        seg_file = sample_dir / f"{sample_id}_seg.nii"
        if not seg_file.exists():
            seg_file = sample_dir / f"{sample_id}_seg.nii.gz"
        
        # Check for alternative segmentation file names
        if not seg_file.exists():
            seg_files = list(sample_dir.glob("*[Ss]eg*.nii")) + list(sample_dir.glob("*[Ss]eg*.nii.gz"))
            if seg_files:
                seg_file = seg_files[0]
        
        if seg_file.exists():
            img = nib.load(str(seg_file))
            data['seg'] = img.get_fdata()
    
    return data


# Export main classes and functions
__all__ = [
    'BraTS2020Preprocessor',
    'BraTS2020Augmentor', 
    'BraTS2020Visualizer',
    'BraTS2020Analyzer',
    'save_nifti',
    'load_brats_sample'
]