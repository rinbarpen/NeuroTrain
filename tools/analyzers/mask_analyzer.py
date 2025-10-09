"""
Mask分析器模块

该模块提供了对图像和文本mask信息的综合分析功能，支持：
- 图像分割mask分析
- 文本attention mask分析  
- 跨模态mask关联分析
- mask质量评估和可视化
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import cv2
from PIL import Image
import logging
from sklearn.metrics import jaccard_score, f1_score
from scipy import ndimage
from collections import defaultdict
import json
from datetime import datetime

class MaskAnalyzer:
    """
    Mask分析器，支持图像和文本mask的综合分析
    
    功能包括：
    - 图像分割mask分析（语义分割、实例分割等）
    - 文本attention mask分析
    - mask质量评估（IoU、Dice等指标）
    - 跨模态mask关联分析
    - mask可视化和报告生成
    """
    
    def __init__(self, 
                 device: str = 'auto',
                 save_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化Mask分析器
        
        Args:
            device: 计算设备 ('cuda', 'cpu', 'auto')
            save_dir: 结果保存目录
            logger: 日志记录器
        """
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        self.save_dir = Path(save_dir) if save_dir else Path('./output/analysis')
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logger or self._setup_logger()
        
        # 支持的mask类型
        self.supported_mask_types = {
            'image_segmentation': ['semantic', 'instance', 'panoptic'],
            'text_attention': ['padding', 'causal', 'bidirectional'],
            'cross_modal': ['alignment', 'correspondence']
        }
        
        self.logger.info(f"MaskAnalyzer initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('MaskAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_image_mask(self,
                          mask: Union[torch.Tensor, np.ndarray],
                          image: Optional[Union[torch.Tensor, np.ndarray]] = None,
                          mask_type: str = 'semantic',
                          class_names: Optional[List[str]] = None,
                          save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析图像分割mask
        
        Args:
            mask: 分割mask [H, W] 或 [B, H, W] 或 [B, C, H, W]
            image: 原始图像（可选）
            mask_type: mask类型 ('semantic', 'instance', 'panoptic')
            class_names: 类别名称列表
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 转换为numpy数组
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy()
        else:
            mask_np = mask.copy()
        
        # 处理维度
        if mask_np.ndim == 4:  # [B, C, H, W]
            mask_np = mask_np[0]  # 取第一个batch
            if mask_np.shape[0] > 1:  # 多类别
                mask_np = np.argmax(mask_np, axis=0)
            else:
                mask_np = mask_np[0]
        elif mask_np.ndim == 3:  # [B, H, W] 或 [C, H, W]
            if mask_np.shape[0] > 1 and mask_type == 'semantic':
                mask_np = np.argmax(mask_np, axis=0)
            else:
                mask_np = mask_np[0]
        
        height, width = mask_np.shape
        unique_labels = np.unique(mask_np)
        num_classes = len(unique_labels)
        
        # 计算基本统计信息
        analysis_results = {
            'mask_type': mask_type,
            'shape': [height, width],
            'num_classes': num_classes,
            'unique_labels': unique_labels.tolist(),
            'total_pixels': height * width
        }
        
        # 类别分布分析
        class_stats = {}
        for label in unique_labels:
            mask_area = np.sum(mask_np == label)
            class_stats[int(label)] = {
                'pixel_count': int(mask_area),
                'percentage': float(mask_area / (height * width) * 100),
                'bbox': self._get_bbox(mask_np == label) if mask_area > 0 else None
            }
        
        analysis_results['class_statistics'] = class_stats
        
        # 形态学分析
        morphology_stats = self._analyze_mask_morphology(mask_np, unique_labels)
        analysis_results['morphology'] = morphology_stats
        
        # 连通性分析
        connectivity_stats = self._analyze_mask_connectivity(mask_np, unique_labels)
        analysis_results['connectivity'] = connectivity_stats
        
        # 可视化
        self._visualize_image_mask(
            mask_np, image, mask_type, class_names, 
            unique_labels, save_path
        )
        
        return analysis_results
    
    def analyze_text_mask(self,
                         attention_mask: Union[torch.Tensor, np.ndarray],
                         tokens: Optional[List[str]] = None,
                         mask_type: str = 'padding',
                         save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析文本attention mask
        
        Args:
            attention_mask: 注意力mask [seq_len] 或 [batch_size, seq_len]
            tokens: token列表（可选）
            mask_type: mask类型 ('padding', 'causal', 'bidirectional')
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 分析结果
        """
        # 转换为numpy数组
        if isinstance(attention_mask, torch.Tensor):
            mask_np = attention_mask.cpu().numpy()
        else:
            mask_np = attention_mask.copy()
        
        # 处理维度
        if mask_np.ndim == 2:
            # 取第一个batch或者是2D mask矩阵
            if mask_np.shape[0] == mask_np.shape[1]:  # 方形矩阵，可能是causal mask
                seq_len = mask_np.shape[0]
                batch_mask = mask_np
            else:  # [batch_size, seq_len]
                batch_mask = mask_np[0]
                seq_len = len(batch_mask)
        else:  # 1D
            batch_mask = mask_np
            seq_len = len(mask_np)
        
        # 计算基本统计信息
        analysis_results = {
            'mask_type': mask_type,
            'sequence_length': seq_len,
            'total_tokens': seq_len
        }
        
        if mask_np.ndim == 2 and mask_np.shape[0] == mask_np.shape[1]:
            # 2D attention mask分析
            analysis_results.update(self._analyze_2d_attention_mask(mask_np, tokens))
        else:
            # 1D padding mask分析
            analysis_results.update(self._analyze_1d_attention_mask(batch_mask, tokens))
        
        # 可视化
        self._visualize_text_mask(mask_np, tokens, mask_type, save_path)
        
        return analysis_results
    
    def analyze_cross_modal_alignment(self,
                                    image_features: torch.Tensor,
                                    text_features: torch.Tensor,
                                    image_mask: Optional[torch.Tensor] = None,
                                    text_mask: Optional[torch.Tensor] = None,
                                    similarity_threshold: float = 0.5,
                                    save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析跨模态对齐关系
        
        Args:
            image_features: 图像特征 [H*W, D] 或 [B, H*W, D]
            text_features: 文本特征 [seq_len, D] 或 [B, seq_len, D]
            image_mask: 图像mask（可选）
            text_mask: 文本mask（可选）
            similarity_threshold: 相似度阈值
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 跨模态对齐分析结果
        """
        # 处理维度
        if image_features.dim() == 3:
            image_features = image_features[0]  # [H*W, D]
        if text_features.dim() == 3:
            text_features = text_features[0]    # [seq_len, D]
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(
            F.normalize(image_features, dim=-1),
            F.normalize(text_features, dim=-1).T
        )  # [H*W, seq_len]
        
        similarity_np = similarity_matrix.cpu().numpy()
        
        # 基本统计
        analysis_results = {
            'image_regions': image_features.shape[0],
            'text_tokens': text_features.shape[0],
            'feature_dim': image_features.shape[1],
            'similarity_stats': {
                'mean': float(np.mean(similarity_np)),
                'std': float(np.std(similarity_np)),
                'max': float(np.max(similarity_np)),
                'min': float(np.min(similarity_np))
            }
        }
        
        # 对齐分析
        alignment_stats = self._analyze_alignment_patterns(
            similarity_np, similarity_threshold, image_mask, text_mask
        )
        analysis_results['alignment'] = alignment_stats
        
        # 可视化跨模态对齐
        self._visualize_cross_modal_alignment(
            similarity_np, image_mask, text_mask, save_path
        )
        
        return analysis_results
    
    def evaluate_mask_quality(self,
                             pred_mask: Union[torch.Tensor, np.ndarray],
                             gt_mask: Union[torch.Tensor, np.ndarray],
                             metrics: List[str] = ['iou', 'dice', 'f1'],
                             class_wise: bool = True) -> Dict[str, Any]:
        """
        评估mask质量
        
        Args:
            pred_mask: 预测mask
            gt_mask: 真实mask
            metrics: 评估指标列表
            class_wise: 是否计算类别级别指标
            
        Returns:
            Dict[str, Any]: 质量评估结果
        """
        # 转换为numpy数组
        if isinstance(pred_mask, torch.Tensor):
            pred_np = pred_mask.cpu().numpy()
        else:
            pred_np = pred_mask.copy()
            
        if isinstance(gt_mask, torch.Tensor):
            gt_np = gt_mask.cpu().numpy()
        else:
            gt_np = gt_mask.copy()
        
        # 确保形状一致
        if pred_np.shape != gt_np.shape:
            self.logger.warning(f"Shape mismatch: pred {pred_np.shape} vs gt {gt_np.shape}")
            return {}
        
        results = {}
        
        # 整体指标
        if 'iou' in metrics:
            results['overall_iou'] = self._calculate_iou(pred_np, gt_np)
        
        if 'dice' in metrics:
            results['overall_dice'] = self._calculate_dice(pred_np, gt_np)
        
        if 'f1' in metrics:
            pred_flat = pred_np.flatten()
            gt_flat = gt_np.flatten()
            results['overall_f1'] = f1_score(gt_flat, pred_flat, average='weighted')
        
        # 类别级别指标
        if class_wise:
            unique_classes = np.unique(gt_np)
            class_results = {}
            
            for cls in unique_classes:
                if cls == 0:  # 跳过背景类
                    continue
                    
                pred_cls = (pred_np == cls).astype(int)
                gt_cls = (gt_np == cls).astype(int)
                
                class_metrics = {}
                if 'iou' in metrics:
                    class_metrics['iou'] = self._calculate_iou(pred_cls, gt_cls)
                if 'dice' in metrics:
                    class_metrics['dice'] = self._calculate_dice(pred_cls, gt_cls)
                if 'f1' in metrics:
                    class_metrics['f1'] = f1_score(gt_cls.flatten(), pred_cls.flatten())
                
                class_results[f'class_{int(cls)}'] = class_metrics
            
            results['class_wise'] = class_results
        
        return results
    
    def _analyze_mask_morphology(self, mask: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """分析mask形态学特征"""
        morphology_stats = {}
        
        for label in labels:
            if label == 0:  # 跳过背景
                continue
                
            binary_mask = (mask == label).astype(np.uint8)
            
            # 计算面积和周长
            area = np.sum(binary_mask)
            contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                perimeter = cv2.arcLength(contours[0], True)
                
                # 计算紧凑性（圆形度）
                compactness = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
                
                # 计算凸包
                hull = cv2.convexHull(contours[0])
                hull_area = cv2.contourArea(hull)
                solidity = area / hull_area if hull_area > 0 else 0
                
                morphology_stats[int(label)] = {
                    'area': int(area),
                    'perimeter': float(perimeter),
                    'compactness': float(compactness),
                    'solidity': float(solidity)
                }
        
        return morphology_stats
    
    def _analyze_mask_connectivity(self, mask: np.ndarray, labels: np.ndarray) -> Dict[str, Any]:
        """分析mask连通性"""
        connectivity_stats = {}
        
        for label in labels:
            if label == 0:  # 跳过背景
                continue
                
            binary_mask = (mask == label).astype(np.uint8)
            
            # 连通组件分析
            num_labels, labeled_mask = cv2.connectedComponents(binary_mask)
            num_components = num_labels - 1  # 减去背景
            
            # 计算各连通组件的大小
            component_sizes = []
            for comp_label in range(1, num_labels):
                comp_size = np.sum(labeled_mask == comp_label)
                component_sizes.append(comp_size)
            
            connectivity_stats[int(label)] = {
                'num_components': num_components,
                'component_sizes': component_sizes,
                'largest_component': max(component_sizes) if component_sizes else 0,
                'fragmentation': num_components / max(1, np.sum(binary_mask))
            }
        
        return connectivity_stats
    
    def _analyze_2d_attention_mask(self, mask: np.ndarray, tokens: Optional[List[str]]) -> Dict[str, Any]:
        """分析2D attention mask"""
        seq_len = mask.shape[0]
        
        # 计算attention模式
        attention_density = np.mean(mask)
        diagonal_attention = np.mean(np.diag(mask))
        
        # 分析attention范围
        attention_spans = []
        for i in range(seq_len):
            span = np.sum(mask[i, :])
            attention_spans.append(span)
        
        return {
            'attention_density': float(attention_density),
            'diagonal_attention': float(diagonal_attention),
            'mean_attention_span': float(np.mean(attention_spans)),
            'attention_span_std': float(np.std(attention_spans)),
            'max_attention_span': float(np.max(attention_spans)),
            'min_attention_span': float(np.min(attention_spans))
        }
    
    def _analyze_1d_attention_mask(self, mask: np.ndarray, tokens: Optional[List[str]]) -> Dict[str, Any]:
        """分析1D attention mask（padding mask）"""
        valid_tokens = np.sum(mask)
        padding_tokens = len(mask) - valid_tokens
        
        # 找到padding开始位置
        padding_start = len(mask)
        for i in range(len(mask)):
            if mask[i] == 0:
                padding_start = i
                break
        
        return {
            'valid_tokens': int(valid_tokens),
            'padding_tokens': int(padding_tokens),
            'padding_ratio': float(padding_tokens / len(mask)),
            'effective_length': int(valid_tokens),
            'padding_start': padding_start
        }
    
    def _analyze_alignment_patterns(self, 
                                  similarity_matrix: np.ndarray,
                                  threshold: float,
                                  image_mask: Optional[torch.Tensor],
                                  text_mask: Optional[torch.Tensor]) -> Dict[str, Any]:
        """分析跨模态对齐模式"""
        # 计算强对齐区域
        strong_alignment = similarity_matrix > threshold
        alignment_density = np.mean(strong_alignment)
        
        # 计算每个图像区域的最佳文本匹配
        best_text_matches = np.argmax(similarity_matrix, axis=1)
        best_similarities = np.max(similarity_matrix, axis=1)
        
        # 计算每个文本token的最佳图像匹配
        best_image_matches = np.argmax(similarity_matrix, axis=0)
        best_image_similarities = np.max(similarity_matrix, axis=0)
        
        return {
            'alignment_density': float(alignment_density),
            'mean_max_similarity': float(np.mean(best_similarities)),
            'alignment_concentration': float(np.std(best_similarities)),
            'bidirectional_consistency': self._calculate_bidirectional_consistency(
                best_text_matches, best_image_matches, similarity_matrix.shape
            )
        }
    
    def _calculate_bidirectional_consistency(self, 
                                           text_matches: np.ndarray,
                                           image_matches: np.ndarray,
                                           shape: Tuple[int, int]) -> float:
        """计算双向对齐一致性"""
        consistent_matches = 0
        total_matches = 0
        
        for i in range(shape[0]):  # 图像区域
            best_text = text_matches[i]
            if image_matches[best_text] == i:
                consistent_matches += 1
            total_matches += 1
        
        return consistent_matches / total_matches if total_matches > 0 else 0.0
    
    def _get_bbox(self, binary_mask: np.ndarray) -> List[int]:
        """获取二值mask的边界框"""
        rows = np.any(binary_mask, axis=1)
        cols = np.any(binary_mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]
        
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        return [int(cmin), int(rmin), int(cmax), int(rmax)]
    
    def _calculate_iou(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """计算IoU"""
        intersection = np.logical_and(pred, gt)
        union = np.logical_or(pred, gt)
        
        if np.sum(union) == 0:
            return 1.0 if np.sum(pred) == 0 else 0.0
        
        return np.sum(intersection) / np.sum(union)
    
    def _calculate_dice(self, pred: np.ndarray, gt: np.ndarray) -> float:
        """计算Dice系数"""
        intersection = np.logical_and(pred, gt)
        
        if np.sum(pred) + np.sum(gt) == 0:
            return 1.0
        
        return 2.0 * np.sum(intersection) / (np.sum(pred) + np.sum(gt))
    
    def _visualize_image_mask(self,
                             mask: np.ndarray,
                             image: Optional[np.ndarray],
                             mask_type: str,
                             class_names: Optional[List[str]],
                             unique_labels: np.ndarray,
                             save_path: Optional[Path]):
        """可视化图像mask"""
        num_classes = len(unique_labels)
        
        if image is not None:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # 原始图像
            if isinstance(image, torch.Tensor):
                image_np = image.cpu().numpy()
            else:
                image_np = image
            
            if image_np.ndim == 3 and image_np.shape[0] in [1, 3]:
                image_np = np.transpose(image_np, (1, 2, 0))
            
            axes[0].imshow(image_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Mask
            im1 = axes[1].imshow(mask, cmap='tab20', vmin=0, vmax=max(19, num_classes-1))
            axes[1].set_title(f'{mask_type.title()} Mask')
            axes[1].axis('off')
            
            # 叠加显示
            axes[2].imshow(image_np)
            axes[2].imshow(mask, cmap='tab20', alpha=0.6, vmin=0, vmax=max(19, num_classes-1))
            axes[2].set_title('Overlay')
            axes[2].axis('off')
            
        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Mask
            im1 = axes[0].imshow(mask, cmap='tab20', vmin=0, vmax=max(19, num_classes-1))
            axes[0].set_title(f'{mask_type.title()} Mask')
            axes[0].axis('off')
            
            # 类别分布
            unique, counts = np.unique(mask, return_counts=True)
            axes[1].bar(unique, counts)
            axes[1].set_title('Class Distribution')
            axes[1].set_xlabel('Class Label')
            axes[1].set_ylabel('Pixel Count')
            
            if class_names and len(class_names) >= len(unique):
                axes[1].set_xticks(unique)
                axes[1].set_xticklabels([class_names[i] if i < len(class_names) else f'Class {i}' 
                                       for i in unique], rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Image mask visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_text_mask(self,
                            mask: np.ndarray,
                            tokens: Optional[List[str]],
                            mask_type: str,
                            save_path: Optional[Path]):
        """可视化文本mask"""
        if mask.ndim == 2:
            # 2D attention mask
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 热图
            im = axes[0].imshow(mask, cmap='Blues', aspect='auto')
            axes[0].set_title(f'{mask_type.title()} Attention Mask')
            axes[0].set_xlabel('Token Position')
            axes[0].set_ylabel('Token Position')
            plt.colorbar(im, ax=axes[0])
            
            if tokens:
                # 设置token标签（如果数量不太多）
                if len(tokens) <= 20:
                    axes[0].set_xticks(range(len(tokens)))
                    axes[0].set_yticks(range(len(tokens)))
                    axes[0].set_xticklabels(tokens, rotation=45, ha='right')
                    axes[0].set_yticklabels(tokens)
            
            # 注意力分布统计
            attention_sums = np.sum(mask, axis=1)
            axes[1].bar(range(len(attention_sums)), attention_sums)
            axes[1].set_title('Attention Distribution per Token')
            axes[1].set_xlabel('Token Position')
            axes[1].set_ylabel('Total Attention')
            
            if tokens and len(tokens) <= 20:
                axes[1].set_xticks(range(len(tokens)))
                axes[1].set_xticklabels(tokens, rotation=45, ha='right')
        
        else:
            # 1D padding mask
            fig, axes = plt.subplots(2, 1, figsize=(12, 8))
            
            # Mask可视化
            axes[0].imshow(mask.reshape(1, -1), cmap='RdYlBu', aspect='auto')
            axes[0].set_title(f'{mask_type.title()} Mask (1=Valid, 0=Padding)')
            axes[0].set_xlabel('Token Position')
            axes[0].set_yticks([])
            
            if tokens:
                if len(tokens) <= 30:
                    axes[0].set_xticks(range(len(tokens)))
                    axes[0].set_xticklabels(tokens, rotation=45, ha='right')
            
            # 有效长度统计
            valid_positions = np.where(mask == 1)[0]
            padding_positions = np.where(mask == 0)[0]
            
            axes[1].bar(['Valid Tokens', 'Padding Tokens'], 
                       [len(valid_positions), len(padding_positions)],
                       color=['green', 'red'], alpha=0.7)
            axes[1].set_title('Token Statistics')
            axes[1].set_ylabel('Count')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Text mask visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_cross_modal_alignment(self,
                                       similarity_matrix: np.ndarray,
                                       image_mask: Optional[torch.Tensor],
                                       text_mask: Optional[torch.Tensor],
                                       save_path: Optional[Path]):
        """可视化跨模态对齐"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 相似度矩阵热图
        im1 = axes[0, 0].imshow(similarity_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title('Cross-Modal Similarity Matrix')
        axes[0, 0].set_xlabel('Text Token Index')
        axes[0, 0].set_ylabel('Image Region Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 最佳匹配可视化
        best_matches = np.argmax(similarity_matrix, axis=1)
        best_similarities = np.max(similarity_matrix, axis=1)
        
        axes[0, 1].scatter(best_matches, range(len(best_matches)), 
                          c=best_similarities, cmap='viridis', alpha=0.7)
        axes[0, 1].set_title('Best Text Match for Each Image Region')
        axes[0, 1].set_xlabel('Best Matching Text Token')
        axes[0, 1].set_ylabel('Image Region Index')
        
        # 3. 相似度分布
        axes[1, 0].hist(similarity_matrix.flatten(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Similarity Score Distribution')
        axes[1, 0].set_xlabel('Similarity Score')
        axes[1, 0].set_ylabel('Frequency')
        
        # 4. 对齐强度热图
        alignment_strength = np.mean(similarity_matrix, axis=1)
        text_alignment_strength = np.mean(similarity_matrix, axis=0)
        
        x = np.arange(len(text_alignment_strength))
        y = np.arange(len(alignment_strength))
        
        axes[1, 1].plot(x, text_alignment_strength, 'b-', label='Text Alignment', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(y, alignment_strength, 'r-', label='Image Alignment', alpha=0.7)
        
        axes[1, 1].set_xlabel('Token/Region Index')
        axes[1, 1].set_ylabel('Text Alignment Strength', color='b')
        ax2.set_ylabel('Image Alignment Strength', color='r')
        axes[1, 1].set_title('Alignment Strength by Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Cross-modal alignment visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       save_path: Optional[Path] = None) -> str:
        """
        生成分析报告
        
        Args:
            analysis_results: 分析结果字典
            save_path: 报告保存路径
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "# Mask Analysis Report",
            f"Generated at: {datetime.now()}",
            "",
            "## Analysis Summary"
        ]
        
        # 添加各种分析结果
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                report_lines.append(f"\n### {key.replace('_', ' ').title()}")
                for sub_key, sub_value in value.items():
                    report_lines.append(f"- {sub_key}: {sub_value}")
            else:
                report_lines.append(f"- {key}: {value}")
        
        report_content = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            self.logger.info(f"Report saved to {save_path}")
        
        return report_content


# 便捷函数
def analyze_image_mask(mask: Union[torch.Tensor, np.ndarray],
                      image: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      mask_type: str = 'semantic',
                      **kwargs) -> Dict[str, Any]:
    """
    便捷函数：分析图像mask
    
    Args:
        mask: 分割mask
        image: 原始图像（可选）
        mask_type: mask类型
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    analyzer = MaskAnalyzer(**kwargs)
    return analyzer.analyze_image_mask(mask, image, mask_type)


def analyze_text_mask(attention_mask: Union[torch.Tensor, np.ndarray],
                     tokens: Optional[List[str]] = None,
                     mask_type: str = 'padding',
                     **kwargs) -> Dict[str, Any]:
    """
    便捷函数：分析文本mask
    
    Args:
        attention_mask: 注意力mask
        tokens: token列表（可选）
        mask_type: mask类型
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 分析结果
    """
    analyzer = MaskAnalyzer(**kwargs)
    return analyzer.analyze_text_mask(attention_mask, tokens, mask_type)


# 使用示例
if __name__ == "__main__":
    # 示例1: 图像分割mask分析
    image_mask = torch.randint(0, 5, (256, 256))  # 5类分割
    image = torch.randn(3, 256, 256)
    
    mask_analyzer = MaskAnalyzer()
    
    # 分析图像mask
    image_results = mask_analyzer.analyze_image_mask(
        mask=image_mask,
        image=image,
        mask_type='semantic',
        class_names=['background', 'person', 'car', 'tree', 'building']
    )
    
    print("Image mask analysis results:")
    print(json.dumps(image_results, indent=2, default=str))
    
    # 示例2: 文本attention mask分析
    text_mask = torch.ones(20)
    text_mask[15:] = 0  # padding
    tokens = [f"token_{i}" for i in range(20)]
    
    text_results = mask_analyzer.analyze_text_mask(
        attention_mask=text_mask,
        tokens=tokens,
        mask_type='padding'
    )
    
    print("\nText mask analysis results:")
    print(json.dumps(text_results, indent=2, default=str))
    
    # 示例3: 跨模态对齐分析
    image_features = torch.randn(64, 512)  # 8x8图像区域，512维特征
    text_features = torch.randn(20, 512)   # 20个token，512维特征
    
    alignment_results = mask_analyzer.analyze_cross_modal_alignment(
        image_features=image_features,
        text_features=text_features,
        similarity_threshold=0.5
    )
    
    print("\nCross-modal alignment results:")
    print(json.dumps(alignment_results, indent=2, default=str))