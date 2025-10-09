"""
关系图分析器模块

该模块提供了跨模态关系分析功能，类似于CLIP模型的分析，支持：
- 图像-文本关系分析
- 多模态特征对齐分析
- 关系图构建和可视化
- 相似度矩阵分析
- 跨模态检索性能评估
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any, Callable
import networkx as nx
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import logging
import json
from datetime import datetime
from collections import defaultdict
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class RelationAnalyzer:
    """
    关系图分析器，专门用于跨模态关系分析
    
    功能包括：
    - 图像-文本相似度计算和分析
    - 跨模态特征空间可视化
    - 关系图构建和网络分析
    - 检索性能评估
    - 对齐质量分析
    - 多模态聚类分析
    """
    
    def __init__(self, 
                 device: str = 'auto',
                 save_dir: Optional[Path] = None,
                 logger: Optional[logging.Logger] = None):
        """
        初始化关系图分析器
        
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
        
        # 支持的分析类型
        self.analysis_types = {
            'similarity': ['cosine', 'euclidean', 'dot_product'],
            'alignment': ['linear', 'canonical', 'procrustes'],
            'clustering': ['kmeans', 'hierarchical', 'spectral'],
            'retrieval': ['top_k', 'reciprocal', 'cross_modal']
        }
        
        self.logger.info(f"RelationAnalyzer initialized on device: {self.device}")
    
    def _setup_logger(self) -> logging.Logger:
        """设置日志记录器"""
        logger = logging.getLogger('RelationAnalyzer')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def analyze_cross_modal_similarity(self,
                                     image_features: torch.Tensor,
                                     text_features: torch.Tensor,
                                     image_labels: Optional[List[str]] = None,
                                     text_labels: Optional[List[str]] = None,
                                     similarity_metric: str = 'cosine',
                                     save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析跨模态相似度
        
        Args:
            image_features: 图像特征 [N_img, D]
            text_features: 文本特征 [N_text, D]
            image_labels: 图像标签列表
            text_labels: 文本标签列表
            similarity_metric: 相似度度量方式
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 相似度分析结果
        """
        # 确保特征在同一设备上
        image_features = image_features.to(self.device)
        text_features = text_features.to(self.device)
        
        # 特征归一化
        image_features_norm = F.normalize(image_features, dim=-1)
        text_features_norm = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵
        if similarity_metric == 'cosine':
            similarity_matrix = torch.matmul(image_features_norm, text_features_norm.T)
        elif similarity_metric == 'euclidean':
            # 计算欧氏距离（转换为相似度）
            distances = torch.cdist(image_features, text_features)
            similarity_matrix = 1.0 / (1.0 + distances)
        elif similarity_metric == 'dot_product':
            similarity_matrix = torch.matmul(image_features, text_features.T)
        else:
            raise ValueError(f"Unsupported similarity metric: {similarity_metric}")
        
        similarity_np = similarity_matrix.cpu().numpy()
        
        # 基本统计分析
        analysis_results = {
            'similarity_metric': similarity_metric,
            'matrix_shape': similarity_np.shape,
            'num_image_samples': similarity_np.shape[0],
            'num_text_samples': similarity_np.shape[1],
            'statistics': {
                'mean': float(np.mean(similarity_np)),
                'std': float(np.std(similarity_np)),
                'max': float(np.max(similarity_np)),
                'min': float(np.min(similarity_np)),
                'median': float(np.median(similarity_np))
            }
        }
        
        # 最佳匹配分析
        best_matches = self._analyze_best_matches(similarity_np, image_labels, text_labels)
        analysis_results['best_matches'] = best_matches
        
        # 相似度分布分析
        distribution_stats = self._analyze_similarity_distribution(similarity_np)
        analysis_results['distribution'] = distribution_stats
        
        # 对称性分析（如果矩阵是方形的）
        if similarity_np.shape[0] == similarity_np.shape[1]:
            symmetry_stats = self._analyze_matrix_symmetry(similarity_np)
            analysis_results['symmetry'] = symmetry_stats
        
        # 可视化相似度矩阵
        self._visualize_similarity_matrix(
            similarity_np, image_labels, text_labels, similarity_metric, save_path
        )
        
        return analysis_results
    
    def analyze_feature_alignment(self,
                                image_features: torch.Tensor,
                                text_features: torch.Tensor,
                                alignment_method: str = 'canonical',
                                n_components: int = 2,
                                save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        分析特征空间对齐
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            alignment_method: 对齐方法
            n_components: 降维后的维度
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 对齐分析结果
        """
        image_np = image_features.cpu().numpy()
        text_np = text_features.cpu().numpy()
        
        analysis_results = {
            'alignment_method': alignment_method,
            'original_dims': {
                'image': image_np.shape[1],
                'text': text_np.shape[1]
            },
            'n_components': n_components
        }
        
        if alignment_method == 'canonical':
            # 典型相关分析
            alignment_stats = self._canonical_correlation_analysis(
                image_np, text_np, n_components
            )
        elif alignment_method == 'linear':
            # 线性对齐
            alignment_stats = self._linear_alignment_analysis(
                image_np, text_np, n_components
            )
        elif alignment_method == 'procrustes':
            # Procrustes分析
            alignment_stats = self._procrustes_analysis(
                image_np, text_np
            )
        else:
            raise ValueError(f"Unsupported alignment method: {alignment_method}")
        
        analysis_results.update(alignment_stats)
        
        # 可视化对齐结果
        self._visualize_feature_alignment(
            image_np, text_np, alignment_stats, alignment_method, save_path
        )
        
        return analysis_results
    
    def build_relation_graph(self,
                           image_features: torch.Tensor,
                           text_features: torch.Tensor,
                           image_labels: Optional[List[str]] = None,
                           text_labels: Optional[List[str]] = None,
                           threshold: float = 0.5,
                           save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        构建跨模态关系图
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            image_labels: 图像标签
            text_labels: 文本标签
            threshold: 相似度阈值
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 关系图分析结果
        """
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(
            F.normalize(image_features, dim=-1),
            F.normalize(text_features, dim=-1).T
        ).cpu().numpy()
        
        # 创建关系图
        G = nx.Graph()
        
        # 添加节点
        num_images = len(image_features)
        num_texts = len(text_features)
        
        # 图像节点
        for i in range(num_images):
            label = image_labels[i] if image_labels else f"img_{i}"
            G.add_node(f"img_{i}", type='image', label=label, index=i)
        
        # 文本节点
        for j in range(num_texts):
            label = text_labels[j] if text_labels else f"text_{j}"
            G.add_node(f"text_{j}", type='text', label=label, index=j)
        
        # 添加边（基于相似度阈值）
        edges_added = 0
        for i in range(num_images):
            for j in range(num_texts):
                similarity = similarity_matrix[i, j]
                if similarity > threshold:
                    G.add_edge(f"img_{i}", f"text_{j}", weight=similarity)
                    edges_added += 1
        
        # 图分析
        graph_stats = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_image_nodes': num_images,
            'num_text_nodes': num_texts,
            'edge_density': edges_added / (num_images * num_texts),
            'threshold': threshold
        }
        
        # 网络属性分析
        if G.number_of_edges() > 0:
            # 连通性分析
            connected_components = list(nx.connected_components(G))
            graph_stats['num_connected_components'] = len(connected_components)
            graph_stats['largest_component_size'] = max(len(cc) for cc in connected_components)
            
            # 中心性分析
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            
            graph_stats['centrality'] = {
                'max_degree': max(degree_centrality.values()),
                'mean_degree': np.mean(list(degree_centrality.values())),
                'max_betweenness': max(betweenness_centrality.values()),
                'mean_betweenness': np.mean(list(betweenness_centrality.values()))
            }
        
        # 可视化关系图
        self._visualize_relation_graph(
            G, similarity_matrix, image_labels, text_labels, save_path
        )
        
        analysis_results = {
            'graph_statistics': graph_stats,
            'similarity_matrix': similarity_matrix,
            'graph': G  # 保存图对象供后续分析
        }
        
        return analysis_results
    
    def evaluate_retrieval_performance(self,
                                     image_features: torch.Tensor,
                                     text_features: torch.Tensor,
                                     ground_truth_pairs: List[Tuple[int, int]],
                                     k_values: List[int] = [1, 5, 10],
                                     save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        评估跨模态检索性能
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            ground_truth_pairs: 真实配对 [(img_idx, text_idx), ...]
            k_values: Top-K值列表
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 检索性能评估结果
        """
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(
            F.normalize(image_features, dim=-1),
            F.normalize(text_features, dim=-1).T
        ).cpu().numpy()
        
        # 构建真实标签矩阵
        gt_matrix = np.zeros_like(similarity_matrix)
        for img_idx, text_idx in ground_truth_pairs:
            gt_matrix[img_idx, text_idx] = 1
        
        # 图像到文本检索
        i2t_results = self._evaluate_image_to_text_retrieval(
            similarity_matrix, gt_matrix, k_values
        )
        
        # 文本到图像检索
        t2i_results = self._evaluate_text_to_image_retrieval(
            similarity_matrix, gt_matrix, k_values
        )
        
        # 综合性能指标
        overall_results = {
            'image_to_text': i2t_results,
            'text_to_image': t2i_results,
            'average_recall': {}
        }
        
        # 计算平均召回率
        for k in k_values:
            avg_recall = (i2t_results[f'recall@{k}'] + t2i_results[f'recall@{k}']) / 2
            overall_results['average_recall'][f'recall@{k}'] = avg_recall
        
        # 可视化检索性能
        self._visualize_retrieval_performance(
            overall_results, k_values, save_path
        )
        
        return overall_results
    
    def analyze_multimodal_clustering(self,
                                    image_features: torch.Tensor,
                                    text_features: torch.Tensor,
                                    n_clusters: int = 5,
                                    clustering_method: str = 'kmeans',
                                    save_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        多模态聚类分析
        
        Args:
            image_features: 图像特征
            text_features: 文本特征
            n_clusters: 聚类数量
            clustering_method: 聚类方法
            save_path: 保存路径
            
        Returns:
            Dict[str, Any]: 聚类分析结果
        """
        # 特征降维用于可视化
        image_np = image_features.cpu().numpy()
        text_np = text_features.cpu().numpy()
        
        # 合并特征进行联合聚类
        combined_features = np.vstack([image_np, text_np])
        
        # 执行聚类
        if clustering_method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(combined_features)
        else:
            raise ValueError(f"Unsupported clustering method: {clustering_method}")
        
        # 分离图像和文本的聚类标签
        image_clusters = cluster_labels[:len(image_np)]
        text_clusters = cluster_labels[len(image_np):]
        
        # 聚类质量分析
        clustering_stats = self._analyze_clustering_quality(
            combined_features, cluster_labels, n_clusters
        )
        
        # 跨模态聚类一致性分析
        consistency_stats = self._analyze_cross_modal_consistency(
            image_clusters, text_clusters, image_features, text_features
        )
        
        analysis_results = {
            'clustering_method': clustering_method,
            'n_clusters': n_clusters,
            'image_clusters': image_clusters.tolist(),
            'text_clusters': text_clusters.tolist(),
            'clustering_quality': clustering_stats,
            'cross_modal_consistency': consistency_stats
        }
        
        # 可视化聚类结果
        self._visualize_multimodal_clustering(
            image_np, text_np, image_clusters, text_clusters, save_path
        )
        
        return analysis_results
    
    def _analyze_best_matches(self, 
                            similarity_matrix: np.ndarray,
                            image_labels: Optional[List[str]],
                            text_labels: Optional[List[str]]) -> Dict[str, Any]:
        """分析最佳匹配"""
        # 图像到文本的最佳匹配
        i2t_best = np.argmax(similarity_matrix, axis=1)
        i2t_scores = np.max(similarity_matrix, axis=1)
        
        # 文本到图像的最佳匹配
        t2i_best = np.argmax(similarity_matrix, axis=0)
        t2i_scores = np.max(similarity_matrix, axis=0)
        
        # 互相最佳匹配（双向一致）
        mutual_matches = []
        for i in range(len(i2t_best)):
            j = i2t_best[i]
            if t2i_best[j] == i:
                mutual_matches.append((i, j, similarity_matrix[i, j]))
        
        return {
            'image_to_text_matches': {
                'indices': i2t_best.tolist(),
                'scores': i2t_scores.tolist(),
                'mean_score': float(np.mean(i2t_scores)),
                'std_score': float(np.std(i2t_scores))
            },
            'text_to_image_matches': {
                'indices': t2i_best.tolist(),
                'scores': t2i_scores.tolist(),
                'mean_score': float(np.mean(t2i_scores)),
                'std_score': float(np.std(t2i_scores))
            },
            'mutual_matches': {
                'count': len(mutual_matches),
                'pairs': mutual_matches,
                'ratio': len(mutual_matches) / min(similarity_matrix.shape)
            }
        }
    
    def _analyze_similarity_distribution(self, similarity_matrix: np.ndarray) -> Dict[str, Any]:
        """分析相似度分布"""
        flat_similarities = similarity_matrix.flatten()
        
        # 计算分位数
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_values = np.percentile(flat_similarities, percentiles)
        
        return {
            'percentiles': dict(zip(percentiles, percentile_values.tolist())),
            'histogram': {
                'counts': np.histogram(flat_similarities, bins=20)[0].tolist(),
                'bin_edges': np.histogram(flat_similarities, bins=20)[1].tolist()
            },
            'skewness': float(self._calculate_skewness(flat_similarities)),
            'kurtosis': float(self._calculate_kurtosis(flat_similarities))
        }
    
    def _analyze_matrix_symmetry(self, matrix: np.ndarray) -> Dict[str, Any]:
        """分析矩阵对称性"""
        # 计算对称性指标
        symmetry_error = np.mean(np.abs(matrix - matrix.T))
        frobenius_norm = np.linalg.norm(matrix - matrix.T, 'fro')
        
        return {
            'symmetry_error': float(symmetry_error),
            'frobenius_norm': float(frobenius_norm),
            'is_symmetric': bool(symmetry_error < 1e-6)
        }
    
    def _canonical_correlation_analysis(self, 
                                      X: np.ndarray, 
                                      Y: np.ndarray, 
                                      n_components: int) -> Dict[str, Any]:
        """典型相关分析"""
        from sklearn.cross_decomposition import CCA
        
        cca = CCA(n_components=n_components)
        X_c, Y_c = cca.fit_transform(X, Y)
        
        # 计算相关系数
        correlations = []
        for i in range(n_components):
            corr = np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1]
            correlations.append(corr)
        
        return {
            'canonical_correlations': correlations,
            'mean_correlation': float(np.mean(correlations)),
            'transformed_features': {
                'image': X_c,
                'text': Y_c
            }
        }
    
    def _linear_alignment_analysis(self, 
                                 X: np.ndarray, 
                                 Y: np.ndarray, 
                                 n_components: int) -> Dict[str, Any]:
        """线性对齐分析"""
        # 使用PCA降维
        pca_x = PCA(n_components=n_components)
        pca_y = PCA(n_components=n_components)
        
        X_reduced = pca_x.fit_transform(X)
        Y_reduced = pca_y.fit_transform(Y)
        
        # 计算对齐质量
        alignment_score = np.mean([
            np.corrcoef(X_reduced[:, i], Y_reduced[:, i])[0, 1] 
            for i in range(n_components)
        ])
        
        return {
            'alignment_score': float(alignment_score),
            'explained_variance': {
                'image': pca_x.explained_variance_ratio_.tolist(),
                'text': pca_y.explained_variance_ratio_.tolist()
            },
            'transformed_features': {
                'image': X_reduced,
                'text': Y_reduced
            }
        }
    
    def _procrustes_analysis(self, X: np.ndarray, Y: np.ndarray) -> Dict[str, Any]:
        """Procrustes分析"""
        from scipy.spatial.distance import procrustes
        
        # 确保矩阵大小一致
        min_samples = min(X.shape[0], Y.shape[0])
        X_proc = X[:min_samples]
        Y_proc = Y[:min_samples]
        
        # 执行Procrustes分析
        mtx1, mtx2, disparity = procrustes(X_proc, Y_proc)
        
        return {
            'disparity': float(disparity),
            'alignment_quality': float(1.0 - disparity),
            'transformed_features': {
                'image': mtx1,
                'text': mtx2
            }
        }
    
    def _evaluate_image_to_text_retrieval(self, 
                                        similarity_matrix: np.ndarray,
                                        gt_matrix: np.ndarray,
                                        k_values: List[int]) -> Dict[str, float]:
        """评估图像到文本检索"""
        results = {}
        
        for k in k_values:
            correct = 0
            total = similarity_matrix.shape[0]
            
            for i in range(total):
                # 获取top-k文本
                top_k_indices = np.argsort(similarity_matrix[i])[-k:]
                
                # 检查是否有正确匹配
                if np.any(gt_matrix[i, top_k_indices] == 1):
                    correct += 1
            
            results[f'recall@{k}'] = correct / total
        
        return results
    
    def _evaluate_text_to_image_retrieval(self, 
                                        similarity_matrix: np.ndarray,
                                        gt_matrix: np.ndarray,
                                        k_values: List[int]) -> Dict[str, float]:
        """评估文本到图像检索"""
        results = {}
        
        for k in k_values:
            correct = 0
            total = similarity_matrix.shape[1]
            
            for j in range(total):
                # 获取top-k图像
                top_k_indices = np.argsort(similarity_matrix[:, j])[-k:]
                
                # 检查是否有正确匹配
                if np.any(gt_matrix[top_k_indices, j] == 1):
                    correct += 1
            
            results[f'recall@{k}'] = correct / total
        
        return results
    
    def _analyze_clustering_quality(self, 
                                  features: np.ndarray,
                                  labels: np.ndarray,
                                  n_clusters: int) -> Dict[str, Any]:
        """分析聚类质量"""
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        # 计算聚类质量指标
        silhouette = silhouette_score(features, labels)
        calinski_harabasz = calinski_harabasz_score(features, labels)
        davies_bouldin = davies_bouldin_score(features, labels)
        
        # 计算类内和类间距离
        intra_cluster_distances = []
        inter_cluster_distances = []
        
        for i in range(n_clusters):
            cluster_points = features[labels == i]
            if len(cluster_points) > 1:
                # 类内距离
                intra_dist = np.mean([
                    np.linalg.norm(cluster_points[j] - cluster_points[k])
                    for j in range(len(cluster_points))
                    for k in range(j+1, len(cluster_points))
                ])
                intra_cluster_distances.append(intra_dist)
                
                # 类间距离
                for other_i in range(i+1, n_clusters):
                    other_cluster_points = features[labels == other_i]
                    if len(other_cluster_points) > 0:
                        inter_dist = np.mean([
                            np.linalg.norm(p1 - p2)
                            for p1 in cluster_points
                            for p2 in other_cluster_points
                        ])
                        inter_cluster_distances.append(inter_dist)
        
        return {
            'silhouette_score': float(silhouette),
            'calinski_harabasz_score': float(calinski_harabasz),
            'davies_bouldin_score': float(davies_bouldin),
            'mean_intra_cluster_distance': float(np.mean(intra_cluster_distances)) if intra_cluster_distances else 0.0,
            'mean_inter_cluster_distance': float(np.mean(inter_cluster_distances)) if inter_cluster_distances else 0.0
        }
    
    def _analyze_cross_modal_consistency(self, 
                                       image_clusters: np.ndarray,
                                       text_clusters: np.ndarray,
                                       image_features: torch.Tensor,
                                       text_features: torch.Tensor) -> Dict[str, Any]:
        """分析跨模态聚类一致性"""
        # 计算聚类标签的一致性
        consistency_matrix = np.zeros((len(np.unique(image_clusters)), len(np.unique(text_clusters))))
        
        # 假设图像和文本是配对的
        min_len = min(len(image_clusters), len(text_clusters))
        
        for i in range(min_len):
            img_cluster = image_clusters[i]
            text_cluster = text_clusters[i]
            consistency_matrix[img_cluster, text_cluster] += 1
        
        # 计算一致性指标
        total_pairs = min_len
        consistent_pairs = np.sum(np.diag(consistency_matrix)) if consistency_matrix.shape[0] == consistency_matrix.shape[1] else 0
        consistency_ratio = consistent_pairs / total_pairs if total_pairs > 0 else 0
        
        return {
            'consistency_matrix': consistency_matrix.tolist(),
            'consistency_ratio': float(consistency_ratio),
            'total_pairs': total_pairs,
            'consistent_pairs': int(consistent_pairs)
        }
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """计算偏度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """计算峰度"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _visualize_similarity_matrix(self,
                                   similarity_matrix: np.ndarray,
                                   image_labels: Optional[List[str]],
                                   text_labels: Optional[List[str]],
                                   metric: str,
                                   save_path: Optional[Path]):
        """可视化相似度矩阵"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 相似度矩阵热图
        im1 = axes[0, 0].imshow(similarity_matrix, cmap='viridis', aspect='auto')
        axes[0, 0].set_title(f'{metric.title()} Similarity Matrix')
        axes[0, 0].set_xlabel('Text Index')
        axes[0, 0].set_ylabel('Image Index')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # 2. 相似度分布直方图
        axes[0, 1].hist(similarity_matrix.flatten(), bins=50, alpha=0.7)
        axes[0, 1].set_title('Similarity Score Distribution')
        axes[0, 1].set_xlabel('Similarity Score')
        axes[0, 1].set_ylabel('Frequency')
        
        # 3. 最佳匹配可视化
        best_matches_i2t = np.argmax(similarity_matrix, axis=1)
        best_scores_i2t = np.max(similarity_matrix, axis=1)
        
        axes[1, 0].scatter(best_matches_i2t, range(len(best_matches_i2t)), 
                          c=best_scores_i2t, cmap='viridis', alpha=0.7)
        axes[1, 0].set_title('Best Text Match for Each Image')
        axes[1, 0].set_xlabel('Best Matching Text Index')
        axes[1, 0].set_ylabel('Image Index')
        
        # 4. 相似度统计
        mean_sim_per_image = np.mean(similarity_matrix, axis=1)
        mean_sim_per_text = np.mean(similarity_matrix, axis=0)
        
        x1 = np.arange(len(mean_sim_per_image))
        x2 = np.arange(len(mean_sim_per_text))
        
        axes[1, 1].plot(x1, mean_sim_per_image, 'b-', label='Image Mean Similarity', alpha=0.7)
        ax2 = axes[1, 1].twinx()
        ax2.plot(x2, mean_sim_per_text, 'r-', label='Text Mean Similarity', alpha=0.7)
        
        axes[1, 1].set_xlabel('Index')
        axes[1, 1].set_ylabel('Mean Similarity (Image)', color='b')
        ax2.set_ylabel('Mean Similarity (Text)', color='r')
        axes[1, 1].set_title('Mean Similarity by Position')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Similarity matrix visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_feature_alignment(self,
                                   image_features: np.ndarray,
                                   text_features: np.ndarray,
                                   alignment_stats: Dict[str, Any],
                                   method: str,
                                   save_path: Optional[Path]):
        """可视化特征对齐"""
        if 'transformed_features' in alignment_stats:
            img_transformed = alignment_stats['transformed_features']['image']
            text_transformed = alignment_stats['transformed_features']['text']
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # 原始特征空间（PCA降维到2D用于可视化）
            if image_features.shape[1] > 2:
                pca = PCA(n_components=2)
                img_2d = pca.fit_transform(image_features)
                text_2d = pca.fit_transform(text_features)
            else:
                img_2d = image_features
                text_2d = text_features
            
            axes[0].scatter(img_2d[:, 0], img_2d[:, 1], alpha=0.6, label='Image', s=50)
            axes[0].scatter(text_2d[:, 0], text_2d[:, 1], alpha=0.6, label='Text', s=50)
            axes[0].set_title('Original Feature Space')
            axes[0].set_xlabel('PC1')
            axes[0].set_ylabel('PC2')
            axes[0].legend()
            
            # 对齐后的特征空间
            if img_transformed.shape[1] >= 2:
                axes[1].scatter(img_transformed[:, 0], img_transformed[:, 1], 
                              alpha=0.6, label='Image (Aligned)', s=50)
                axes[1].scatter(text_transformed[:, 0], text_transformed[:, 1], 
                              alpha=0.6, label='Text (Aligned)', s=50)
                axes[1].set_title(f'Aligned Feature Space ({method})')
                axes[1].set_xlabel('Component 1')
                axes[1].set_ylabel('Component 2')
                axes[1].legend()
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature alignment visualization saved to {save_path}")
            
            plt.show()
    
    def _visualize_relation_graph(self,
                                G: nx.Graph,
                                similarity_matrix: np.ndarray,
                                image_labels: Optional[List[str]],
                                text_labels: Optional[List[str]],
                                save_path: Optional[Path]):
        """可视化关系图"""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 1. 网络图可视化
        pos = nx.spring_layout(G, k=1, iterations=50)
        
        # 分离图像和文本节点
        image_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'image']
        text_nodes = [n for n in G.nodes() if G.nodes[n]['type'] == 'text']
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, nodelist=image_nodes, 
                              node_color='lightblue', node_size=300, 
                              alpha=0.8, ax=axes[0], label='Image')
        nx.draw_networkx_nodes(G, pos, nodelist=text_nodes, 
                              node_color='lightcoral', node_size=300, 
                              alpha=0.8, ax=axes[0], label='Text')
        
        # 绘制边
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(G, pos, edgelist=edges, width=weights, 
                              alpha=0.6, ax=axes[0])
        
        # 添加标签
        labels = {n: G.nodes[n]['label'][:10] + '...' if len(G.nodes[n]['label']) > 10 
                 else G.nodes[n]['label'] for n in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=axes[0])
        
        axes[0].set_title('Cross-Modal Relation Graph')
        axes[0].legend()
        axes[0].axis('off')
        
        # 2. 度分布
        degrees = [G.degree(n) for n in G.nodes()]
        axes[1].hist(degrees, bins=20, alpha=0.7)
        axes[1].set_title('Node Degree Distribution')
        axes[1].set_xlabel('Degree')
        axes[1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Relation graph visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_retrieval_performance(self,
                                       results: Dict[str, Any],
                                       k_values: List[int],
                                       save_path: Optional[Path]):
        """可视化检索性能"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # 1. Recall@K曲线
        i2t_recalls = [results['image_to_text'][f'recall@{k}'] for k in k_values]
        t2i_recalls = [results['text_to_image'][f'recall@{k}'] for k in k_values]
        avg_recalls = [results['average_recall'][f'recall@{k}'] for k in k_values]
        
        axes[0].plot(k_values, i2t_recalls, 'b-o', label='Image→Text', linewidth=2)
        axes[0].plot(k_values, t2i_recalls, 'r-s', label='Text→Image', linewidth=2)
        axes[0].plot(k_values, avg_recalls, 'g-^', label='Average', linewidth=2)
        
        axes[0].set_xlabel('K')
        axes[0].set_ylabel('Recall@K')
        axes[0].set_title('Cross-Modal Retrieval Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. 性能对比柱状图
        categories = ['Image→Text', 'Text→Image', 'Average']
        recall_1 = [results['image_to_text']['recall@1'], 
                   results['text_to_image']['recall@1'],
                   results['average_recall']['recall@1']]
        recall_5 = [results['image_to_text']['recall@5'], 
                   results['text_to_image']['recall@5'],
                   results['average_recall']['recall@5']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        axes[1].bar(x - width/2, recall_1, width, label='Recall@1', alpha=0.8)
        axes[1].bar(x + width/2, recall_5, width, label='Recall@5', alpha=0.8)
        
        axes[1].set_xlabel('Retrieval Direction')
        axes[1].set_ylabel('Recall')
        axes[1].set_title('Retrieval Performance Comparison')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(categories)
        axes[1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Retrieval performance visualization saved to {save_path}")
        
        plt.show()
    
    def _visualize_multimodal_clustering(self,
                                       image_features: np.ndarray,
                                       text_features: np.ndarray,
                                       image_clusters: np.ndarray,
                                       text_clusters: np.ndarray,
                                       save_path: Optional[Path]):
        """可视化多模态聚类"""
        # 使用t-SNE降维到2D
        tsne = TSNE(n_components=2, random_state=42)
        
        # 合并特征进行降维
        combined_features = np.vstack([image_features, text_features])
        combined_2d = tsne.fit_transform(combined_features)
        
        # 分离降维后的特征
        image_2d = combined_2d[:len(image_features)]
        text_2d = combined_2d[len(image_features):]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 图像聚类
        scatter1 = axes[0].scatter(image_2d[:, 0], image_2d[:, 1], 
                                  c=image_clusters, cmap='tab10', alpha=0.7, s=50)
        axes[0].set_title('Image Clustering')
        axes[0].set_xlabel('t-SNE 1')
        axes[0].set_ylabel('t-SNE 2')
        plt.colorbar(scatter1, ax=axes[0])
        
        # 2. 文本聚类
        scatter2 = axes[1].scatter(text_2d[:, 0], text_2d[:, 1], 
                                  c=text_clusters, cmap='tab10', alpha=0.7, s=50)
        axes[1].set_title('Text Clustering')
        axes[1].set_xlabel('t-SNE 1')
        axes[1].set_ylabel('t-SNE 2')
        plt.colorbar(scatter2, ax=axes[1])
        
        # 3. 联合可视化
        # 图像用圆形，文本用方形
        for cluster in np.unique(image_clusters):
            mask = image_clusters == cluster
            axes[2].scatter(image_2d[mask, 0], image_2d[mask, 1], 
                           c=f'C{cluster}', marker='o', alpha=0.7, s=50,
                           label=f'Image Cluster {cluster}')
        
        for cluster in np.unique(text_clusters):
            mask = text_clusters == cluster
            axes[2].scatter(text_2d[mask, 0], text_2d[mask, 1], 
                           c=f'C{cluster}', marker='s', alpha=0.7, s=50,
                           label=f'Text Cluster {cluster}')
        
        axes[2].set_title('Joint Multimodal Clustering')
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Multimodal clustering visualization saved to {save_path}")
        
        plt.show()
    
    def generate_report(self, 
                       analysis_results: Dict[str, Any],
                       save_path: Optional[Path] = None) -> str:
        """
        生成关系分析报告
        
        Args:
            analysis_results: 分析结果字典
            save_path: 报告保存路径
            
        Returns:
            str: 报告内容
        """
        report_lines = [
            "# Cross-Modal Relation Analysis Report",
            f"Generated at: {datetime.now()}",
            "",
            "## Analysis Summary"
        ]
        
        # 添加各种分析结果
        for key, value in analysis_results.items():
            if isinstance(value, dict):
                report_lines.append(f"\n### {key.replace('_', ' ').title()}")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, dict):
                        report_lines.append(f"\n#### {sub_key.replace('_', ' ').title()}")
                        for sub_sub_key, sub_sub_value in sub_value.items():
                            report_lines.append(f"- {sub_sub_key}: {sub_sub_value}")
                    else:
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
def analyze_cross_modal_similarity(image_features: torch.Tensor,
                                 text_features: torch.Tensor,
                                 **kwargs) -> Dict[str, Any]:
    """
    便捷函数：分析跨模态相似度
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 相似度分析结果
    """
    analyzer = RelationAnalyzer(**kwargs)
    return analyzer.analyze_cross_modal_similarity(image_features, text_features)


def build_relation_graph(image_features: torch.Tensor,
                        text_features: torch.Tensor,
                        **kwargs) -> Dict[str, Any]:
    """
    便捷函数：构建关系图
    
    Args:
        image_features: 图像特征
        text_features: 文本特征
        **kwargs: 其他参数
        
    Returns:
        Dict[str, Any]: 关系图分析结果
    """
    analyzer = RelationAnalyzer(**kwargs)
    return analyzer.build_relation_graph(image_features, text_features)


# 使用示例
if __name__ == "__main__":
    # 示例数据
    batch_size = 32
    feature_dim = 512
    
    # 模拟CLIP风格的特征
    image_features = torch.randn(batch_size, feature_dim)
    text_features = torch.randn(batch_size, feature_dim)
    
    # 创建分析器
    relation_analyzer = RelationAnalyzer()
    
    # 1. 跨模态相似度分析
    similarity_results = relation_analyzer.analyze_cross_modal_similarity(
        image_features=image_features,
        text_features=text_features,
        similarity_metric='cosine'
    )
    
    print("Cross-modal similarity analysis:")
    print(json.dumps(similarity_results, indent=2, default=str))
    
    # 2. 构建关系图
    graph_results = relation_analyzer.build_relation_graph(
        image_features=image_features,
        text_features=text_features,
        threshold=0.3
    )
    
    print("\nRelation graph analysis:")
    print(json.dumps(graph_results['graph_statistics'], indent=2, default=str))
    
    # 3. 检索性能评估（模拟真实配对）
    ground_truth_pairs = [(i, i) for i in range(batch_size)]  # 假设图像和文本是配对的
    
    retrieval_results = relation_analyzer.evaluate_retrieval_performance(
        image_features=image_features,
        text_features=text_features,
        ground_truth_pairs=ground_truth_pairs,
        k_values=[1, 5, 10]
    )
    
    print("\nRetrieval performance:")
    print(json.dumps(retrieval_results, indent=2, default=str))
    
    # 4. 多模态聚类分析
    clustering_results = relation_analyzer.analyze_multimodal_clustering(
        image_features=image_features,
        text_features=text_features,
        n_clusters=5
    )
    
    print("\nMultimodal clustering analysis:")
    print(json.dumps(clustering_results['clustering_quality'], indent=2, default=str))