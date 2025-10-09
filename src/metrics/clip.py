"""
CLIP (Contrastive Language-Image Pre-training) 相关评估指标实现

本模块实现了CLIP模型的各种评估指标，包括：
- CLIP准确率
- 图像检索召回率@1和@5
- 文本检索召回率@1和@5  
- 图像-文本相似度
- 零样本分类准确率
"""

import numpy as np
import torch
from typing import Union, List, Tuple, Optional

def _compute_similarity_matrix(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> np.ndarray:
    """
    计算图像特征和文本特征之间的相似度矩阵
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (M, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        相似度矩阵，形状为 (N, M)
    """
    # 归一化特征向量
    image_features = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    
    # 计算余弦相似度
    similarity_matrix = np.dot(image_features, text_features.T) / temperature
    return similarity_matrix


def _compute_recall_at_k(similarity_matrix: np.ndarray, k: int, mode: str = 'i2t') -> float:
    """
    计算Recall@K指标
    
    Args:
        similarity_matrix: 相似度矩阵，形状为 (N, M)
        k: Top-K值
        mode: 模式，'i2t'表示图像到文本检索，'t2i'表示文本到图像检索
    
    Returns:
        Recall@K值
    """
    if mode == 'i2t':
        # 图像到文本检索：对每个图像，找到最相似的k个文本
        ranks = np.argsort(-similarity_matrix, axis=1)[:, :k]
        # 假设对角线元素为正确匹配（即第i个图像对应第i个文本）
        correct_matches = np.arange(similarity_matrix.shape[0])[:, np.newaxis]
        recall = np.mean(np.any(ranks == correct_matches, axis=1))
    else:  # 't2i'
        # 文本到图像检索：对每个文本，找到最相似的k个图像
        ranks = np.argsort(-similarity_matrix, axis=0)[:k, :]
        correct_matches = np.arange(similarity_matrix.shape[1])[np.newaxis, :]
        recall = np.mean(np.any(ranks == correct_matches, axis=0))
    
    return float(recall)


def clip_accuracy(image_features: np.ndarray, text_features: np.ndarray) -> float:
    """
    计算CLIP准确率
    
    CLIP准确率定义为图像-文本对的正确匹配率。
    对于每个图像，计算其与所有文本的相似度，如果与对应文本的相似度最高，则认为匹配正确。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)，与图像一一对应
    
    Returns:
        CLIP准确率 (0-1之间的浮点数)
    """
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError("图像特征和文本特征的数量必须相等")
    
    # 计算相似度矩阵
    similarity_matrix = _compute_similarity_matrix(image_features, text_features)
    
    # 计算准确率：对角线元素是否为每行的最大值
    predicted_indices = np.argmax(similarity_matrix, axis=1)
    true_indices = np.arange(len(image_features))
    accuracy = np.mean(predicted_indices == true_indices)
    
    return float(accuracy)


def image_retrieval_recall_at_1(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> float:
    """
    计算图像检索召回率@1
    
    给定文本查询，在图像库中检索最相似的1张图像，计算正确检索的比例。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        图像检索召回率@1
    """
    similarity_matrix = _compute_similarity_matrix(image_features, text_features, temperature)
    return _compute_recall_at_k(similarity_matrix, k=1, mode='t2i')


def image_retrieval_recall_at_5(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> float:
    """
    计算图像检索召回率@5
    
    给定文本查询，在图像库中检索最相似的5张图像，计算正确检索的比例。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        图像检索召回率@5
    """
    similarity_matrix = _compute_similarity_matrix(image_features, text_features, temperature)
    return _compute_recall_at_k(similarity_matrix, k=5, mode='t2i')


def text_retrieval_recall_at_1(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> float:
    """
    计算文本检索召回率@1
    
    给定图像查询，在文本库中检索最相似的1个文本，计算正确检索的比例。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        文本检索召回率@1
    """
    similarity_matrix = _compute_similarity_matrix(image_features, text_features, temperature)
    return _compute_recall_at_k(similarity_matrix, k=1, mode='i2t')


def text_retrieval_recall_at_5(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> float:
    """
    计算文本检索召回率@5
    
    给定图像查询，在文本库中检索最相似的5个文本，计算正确检索的比例。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        文本检索召回率@5
    """
    similarity_matrix = _compute_similarity_matrix(image_features, text_features, temperature)
    return _compute_recall_at_k(similarity_matrix, k=5, mode='i2t')


def image_text_similarity(image_features: np.ndarray, text_features: np.ndarray, temperature: float=0.07) -> float:
    """
    计算图像-文本相似度
    
    计算对应图像-文本对之间的平均余弦相似度。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        text_features: 文本特征矩阵，形状为 (N, D)
        temperature: 温度参数，用于控制相似度矩阵的缩放
    
    Returns:
        平均图像-文本相似度
    """
    if image_features.shape[0] != text_features.shape[0]:
        raise ValueError("图像特征和文本特征的数量必须相等")
    
    # 归一化特征向量
    image_features_norm = image_features / np.linalg.norm(image_features, axis=1, keepdims=True)
    text_features_norm = text_features / np.linalg.norm(text_features, axis=1, keepdims=True)
    
    # 计算对应对的余弦相似度
    similarities = np.sum(image_features_norm * text_features_norm, axis=1)
    
    # 应用温度缩放
    similarities = similarities / temperature
    
    return float(np.mean(similarities))


def zero_shot_classification_accuracy(
    image_features: np.ndarray, 
    class_text_features: np.ndarray, 
    true_labels: np.ndarray,
    temperature: float=0.07
) -> float:
    """
    计算零样本分类准确率
    
    使用CLIP模型进行零样本分类，计算分类准确率。
    对于每个图像，计算其与所有类别文本描述的相似度，选择相似度最高的类别作为预测结果。
    
    Args:
        image_features: 图像特征矩阵，形状为 (N, D)
        class_text_features: 类别文本特征矩阵，形状为 (C, D)，C为类别数
        true_labels: 真实标签，形状为 (N,)，值范围为 [0, C-1]
    
    Returns:
        零样本分类准确率
    """
    # 计算图像与所有类别文本的相似度
    similarity_matrix = _compute_similarity_matrix(image_features, class_text_features, temperature)
    
    # 预测类别为相似度最高的类别
    predicted_labels = np.argmax(similarity_matrix, axis=1)
    
    # 计算准确率
    accuracy = np.mean(predicted_labels == true_labels)
    
    return float(accuracy)


# 为了保持向后兼容性，创建别名
image_retrieval_recall_1 = image_retrieval_recall_at_1
image_retrieval_recall_5 = image_retrieval_recall_at_5
text_retrieval_recall_1 = text_retrieval_recall_at_1
text_retrieval_recall_5 = text_retrieval_recall_at_5