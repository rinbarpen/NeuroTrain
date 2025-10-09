"""
BLEU (Bilingual Evaluation Understudy) 评估指标实现

本模块实现了BLEU评估指标，用于评估机器翻译和文本生成任务的质量。
BLEU指标基于n-gram精度和简洁惩罚因子来衡量候选文本与参考文本的相似度。

实现的指标包括：
- BLEU-1 到 BLEU-4 (单独的n-gram精度)
- 累积BLEU分数 (综合1-gram到4-gram的几何平均)
- 简洁惩罚因子计算
- n-gram精度计算辅助函数
"""

import numpy as np
from typing import List, Union, Dict, Tuple, Optional
from collections import Counter, defaultdict
import math


def _tokenize(text: str) -> List[str]:
    """
    简单的文本分词函数
    
    Args:
        text: 输入文本字符串
    
    Returns:
        分词后的token列表
    """
    # 简单的空格分词，实际应用中可能需要更复杂的分词器
    return text.strip().split()


def _get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    从token列表中提取n-gram
    
    Args:
        tokens: token列表
        n: n-gram的n值
    
    Returns:
        n-gram的计数器
    """
    if len(tokens) < n:
        return Counter()
    
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngrams.append(ngram)
    
    return Counter(ngrams)


def _compute_ngram_precision(candidate_tokens: List[str], 
                           reference_tokens_list: List[List[str]], 
                           n: int) -> float:
    """
    计算n-gram精度
    
    Args:
        candidate_tokens: 候选文本的token列表
        reference_tokens_list: 参考文本的token列表的列表
        n: n-gram的n值
    
    Returns:
        n-gram精度值
    """
    candidate_ngrams = _get_ngrams(candidate_tokens, n)
    
    if not candidate_ngrams:
        return 0.0
    
    # 计算每个参考文本的n-gram，并取最大匹配数
    max_ref_counts = Counter()
    for ref_tokens in reference_tokens_list:
        ref_ngrams = _get_ngrams(ref_tokens, n)
        for ngram in candidate_ngrams:
            max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
    
    # 计算匹配的n-gram数量
    matched_count = 0
    total_count = sum(candidate_ngrams.values())
    
    for ngram, count in candidate_ngrams.items():
        matched_count += min(count, max_ref_counts[ngram])
    
    return matched_count / total_count if total_count > 0 else 0.0


def _compute_brevity_penalty(candidate_length: int, 
                           reference_lengths: List[int]) -> float:
    """
    计算简洁惩罚因子 (Brevity Penalty)
    
    Args:
        candidate_length: 候选文本长度
        reference_lengths: 参考文本长度列表
    
    Returns:
        简洁惩罚因子
    """
    if candidate_length == 0:
        return 0.0
    
    # 选择最接近候选文本长度的参考文本长度
    closest_ref_length = min(reference_lengths, 
                           key=lambda x: abs(x - candidate_length))
    
    if candidate_length >= closest_ref_length:
        return 1.0
    else:
        return math.exp(1 - closest_ref_length / candidate_length)


def bleu_1(candidate: str, references: List[str]) -> float:
    """
    计算BLEU-1分数 (1-gram精度)
    
    Args:
        candidate: 候选文本
        references: 参考文本列表
    
    Returns:
        BLEU-1分数
    """
    candidate_tokens = _tokenize(candidate)
    reference_tokens_list = [_tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # 计算1-gram精度
    precision_1 = _compute_ngram_precision(candidate_tokens, reference_tokens_list, 1)
    
    # 计算简洁惩罚因子
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    bp = _compute_brevity_penalty(candidate_length, reference_lengths)
    
    return bp * precision_1


def bleu_2(candidate: str, references: List[str]) -> float:
    """
    计算BLEU-2分数 (1-gram和2-gram精度的几何平均)
    
    Args:
        candidate: 候选文本
        references: 参考文本列表
    
    Returns:
        BLEU-2分数
    """
    candidate_tokens = _tokenize(candidate)
    reference_tokens_list = [_tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # 计算1-gram和2-gram精度
    precision_1 = _compute_ngram_precision(candidate_tokens, reference_tokens_list, 1)
    precision_2 = _compute_ngram_precision(candidate_tokens, reference_tokens_list, 2)
    
    # 避免对数计算中的零值
    if precision_1 == 0 or precision_2 == 0:
        return 0.0
    
    # 计算几何平均
    geometric_mean = math.exp(0.5 * (math.log(precision_1) + math.log(precision_2)))
    
    # 计算简洁惩罚因子
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    bp = _compute_brevity_penalty(candidate_length, reference_lengths)
    
    return bp * geometric_mean


def bleu_3(candidate: str, references: List[str]) -> float:
    """
    计算BLEU-3分数 (1-gram到3-gram精度的几何平均)
    
    Args:
        candidate: 候选文本
        references: 参考文本列表
    
    Returns:
        BLEU-3分数
    """
    candidate_tokens = _tokenize(candidate)
    reference_tokens_list = [_tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # 计算1-gram到3-gram精度
    precisions = []
    for n in range(1, 4):
        precision = _compute_ngram_precision(candidate_tokens, reference_tokens_list, n)
        if precision == 0:
            return 0.0
        precisions.append(precision)
    
    # 计算几何平均
    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum / 3)
    
    # 计算简洁惩罚因子
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    bp = _compute_brevity_penalty(candidate_length, reference_lengths)
    
    return bp * geometric_mean


def bleu_4(candidate: str, references: List[str]) -> float:
    """
    计算BLEU-4分数 (1-gram到4-gram精度的几何平均)
    
    Args:
        candidate: 候选文本
        references: 参考文本列表
    
    Returns:
        BLEU-4分数
    """
    candidate_tokens = _tokenize(candidate)
    reference_tokens_list = [_tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # 计算1-gram到4-gram精度
    precisions = []
    for n in range(1, 5):
        precision = _compute_ngram_precision(candidate_tokens, reference_tokens_list, n)
        if precision == 0:
            return 0.0
        precisions.append(precision)
    
    # 计算几何平均 (标准BLEU-4使用等权重)
    log_sum = sum(math.log(p) for p in precisions)
    geometric_mean = math.exp(log_sum / 4)
    
    # 计算简洁惩罚因子
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    bp = _compute_brevity_penalty(candidate_length, reference_lengths)
    
    return bp * geometric_mean


def cumulative_bleu(candidate: str, 
                   references: List[str], 
                   weights: Optional[List[float]] = None) -> float:
    """
    计算累积BLEU分数 (可自定义权重的n-gram精度几何平均)
    
    Args:
        candidate: 候选文本
        references: 参考文本列表
        weights: n-gram权重列表，默认为[0.25, 0.25, 0.25, 0.25]
    
    Returns:
        累积BLEU分数
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]  # 标准BLEU-4权重
    
    candidate_tokens = _tokenize(candidate)
    reference_tokens_list = [_tokenize(ref) for ref in references]
    
    if not candidate_tokens:
        return 0.0
    
    # 计算各个n-gram精度
    precisions = []
    for n in range(1, len(weights) + 1):
        precision = _compute_ngram_precision(candidate_tokens, reference_tokens_list, n)
        if precision == 0:
            return 0.0
        precisions.append(precision)
    
    # 计算加权几何平均
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    geometric_mean = math.exp(log_sum)
    
    # 计算简洁惩罚因子
    candidate_length = len(candidate_tokens)
    reference_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
    bp = _compute_brevity_penalty(candidate_length, reference_lengths)
    
    return bp * geometric_mean


def corpus_bleu(candidates: List[str], 
               references_list: List[List[str]], 
               weights: Optional[List[float]] = None) -> float:
    """
    计算语料库级别的BLEU分数
    
    Args:
        candidates: 候选文本列表
        references_list: 参考文本列表的列表，每个元素对应一个候选文本的多个参考
        weights: n-gram权重列表，默认为[0.25, 0.25, 0.25, 0.25]
    
    Returns:
        语料库级别的BLEU分数
    """
    if weights is None:
        weights = [0.25, 0.25, 0.25, 0.25]
    
    if len(candidates) != len(references_list):
        raise ValueError("候选文本数量必须与参考文本列表数量相等")
    
    # 累积统计信息
    total_candidate_length = 0
    total_reference_lengths = []
    
    # 为每个n-gram级别累积匹配数和总数
    total_matches = [0] * len(weights)
    total_counts = [0] * len(weights)
    
    for candidate, references in zip(candidates, references_list):
        candidate_tokens = _tokenize(candidate)
        reference_tokens_list = [_tokenize(ref) for ref in references]
        
        if not candidate_tokens:
            continue
        
        total_candidate_length += len(candidate_tokens)
        
        # 收集参考文本长度
        ref_lengths = [len(ref_tokens) for ref_tokens in reference_tokens_list]
        total_reference_lengths.extend(ref_lengths)
        
        # 为每个n-gram级别累积统计
        for n in range(1, len(weights) + 1):
            candidate_ngrams = _get_ngrams(candidate_tokens, n)
            
            if candidate_ngrams:
                # 计算最大参考匹配数
                max_ref_counts = Counter()
                for ref_tokens in reference_tokens_list:
                    ref_ngrams = _get_ngrams(ref_tokens, n)
                    for ngram in candidate_ngrams:
                        max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_ngrams[ngram])
                
                # 累积匹配数和总数
                matched_count = sum(min(count, max_ref_counts[ngram]) 
                                  for ngram, count in candidate_ngrams.items())
                total_count = sum(candidate_ngrams.values())
                
                total_matches[n-1] += matched_count
                total_counts[n-1] += total_count
    
    # 计算精度
    precisions = []
    for matches, counts in zip(total_matches, total_counts):
        if counts == 0:
            return 0.0
        precisions.append(matches / counts)
    
    # 检查是否有零精度
    if any(p == 0 for p in precisions):
        return 0.0
    
    # 计算加权几何平均
    log_sum = sum(w * math.log(p) for w, p in zip(weights, precisions))
    geometric_mean = math.exp(log_sum)
    
    # 计算语料库级别的简洁惩罚因子
    if total_candidate_length == 0:
        return 0.0
    
    # 选择最接近的总参考长度
    avg_ref_length = sum(total_reference_lengths) / len(total_reference_lengths) if total_reference_lengths else 0
    
    if total_candidate_length >= avg_ref_length:
        bp = 1.0
    else:
        bp = math.exp(1 - avg_ref_length / total_candidate_length)
    
    return bp * geometric_mean


# 为了向后兼容，提供别名
bleu = bleu_4  # 默认BLEU指标为BLEU-4
sentence_bleu = cumulative_bleu  # 句子级别BLEU