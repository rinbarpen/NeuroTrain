"""
LLM驱动的数据集分析器

使用大语言模型来驱动数据集的筛选、分析和处理
支持自然语言查询和智能数据处理
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Callable
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict
import torch
from torch.utils.data import Dataset

from .custom_dataset import CustomDataset


@dataclass
class DatasetStats:
    """数据集统计信息"""
    name: str
    size: int
    split: str
    num_classes: Optional[int] = None
    class_distribution: Optional[Dict[int, int]] = None
    image_shape: Optional[tuple] = None
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self):
        """转换为字典"""
        return asdict(self)
    
    def to_json(self):
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)


class DatasetAnalyzer:
    """数据集分析器基类"""
    
    def __init__(self, dataset: Dataset, name: str = "dataset"):
        """
        Args:
            dataset: PyTorch数据集
            name: 数据集名称
        """
        self.dataset = dataset
        self.name = name
        self.logger = logging.getLogger(__name__)
        self._stats = None
    
    def compute_stats(self, max_samples: int = 1000) -> DatasetStats:
        """计算数据集统计信息
        
        Args:
            max_samples: 用于计算统计的最大样本数
            
        Returns:
            DatasetStats对象
        """
        if self._stats is not None:
            return self._stats
        
        self.logger.info(f"计算数据集 '{self.name}' 的统计信息...")
        
        # 基本信息
        size = len(self.dataset)
        split = getattr(self.dataset, 'split', 'unknown')
        
        # 采样数据
        num_samples = min(max_samples, size)
        indices = np.random.choice(size, num_samples, replace=False)
        
        # 收集样本信息
        images = []
        labels = []
        
        for idx in indices:
            try:
                item = self.dataset[idx]
                if isinstance(item, (tuple, list)):
                    image = item[0]
                    label = item[1] if len(item) > 1 else None
                else:
                    image = item
                    label = None
                
                if isinstance(image, torch.Tensor):
                    images.append(image)
                
                if label is not None:
                    if isinstance(label, torch.Tensor):
                        labels.append(label.item())
                    else:
                        labels.append(label)
            except Exception as e:
                self.logger.warning(f"处理样本 {idx} 时出错: {e}")
                continue
        
        # 计算图像统计
        image_shape = None
        mean = None
        std = None
        
        if images:
            image_shape = tuple(images[0].shape)
            # 转换为numpy进行计算
            images_array = torch.stack(images).numpy()
            mean = images_array.mean(axis=(0, 2, 3)).tolist()
            std = images_array.std(axis=(0, 2, 3)).tolist()
        
        # 计算类别分布
        num_classes = None
        class_distribution = None
        
        if labels:
            unique_labels = set(labels)
            num_classes = len(unique_labels)
            class_distribution = {
                int(label): labels.count(label) 
                for label in unique_labels
            }
        
        # 获取元数据
        metadata = {}
        if hasattr(self.dataset, 'metadata'):
            try:
                metadata = self.dataset.metadata()
            except:
                pass
        
        self._stats = DatasetStats(
            name=self.name,
            size=size,
            split=split,
            num_classes=num_classes,
            class_distribution=class_distribution,
            image_shape=image_shape,
            mean=mean,
            std=std,
            metadata=metadata
        )
        
        return self._stats
    
    def filter_by_label(self, labels: List[int]) -> 'FilteredDataset':
        """按标签筛选数据
        
        Args:
            labels: 要保留的标签列表
            
        Returns:
            FilteredDataset对象
        """
        indices = []
        for idx in range(len(self.dataset)):
            try:
                item = self.dataset[idx]
                if isinstance(item, (tuple, list)) and len(item) > 1:
                    label = item[1]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    if label in labels:
                        indices.append(idx)
            except:
                continue
        
        self.logger.info(f"按标签 {labels} 筛选: {len(indices)}/{len(self.dataset)} 样本")
        return FilteredDataset(self.dataset, indices, f"{self.name}_filtered")
    
    def filter_by_function(self, filter_fn: Callable) -> 'FilteredDataset':
        """按自定义函数筛选数据
        
        Args:
            filter_fn: 筛选函数，接收(image, label)，返回bool
            
        Returns:
            FilteredDataset对象
        """
        indices = []
        for idx in range(len(self.dataset)):
            try:
                item = self.dataset[idx]
                if isinstance(item, (tuple, list)):
                    image = item[0]
                    label = item[1] if len(item) > 1 else None
                else:
                    image = item
                    label = None
                
                if filter_fn(image, label):
                    indices.append(idx)
            except Exception as e:
                self.logger.warning(f"筛选样本 {idx} 时出错: {e}")
                continue
        
        self.logger.info(f"自定义筛选: {len(indices)}/{len(self.dataset)} 样本")
        return FilteredDataset(self.dataset, indices, f"{self.name}_filtered")
    
    def sample_balanced(self, samples_per_class: int) -> 'FilteredDataset':
        """类别平衡采样
        
        Args:
            samples_per_class: 每个类别采样的样本数
            
        Returns:
            FilteredDataset对象
        """
        # 按类别收集索引
        class_indices = {}
        for idx in range(len(self.dataset)):
            try:
                item = self.dataset[idx]
                if isinstance(item, (tuple, list)) and len(item) > 1:
                    label = item[1]
                    if isinstance(label, torch.Tensor):
                        label = label.item()
                    
                    if label not in class_indices:
                        class_indices[label] = []
                    class_indices[label].append(idx)
            except:
                continue
        
        # 每个类别采样
        selected_indices = []
        for label, indices in class_indices.items():
            n_samples = min(samples_per_class, len(indices))
            sampled = np.random.choice(indices, n_samples, replace=False)
            selected_indices.extend(sampled.tolist())
        
        self.logger.info(
            f"类别平衡采样: {len(selected_indices)} 样本 "
            f"({samples_per_class} per class x {len(class_indices)} classes)"
        )
        return FilteredDataset(self.dataset, selected_indices, f"{self.name}_balanced")
    
    def generate_report(self) -> str:
        """生成数据集分析报告
        
        Returns:
            Markdown格式的报告
        """
        stats = self.compute_stats()
        
        report = f"""# 数据集分析报告: {stats.name}

## 基本信息
- **数据集名称**: {stats.name}
- **划分**: {stats.split}
- **样本数量**: {stats.size:,}
- **类别数量**: {stats.num_classes or 'N/A'}

## 图像信息
- **图像形状**: {stats.image_shape}
- **均值**: {[f'{x:.4f}' for x in stats.mean] if stats.mean else 'N/A'}
- **标准差**: {[f'{x:.4f}' for x in stats.std] if stats.std else 'N/A'}

## 类别分布
"""
        
        if stats.class_distribution:
            report += "\n| 类别 | 样本数 | 占比 |\n"
            report += "|------|--------|------|\n"
            for label, count in sorted(stats.class_distribution.items()):
                ratio = count / stats.size * 100
                report += f"| {label} | {count:,} | {ratio:.2f}% |\n"
        else:
            report += "\n无类别信息\n"
        
        if stats.metadata:
            report += "\n## 元数据\n"
            for key, value in stats.metadata.items():
                report += f"- **{key}**: {value}\n"
        
        return report


class FilteredDataset(Dataset):
    """筛选后的数据集"""
    
    def __init__(self, base_dataset: Dataset, indices: List[int], name: str = "filtered"):
        """
        Args:
            base_dataset: 基础数据集
            indices: 保留的索引列表
            name: 数据集名称
        """
        self.base_dataset = base_dataset
        self.indices = indices
        self.name = name
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        return self.base_dataset[self.indices[idx]]


class LLMDataAnalyzer:
    """LLM驱动的数据分析器
    
    使用自然语言指令来分析和处理数据集
    """
    
    def __init__(
        self,
        llm_model: Optional[Any] = None,
        llm_config: Optional[Dict] = None
    ):
        """
        Args:
            llm_model: LLM模型实例（可选）
            llm_config: LLM配置（可选）
        """
        self.llm_model = llm_model
        self.llm_config = llm_config or {}
        self.logger = logging.getLogger(__name__)
        self.datasets = {}
    
    def add_dataset(self, dataset: Dataset, name: str):
        """添加数据集
        
        Args:
            dataset: PyTorch数据集
            name: 数据集名称
        """
        analyzer = DatasetAnalyzer(dataset, name)
        self.datasets[name] = {
            'dataset': dataset,
            'analyzer': analyzer
        }
        self.logger.info(f"已添加数据集: {name}")
    
    def analyze(self, dataset_name: str, query: str) -> Dict[str, Any]:
        """使用LLM分析数据集
        
        Args:
            dataset_name: 数据集名称
            query: 自然语言查询
            
        Returns:
            分析结果字典
        """
        if dataset_name not in self.datasets:
            raise ValueError(f"数据集 '{dataset_name}' 不存在")
        
        analyzer = self.datasets[dataset_name]['analyzer']
        stats = analyzer.compute_stats()
        
        # 解析查询意图
        intent = self._parse_query(query)
        
        result = {
            'dataset': dataset_name,
            'query': query,
            'intent': intent,
            'stats': stats.to_dict()
        }
        
        # 根据意图执行操作
        if intent == 'show_stats':
            result['response'] = analyzer.generate_report()
        
        elif intent == 'filter_by_label':
            # 从查询中提取标签
            labels = self._extract_labels(query)
            if labels:
                filtered = analyzer.filter_by_label(labels)
                result['response'] = f"已筛选出标签为 {labels} 的数据，共 {len(filtered)} 个样本"
                result['filtered_dataset'] = filtered
        
        elif intent == 'show_distribution':
            if stats.class_distribution:
                result['response'] = f"类别分布:\n{json.dumps(stats.class_distribution, indent=2)}"
            else:
                result['response'] = "该数据集没有类别信息"
        
        elif intent == 'balance':
            # 类别平衡
            samples_per_class = self._extract_number(query, default=100)
            balanced = analyzer.sample_balanced(samples_per_class)
            result['response'] = f"已进行类别平衡采样，每类 {samples_per_class} 个样本，共 {len(balanced)} 个样本"
            result['filtered_dataset'] = balanced
        
        else:
            result['response'] = self._generate_llm_response(query, stats)
        
        return result
    
    def _parse_query(self, query: str) -> str:
        """解析查询意图
        
        Args:
            query: 查询字符串
            
        Returns:
            意图类型
        """
        query_lower = query.lower()
        
        # 关键词匹配（按优先级顺序检查，更具体的在前面）
        if any(kw in query_lower for kw in ['平衡', 'balance', '均衡', '平衡采样']):
            return 'balance'
        
        elif any(kw in query_lower for kw in ['筛选', 'filter', '选择', 'select']) and any(kw in query_lower for kw in ['标签', 'label']):
            return 'filter_by_label'
        
        elif any(kw in query_lower for kw in ['分布', 'distribution']) or (any(kw in query_lower for kw in ['类别', 'class']) and '分布' in query_lower):
            return 'show_distribution'
        
        elif any(kw in query_lower for kw in ['统计', 'stats', '信息', 'info', '报告', 'report']):
            return 'show_stats'
        
        else:
            return 'general'
    
    def _extract_labels(self, query: str) -> List[int]:
        """从查询中提取标签
        
        Args:
            query: 查询字符串
            
        Returns:
            标签列表
        """
        import re
        # 查找所有数字
        numbers = re.findall(r'\d+', query)
        return [int(n) for n in numbers]
    
    def _extract_number(self, query: str, default: int = 100) -> int:
        """从查询中提取数字
        
        Args:
            query: 查询字符串
            default: 默认值
            
        Returns:
            提取的数字
        """
        import re
        numbers = re.findall(r'\d+', query)
        return int(numbers[0]) if numbers else default
    
    def _generate_llm_response(self, query: str, stats: DatasetStats) -> str:
        """使用LLM生成响应
        
        Args:
            query: 查询
            stats: 数据集统计
            
        Returns:
            LLM响应
        """
        if self.llm_model is None:
            # 没有LLM时的简单响应
            return f"数据集 '{stats.name}' 包含 {stats.size} 个样本"
        
        # 构建prompt
        prompt = f"""你是一个数据集分析助手。根据以下数据集信息回答用户问题。

数据集信息:
{stats.to_json()}

用户问题: {query}

请简洁地回答用户的问题。
"""
        
        try:
            # 调用LLM
            response = self._call_llm(prompt)
            return response
        except Exception as e:
            self.logger.error(f"LLM调用失败: {e}")
            return f"无法生成响应: {e}"
    
    def _call_llm(self, prompt: str) -> str:
        """调用LLM
        
        Args:
            prompt: 输入提示
            
        Returns:
            LLM响应
        """
        # 这里可以集成各种LLM API
        # 例如: OpenAI, Claude, 本地LLM等
        
        if hasattr(self.llm_model, 'generate'):
            return self.llm_model.generate(prompt)
        elif hasattr(self.llm_model, '__call__'):
            return self.llm_model(prompt)
        else:
            return "LLM模型未正确配置"
    
    def batch_analyze(
        self,
        dataset_name: str,
        queries: List[str]
    ) -> List[Dict[str, Any]]:
        """批量分析
        
        Args:
            dataset_name: 数据集名称
            queries: 查询列表
            
        Returns:
            分析结果列表
        """
        results = []
        for query in queries:
            try:
                result = self.analyze(dataset_name, query)
                results.append(result)
            except Exception as e:
                self.logger.error(f"处理查询 '{query}' 时出错: {e}")
                results.append({
                    'query': query,
                    'error': str(e)
                })
        return results
    
    def export_analysis(
        self,
        results: Union[Dict, List[Dict]],
        output_path: Path,
        format: str = 'json'
    ):
        """导出分析结果
        
        Args:
            results: 分析结果
            output_path: 输出路径
            format: 输出格式 ('json', 'markdown', 'csv')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        elif format == 'markdown':
            if isinstance(results, dict):
                results = [results]
            
            markdown = "# 数据集分析结果\n\n"
            for i, result in enumerate(results, 1):
                markdown += f"## 查询 {i}: {result.get('query', 'N/A')}\n\n"
                markdown += f"**响应**: {result.get('response', 'N/A')}\n\n"
                markdown += "---\n\n"
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(markdown)
        
        elif format == 'csv':
            if isinstance(results, dict):
                results = [results]
            
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False, encoding='utf-8')
        
        self.logger.info(f"分析结果已导出到: {output_path}")


def create_analyzer(dataset: Dataset, name: str = "dataset") -> DatasetAnalyzer:
    """创建数据集分析器的便捷函数
    
    Args:
        dataset: PyTorch数据集
        name: 数据集名称
        
    Returns:
        DatasetAnalyzer实例
    """
    return DatasetAnalyzer(dataset, name)


def create_llm_analyzer(
    llm_model: Optional[Any] = None,
    llm_config: Optional[Dict] = None
) -> LLMDataAnalyzer:
    """创建LLM驱动的分析器的便捷函数
    
    Args:
        llm_model: LLM模型实例
        llm_config: LLM配置
        
    Returns:
        LLMDataAnalyzer实例
    """
    return LLMDataAnalyzer(llm_model, llm_config)

