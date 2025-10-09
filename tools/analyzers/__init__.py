"""NeuroTrain Analyzers Package

该包提供了三个核心分析器模块：
- AttentionAnalyzer: 注意力机制分析器
- MetricsAnalyzer: 数据指标分析器  
- DatasetAnalyzer: 数据集分析器

每个分析器都提供独立的功能和清晰的接口，便于单独维护和调用。
"""

from pathlib import Path
from .attention_analyzer import AttentionAnalyzer, analyze_model_attention
from .metrics_analyzer import MetricsAnalyzer, analyze_model_metrics
from .dataset_analyzer import DatasetAnalyzer, analyze_dataset
from .mask_analyzer import MaskAnalyzer, analyze_image_mask, analyze_text_mask
from .relation_analyzer import RelationAnalyzer, analyze_cross_modal_similarity, build_relation_graph

__all__ = [
    'AttentionAnalyzer', 'analyze_model_attention',
    'MetricsAnalyzer', 'analyze_model_metrics',
    'DatasetAnalyzer', 'analyze_dataset',
    'MaskAnalyzer', 'analyze_image_mask', 'analyze_text_mask',
    'RelationAnalyzer', 'analyze_cross_modal_similarity', 'build_relation_graph',
    'UnifiedAnalyzer', 'create_unified_analyzer', 'run_comprehensive_analysis'
]

__version__ = '1.0.0'


class UnifiedAnalyzer:
    """
    统一分析器，整合所有分析功能
    
    提供一站式的模型分析服务，包括：
    - 注意力机制分析（包括SE模块等常规注意力）
    - 训练数据和指标分析
    - 数据集质量分析
    - Mask信息分析（图像和文本）
    - 跨模态关系分析（类似CLIP）
    """
    
    def __init__(self, 
                 model=None,
                 dataset_name: str = None,
                 dataset_config: dict = None,
                 output_dir: str = "output/analysis"):
        """
        初始化统一分析器
        
        Args:
            model: 要分析的模型
            dataset_name: 数据集名称
            dataset_config: 数据集配置
            output_dir: 输出目录
        """
        self.model = model
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config or {}
        self.output_dir = output_dir
        
        # 初始化各个分析器
        self.attention_analyzer = AttentionAnalyzer(output_dir=output_dir)
        self.metrics_analyzer = MetricsAnalyzer(output_dir=output_dir)
        self.dataset_analyzer = DatasetAnalyzer(output_dir=output_dir)
        self.mask_analyzer = MaskAnalyzer(save_dir=Path(output_dir))
        self.relation_analyzer = RelationAnalyzer(save_dir=Path(output_dir))
        
    def analyze_attention(self, model=None, **kwargs):
        """分析注意力机制"""
        model = model or self.model
        if model is None:
            raise ValueError("需要提供模型进行注意力分析")
        return self.attention_analyzer.analyze_model(model, **kwargs)
    
    def analyze_metrics(self, metrics_data=None, **kwargs):
        """分析训练指标"""
        return self.metrics_analyzer.analyze_metrics(metrics_data, **kwargs)
    
    def analyze_dataset(self, dataset_name=None, dataset_config=None, **kwargs):
        """分析数据集"""
        dataset_name = dataset_name or self.dataset_name
        dataset_config = dataset_config or self.dataset_config
        return self.dataset_analyzer.run_full_analysis(
            dataset_name=dataset_name,
            dataset_config=dataset_config,
            **kwargs
        )
    
    def analyze_masks(self, masks_data, mask_type='image', **kwargs):
        """分析Mask信息"""
        if mask_type == 'image':
            return self.mask_analyzer.analyze_image_mask(masks_data, **kwargs)
        elif mask_type == 'text':
            return self.mask_analyzer.analyze_text_mask(masks_data, **kwargs)
        else:
            raise ValueError(f"不支持的mask类型: {mask_type}")
    
    def analyze_relations(self, image_features, text_features, **kwargs):
        """分析跨模态关系"""
        return self.relation_analyzer.analyze_cross_modal_similarity(image_features, text_features, **kwargs)
    
    def run_comprehensive_analysis(self, **kwargs):
        """运行全面分析"""
        results = {}
        
        # 数据集分析
        if self.dataset_name:
            try:
                results['dataset'] = self.analyze_dataset(**kwargs.get('dataset_kwargs', {}))
            except Exception as e:
                print(f"数据集分析失败: {e}")
        
        # 模型分析
        if self.model:
            try:
                results['attention'] = self.analyze_attention(**kwargs.get('attention_kwargs', {}))
            except Exception as e:
                print(f"注意力分析失败: {e}")
        
        # 指标分析
        metrics_data = kwargs.get('metrics_data')
        if metrics_data:
            try:
                results['metrics'] = self.analyze_metrics(metrics_data, **kwargs.get('metrics_kwargs', {}))
            except Exception as e:
                print(f"指标分析失败: {e}")
        
        return results


def create_unified_analyzer(model=None, dataset_name=None, dataset_config=None, output_dir="output/analysis"):
    """
    创建统一分析器的便捷函数
    
    Args:
        model: 要分析的模型
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        output_dir: 输出目录
        
    Returns:
        UnifiedAnalyzer: 统一分析器实例
    """
    return UnifiedAnalyzer(
        model=model,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=output_dir
    )


def run_comprehensive_analysis(model=None, dataset_name=None, dataset_config=None, 
                             metrics_data=None, output_dir="output/analysis", **kwargs):
    """
    运行全面分析的便捷函数
    
    Args:
        model: 要分析的模型
        dataset_name: 数据集名称
        dataset_config: 数据集配置
        metrics_data: 训练指标数据
        output_dir: 输出目录
        **kwargs: 其他参数
        
    Returns:
        dict: 分析结果
    """
    analyzer = create_unified_analyzer(
        model=model,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=output_dir
    )
    
    return analyzer.run_comprehensive_analysis(metrics_data=metrics_data, **kwargs)


# 为了向后兼容，保留旧的别名
DataAnalyzer = MetricsAnalyzer
analyze_attention_weights = analyze_model_attention  # 向后兼容别名
analyze_image_masks = analyze_image_mask  # 向后兼容别名
analyze_text_masks = analyze_text_mask  # 向后兼容别名