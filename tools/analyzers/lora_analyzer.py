"""
LoRA 分析器模块

提供 LoRA 模型合并、分析和可视化功能，支持多种合并策略和权重分析。

主要功能：
- LoRA 适配器合并（顺序合并、加权合并、平均合并）
- LoRA 权重分析和可视化
- 模型大小和参数统计
- 合并策略比较
- 权重分布分析
"""

import json
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import warnings

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import torch.nn as nn
    from matplotlib import font_manager
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some visualization dependencies are not available: {e}")
    MATPLOTLIB_AVAILABLE = False
    # 创建占位符
    plt = None
    np = None
    torch = None
    nn = None
    sns = None

# 设置matplotlib字体为英语
if MATPLOTLIB_AVAILABLE:
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    sns.set_palette("husl")

logger = logging.getLogger(__name__)


class LoRAAnalyzer:
    """LoRA 分析器类，提供 LoRA 模型合并和分析功能。"""
    
    def __init__(
        self,
        output_dir: Union[str, Path] = "runs/lora_analysis",
        device: str = "auto",
        use_proxy: bool = False,
        trust_remote_code: bool = False,
        local_files_only: bool = False
    ):
        """
        初始化 LoRA 分析器。
        
        Args:
            output_dir: 输出目录
            device: 设备类型 ('auto', 'cpu', 'cuda')
            use_proxy: 是否使用代理
            trust_remote_code: 是否信任远程代码
            local_files_only: 是否仅使用本地文件
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.use_proxy = use_proxy
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only
        
        # 创建子目录
        self.merge_dir = self.output_dir / "merged_models"
        self.analysis_dir = self.output_dir / "analysis"
        self.visualization_dir = self.output_dir / "visualizations"
        
        for dir_path in [self.merge_dir, self.analysis_dir, self.visualization_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def _enable_proxy(self) -> None:
        """启用代理（如果配置了）。"""
        if not self.use_proxy:
            return
        try:
            subprocess.run(["proxy_on"], check=False, capture_output=True)
            logger.info("Proxy enabled successfully")
        except Exception as e:
            logger.warning(f"Failed to enable proxy: {e}")
    
    def _load_model(self, model_path: str, torch_dtype: Optional[torch.dtype] = None):
        """加载模型。"""
        try:
            from transformers import AutoModelForCausalLM, AutoModel
            
            kwargs = {
                "torch_dtype": torch_dtype,
                "device_map": self.device,
                "trust_remote_code": self.trust_remote_code,
                "local_files_only": self.local_files_only,
            }
            
            # 尝试加载因果语言模型
            try:
                return AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
            except Exception:
                # 回退到通用模型
                return AutoModel.from_pretrained(model_path, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise
    
    def _load_peft_model(self, model, adapter_path: str):
        """加载 PEFT 模型。"""
        try:
            from peft import PeftModel
            return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
        except Exception as e:
            logger.error(f"Failed to load PEFT model from {adapter_path}: {e}")
            raise
    
    def merge_adapters(
        self,
        base_model: str,
        adapters: List[str],
        output_name: str,
        merge_strategy: str = "sequential",
        weights: Optional[List[float]] = None,
        merge_dtype: Optional[torch.dtype] = None,
        save_tokenizer: bool = True,
        save_safetensors: bool = True
    ) -> Dict[str, Any]:
        """
        合并 LoRA 适配器。
        
        Args:
            base_model: 基础模型路径或 Hugging Face 模型 ID
            adapters: LoRA 适配器路径列表
            output_name: 输出模型名称
            merge_strategy: 合并策略 ('sequential', 'weighted', 'average')
            weights: 权重列表（用于加权合并）
            merge_dtype: 合并时的数据类型
            save_tokenizer: 是否保存 tokenizer
            save_safetensors: 是否使用 safetensors 格式保存
            
        Returns:
            合并结果信息
        """
        self._enable_proxy()
        
        logger.info(f"Starting LoRA merge with strategy: {merge_strategy}")
        logger.info(f"Base model: {base_model}")
        logger.info(f"Adapters: {adapters}")
        
        # 加载基础模型
        logger.info("Loading base model...")
        model = self._load_model(base_model, merge_dtype)
        
        # 根据策略合并适配器
        if merge_strategy == "sequential":
            merged_model = self._sequential_merge(model, adapters)
        elif merge_strategy == "weighted":
            merged_model = self._weighted_merge(model, adapters, weights)
        elif merge_strategy == "average":
            merged_model = self._average_merge(model, adapters)
        else:
            raise ValueError(f"Unsupported merge strategy: {merge_strategy}")
        
        # 保存合并后的模型
        output_path = self.merge_dir / output_name
        logger.info(f"Saving merged model to: {output_path}")
        merged_model.save_pretrained(output_path, safe_serialization=save_safetensors)
        
        # 保存 tokenizer
        if save_tokenizer:
            self._save_tokenizer(base_model, output_path)
        
        # 收集合并信息
        merge_info = {
            "base_model": base_model,
            "adapters": adapters,
            "merge_strategy": merge_strategy,
            "weights": weights,
            "output_path": str(output_path),
            "model_size_mb": self._get_model_size(output_path),
            "total_parameters": self._count_parameters(merged_model)
        }
        
        # 保存合并信息
        info_path = output_path / "merge_info.json"
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(merge_info, f, indent=2, ensure_ascii=False)
        
        logger.info("LoRA merge completed successfully")
        return merge_info
    
    def _sequential_merge(self, model, adapters: List[str]):
        """顺序合并适配器。"""
        merged_model = model
        for i, adapter_path in enumerate(adapters):
            logger.info(f"Loading adapter {i+1}/{len(adapters)}: {adapter_path}")
            peft_model = self._load_peft_model(merged_model, adapter_path)
            merged_model = peft_model.merge_and_unload()
        return merged_model
    
    def _weighted_merge(self, model, adapters: List[str], weights: Optional[List[float]] = None):
        """加权合并适配器。"""
        if weights is None:
            weights = [1.0] * len(adapters)
        
        if len(weights) != len(adapters):
            raise ValueError("Weights length must match adapters length")
        
        # 加载所有适配器
        peft_models = []
        for adapter_path in adapters:
            peft_model = self._load_peft_model(model, adapter_path)
            peft_models.append(peft_model)
        
        # 加权合并
        merged_model = self._merge_with_weights(model, peft_models, weights)
        return merged_model
    
    def _average_merge(self, model, adapters: List[str]):
        """平均合并适配器。"""
        weights = [1.0 / len(adapters)] * len(adapters)
        return self._weighted_merge(model, adapters, weights)
    
    def _merge_with_weights(self, base_model, peft_models: List, weights: List[float]):
        """使用权重合并多个 PEFT 模型。"""
        # 获取基础模型状态字典
        merged_state_dict = base_model.state_dict().copy()
        
        # 对每个适配器应用权重
        for peft_model, weight in zip(peft_models, weights):
            adapter_state_dict = peft_model.state_dict()
            
            for name, param in adapter_state_dict.items():
                if name in merged_state_dict:
                    if 'lora_A' in name or 'lora_B' in name:
                        # LoRA 参数需要特殊处理
                        merged_state_dict[name] += weight * param
                    else:
                        # 其他参数直接加权
                        merged_state_dict[name] += weight * param
        
        # 创建新的模型实例
        merged_model = type(base_model)(base_model.config)
        merged_model.load_state_dict(merged_state_dict)
        
        return merged_model
    
    def _save_tokenizer(self, model_path: str, output_path: Path):
        """保存 tokenizer。"""
        try:
            from transformers import AutoTokenizer
            
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=self.trust_remote_code,
                local_files_only=self.local_files_only,
                use_fast=False,
            )
            tokenizer.save_pretrained(output_path)
            logger.info("Tokenizer saved successfully")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")
    
    def _get_model_size(self, model_path: Path) -> float:
        """获取模型大小（MB）。"""
        total_size = 0
        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        return total_size / (1024 * 1024)  # 转换为MB
    
    def _count_parameters(self, model) -> int:
        """计算模型参数数量。"""
        return sum(p.numel() for p in model.parameters())
    
    def analyze_lora_weights(
        self,
        adapter_path: str,
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        分析 LoRA 权重。
        
        Args:
            adapter_path: LoRA 适配器路径
            save_plots: 是否保存图表
            
        Returns:
            分析结果
        """
        logger.info(f"Analyzing LoRA weights from: {adapter_path}")
        
        try:
            from peft import PeftConfig, PeftModel
            
            # 加载适配器配置
            config = PeftConfig.from_pretrained(adapter_path)
            
            # 加载适配器权重
            adapter_state_dict = torch.load(
                Path(adapter_path) / "adapter_model.bin",
                map_location="cpu"
            )
            
            # 分析权重
            analysis_results = {
                "adapter_path": adapter_path,
                "config": config.to_dict(),
                "weight_statistics": {},
                "layer_analysis": {}
            }
            
            # 统计权重信息
            lora_weights = {}
            for name, param in adapter_state_dict.items():
                if 'lora_A' in name or 'lora_B' in name:
                    layer_name = name.split('.lora_')[0]
                    if layer_name not in lora_weights:
                        lora_weights[layer_name] = {}
                    lora_weights[layer_name][name] = param.numpy()
            
            # 计算统计信息
            for layer_name, weights in lora_weights.items():
                layer_stats = {}
                for weight_name, weight_array in weights.items():
                    layer_stats[weight_name] = {
                        "mean": float(np.mean(weight_array)),
                        "std": float(np.std(weight_array)),
                        "min": float(np.min(weight_array)),
                        "max": float(np.max(weight_array)),
                        "shape": list(weight_array.shape)
                    }
                
                analysis_results["layer_analysis"][layer_name] = layer_stats
            
            # 计算总体统计
            all_weights = np.concatenate([
                param.numpy().flatten() 
                for param in adapter_state_dict.values()
                if 'lora_A' in param.name or 'lora_B' in param.name
            ])
            
            analysis_results["weight_statistics"] = {
                "total_parameters": len(all_weights),
                "mean": float(np.mean(all_weights)),
                "std": float(np.std(all_weights)),
                "min": float(np.min(all_weights)),
                "max": float(np.max(all_weights)),
                "percentiles": {
                    "25th": float(np.percentile(all_weights, 25)),
                    "50th": float(np.percentile(all_weights, 50)),
                    "75th": float(np.percentile(all_weights, 75)),
                    "95th": float(np.percentile(all_weights, 95)),
                    "99th": float(np.percentile(all_weights, 99))
                }
            }
            
            # 保存分析结果
            results_path = self.analysis_dir / f"{Path(adapter_path).name}_analysis.json"
            with open(results_path, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            # 生成可视化
            if save_plots:
                self._visualize_lora_weights(adapter_path, lora_weights, analysis_results)
            
            logger.info("LoRA weight analysis completed")
            return analysis_results
            
        except Exception as e:
            logger.error(f"Failed to analyze LoRA weights: {e}")
            raise
    
    def _visualize_lora_weights(
        self,
        adapter_path: str,
        lora_weights: Dict[str, Dict[str, np.ndarray]],
        analysis_results: Dict[str, Any]
    ):
        """可视化 LoRA 权重。"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping visualization")
            return
            
        adapter_name = Path(adapter_path).name
        
        # 1. 权重分布直方图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'LoRA Weight Analysis: {adapter_name}', fontsize=16)
        
        # 收集所有权重
        all_weights = []
        for layer_weights in lora_weights.values():
            for weight_array in layer_weights.values():
                all_weights.extend(weight_array.flatten())
        
        all_weights = np.array(all_weights)
        
        # 权重分布直方图
        axes[0, 0].hist(all_weights, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Weight Distribution')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 对数尺度分布
        axes[0, 1].hist(np.log10(np.abs(all_weights) + 1e-8), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Log-scale Weight Distribution')
        axes[0, 1].set_xlabel('Log10(|Weight| + 1e-8)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 每层权重统计
        layer_names = list(lora_weights.keys())
        layer_means = []
        layer_stds = []
        
        for layer_name in layer_names:
            layer_weights = []
            for weight_array in lora_weights[layer_name].values():
                layer_weights.extend(weight_array.flatten())
            layer_means.append(np.mean(layer_weights))
            layer_stds.append(np.std(layer_weights))
        
        # 层权重均值
        axes[1, 0].bar(range(len(layer_names)), layer_means, alpha=0.7, edgecolor='black')
        axes[1, 0].set_title('Mean Weight by Layer')
        axes[1, 0].set_xlabel('Layer')
        axes[1, 0].set_ylabel('Mean Weight')
        axes[1, 0].set_xticks(range(len(layer_names)))
        axes[1, 0].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 层权重标准差
        axes[1, 1].bar(range(len(layer_names)), layer_stds, alpha=0.7, edgecolor='black', color='orange')
        axes[1, 1].set_title('Weight Std by Layer')
        axes[1, 1].set_xlabel('Layer')
        axes[1, 1].set_ylabel('Weight Std')
        axes[1, 1].set_xticks(range(len(layer_names)))
        axes[1, 1].set_xticklabels([name.split('.')[-1] for name in layer_names], rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        plot_path = self.visualization_dir / f"{adapter_name}_weight_analysis.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 权重热图（如果有多个层）
        if len(layer_names) > 1:
            self._create_weight_heatmap(adapter_name, lora_weights, layer_names)
        
        logger.info(f"Visualizations saved to: {self.visualization_dir}")
    
    def _create_weight_heatmap(
        self,
        adapter_name: str,
        lora_weights: Dict[str, Dict[str, np.ndarray]],
        layer_names: List[str]
    ):
        """创建权重热图。"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        # 创建权重矩阵用于热图
        weight_matrix = []
        layer_labels = []
        
        for layer_name in layer_names:
            layer_weights = lora_weights[layer_name]
            for weight_name, weight_array in layer_weights.items():
                if weight_array.ndim == 2:  # 只处理2D权重
                    weight_matrix.append(weight_array)
                    layer_labels.append(f"{layer_name.split('.')[-1]}_{weight_name.split('.')[-1]}")
        
        if not weight_matrix:
            return
        
        # 创建热图
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 如果权重矩阵大小不同，取平均值
        if len(set(w.shape for w in weight_matrix)) > 1:
            # 计算每层的平均权重
            layer_means = [np.mean(w) for w in weight_matrix]
            layer_stds = [np.std(w) for w in weight_matrix]
            
            # 创建条形图
            x_pos = range(len(layer_labels))
            ax.bar(x_pos, layer_means, yerr=layer_stds, alpha=0.7, capsize=5)
            ax.set_title(f'LoRA Weight Statistics by Layer: {adapter_name}')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Weight Value')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(layer_labels, rotation=45, ha='right')
        else:
            # 所有权重矩阵大小相同，创建热图
            weight_matrix = np.array(weight_matrix)
            im = ax.imshow(weight_matrix, cmap='RdBu_r', aspect='auto')
            ax.set_title(f'LoRA Weight Heatmap: {adapter_name}')
            ax.set_xlabel('Weight Index')
            ax.set_ylabel('Layer')
            ax.set_yticks(range(len(layer_labels)))
            ax.set_yticklabels(layer_labels)
            plt.colorbar(im, ax=ax)
        
        plt.tight_layout()
        
        # 保存热图
        heatmap_path = self.visualization_dir / f"{adapter_name}_weight_heatmap.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_adapters(
        self,
        adapters: List[str],
        save_plots: bool = True
    ) -> Dict[str, Any]:
        """
        比较多个 LoRA 适配器。
        
        Args:
            adapters: 适配器路径列表
            save_plots: 是否保存图表
            
        Returns:
            比较结果
        """
        logger.info(f"Comparing {len(adapters)} LoRA adapters")
        
        comparison_results = {
            "adapters": adapters,
            "comparison_data": {}
        }
        
        # 分析每个适配器
        for adapter_path in adapters:
            adapter_name = Path(adapter_path).name
            analysis = self.analyze_lora_weights(adapter_path, save_plots=False)
            comparison_results["comparison_data"][adapter_name] = analysis
        
        # 创建比较可视化
        if save_plots:
            self._create_comparison_plots(comparison_results)
        
        # 保存比较结果
        comparison_path = self.analysis_dir / "adapter_comparison.json"
        with open(comparison_path, 'w', encoding='utf-8') as f:
            json.dump(comparison_results, f, indent=2, ensure_ascii=False)
        
        logger.info("Adapter comparison completed")
        return comparison_results
    
    def _create_comparison_plots(self, comparison_results: Dict[str, Any]):
        """创建比较图表。"""
        if not MATPLOTLIB_AVAILABLE:
            logger.warning("Matplotlib not available, skipping comparison plots")
            return
            
        adapters = list(comparison_results["comparison_data"].keys())
        comparison_data = comparison_results["comparison_data"]
        
        # 1. 参数数量比较
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('LoRA Adapters Comparison', fontsize=16)
        
        # 参数数量
        param_counts = [
            comparison_data[adapter]["weight_statistics"]["total_parameters"]
            for adapter in adapters
        ]
        
        axes[0, 0].bar(adapters, param_counts, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Total Parameters')
        axes[0, 0].set_xlabel('Adapter')
        axes[0, 0].set_ylabel('Parameter Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        axes[0, 0].grid(True, alpha=0.3)
        
        # 权重均值比较
        weight_means = [
            comparison_data[adapter]["weight_statistics"]["mean"]
            for adapter in adapters
        ]
        
        axes[0, 1].bar(adapters, weight_means, alpha=0.7, edgecolor='black', color='orange')
        axes[0, 1].set_title('Mean Weight Value')
        axes[0, 1].set_xlabel('Adapter')
        axes[0, 1].set_ylabel('Mean Weight')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 权重标准差比较
        weight_stds = [
            comparison_data[adapter]["weight_statistics"]["std"]
            for adapter in adapters
        ]
        
        axes[1, 0].bar(adapters, weight_stds, alpha=0.7, edgecolor='black', color='green')
        axes[1, 0].set_title('Weight Standard Deviation')
        axes[1, 0].set_xlabel('Adapter')
        axes[1, 0].set_ylabel('Weight Std')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # 权重范围比较
        weight_ranges = [
            comparison_data[adapter]["weight_statistics"]["max"] - 
            comparison_data[adapter]["weight_statistics"]["min"]
            for adapter in adapters
        ]
        
        axes[1, 1].bar(adapters, weight_ranges, alpha=0.7, edgecolor='black', color='red')
        axes[1, 1].set_title('Weight Range')
        axes[1, 1].set_xlabel('Adapter')
        axes[1, 1].set_ylabel('Weight Range')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存比较图表
        comparison_plot_path = self.visualization_dir / "adapter_comparison.png"
        plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 权重分布比较
        self._create_distribution_comparison(adapters, comparison_data)
    
    def _create_distribution_comparison(
        self,
        adapters: List[str],
        comparison_data: Dict[str, Any]
    ):
        """创建权重分布比较图。"""
        if not MATPLOTLIB_AVAILABLE:
            return
            
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(adapters)))
        
        for i, adapter in enumerate(adapters):
            # 这里我们使用统计信息来近似分布
            stats = comparison_data[adapter]["weight_statistics"]
            mean = stats["mean"]
            std = stats["std"]
            
            # 生成正态分布样本用于可视化
            x = np.linspace(mean - 4*std, mean + 4*std, 1000)
            y = (1/(std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)
            
            ax.plot(x, y, label=adapter, color=colors[i], linewidth=2)
            ax.fill_between(x, y, alpha=0.3, color=colors[i])
        
        ax.set_title('LoRA Weight Distribution Comparison')
        ax.set_xlabel('Weight Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存分布比较图
        dist_plot_path = self.visualization_dir / "weight_distribution_comparison.png"
        plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_report(self, analysis_results: Dict[str, Any]) -> str:
        """生成分析报告。"""
        report_lines = [
            "# LoRA Analysis Report",
            "",
            f"**Analysis Date**: {torch.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Summary",
            ""
        ]
        
        # 添加统计信息
        if "weight_statistics" in analysis_results:
            stats = analysis_results["weight_statistics"]
            report_lines.extend([
                f"- **Total Parameters**: {stats['total_parameters']:,}",
                f"- **Mean Weight**: {stats['mean']:.6f}",
                f"- **Weight Std**: {stats['std']:.6f}",
                f"- **Weight Range**: [{stats['min']:.6f}, {stats['max']:.6f}]",
                ""
            ])
        
        # 添加层分析
        if "layer_analysis" in analysis_results:
            report_lines.extend([
                "## Layer Analysis",
                ""
            ])
            
            for layer_name, layer_stats in analysis_results["layer_analysis"].items():
                report_lines.extend([
                    f"### {layer_name}",
                    ""
                ])
                
                for weight_name, weight_info in layer_stats.items():
                    report_lines.extend([
                        f"- **{weight_name}**:",
                        f"  - Shape: {weight_info['shape']}",
                        f"  - Mean: {weight_info['mean']:.6f}",
                        f"  - Std: {weight_info['std']:.6f}",
                        f"  - Range: [{weight_info['min']:.6f}, {weight_info['max']:.6f}]",
                        ""
                    ])
        
        report_content = "\n".join(report_lines)
        
        # 保存报告
        report_path = self.analysis_dir / "analysis_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_content


# 便捷函数
def analyze_lora_weights(
    adapter_path: str,
    output_dir: Union[str, Path] = "runs/lora_analysis",
    **kwargs
) -> Dict[str, Any]:
    """便捷函数：分析 LoRA 权重。"""
    analyzer = LoRAAnalyzer(output_dir=output_dir, **kwargs)
    return analyzer.analyze_lora_weights(adapter_path)


def merge_lora_adapters(
    base_model: str,
    adapters: List[str],
    output_name: str,
    output_dir: Union[str, Path] = "runs/lora_analysis",
    **kwargs
) -> Dict[str, Any]:
    """便捷函数：合并 LoRA 适配器。"""
    analyzer = LoRAAnalyzer(output_dir=output_dir, **kwargs)
    return analyzer.merge_adapters(base_model, adapters, output_name, **kwargs)


def compare_lora_adapters(
    adapters: List[str],
    output_dir: Union[str, Path] = "runs/lora_analysis",
    **kwargs
) -> Dict[str, Any]:
    """便捷函数：比较 LoRA 适配器。"""
    analyzer = LoRAAnalyzer(output_dir=output_dir, **kwargs)
    return analyzer.compare_adapters(adapters, **kwargs)
