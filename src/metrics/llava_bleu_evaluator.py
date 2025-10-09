"""
LLaVA-1.5 模型 BLEU 评估器

专门为 LLaVA-1.5 多模态模型设计的 BLEU 评估工具。
LLaVA-1.5 是一个视觉-语言模型，能够根据图像生成描述性文本。

主要功能：
1. 处理 LLaVA-1.5 模型的输出格式
2. 提供批量评估功能
3. 支持多种文本预处理选项
4. 生成详细的评估报告和可视化
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union
from datetime import datetime
import pandas as pd
from pathlib import Path

# 导入我们实现的 BLEU 模块
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from bleu import bleu_score, corpus_bleu, tokenize


class LLaVABLEUEvaluator:
    """
    LLaVA-1.5 模型专用的 BLEU 评估器
    
    特点：
    - 专门处理视觉描述任务的文本生成评估
    - 支持多参考句子评估
    - 提供详细的分析报告
    - 自动保存评估结果
    """
    
    def __init__(self, 
                 output_dir: str = "runs",
                 run_id: Optional[str] = None,
                 smoothing: bool = True,
                 weights: List[float] = None):
        """
        初始化评估器
        
        Args:
            output_dir: 输出目录
            run_id: 运行ID，如果为None则自动生成
            smoothing: 是否使用平滑处理
            weights: BLEU-4权重，默认为均等权重
        """
        self.smoothing = smoothing
        self.weights = weights or [0.25, 0.25, 0.25, 0.25]
        
        # 设置输出目录
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.output_dir = Path(output_dir) / run_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化结果存储
        self.evaluation_results = []
        self.metadata = {
            "model": "LLaVA-1.5",
            "metric": "BLEU-4",
            "smoothing": smoothing,
            "weights": weights,
            "created_at": datetime.now().isoformat()
        }
    
    def preprocess_text(self, text: str, 
                       remove_special_tokens: bool = True,
                       normalize_whitespace: bool = True,
                       lowercase: bool = True) -> str:
        """
        预处理 LLaVA 生成的文本
        
        LLaVA-1.5 的输出可能包含特殊标记和格式，需要清理
        
        Args:
            text: 原始文本
            remove_special_tokens: 是否移除特殊标记
            normalize_whitespace: 是否标准化空白字符
            lowercase: 是否转换为小写
            
        Returns:
            清理后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 移除常见的特殊标记
        if remove_special_tokens:
            # LLaVA 可能的特殊标记
            special_tokens = [
                "<image>", "</image>", 
                "<s>", "</s>",
                "<unk>", "<pad>",
                "USER:", "ASSISTANT:",
                "[IMG]", "[/IMG]"
            ]
            for token in special_tokens:
                text = text.replace(token, "")
        
        # 标准化空白字符
        if normalize_whitespace:
            import re
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        # 转换为小写
        if lowercase:
            text = text.lower()
        
        return text
    
    def evaluate_single(self, 
                       candidate: str, 
                       references: List[str],
                       image_id: Optional[str] = None,
                       preprocess: bool = True) -> Dict:
        """
        评估单个样本
        
        Args:
            candidate: LLaVA 生成的候选文本
            references: 参考文本列表
            image_id: 图像ID（可选）
            preprocess: 是否预处理文本
            
        Returns:
            评估结果字典
        """
        # 预处理文本
        if preprocess:
            candidate = self.preprocess_text(candidate)
            references = [self.preprocess_text(ref) for ref in references]
        
        # 计算 BLEU 分数
        bleu = bleu_score(
            candidate=candidate,
            references=references,
            weights=self.weights,
            smoothing=self.smoothing
        )
        
        # 计算各个 n-gram 的精确度（用于详细分析）
        from .bleu import modified_precision
        candidate_tokens = tokenize(candidate)
        reference_tokens_list = [tokenize(ref) for ref in references]
        
        precisions = []
        for n in range(1, 5):
            precision = modified_precision(candidate_tokens, reference_tokens_list, n)
            precisions.append(precision)
        
        # 构建结果
        result = {
            "image_id": image_id,
            "candidate": candidate,
            "references": references,
            "bleu_score": bleu,
            "precisions": {
                "1-gram": precisions[0],
                "2-gram": precisions[1],
                "3-gram": precisions[2],
                "4-gram": precisions[3]
            },
            "candidate_length": len(candidate_tokens),
            "reference_lengths": [len(tokenize(ref)) for ref in references]
        }
        
        return result
    
    def evaluate_batch(self, 
                      candidates: List[str],
                      references_list: List[List[str]],
                      image_ids: Optional[List[str]] = None,
                      preprocess: bool = True,
                      save_results: bool = True) -> Dict:
        """
        批量评估
        
        Args:
            candidates: 候选文本列表
            references_list: 每个候选对应的参考文本列表
            image_ids: 图像ID列表（可选）
            preprocess: 是否预处理文本
            save_results: 是否保存结果
            
        Returns:
            批量评估结果
        """
        if len(candidates) != len(references_list):
            raise ValueError("候选文本数量必须与参考文本组数量相等")
        
        if image_ids and len(image_ids) != len(candidates):
            raise ValueError("图像ID数量必须与候选文本数量相等")
        
        # 单个样本评估
        individual_results = []
        processed_candidates = []
        processed_references_list = []
        
        for i, (candidate, references) in enumerate(zip(candidates, references_list)):
            image_id = image_ids[i] if image_ids else f"sample_{i}"
            
            result = self.evaluate_single(
                candidate=candidate,
                references=references,
                image_id=image_id,
                preprocess=preprocess
            )
            
            individual_results.append(result)
            processed_candidates.append(result["candidate"])
            processed_references_list.append(result["references"])
        
        # 语料库级别评估
        corpus_bleu_score = corpus_bleu(
            candidates=processed_candidates,
            references_list=processed_references_list,
            weights=self.weights,
            smoothing=self.smoothing
        )
        
        # 统计分析
        bleu_scores = [r["bleu_score"] for r in individual_results]
        
        batch_results = {
            "corpus_bleu": corpus_bleu_score,
            "individual_results": individual_results,
            "statistics": {
                "mean_bleu": np.mean(bleu_scores),
                "std_bleu": np.std(bleu_scores),
                "min_bleu": np.min(bleu_scores),
                "max_bleu": np.max(bleu_scores),
                "median_bleu": np.median(bleu_scores),
                "total_samples": len(candidates)
            },
            "metadata": self.metadata
        }
        
        # 保存结果
        if save_results:
            self.save_evaluation_results(batch_results)
        
        self.evaluation_results.append(batch_results)
        return batch_results
    
    def save_evaluation_results(self, results: Dict, filename: str = None):
        """
        保存评估结果到文件
        
        Args:
            results: 评估结果
            filename: 文件名，如果为None则自动生成
        """
        if filename is None:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"bleu_evaluation_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            return obj
        
        # 深度转换结果
        import copy
        serializable_results = copy.deepcopy(results)
        
        def recursive_convert(obj):
            if isinstance(obj, dict):
                return {k: recursive_convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [recursive_convert(item) for item in obj]
            else:
                return convert_numpy(obj)
        
        serializable_results = recursive_convert(serializable_results)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"评估结果已保存到: {filepath}")
    
    def create_visualization(self, results: Dict, save_plots: bool = True):
        """
        创建评估结果可视化
        
        Args:
            results: 评估结果
            save_plots: 是否保存图表
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 提取数据
        individual_results = results["individual_results"]
        bleu_scores = [r["bleu_score"] for r in individual_results]
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'LLaVA-1.5 BLEU-4 Evaluation Results\nCorpus BLEU: {results["corpus_bleu"]:.4f}', 
                     fontsize=16, fontweight='bold')
        
        # 1. BLEU分数分布直方图
        axes[0, 0].hist(bleu_scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(np.mean(bleu_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(bleu_scores):.4f}')
        axes[0, 0].set_xlabel('BLEU-4 Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('BLEU Score Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. BLEU分数序列图
        axes[0, 1].plot(range(len(bleu_scores)), bleu_scores, 'o-', alpha=0.7, markersize=4)
        axes[0, 1].axhline(np.mean(bleu_scores), color='red', linestyle='--', 
                          label=f'Mean: {np.mean(bleu_scores):.4f}')
        axes[0, 1].set_xlabel('Sample Index')
        axes[0, 1].set_ylabel('BLEU-4 Score')
        axes[0, 1].set_title('BLEU Scores by Sample')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. N-gram精确度分析
        precision_data = {
            '1-gram': [r["precisions"]["1-gram"] for r in individual_results],
            '2-gram': [r["precisions"]["2-gram"] for r in individual_results],
            '3-gram': [r["precisions"]["3-gram"] for r in individual_results],
            '4-gram': [r["precisions"]["4-gram"] for r in individual_results]
        }
        
        precision_means = [np.mean(precision_data[key]) for key in precision_data.keys()]
        x_pos = range(len(precision_means))
        
        bars = axes[1, 0].bar(x_pos, precision_means, alpha=0.7, 
                             color=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99'])
        axes[1, 0].set_xlabel('N-gram Type')
        axes[1, 0].set_ylabel('Average Precision')
        axes[1, 0].set_title('Average N-gram Precisions')
        axes[1, 0].set_xticks(x_pos)
        axes[1, 0].set_xticklabels(precision_data.keys())
        axes[1, 0].grid(True, alpha=0.3)
        
        # 添加数值标签
        for bar, value in zip(bars, precision_means):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # 4. 文本长度 vs BLEU分数散点图
        candidate_lengths = [r["candidate_length"] for r in individual_results]
        axes[1, 1].scatter(candidate_lengths, bleu_scores, alpha=0.6, s=30)
        axes[1, 1].set_xlabel('Candidate Text Length (tokens)')
        axes[1, 1].set_ylabel('BLEU-4 Score')
        axes[1, 1].set_title('Text Length vs BLEU Score')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 添加趋势线
        z = np.polyfit(candidate_lengths, bleu_scores, 1)
        p = np.poly1d(z)
        axes[1, 1].plot(sorted(candidate_lengths), p(sorted(candidate_lengths)), 
                       "r--", alpha=0.8, label=f'Trend: y={z[0]:.4f}x+{z[1]:.4f}')
        axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_plots:
            plot_path = self.output_dir / f"bleu_evaluation_plots_{datetime.now().strftime('%H%M%S')}.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            print(f"可视化图表已保存到: {plot_path}")
        
        plt.show()
    
    def generate_report(self, results: Dict) -> str:
        """
        生成详细的评估报告
        
        Args:
            results: 评估结果
            
        Returns:
            报告文本
        """
        stats = results["statistics"]
        
        report = f"""
# LLaVA-1.5 BLEU-4 评估报告

## 基本信息
- 模型: {self.metadata['model']}
- 评估指标: {self.metadata['metric']}
- 评估时间: {self.metadata['created_at']}
- 样本数量: {stats['total_samples']}
- 平滑处理: {'启用' if self.smoothing else '禁用'}

## 整体评估结果
- **语料库BLEU分数**: {results['corpus_bleu']:.4f}
- **平均BLEU分数**: {stats['mean_bleu']:.4f} ± {stats['std_bleu']:.4f}
- **BLEU分数范围**: {stats['min_bleu']:.4f} - {stats['max_bleu']:.4f}
- **中位数BLEU分数**: {stats['median_bleu']:.4f}

## 性能分析
"""
        
        # 性能等级分析
        bleu_scores = [r["bleu_score"] for r in results["individual_results"]]
        excellent = sum(1 for score in bleu_scores if score >= 0.4)
        good = sum(1 for score in bleu_scores if 0.2 <= score < 0.4)
        fair = sum(1 for score in bleu_scores if 0.1 <= score < 0.2)
        poor = sum(1 for score in bleu_scores if score < 0.1)
        
        report += f"""
### BLEU分数分布
- 优秀 (≥0.4): {excellent} 样本 ({excellent/len(bleu_scores)*100:.1f}%)
- 良好 (0.2-0.4): {good} 样本 ({good/len(bleu_scores)*100:.1f}%)
- 一般 (0.1-0.2): {fair} 样本 ({fair/len(bleu_scores)*100:.1f}%)
- 较差 (<0.1): {poor} 样本 ({poor/len(bleu_scores)*100:.1f}%)

### N-gram精确度分析
"""
        
        # N-gram分析
        precision_stats = {}
        for n in ['1-gram', '2-gram', '3-gram', '4-gram']:
            precisions = [r["precisions"][n] for r in results["individual_results"]]
            precision_stats[n] = {
                'mean': np.mean(precisions),
                'std': np.std(precisions)
            }
            report += f"- {n}平均精确度: {precision_stats[n]['mean']:.4f} ± {precision_stats[n]['std']:.4f}\n"
        
        report += f"""

## 建议和改进方向

### 基于评估结果的建议:
"""
        
        if results['corpus_bleu'] >= 0.3:
            report += "- ✅ 模型整体表现良好，生成文本质量较高\n"
        elif results['corpus_bleu'] >= 0.2:
            report += "- ⚠️ 模型表现中等，建议优化训练数据或调整模型参数\n"
        else:
            report += "- ❌ 模型表现需要改进，建议检查训练策略和数据质量\n"
        
        if precision_stats['1-gram']['mean'] < 0.5:
            report += "- 建议改进词汇选择和基础语言理解能力\n"
        
        if precision_stats['4-gram']['mean'] < 0.1:
            report += "- 建议提高长序列生成的连贯性和流畅度\n"
        
        report += f"""
### 技术建议:
- 考虑使用更大的训练数据集
- 尝试不同的解码策略（beam search, nucleus sampling等）
- 调整模型的温度参数以平衡创造性和准确性
- 使用多样化的参考文本进行训练

---
报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report
    
    def save_report(self, results: Dict, filename: str = None):
        """
        保存评估报告
        
        Args:
            results: 评估结果
            filename: 文件名
        """
        if filename is None:
            timestamp = datetime.now().strftime("%H%M%S")
            filename = f"bleu_evaluation_report_{timestamp}.md"
        
        report = self.generate_report(results)
        report_path = self.output_dir / filename
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"评估报告已保存到: {report_path}")
        return report


# 使用示例和最佳实践
def example_usage():
    """
    LLaVA-1.5 BLEU评估的使用示例
    """
    print("=== LLaVA-1.5 BLEU评估使用示例 ===\n")
    
    # 1. 创建评估器
    evaluator = LLaVABLEUEvaluator(
        output_dir="runs",
        run_id="llava_test_001",
        smoothing=True
    )
    
    # 2. 准备测试数据（模拟LLaVA-1.5的输出）
    # 这些是图像描述任务的示例
    candidates = [
        "A cat is sitting on a wooden table next to a laptop computer.",
        "The image shows a beautiful sunset over the ocean with orange and pink colors in the sky.",
        "A person is walking down a busy street with tall buildings on both sides.",
        "There are several books stacked on a shelf in what appears to be a library.",
        "A red car is parked in front of a modern house with large windows."
    ]
    
    # 对应的参考描述（通常来自人工标注）
    references_list = [
        [
            "A cat sits on a wooden desk beside a laptop.",
            "There is a cat on the table next to a computer.",
            "A feline is positioned on a wooden surface near a laptop."
        ],
        [
            "The sunset creates beautiful orange and pink hues across the ocean horizon.",
            "A stunning sunset with warm colors reflects over the water.",
            "The sky displays vibrant sunset colors above the sea."
        ],
        [
            "A person walks along a street lined with tall buildings.",
            "Someone is walking down an urban street with skyscrapers.",
            "A pedestrian moves through a city street surrounded by high-rise buildings."
        ],
        [
            "Multiple books are arranged on a library shelf.",
            "Several books are stacked together on a bookshelf.",
            "Books are organized on shelves in a library setting."
        ],
        [
            "A red automobile is parked outside a contemporary home.",
            "A red vehicle sits in front of a modern residential building.",
            "A red car is positioned before a house with contemporary architecture."
        ]
    ]
    
    # 图像ID（可选）
    image_ids = [f"img_{i:03d}" for i in range(len(candidates))]
    
    # 3. 执行批量评估
    print("正在执行批量评估...")
    results = evaluator.evaluate_batch(
        candidates=candidates,
        references_list=references_list,
        image_ids=image_ids,
        preprocess=True,
        save_results=True
    )
    
    # 4. 显示结果摘要
    print(f"\n=== 评估结果摘要 ===")
    print(f"语料库BLEU分数: {results['corpus_bleu']:.4f}")
    print(f"平均BLEU分数: {results['statistics']['mean_bleu']:.4f}")
    print(f"标准差: {results['statistics']['std_bleu']:.4f}")
    print(f"评估样本数: {results['statistics']['total_samples']}")
    
    # 5. 生成可视化
    print(f"\n正在生成可视化图表...")
    evaluator.create_visualization(results, save_plots=True)
    
    # 6. 生成并保存报告
    print(f"\n正在生成评估报告...")
    evaluator.save_report(results)
    
    print(f"\n✅ 评估完成！所有结果已保存到: {evaluator.output_dir}")
    
    return results


if __name__ == "__main__":
    # 运行示例
    example_results = example_usage()